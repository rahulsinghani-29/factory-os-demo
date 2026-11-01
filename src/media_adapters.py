"""Local-only voice/OCR adapters. No cloud SDKs and no implicit model downloads."""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time


class MediaDependencyError(RuntimeError):
    pass


class MediaProcessingError(RuntimeError):
    pass


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def preserve_upload(content, suffix, root):
    suffix = suffix.lower()
    if suffix not in (".wav", ".mp3", ".m4a", ".aac", ".png", ".jpg", ".jpeg"):
        raise ValueError("Only common audio files and JPG/PNG photos are accepted")
    if not content or len(content) > 25 * 1024 * 1024:
        raise ValueError("File is empty or exceeds the 25 MB demo limit")
    digest = sha256_bytes(content)
    directory = pathlib.Path(root) / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (digest + suffix)
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=str(directory), prefix="upload-", delete=False) as handle:
            temp = pathlib.Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(destination)
    return {"sha256": digest, "path": str(destination), "suffix": suffix,
            "size_bytes": len(content)}


def transcribe_with_local_whisper(path, executable="opensuperwhisper"):
    """Use the installed local OpenSuperWhisper CLI (which must have a model configured)."""
    binary = shutil.which(executable)
    if not binary:
        raise MediaDependencyError("Local Whisper CLI missing: install/configure OpenSuperWhisper and ensure `opensuperwhisper` is on PATH. No audio was sent elsewhere.")
    started = time.monotonic()
    try:
        result = subprocess.run([binary, "transcribe", str(path), "--json"],
                                capture_output=True, text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError("Local Whisper transcription timed out after 300 seconds") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()[-4:]
        message = " | ".join(detail) or "transcriber returned a nonzero status"
        raise MediaProcessingError("Local Whisper failed: " + message)
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise MediaProcessingError("Local Whisper returned invalid JSON") from exc
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise MediaProcessingError("Local Whisper returned no speech text; no count proposal created")
    # OpenSuperWhisper's CLI currently returns text but no word/segment timestamps.
    segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    return {"text": text.strip(), "segments": segments,
            "segment_timing": "provided by local CLI" if segments else "not provided by installed CLI",
            "engine": "OpenSuperWhisper local Whisper engine",
            "engine_reported_model": payload.get("model") or payload.get("model_name"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "raw_cli_metadata": {k: v for k, v in payload.items() if k != "file"}}


def _row_ids(blocks):
    """Group OCR line boxes by vertical overlap; this is a review hint, not table truth."""
    indexed = []
    for block in sorted(blocks, key=lambda b: ((b["bbox"][1] + b["bbox"][3]) / 2, b["bbox"][0])):
        y0, y1 = block["bbox"][1], block["bbox"][3]
        center = (y0 + y1) / 2
        height = max(1.0, y1 - y0)
        target = None
        for row in indexed:
            overlap = max(0.0, min(y1, row["y1"]) - max(y0, row["y0"]))
            if overlap / min(height, max(1.0, row["y1"] - row["y0"])) >= 0.45 or abs(center - row["center"]) <= 0.35 * max(height, row["height"]):
                target = row
                break
        if target is None:
            target = {"blocks": [], "y0": y0, "y1": y1, "center": center, "height": height}
            indexed.append(target)
        target["blocks"].append(block)
        target["y0"] = min(target["y0"], y0)
        target["y1"] = max(target["y1"], y1)
        target["center"] = sum((b["bbox"][1] + b["bbox"][3]) / 2 for b in target["blocks"]) / len(target["blocks"])
        target["height"] = max(target["height"], height)
    indexed.sort(key=lambda r: r["center"])
    for row_no, row in enumerate(indexed, 1):
        row_id = f"OCR-ROW-{row_no:03d}"
        row["blocks"].sort(key=lambda b: b["bbox"][0])
        for block in row["blocks"]:
            block["row_id"] = row_id
    return indexed


def _extract_paddle_result(result):
    blocks = []
    for page_no, page in enumerate(result, 1):
        data = getattr(page, "json", None)
        if callable(data):
            data = data()
        if not isinstance(data, dict):
            raise MediaProcessingError("PaddleOCR returned an unsupported result object")
        res = data.get("res", data)
        polygons = res.get("dt_polys") or res.get("rec_polys") or []
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        if len(polygons) != len(texts):
            raise MediaProcessingError("OCR text and bounding-box counts do not match")
        for idx, (poly, text) in enumerate(zip(polygons, texts), 1):
            points = [[float(point[0]), float(point[1])] for point in poly]
            if not points or not str(text).strip():
                continue
            blocks.append({"block_id": f"OCR-P{page_no}-B{idx:03d}", "page": page_no,
                           "text": str(text), "bbox": [min(p[0] for p in points), min(p[1] for p in points),
                                                         max(p[0] for p in points), max(p[1] for p in points)],
                           "polygon": points, "ocr_score": float(scores[idx - 1]) if idx <= len(scores) else None})
    return blocks


def ocr_photo_with_local_paddle(path):
    """Local PaddleOCR PP-OCRv5; requires existing Tamil det/rec models to avoid downloads."""
    try:
        import paddle
        import paddleocr
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise MediaDependencyError("Photo OCR unavailable: install PaddlePaddle + PaddleOCR 3.x and the local Tamil-English PP-OCRv5 models. No OCR or text-only fallback was claimed.") from exc
    det_dir = os.environ.get("QUINCE_OCR_DET_MODEL_DIR")
    rec_dir = os.environ.get("QUINCE_OCR_TA_MODEL_DIR")
    if not det_dir or not rec_dir or not pathlib.Path(det_dir).is_dir() or not pathlib.Path(rec_dir).is_dir():
        raise MediaDependencyError("Photo OCR models are not configured locally. Set QUINCE_OCR_DET_MODEL_DIR and QUINCE_OCR_TA_MODEL_DIR to existing model directories. This adapter will not download weights automatically.")
    try:
        ocr = PaddleOCR(lang="ta", ocr_version="PP-OCRv5",
                        text_detection_model_name="PP-OCRv5_mobile_det",
                        text_recognition_model_name="ta_PP-OCRv5_mobile_rec",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False, use_textline_orientation=False,
                        text_detection_model_dir=det_dir,
                        text_recognition_model_dir=rec_dir)
        blocks = _extract_paddle_result(ocr.predict(str(path)))
    except Exception as exc:
        raise MediaProcessingError(f"Local PaddleOCR failed closed: {type(exc).__name__}: {exc}") from exc
    rows = _row_ids(blocks)
    return {"engine": "PaddleOCR PP-OCRv5 local", "paddle_version": getattr(paddle, "__version__", None),
            "paddleocr_version": getattr(paddleocr, "__version__", None), "language_model": "Tamil + English",
            "blocks": blocks, "rows": [{"row_id": f"OCR-ROW-{idx:03d}",
                         "block_ids": [b["block_id"] for b in row["blocks"]],
                         "text": " | ".join(b["text"] for b in row["blocks"]),
                         "bbox": [min(b["bbox"][0] for b in row["blocks"]), row["y0"],
                                  max(b["bbox"][2] for b in row["blocks"]), row["y1"]],
                         "association_status": "heuristic row grouping; verify visually"}
                        for idx, row in enumerate(rows, 1)],
            "table_cell_mapping": "not inferred without clear reviewed header/cell alignment"}


def remarks_crop_box(image_size, blocks):
    """Find the handwritten remarks column from printed headers and its text boxes."""
    width, height = image_size
    if blocks is None:
        # The built-in synthetic handoff form has a fixed printed layout. This
        # lets vision OCR begin while Paddle reads table fields in parallel.
        return (int(width * 0.733), int(height * 0.458),
                int(width * 0.974), int(height * 0.687))
    remarks = [b for b in blocks if b.get("text", "").strip().lower() == "remarks"]
    footer = [b for b in blocks if "supervisor sign" in b.get("text", "").lower()]
    if not remarks or not footer:
        raise MediaProcessingError("Could not locate both Remarks and Supervisor Sign headers; focused remarks OCR was held")
    header = min(remarks, key=lambda b: b["bbox"][1])
    footer_y = min(b["bbox"][1] for b in footer)
    hx0, hy0, hx1, hy1 = header["bbox"]
    # Use the right-hand remarks column, bounded above the printed supervisor footer.
    x0 = max(0, int(hx0 - max(100, (hx1 - hx0) * 1.2)))
    x1 = min(width, int(width - max(30, width * 0.025)))
    bilingual_headers = [b for b in blocks if "குறிப்புகள்" in b.get("text", "")]
    y0 = max(0, int(max([hy1] + [b["bbox"][3] for b in bilingual_headers]) + 8))
    candidates = [b["bbox"][3] for b in blocks
                  if b.get("bbox") and b["bbox"][1] > hy1 and b["bbox"][1] < footer_y
                  and b["bbox"][2] >= x0 and b["bbox"][0] <= x1]
    if not candidates:
        raise MediaProcessingError("No OCR text boxes were found below the Remarks header; focused remarks OCR was held")
    y1 = min(footer_y - 20, int(max(candidates) + 24))
    if x1 <= x0 or y1 <= y0:
        raise MediaProcessingError("Remarks region has invalid dimensions; focused remarks OCR was held")
    return x0, y0, x1, y1


def read_remarks_with_local_vision(path, blocks, log_path, crop_path):
    """Transcribe the remarks area with an installed Ollama vision model."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError as exc:
        raise MediaDependencyError("Pillow is required for focused remarks OCR") from exc
    from llm import call_ollama_vision_json
    model = os.environ.get("QUINCE_OLLAMA_VISION_MODEL", "gemma4:latest")
    image = Image.open(path).convert("RGB")
    box = remarks_crop_box(image.size, blocks)
    crop = image.crop(box)
    crop = crop.resize((int(crop.width * 1.5), int(crop.height * 1.5)))
    crop = ImageEnhance.Contrast(crop).enhance(1.15).filter(ImageFilter.SHARPEN)
    crop_path = pathlib.Path(crop_path)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path)
    prompt = ("Read only the handwritten remarks in this crop. Transcribe Hindi and Tamil exactly where readable. "
              "Give a short English meaning. Do not infer missing words or facts. Use an empty string for an unreadable language. "
              "List unclear words in uncertain_spans. Return JSON with exactly hindi, tamil, english_meaning, uncertain_spans.")
    system = "You are a careful local OCR reader. Printed labels are context, not remarks. Never guess unreadable handwriting."
    result = call_ollama_vision_json(system, prompt, crop_path, model, log_path,
                                    timeout=int(os.environ.get("QUINCE_OLLAMA_VISION_TIMEOUT", "90")))
    try:
        parsed = json.loads(result["text"])
    except (TypeError, ValueError) as exc:
        raise MediaProcessingError("Local vision OCR returned invalid JSON; remarks were held") from exc
    # Ollama's JSON mode can serialize an empty array as an empty string. Treat
    # only that empty sentinel as no uncertain spans; never discard non-empty text.
    if isinstance(parsed, dict) and parsed.get("uncertain_spans") == "":
        parsed["uncertain_spans"] = []
    required = {"hindi", "tamil", "english_meaning", "uncertain_spans"}
    if (not isinstance(parsed, dict) or set(parsed) != required or
            any(not isinstance(parsed[k], str) for k in ("hindi", "tamil", "english_meaning")) or
            not isinstance(parsed["uncertain_spans"], list) or
            any(not isinstance(item, str) for item in parsed["uncertain_spans"]) or
            not any(parsed[k].strip() for k in ("hindi", "tamil", "english_meaning"))):
        raise MediaProcessingError("Local vision OCR returned an unsupported remarks schema; remarks were held")
    return {"engine": "Ollama local vision", "model": result["model"], "endpoint": result["endpoint"],
            "latency_seconds": result["latency_seconds"], "log_file": result["log_file"],
            "crop_file": str(crop_path), "crop_bbox": list(box), **parsed}


def normalize_po_identifier(value):
    """Remove OCR-inserted spacing in a printed PO token; never change glyphs."""
    return re.sub(r"\s+", "", value).strip() or None


def read_po_and_line_with_local_vision(path, blocks, log_path, crop_path):
    """Read only the printed PO and line fields, retaining the crop and model trace."""
    from PIL import Image, ImageEnhance, ImageFilter
    from llm import call_ollama_vision_json
    image = Image.open(path).convert("RGB")
    if blocks is None:
        width, height = image.size
        boxes = [(int(width * 0.09), int(height * 0.265), int(width * 0.36), int(height * 0.345)),
                 (int(width * 0.48), int(height * 0.19), int(width * 0.73), int(height * 0.275))]
    else:
        po_headers = [b for b in blocks if any(term in b.get("text", "").lower()
                      for term in ("po no", "po எண்"))]
        line_headers = [b for b in blocks if any(term in b.get("text", "").lower()
                        for term in ("line / section", "வரி / பகுதி"))]
        if not po_headers or not line_headers:
            raise MediaProcessingError("PO and Line / Section labels were not both located; identifiers are unknown")
        po_header = min(po_headers, key=lambda b: b["bbox"][1])
        line_header = min(line_headers, key=lambda b: b["bbox"][1])
        px0, py0, px1, py1 = po_header["bbox"]
        lx0, ly0, lx1, ly1 = line_header["bbox"]
        boxes = [(max(0, int(px0 - 24)), max(0, int(py0 - 20)), min(image.width, int(px1 + 270)),
                  min(image.height, int(py1 + 75))),
                 (max(0, int(lx0 - 15)), max(0, int(ly0 - 20)), min(image.width, int(lx1 + 240)),
                  min(image.height, int(ly1 + 70)))]
    # Keep the two labeled cells separate. The earlier wide crop mixed the PO
    # with adjacent fields and Gemma confused the letter O with a zero.
    crops = []
    for box in boxes:
        crop = image.crop(box)
        crop = crop.resize((int(crop.width * 2), int(crop.height * 2)))
        crop = ImageEnhance.Contrast(crop).enhance(1.12).filter(ImageFilter.SHARPEN)
        crops.append(crop)
    canvas_width = max(crop.width for crop in crops)
    gap = 18
    from PIL import Image as PILImage
    composite = PILImage.new("RGB", (canvas_width, sum(crop.height for crop in crops) + gap), "white")
    y = 0
    for crop in crops:
        composite.paste(crop, (0, y))
        y += crop.height + gap
    crop_path = pathlib.Path(crop_path)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(crop_path)
    system = ("Read the values beside the printed PO No. and Line / Section labels. Return the handwritten/filled values exactly. "
              "Do not infer values from another field or from factory records. If a value is not clear, use an empty string and "
              "list the field name in uncertain_fields.")
    prompt = ("This image contains two enlarged crops, in order: first the PO No. cell, then the Line / Section cell. "
              "Return JSON with exactly po_id, line, uncertain_fields. Read only the value in each labeled cell. "
              "Pay close attention to letter O versus digit zero. Do not copy a zero if the handwriting is a letter O.")
    model = os.environ.get("QUINCE_OLLAMA_VISION_MODEL", "gemma4:latest")
    result = call_ollama_vision_json(system, prompt, crop_path, model, log_path,
                                    timeout=int(os.environ.get("QUINCE_OLLAMA_VISION_TIMEOUT", "90")), max_tokens=100)
    try:
        parsed = json.loads(result["text"])
    except (TypeError, ValueError) as exc:
        raise MediaProcessingError("Local vision did not return valid PO and line JSON") from exc
    if isinstance(parsed, dict) and parsed.get("uncertain_fields") == "":
        parsed["uncertain_fields"] = []
    required = {"po_id", "line", "uncertain_fields"}
    if (not isinstance(parsed, dict) or set(parsed) != required or
            not isinstance(parsed["po_id"], str) or not isinstance(parsed["line"], str) or
            not isinstance(parsed["uncertain_fields"], list) or
            any(not isinstance(item, str) for item in parsed["uncertain_fields"])):
        raise MediaProcessingError("Local vision returned an unsupported PO and line schema")
    parsed["po_id"] = normalize_po_identifier(parsed["po_id"])
    parsed["line"] = parsed["line"].strip() or None
    if not parsed["po_id"] and "po_id" not in parsed["uncertain_fields"]:
        parsed["uncertain_fields"].append("po_id")
    if not parsed["line"] and "line" not in parsed["uncertain_fields"]:
        parsed["uncertain_fields"].append("line")
    return {"engine": "Ollama local vision", "model": result["model"], "endpoint": result["endpoint"],
            "latency_seconds": result["latency_seconds"], "log_file": result["log_file"],
            "crop_file": str(crop_path), "crop_bbox": [list(box) for box in boxes], **parsed}
