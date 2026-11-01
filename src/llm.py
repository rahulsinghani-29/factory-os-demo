"""Minimal, dependency-free LLM client (stdlib only).

Supports Gemini and OpenAI-compatible APIs, selected by environment variables. The
local run script configures the OpenAI-compatible API to local Ollama.

Every call is logged verbatim (request + response) to logs/ so the recording and the
reviewer can see the real API traffic. No SDKs: the whole integration is readable here.
"""
import base64, json, os, time, urllib.request, urllib.error, pathlib, urllib.parse
from observability import langfuse_generation

LOGDIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
LOGDIR.mkdir(exist_ok=True)

class LLMError(Exception):
    pass

def _post_json(url, payload, headers=None, timeout=120):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        raise LLMError(f"HTTP {e.code} from {url}: {body}")
    except urllib.error.URLError as e:
        raise LLMError(f"Connection error calling {url}: {e}")

def call_llm(stage: str, system: str, user: str, json_mode: bool = True) -> dict:
    """One real LLM call. Returns {'text': str, 'meta': {...}}. Logs raw traffic."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    ts = time.strftime("%Y%m%d-%H%M%S")
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set")
        model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.1, **({"response_mime_type": "application/json"} if json_mode else {})},
        }
        resp = _post_json(url, payload)
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
        meta = {"provider": "gemini", "model": model, "usage": resp.get("usageMetadata", {})}
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,
            **({"response_format": {"type": "json_object"}} if json_mode else {}),
        }
        try:
            resp = _post_json(url, payload, headers={"Authorization": f"Bearer {key}"})
        except LLMError as e:
            # some local servers (e.g. older Ollama) reject response_format; retry without it
            if json_mode and ("response_format" in str(e) or "400" in str(e)):
                payload.pop("response_format", None)
                resp = _post_json(url, payload, headers={"Authorization": f"Bearer {key}"})
            else:
                raise
        text = resp["choices"][0]["message"]["content"]
        meta = {"provider": "openai", "model": model, "usage": resp.get("usage", {})}
    else:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}' (use 'gemini' or 'openai')")
    parsed_url = urllib.parse.urlsplit(url)
    meta["endpoint"] = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    log_path = LOGDIR / f"{ts}-{stage}.json"
    meta["log_file"] = str(log_path.relative_to(LOGDIR.parent))
    log_path.write_text(json.dumps(
        {"stage": stage, "request": {"system": system, "user": user}, "response_text": text, "meta": meta}, indent=2))
    return {"text": text, "meta": meta}


def call_ollama_tools(messages, tools, model, log_path, round_number, timeout=120):
    """One local Ollama OpenAI-compatible tool-calling round; no cloud fallback."""
    base = "http://localhost:11434/v1"
    url = base + "/chat/completions"
    payload = {"model": model, "messages": messages, "tools": tools,
               "tool_choice": "auto", "temperature": 0, "stream": False}
    response = _post_json(url, payload, headers={"Authorization": "Bearer ollama"}, timeout=timeout)
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Ollama response did not contain a chat message") from exc
    safe_log = {"round": round_number, "endpoint": url, "model_requested": model,
                "model_returned": response.get("model"), "request": payload,
                "response": response}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(safe_log, indent=2, ensure_ascii=False) + "\n")
    return {"message": message, "model": response.get("model", model),
            "usage": response.get("usage"), "endpoint": url}


def call_ollama_json(system, user, model, log_path, timeout=180, max_tokens=384):
    """Single local Ollama JSON extraction call. This never falls back to a hosted API."""
    base = "http://localhost:11434/v1"
    url = base + "/chat/completions"
    payload = {"model": model, "messages": [{"role": "system", "content": system},
              {"role": "user", "content": user}], "temperature": 0,
              "stream": False, "max_tokens": max_tokens,
              "response_format": {"type": "json_object"}}
    with langfuse_generation("ollama-structured-extraction", model,
                             {"system": system, "user": user}) as generation:
        response = _post_json(url, payload, headers={"Authorization": "Bearer ollama"}, timeout=timeout)
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Ollama response did not contain JSON chat content") from exc
        usage = response.get("usage") or {}
        if generation is not None:
            usage_details = {}
            if usage.get("prompt_tokens") is not None:
                usage_details["input"] = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                usage_details["output"] = int(usage["completion_tokens"])
            generation.update(output=content, usage_details=usage_details,
                              cost_details={"input": 0.0, "output": 0.0},
                              metadata={"endpoint": url, "billing": "local Ollama; API charge $0"})
    meta = {"endpoint": url, "model_requested": model, "model_returned": response.get("model"),
            "usage": response.get("usage"), "response_format": "json_object"}
    payload_log = {"request": payload, "response": response, "meta": meta}
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload_log, indent=2, ensure_ascii=False) + "\n")
    return {"text": content, "model": response.get("model", model), "endpoint": url,
            "usage": response.get("usage"), "log_file": str(log_path)}


def call_ollama_vision_json(system, user, image_path, model, log_path, timeout=90, max_tokens=220):
    """Send a focused local image crop to Ollama and log response without image bytes."""
    image_bytes = pathlib.Path(image_path).read_bytes()
    started = time.monotonic()
    url = "http://localhost:11434/api/chat"
    payload = {"model": model, "stream": False, "think": False,
               "format": "json", "options": {"temperature": 0, "num_predict": max_tokens},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user,
                             "images": [base64.b64encode(image_bytes).decode("ascii")]}]}
    with langfuse_generation("ollama-remarks-vision", model, {"system": system, "user": user}) as generation:
        response = _post_json(url, payload, headers={"Authorization": "Bearer ollama"}, timeout=timeout)
        message = response.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama vision response did not contain JSON chat content")
        elapsed = time.monotonic() - started
        if generation is not None:
            generation.update(output=content, usage_details={},
                              cost_details={"input": 0.0, "output": 0.0},
                              metadata={"endpoint": url, "billing": "local Ollama; API charge $0"})
    meta = {"endpoint": url, "model_requested": model, "model_returned": response.get("model"),
            "created_at": response.get("created_at"), "done": response.get("done"),
            "done_reason": response.get("done_reason"),
            "total_duration_ns": response.get("total_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"), "eval_count": response.get("eval_count"),
            "elapsed_seconds": round(elapsed, 3), "image_bytes": len(image_bytes),
            "response_format": "json"}
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"request": {"model": model, "system": system, "user": user,
                                           "image_file": pathlib.Path(image_path).name},
                                   "response": response, "meta": meta},
                                  indent=2, ensure_ascii=False) + "\n")
    return {"text": content, "model": response.get("model", model), "endpoint": url,
            "usage": {"prompt_tokens": response.get("prompt_eval_count"),
                      "completion_tokens": response.get("eval_count")},
            "log_file": str(log_path), "latency_seconds": round(elapsed, 3)}
