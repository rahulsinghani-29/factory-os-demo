"""Local Factory OS demo: real image/text inference and evidence-linked results."""
import datetime as dt
import json
import pathlib
import sys
import uuid
from zoneinfo import ZoneInfo
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from live_workflow import run_live, build_live_graph, INFERENCE_PROMPT, OCR_PROMPT
from report import build_report, short_time
from media_adapters import preserve_upload
from s10_demo import load_examples
from wip_demo import preflight_local_ollama
from observability import langfuse_run, langfuse_configured

st.set_page_config(page_title='Quince Factory OS', layout='wide')
st.markdown('''<style>
.block-container {max-width:1400px;padding-top:5rem;padding-bottom:3rem;}
[data-testid="stButtonGroup"] {padding-top:0.25rem;padding-bottom:0.5rem;}
[data-testid="stButtonGroup"] button {min-height:2.75rem;height:auto;padding:0.625rem 1rem;line-height:1.4;}
@media (max-width:640px) {
  .block-container {padding-top:4.75rem;padding-left:1rem;padding-right:1rem;}
  [data-testid="stButtonGroup"] button {padding:0.625rem 0.75rem;}
}
</style>''', unsafe_allow_html=True)


def ist(value):
    try: return dt.datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(ZoneInfo('Asia/Kolkata')).strftime('%d %b %Y · %H:%M:%S IST')
    except (ValueError, AttributeError): return 'Unknown'


def workflow():
    st.title('Workflow design')
    st.write('WhatsApp supplies a continuous stream of messages. This demo tests one photo or text message per run. Each run combines that message with PO details, a dated plan, prior notes, and department handoffs.')
    st.graphviz_chart('''digraph workflow {
      rankdir=LR; bgcolor="white";
      node [shape=box,style="rounded,filled",fillcolor="#f7fafc",color="#334155",fontcolor="#111827",fontname="Helvetica",fontsize=14,margin="0.20,0.15"];
      edge [color="#475569",fontcolor="#111827",fontname="Helvetica",fontsize=12];
      intake [label="WhatsApp stream\nPhoto · text · voice",fillcolor="#e8f1ff"];
      event [label="Accept one message\nUpload photo or paste text"];
      read [label="Read message\nPhoto → local vision OCR\nText → preserve original"];
      context [label="Read source context\nPO + due date · plan\nHandoffs · prior WhatsApp notes"];
      llm [label="Calculate lowest / average / best pace\nThen compare facts with Qwen",fillcolor="#fff4d6"];
      validate [label="Check quotes and sources\nKeep uncertainty visible"];
      clear [label="CLEAR\nDelay risk: yes or no",shape=ellipse,fillcolor="#e6f4ea"];
      unclear [label="UNCLEAR\nAssign to line supervisor",shape=ellipse,fillcolor="#fff0eb"];
      intake -> event -> read -> context -> llm -> validate;
      validate -> clear [label="Supported"];
      validate -> unclear [label="Missing / conflicting / failed"];
    }''', width='stretch')
    st.caption('The live WhatsApp connector and voice transcription are outside this demo. Photo OCR and Qwen calls run locally through Ollama. PO, plan, and handoff context can be edited below; no live factory APIs are connected.')
    with st.expander('Inspect the configured LangGraph and exact prompts'):
        st.code(build_live_graph().get_graph().draw_mermaid(), language='text')
        st.download_button('Download configured LangGraph', build_live_graph().get_graph().draw_mermaid(), 'factory-live-langgraph.mmd')
        st.markdown('**OCR prompt · Gemma 3 4B**')
        st.code(OCR_PROMPT, language='text')
        st.markdown('**Inference prompt · Qwen 2.5 7B**')
        st.code(INFERENCE_PROMPT, language='text')


def inputs():
    st.title('Input')
    st.write('Choose a photographed shift sheet, upload your own photo, or paste a production message. Each sheet covers eight hours; the final two hourly entries are blank because they have not happened yet.')
    examples = load_examples()['examples']
    mode = st.radio('Input type', ['Sample sheet', 'Upload photo', 'Paste text'], horizontal=True)
    selected = st.selectbox('Sheet / source context', [x['label'] for x in examples], key='live-case')
    example = next(x for x in examples if x['label']==selected)
    synthetic = st.checkbox('All content in this run is synthetic', value=(mode=='Sample sheet'))
    configured = langfuse_configured()
    trace_cloud = st.checkbox('Also send this run to Langfuse', value=False) if configured else False
    st.caption('Local node tracing is always on. ' + ('Langfuse settings are present; cloud export is optional for synthetic data.' if configured else 'Langfuse is not configured in this app process. Local traces will still be saved.'))
    run_clicked = st.button('Run input → view output', type='primary')
    image_path=None; text=''; upload=None
    left, right = st.columns([1.15,1])
    with left:
        if mode=='Sample sheet':
            image_path=str(ROOT/'data'/'s10_samples'/example.get('image_file',f"{example['id']}.jpg"))
            st.image(image_path, caption='Synthetic camera-style example. Six hours filled; two future hours blank. Image text can contain generation errors; compare with OCR.', width='stretch')
        elif mode=='Upload photo':
            upload=st.file_uploader('Upload a sheet or WhatsApp photo',type=['png','jpg','jpeg'])
            if upload: st.image(upload.getvalue(),width='stretch')
        else:
            text=st.text_area('Custom factory message for the LLM',height=250,placeholder='Line S10, PO-SYN-101, as of 14:00 IST.\n08:00–09:00: 72\n09:00–10:00: 70\nInclude all observed hours and remarks. Future blank hours are not zero.',key='live-custom-text')
    context={key:example.get(key) for key in ('as_of','plan_api','po_details_api','handoffs','whatsapp_updates')}
    with right:
        st.subheader('Source context and working calendar')
        st.write('Edit the PO requirement, due date, as-of time, handoffs and planned working windows. The forecast needs the number sewn before this shift. It never assumes the factory works 24 hours per day.')
        context_text=st.text_area('Source context · JSON',json.dumps(context,ensure_ascii=False,indent=2),height=480,key='context-v2-'+example['id'])
        st.caption('All default records are synthetic. Expected test answers are excluded from inference.')
    if run_clicked:
        try:
            parsed=json.loads(context_text)
            if not isinstance(parsed,dict): raise ValueError('Context must be a JSON object')
            if trace_cloud and not synthetic: raise ValueError('Cloud tracing requires synthetic-only input. Turn off Langfuse for other input.')
            if mode=='Upload photo':
                if not upload: raise ValueError('Upload a photo first')
                image_path=preserve_upload(upload.getvalue(),pathlib.Path(upload.name).suffix,ROOT/'data')['path']
            if mode=='Paste text' and not text.strip(): raise ValueError('Enter a factory message first')
            st.session_state['pending_run']={'text':text,'image_path':image_path,'context':parsed,'synthetic':synthetic,'trace_cloud':trace_cloud}
            st.session_state['navigate_to']='Output + trace'
            st.rerun()
        except Exception as exc: st.error(f'Input could not be started: {type(exc).__name__}: {exc}')


def output():
    st.title('Output + trace')
    pending=st.session_state.pop('pending_run',None)
    if pending:
        progress_box=st.empty()
        def progress(name): progress_box.info('Completed: '+name.replace('_',' ')+'. Continuing…')
        with st.spinner('Running local OCR, pace calculations and Qwen. Please wait…',show_time=True):
            with langfuse_run(pending['synthetic'],pending['trace_cloud'],uuid.uuid4().hex,'photo' if pending['image_path'] else 'text','live-input') as cloud_trace:
                saved_run=run_live(pending['text'],pending['image_path'],pending['context'],synthetic=pending['synthetic'],origin='input_tab',progress=progress)
                if cloud_trace is not None: cloud_trace.update(output={'run_id':saved_run['run_id'],'status':saved_run['status'],'outcome':saved_run['result']['outcome']})
        progress_box.empty()
        st.session_state['last_live_run']=saved_run
    saved = st.session_state.get('last_live_run')
    if not saved:
        paths = sorted((ROOT/'logs'/'live-demo').glob('*/run-outcome.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        records = [json.loads(path.read_text()) for path in paths]
        saved = next((record for record in records if record.get('origin')=='input_tab'), records[0] if records else None)
    if not saved:
        st.info('Run an input first. Its trace table appears here above the outcome.')
        return
    provenance = 'Latest input run' if saved.get('origin')=='input_tab' else 'Saved local verification run · submit your own message in Input'
    st.caption(f"{provenance} · {saved['run_id']} · {ist(saved['timestamp'])}")
    st.subheader('Trace')
    st.dataframe([{'Time':ist(row['timestamp']), 'Step':row['node'], 'Status':row['status'], 'Latency (s)':row['latency_seconds']} for row in saved.get('trace',[])], hide_index=True, width='stretch')
    result=saved.get('result') or {}
    pace=result.get('pace_forecast') or {}
    context=saved.get('context') or {}
    if not context:
        input_path=ROOT/'logs'/'live-demo'/str(saved.get('run_id',''))/'input.json'
        if input_path.is_file():
            try: context=json.loads(input_path.read_text()).get('context') or {}
            except (ValueError,OSError): pass
    report=build_report(saved,context)
    po=(context.get('po_details_api') or {}).get('po_id','Order outlook')
    line=(context.get('plan_api') or {}).get('line')
    st.subheader(po + (' · '+line if line else ''))
    if report['status']=='on_track': st.success(report['title'])
    elif report['status']=='at_risk': st.error(report['title'])
    else: st.warning(report['title'])
    for sentence in report['sentences']: st.write(sentence)
    if report.get('check'): st.write(report['check'])
    if report['status']!='on_track': st.write(report['action'])
    st.caption(report['scope'])
    with st.expander('See the calculation'):
        if pace.get('status')=='calculated':
            st.write(f"{pace['remaining_units']:,.0f} pieces remain. There are {pace['available_work_hours']:g} planned work hours before the due time.")
            labels={'lowest_observed':'Slowest recorded','average':'Average','highest_observed':'Fastest recorded'}
            st.dataframe([{'Pace':labels.get(row['scenario'],row['scenario']), 'Pieces/hour':round(row['rate_per_hour'],1),
                'Finish estimate':short_time(row['finish_by_due']) if row['finish_by_due'] else 'After the due time',
                'Meets due time':'Yes' if row['meets_requirement'] else 'No'} for row in pace['scenarios']],hide_index=True,width='stretch')
            if pace.get('shift_forecast'):
                st.markdown('**Today’s shift**')
                st.dataframe([{'Pace':row['Scenario'].replace('lowest observed','Slowest recorded').replace('highest observed','Fastest recorded').capitalize(),
                    'Expected total':round(row['Projected shift sewn']), 'Shift target':row['Shift target'], 'Meets target':'Yes' if row['Target met'] else 'No'}
                    for row in pace['shift_forecast']],hide_index=True,width='stretch')
            st.caption('The slowest pace includes recorded zero-output hours. Blank future hours are excluded. Work follows the supplied shift calendar.')
        else: st.write(pace.get('reason','No hourly calculation was available for this run.'))
    with st.expander('Check the source text'):
        st.text(saved.get('source_text') or 'No text was read.')
        for flag in (saved.get('ocr') or {}).get('uncertain_fields',[]): st.write(str(flag))
        for flag in pace.get('warnings',[]): st.write(str(flag))
    with st.expander('Run details and model notes'):
        if saved.get('error'): st.error(saved['error'])
        for reason in result.get('validation_holds',[]): st.write(reason)
        st.caption(f"Qwen: {saved.get('model_returned') or 'not verified'} · OCR: {(saved.get('ocr') or {}).get('model') or 'not used / not verified'}")
        st.caption('Synthetic data: '+str(saved.get('synthetic_data'))+'. No factory write or human approval occurred.')
        st.caption('Log: '+str(saved.get('log_path','unavailable')))
        st.json(saved)


def evaluation():
    st.title('Eval criteria')
    st.write('Score a sampled run against a reviewed reference: original image transcription, source facts, and correct risk decision. Each criterion scores 0, 1, or 2. Weighted score = sum(weight × score / 2).')
    rows = [
      ['Identifier and time accuracy',15,'Wrong PO, line, or event time','One missing identifier; explicitly unknown','Exact PO, line, date, hour and timezone'],
      ['Counts, rates and completion math',20,'Wrong count, double count, or blank treated as zero','Counts correct but calendar or rate assumptions incomplete','Exact sum and rates; blanks excluded; available work hours and finish calculation correct'],
      ['Tamil remarks and event meaning',20,'Misses stoppage or turns intent into completion','Core issue captured; uncertainty or nuance missing','Original remark preserved; translation and current/future meaning correct'],
      ['Cross-source reconciliation',15,'Combines mismatched POs or ignores conflicts','Finds some differences but misses relevant sources','Links PO, plan, WhatsApp and handoffs; explains conflict/unknown'],
      ['Risk decision and abstention',20,'False clear, unsupported delay, or invented numerical impact','Safe unclear but misses a decidable case','Correct risk yes/no or necessary abstention with missing facts'],
      ['Evidence and explanation',10,'Fabricated quote/source or unsupported causal claim','Real sources but incomplete support or verbose output','Exact quotes and valid IDs; concise explanation; limits stated'],
    ]
    st.dataframe([dict(zip(['Criterion','Weight','0 · fail','1 · partial','2 · pass'],r)) for r in rows], hide_index=True, width='stretch')
    st.markdown('**Release gate.** Require at least 90/100 and no critical error. A wrong PO/line, fabricated source, invented shipment delay, future plan treated as completion, false clear on conflicting facts, or unauthorized write fails the run regardless of score.')
    st.markdown('**Dataset.** Version the source image, exact transcription, row quantities, remark translation, PO/plan/handoff snapshot, expected risk and rationale. Include clean print, handwriting, Tamil, mixed language, partial work, stopped machines, conflicting updates, missing dates and unreadable photos. Split by document template and PO so near-duplicate pages cannot leak into the test set.')
    st.markdown('**Measure by group.** Report PO/line exact match, per-row quantity accuracy, character error rate, remark meaning accuracy, risk precision/recall, false-clear rate and abstention rate. Report both correct decisions and unnecessary abstentions; blocking every case must not pass.')
    st.markdown('**Proposed targets, not measured results.** PO/line and quantity accuracy ≥98%; risk recall ≥95%; false-clear rate ≤1%; zero fabricated evidence or writes. Track OCR, LLM and total p50/p95 latency separately. OCR p95 target: 15 seconds on a warm local model; total latency has no validated target yet. Local API charge is zero; hardware cost is unmeasured.')
    st.markdown('**Async operation.** Sample 10% of live runs plus every unclear/failed run for later supervisor scoring. Score the fixed golden set on every model, prompt, OCR or code change. Save rubric version, reviewer, model/prompt version and trace ID. Compare regression metrics before release; do not block routine live OCR for per-case approval.')
    st.info('This is the rubric and release policy. A reviewed media golden set and automated sampling job are not configured. Existing regression tests are not an OCR benchmark.')


pages=['Workflow design','Input','Output + trace','Eval criteria']
if 'navigate_to' in st.session_state: st.session_state['page']=st.session_state.pop('navigate_to')
if 'page' not in st.session_state: st.session_state['page']='Workflow design'
page=st.segmented_control('Pages',pages,key='page',selection_mode='single',label_visibility='collapsed')
if page=='Input': inputs()
elif page=='Output + trace': output()
elif page=='Eval criteria': evaluation()
else: workflow()
