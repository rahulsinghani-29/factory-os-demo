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
# An existing Streamlit process may retain modules imported before a source upgrade.
import importlib
import live_workflow
if getattr(live_workflow, 'WORKFLOW_VERSION', None) != 'typed-steps-v1':
    importlib.reload(live_workflow)
from live_workflow import run_live, build_live_graph, INFERENCE_PROMPT, OCR_PROMPT
from workflow_view import workflow_dot
from golden_data import load_golden, golden_rows, golden_csv, LABEL_FIELDS
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
    st.write('Each box is one configured step. Its label shows whether it is code, an if/else decision, a model call, or a tool call.')
    st.caption('An incoming WhatsApp-style message is represented by an upload or pasted text. The live WhatsApp connector is not connected. Audio and video take the unsupported-input branch and end with an unclear report.')
    st.graphviz_chart(workflow_dot(), width='stretch')
    st.caption('PO details, handoff/QC status, the plan, and preceding messages have separate tool steps. These are read-only local fixture adapters for future APIs. No live factory API is called. Every route reaches END. Failures remain unclear.')
    st.caption('A supervisor hold is a suggested follow-up. It does not send a task or record human approval. Graph detail does not establish model accuracy; known reasoning gaps remain.')
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
    st.dataframe([{'Time':ist(row['timestamp']), 'Step':row['node'], 'Type':row.get('type','legacy trace'), 'Status':row['status'], 'Latency (s)':row['latency_seconds']} for row in saved.get('trace',[])], hide_index=True, width='stretch')
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
    data = load_golden()
    st.subheader('Golden dataset · 10 draft reference cases')
    st.write('These synthetic examples show the expected answer before any model is scored. Clear cases and necessary holds are both included.')
    st.warning('Draft references, pending independent review. These are not ten passing model runs or an OCR benchmark. G08 and G10 represent known validation gaps.')
    st.dataframe(golden_rows(data), hide_index=True, width='stretch')
    selected = st.selectbox('Inspect the exact input and its labels', [c['case_id']+' · '+c['title'] for c in data['cases']])
    case = next(c for c in data['cases'] if selected.startswith(c['case_id']+' ·'))
    with st.expander('Source text, PO facts and expected answer', expanded=False):
        st.text(case['reference_text'])
        st.json({'expected_labels': case['labels'], 'source_id': case['source_id'], 'source_kind': case['source_kind'],
                 'context_snapshot': case['context_snapshot']})
    left, right = st.columns(2)
    left.download_button('Download all 10 reference cases · JSON', json.dumps(data,ensure_ascii=False,indent=2), 'golden-cases-v1.json', 'application/json')
    right.download_button('Download the labelled table · CSV', golden_csv(data), 'golden-cases-v1.csv', 'text/csv')
    with st.expander('What each label means'):
        st.dataframe([{'Label': key, 'Meaning': value} for key,value in LABEL_FIELDS.items()], hide_index=True, width='stretch')
    with st.expander('LLM-as-judge rubric and release gate'):
        st.write('An independent reviewer must check the references first. A judge can then score meaning and explanation. Code checks identity, arithmetic, routes and forbidden writes. A judge score alone does not certify a run.')
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
    st.caption('Reference labels never enter the inference prompt. Text and prepared OCR transcripts test reasoning only. Media evaluation also needs original assets, reviewed transcription and translation, template/PO holdout splits, and measured p50/p95 latency.')
    st.caption('Async sampling and automatic judge scores are not configured. Proposed policy: sample 10% of routine runs and all unclear/failed runs for later evaluation. Routine OCR does not pause for case-by-case approval.')


pages=['Workflow design','Input','Output + trace','Eval criteria']
if 'navigate_to' in st.session_state: st.session_state['page']=st.session_state.pop('navigate_to')
if 'page' not in st.session_state: st.session_state['page']='Workflow design'
page=st.segmented_control('Pages',pages,key='page',selection_mode='single',label_visibility='collapsed')
if page=='Input': inputs()
elif page=='Output + trace': output()
elif page=='Eval criteria': evaluation()
else: workflow()
