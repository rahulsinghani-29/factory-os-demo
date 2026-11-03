import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import live_workflow as live


class LiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.context = {'as_of':'2026-11-01T14:00:00+05:30',
            'po_details_api':{'due_at':'2026-11-03T17:00:00+05:30','source_id':'PO'},
            'plan_api':{'source_id':'PLAN'}, 'handoffs':{}, 'whatsapp_updates':[],
            'expected_delay_risk':'no', 'expected_clear':True}

    def response(self, system, payload, model, log, **kwargs):
        data = json.loads(payload)
        self.assertNotIn('expected_delay_risk', data['context'])
        return {'text':json.dumps({'outcome':'clear','delay_risk':'yes','explanation':'A stoppage creates risk.',
            'observations':[{'quote':data['source_text'],'meaning':'Machine stopped.'}],
            'evidence_ids':[data['source_id']], 'missing_facts':[]}),
            'model':'qwen2.5:7b','endpoint':'http://localhost:11434/v1/chat/completions'}

    def test_custom_text_runs_model_without_golden_answer_and_can_be_clear(self):
        with patch.object(live,'ROOT',pathlib.Path(self.tmp.name)), patch.object(live,'call_ollama_json',side_effect=self.response) as llm:
            result = live.run_live('Machine stopped.',None,self.context)
        self.assertTrue(llm.called)
        self.assertEqual(result['result']['outcome'],'clear')
        self.assertEqual(result['result']['delay_risk'],'yes')
        self.assertTrue(pathlib.Path(result['log_path']).is_file())

    def test_unknown_quote_cannot_be_promoted_to_clear(self):
        state={'analysis':{'candidate':{'outcome':'clear','delay_risk':'no','explanation':'Fine',
                'evidence_ids':['INPUT'], 'observations':[{'quote':'Invented text'}]},
                'allowed_ids':['INPUT'],'model_returned':'test','endpoint':'local'},
                'source_text':'Machine stopped.', 'context':self.context,'ocr':{}}
        self.assertEqual(live._validate(state)['result']['outcome'],'unclear')

    def test_image_calls_ocr_and_passes_actual_read_text_to_qwen(self):
        image=pathlib.Path(self.tmp.name)/'test.jpg'; image.write_bytes(b'fake image for mocked client')
        with patch.object(live,'ROOT',pathlib.Path(self.tmp.name)), patch.object(live,'call_ollama_vision_json',return_value={
                'text':json.dumps({'text':'Machine stopped.','uncertain_fields':[]}), 'model':'vision','endpoint':'local'}) as ocr, \
                patch.object(live,'call_ollama_json',side_effect=self.response):
            result=live.run_live('',str(image),self.context)
        self.assertTrue(ocr.called)
        self.assertEqual(result['source_text'],'Machine stopped.')

    def test_failed_model_call_saves_failed_stage_and_never_writes_outbox(self):
        with patch.object(live,'ROOT',pathlib.Path(self.tmp.name)), patch.object(live,'call_ollama_json',side_effect=RuntimeError('offline')):
            result=live.run_live('Machine stopped.',None,self.context)
        self.assertEqual(result['status'],'failed_closed')
        self.assertEqual([row['node'] for row in result['trace'] if row['status']=='failed'], ['qwen_inference'])
        self.assertEqual(result['trace'][-1]['node'], 'prepare_report')
        self.assertFalse(result['outbox_written'])

    def test_unrelated_programming_request_never_reaches_the_model(self):
        with patch.object(live,'ROOT',pathlib.Path(self.tmp.name)), patch.object(live,'call_ollama_json') as llm:
            result=live.run_live('Write Python code to sort a list.',None,self.context)
        self.assertFalse(llm.called)
        self.assertEqual(result['result']['outcome'],'unclear')
        self.assertIn('Factory scope only',result['error'])

    def test_logged_restart_abstention_does_not_hide_valid_conditional_rate_decision(self):
        state={'analysis':{'candidate':{'outcome':'unclear','delay_risk':None,
            'explanation':'The pace forecast is conditional and cannot be confirmed without a restart time.',
            'observations':[{'quote':'Machine stopped.'}], 'evidence_ids':['INPUT'], 'missing_facts':['Restart time']},
            'allowed_ids':['INPUT'],'model_returned':'qwen2.5:7b','endpoint':'local'},
            'source_text':'Machine stopped.','context':self.context,'ocr':{},
            'pace':{'status':'calculated','assessment':'delay_risk','total_required':1000,
                    'scenarios':[{'rate_per_hour':0,'projected_sewn_by_due':188}]}}
        result=live._validate(state)['result']
        self.assertEqual(result['outcome'],'clear');self.assertEqual(result['delay_risk'],'yes')
        self.assertEqual(result['assessment_scope'],'conditional_sewing_pace')
        self.assertEqual(result['llm_outcome'],'unclear')
        self.assertEqual(result['unresolved_context'],['Restart time'])


if __name__=='__main__': unittest.main()
