import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import live_workflow as live
from golden_data import load_golden, golden_rows, golden_csv
from pace import forecast, text_rows
from source_tools import read_fixture
from workflow_view import workflow_dot


class WorkflowReferenceTests(unittest.TestCase):
    def test_every_configured_node_is_reachable_and_can_reach_end(self):
        g=live.build_live_graph().get_graph()
        edges={key:[] for key in g.nodes}
        reverse={key:[] for key in g.nodes}
        for e in g.edges:
            edges[e.source].append(e.target);reverse[e.target].append(e.source)
        def visit(start, adjacency):
            seen=set();pending=[start]
            while pending:
                node=pending.pop()
                if node not in seen:seen.add(node);pending.extend(adjacency[node])
            return seen
        self.assertEqual(visit('__start__', edges),set(g.nodes))
        self.assertEqual(visit('__end__', reverse),set(g.nodes))
        self.assertEqual(set(live.NODE_LABELS),set(g.nodes)-{'__start__','__end__'})
        for name in live.NODE_LABELS:self.assertIn(name,workflow_dot())

    def test_audio_and_video_have_a_real_hold_path_without_model_calls(self):
        for media in ['audio','video']:
            with tempfile.TemporaryDirectory() as tmp, patch.object(live,'ROOT',pathlib.Path(tmp)), patch.object(live,'call_ollama_json') as llm, patch.object(live,'call_ollama_vision_json') as vision:
                r=live.run_live('',None,{'as_of':'2026-11-01T14:00:00+05:30'},modality=media)
                self.assertEqual(r['status'],'unsupported_input')
                self.assertEqual(r['result']['outcome'],'unclear')
                self.assertEqual(r['trace'][-1]['node'],'prepare_report')
                self.assertFalse(r['outbox_written']);llm.assert_not_called();vision.assert_not_called()

    def test_empty_ocr_and_exception_reach_failed_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=pathlib.Path(tmp)/'x.png';image.write_bytes(b'test stub')
            with patch.object(live,'ROOT',pathlib.Path(tmp)), patch.object(live,'call_ollama_vision_json',return_value={'text':json.dumps({'text':'','uncertain_fields':[]})}), patch.object(live,'call_ollama_json') as llm:
                r=live.run_live('',str(image),{})
            self.assertEqual(r['status'],'failed_closed')
            self.assertEqual([x['node'] for x in r['trace'] if x['status']=='failed'],['ocr_photo'])
            self.assertIn('run_failed',[x['node'] for x in r['trace']]);llm.assert_not_called()

    def test_each_fixture_reader_is_separately_traced_and_labels_are_not_in_prompt(self):
        case=load_golden()['cases'][0]
        context={**case['context_snapshot'], 'labels':case['labels']}
        def response(system,payload,*args,**kwargs):
            value=json.loads(payload)
            self.assertNotIn('labels',value['context'])
            self.assertNotIn('expected_outcome',payload)
            return {'text':json.dumps({'outcome':'clear','delay_risk':'no','explanation':'The supplied pace can meet the due time.',
                'observations':[{'quote':'Work continues.'}], 'evidence_ids':[value['source_id']], 'missing_facts':[]}), 'model':'mocked','endpoint':'mocked'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(live,'ROOT',pathlib.Path(tmp)), patch.object(live,'call_ollama_json',side_effect=response):
            result=live.run_live(case['reference_text'],None,context)
        names=[r['node'] for r in result['trace']]
        for name in ['get_po_details','get_handoff_status','get_production_plan','get_prior_messages']:
            self.assertIn(name,names)
            self.assertEqual(next(r['type'] for r in result['trace'] if r['node']==name),'TOOL CALL')
        self.assertEqual(result['result']['outcome'],'clear')

    def test_fixture_lookup_is_read_only_and_rejects_wrong_identity(self):
        context=load_golden()['cases'][0]['context_snapshot'];old=json.dumps(context)
        value=read_fixture(context,'po_details_api',{'po_id':'PO-SYN-101'})
        value['total_required']=0
        self.assertEqual(json.dumps(context),old)
        with self.assertRaises(ValueError):read_fixture(context,'po_details_api',{'po_id':'PO-OTHER'})

    def test_reference_labels_cover_clear_hold_and_match_authored_arithmetic(self):
        data=load_golden();cases=data['cases']
        self.assertEqual(len(cases),10);self.assertEqual(len(golden_rows(data)),10)
        self.assertEqual(len({c['case_id'] for c in cases}),10)
        self.assertEqual({c['labels']['expected_outcome'] for c in cases},{'clear','unclear'})
        self.assertEqual({c['labels']['expected_delay_risk'] for c in cases},{'yes','no',None})
        self.assertIn('G10',golden_csv(data))
        for case in cases:
            label=case['labels']
            self.assertEqual(label['review_status'],'pending_independent_review')
            self.assertIn(case['source_id'],label['evidence_ids'])
            self.assertTrue(label['rationale']);self.assertTrue(label['forbidden_claims'])
            values=label['hourly_quantities']
            if values is not None:
                self.assertEqual(values[-2:],[None,None])
                self.assertEqual(sum(x for x in values if x is not None),label['total_reported_sewn'])
            if case['case_id'] in {'G01','G02','G04','G05','G06'}:
                f=forecast(text_rows(case['reference_text']),case['reference_text'],case['context_snapshot'])
                self.assertEqual(f['status'],'calculated')
                self.assertEqual('yes' if f['assessment']=='delay_risk' else 'no',label['expected_delay_risk'])


if __name__=='__main__':unittest.main()
