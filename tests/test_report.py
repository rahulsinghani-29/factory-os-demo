import copy,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from report import build_report

class ReportTests(unittest.TestCase):
 def sample(self):
  return {'source_text':'Date: 2023-11-01','ocr':{'uncertain_fields':['OCR document date differs from supplied as-of production date; pace uses supplied date conditionally']},
   'context':{'as_of':'2026-11-01T14:00:00+05:30','po_details_api':{'due_at':'2026-11-03T17:00:00+05:30'}},
   'result':{'outcome':'unclear','validation_holds':['OCR reported ambiguous fields'],
    'explanation':'The source checks could not validate the LLM interpretation.',
    'pace_forecast':{'status':'calculated','assessment':'no_observed_pace_risk','current_reported_sewn':400,
     'total_required':1000,'remaining_units':600,'due_at':'2026-11-03T17:00:00+05:30',
     'scenarios':[{'scenario':'lowest_observed','rate_per_hour':62,'finish_by_due':'2026-11-02T15:40:38+05:30','meets_requirement':True}]}}}
 def test_logged_date_conflict_is_one_check_verdict_not_a_green_and_yellow_contradiction(self):
  r=build_report(self.sample())
  self.assertEqual(r['status'],'check_input');self.assertIn('date',r['title'])
  self.assertIn('1 Nov 2023',r['check']);self.assertIn('1 Nov 2026',r['check'])
  self.assertIn('15:41',r['sentences'][2]);self.assertNotIn('validate',' '.join(r['sentences']))
 def test_good_pace_and_no_holds_have_one_on_track_verdict(self):
  s=self.sample();s['source_text']='Date: 2026-11-01';s['ocr']['uncertain_fields']=[];s['result']['validation_holds']=[]
  r=build_report(s);self.assertEqual(r['status'],'on_track');self.assertIsNone(r['check'])
 def test_invalid_ocr_date_is_reported_without_crashing_the_output(self):
  s=self.sample();s['source_text']='Date: 2023-99-99'
  r=build_report(s);self.assertEqual(r['status'],'check_input');self.assertIn('2023-99-99',r['check'])
 def test_risk_uses_the_calculation_even_if_raw_llm_says_smooth_work(self):
  s=self.sample();s['source_text']='Date: 2026-11-01';s['ocr']['uncertain_fields']=[];s['result']['validation_holds']=[]
  s['result']['explanation']='Work is proceeding smoothly.'
  p=s['result']['pace_forecast'];p['assessment']='delay_risk';p['scenarios'][0].update(rate_per_hour=0,finish_by_due=None,meets_requirement=False)
  r=build_report(s);self.assertEqual(r['status'],'at_risk');self.assertNotIn('smoothly',' '.join(r['sentences']))
 def test_missing_due_date_names_the_missing_fact_in_plain_english(self):
  s=self.sample();s['context']['po_details_api']['due_at']=None;s['result']['pace_forecast']={'status':'unknown','reason':'fromisoformat: argument must be str','hourly_sum':380}
  r=build_report(s);self.assertEqual(r['status'],'unknown');self.assertEqual(r['action'],'Add the PO due date.')
  self.assertIn('380',r['sentences'][0])
