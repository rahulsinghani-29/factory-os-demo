import copy,json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from pace import forecast,text_rows
from s10_demo import load_examples

class PaceTests(unittest.TestCase):
 def case(self,index):
  ex=load_examples()['examples'][index]
  text=ex['po_details_api']['po_id']+' line S10\n'+'\n'.join(r['time']+': '+str(r['sewn']) for r in ex['rows'])
  context={k:ex[k] for k in ('as_of','plan_api','po_details_api')}
  return text,context
 def test_six_observed_hours_project_with_eighteen_work_hours_not_calendar_hours(self):
  t,c=self.case(0); r=forecast(text_rows(t),t,c)
  self.assertEqual(r['status'],'calculated');self.assertEqual(r['hourly_sum'],400)
  self.assertEqual(r['available_work_hours'],18)
  self.assertEqual(r['scenarios'][0]['projected_sewn_by_due'],1516)
  self.assertEqual(r['assessment'],'no_observed_pace_risk')
 def test_real_zero_hours_change_the_low_case_but_blank_future_hours_do_not(self):
  t,c=self.case(1); rows=text_rows(t)+[{'start':'14:00','end':'15:00','quantity':None},{'start':'15:00','end':'16:00','quantity':None}]
  r=forecast(rows,t,c)
  self.assertEqual(r['hourly_sum'],188); self.assertEqual(r['observed_hours'],6)
  self.assertEqual(r['rates_per_hour']['lowest_observed'],0)
  self.assertEqual(r['scenarios'][1]['projected_sewn_by_due'],752)
  self.assertEqual(r['assessment'],'delay_risk')
 def test_missing_due_date_abstains_without_invented_eta(self):
  t,c=self.case(3); self.assertEqual(forecast(text_rows(t),t,c)['status'],'unknown')
 def test_actual_ocr_table_parsing_uses_incremental_column_not_running_total(self):
  t,c=self.case(0)
  t='PO - SYN - 101 line S10\nHour | Pieces sewn | Running total | Remarks\n08:00 – 09:00 | 72 | 72 | ok\n09:00 – 10:00 | 70 | 142 | ok\n14:00 – 15:00 | | |\nTotal sewn so far : 142'
  rows=text_rows(t);r=forecast(rows,t,c)
  self.assertEqual(r['hourly_sum'],142);self.assertEqual(r['observed_hours'],2)
  self.assertIsNone(rows[-1]['quantity'])
  self.assertEqual(forecast(rows,t.replace(': 142',': 999'),c)['status'],'unknown')
 def test_mismatched_po_and_overlapping_rows_are_not_valid_forecasts(self):
  t,c=self.case(0);r=text_rows(t)
  self.assertEqual(forecast(r,t.replace('PO-SYN-101','PO-OTHER'),c)['status'],'unknown')
  self.assertEqual(forecast(r+r[:1],t,c)['status'],'unknown')
 def test_future_quantity_is_not_counted_and_prior_production_is_required(self):
  t,c=self.case(0); rows=text_rows(t)
  rows.append({'start':'14:00','end':'15:00','quantity':50,'source_quote':t[:5]})
  result=forecast(rows,t,c)
  self.assertEqual(result['status'],'calculated');self.assertEqual(result['hourly_sum'],400)
  self.assertTrue(result['warnings'])
  c=copy.deepcopy(c); c['plan_api'].pop('sewn_before_shift')
  self.assertEqual(forecast(text_rows(t),t,c)['status'],'unknown')
