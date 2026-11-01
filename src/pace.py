"""Conditional arithmetic forecast. Blank future hours never mean zero production."""
import datetime as dt
import re


def text_rows(text):
    if re.search(r'Hour\s*\|\s*Pieces sewn\s*\|\s*Running total',text,re.I):
        pattern=r'(?m)^\s*\|?\s*(\d\d:\d\d)\s*[–—-]\s*(\d\d:\d\d)\s*\|\s*(\d*)\s*\|[^\n]*'
        return [{'start':m[1],'end':m[2],'quantity':int(m[3]) if m[3] else None,'source_quote':m[0].strip()} for m in re.finditer(pattern,text)]
    pattern = r'(?m)^\s*(\d\d:\d\d)\s*[–—-]\s*(\d\d:\d\d)\s*[:|]\s*(\d+)\s*(?:pieces|pcs)?\s*$'
    return [{'start':m[1], 'end':m[2], 'quantity':int(m[3]), 'source_quote':m[0].strip()} for m in re.finditer(pattern,text)]


def forecast(rows, text, context):
    result={'status':'unknown','warnings':[],'assumptions':['Rates continue only within the supplied working windows.',
        'Lowest observed rate includes observed zero-output hours. It is not an absolute worst case.',
        'Future blank hours are excluded, not counted as zero. Sewing completion is not shipment readiness.',
        'Continuing any observed rate assumes material and machine availability; handoff records do not guarantee that.']}
    try:
        as_of=dt.datetime.fromisoformat(context['as_of'])
        plan=context['plan_api']; po=context['po_details_api']
        if as_of.tzinfo is None: raise ValueError('As-of time needs a timezone')
        normalized_text=re.sub(r'\s*[-–]\s*','-',text.upper())
        if not po.get('po_id') or po['po_id'].upper() not in normalized_text: raise ValueError('Input PO does not explicitly match the context PO')
        if not plan.get('line') or not re.search(r'(?<!\w)'+re.escape(plan['line'])+r'(?!\w)',text): raise ValueError('Input line does not explicitly match the planned line')
        seen=set(); counts=[]; hours=0; last_end=None
        for row in rows:
            quantity=row.get('quantity')
            if quantity is None: continue
            if type(quantity) is not int or quantity<0: raise ValueError('Invalid hourly count')
            if not row.get('source_quote') or row['source_quote'] not in text: raise ValueError('Hourly row has no matching OCR/text quote')
            start=dt.datetime.fromisoformat(as_of.date().isoformat()+'T'+row['start']).replace(tzinfo=as_of.tzinfo)
            end=dt.datetime.fromisoformat(as_of.date().isoformat()+'T'+row['end']).replace(tzinfo=as_of.tzinfo)
            if end>as_of:
                result['warnings'].append(f"Ignored a value reported for future interval {row['start']}–{row['end']}; it is not observed production.")
                continue
            if end<=start or (last_end and start<last_end): raise ValueError('Overlapping or invalid hours')
            if (start,end) in seen: raise ValueError('Duplicate hourly row')
            seen.add((start,end)); last_end=end
            length=(end-start).total_seconds()/3600
            counts.append((quantity,length)); hours+=length
        if not counts: raise ValueError('No evidence-linked hourly rows were extracted')
        units=sum(q for q,h in counts)
        declared=re.search(r'Total sewn(?:\s*\(so far\)| so far)?\s*[:|]\s*(\d+)',text,re.I)
        if declared and int(declared[1])!=units: raise ValueError('Hourly sum does not match the sheet total')
        rates={'lowest_observed':min(q/h for q,h in counts),'average':units/hours,'highest_observed':max(q/h for q,h in counts)}
        result.update(hourly_sum=units,observed_hours=hours,rates_per_hour={k:round(v,2) for k,v in rates.items()})
        if plan.get('shift_end') and type(plan.get('shift_target_units')) is int:
            shift_end=dt.datetime.fromisoformat(plan['shift_end'])
            remaining_shift=max(0,(shift_end-as_of).total_seconds()/3600)
            result['shift_forecast']=[{'Scenario':name.replace('_',' '),'Observed sewn':units,
                'Hours left':remaining_shift,'Projected shift sewn':round(units+rate*remaining_shift,1),
                'Shift target':plan['shift_target_units'],'Target met':units+rate*remaining_shift>=plan['shift_target_units']}
                for name,rate in rates.items()]
        due=dt.datetime.fromisoformat(po['due_at'])
        if due.tzinfo is None: raise ValueError('Due date needs a timezone')
        prior=plan.get('sewn_before_shift')
        if type(prior) is not int or prior<0: raise ValueError('Explicit sewn_before_shift is required for a PO forecast')
        required=po['total_required']
        if type(required) is not int or required<0: raise ValueError('Invalid PO requirement')
        windows=sorted((dt.datetime.fromisoformat(w['start']),dt.datetime.fromisoformat(w['end'])) for w in plan['working_windows'])
        available=0; previous=None
        for start,end in windows:
            if start.tzinfo is None or end.tzinfo is None or end<=start or (previous and start<previous): raise ValueError('Invalid or overlapping working windows')
            previous=end
            available+=max(0,(min(end,due)-max(start,as_of)).total_seconds()/3600)
        current=prior+units; remaining=max(0,required-current)
        scenarios=[]
        for name,rate in rates.items():
            projected=current+rate*available; needed=remaining/rate if rate>0 else (0 if remaining==0 else None)
            finish=None; left=needed
            if left is not None:
                for start,end in windows:
                    start=max(start,as_of); end=min(end,due)
                    capacity=max(0,(end-start).total_seconds()/3600)
                    if left<=capacity and capacity>0:
                        finish=(start+dt.timedelta(hours=left)).isoformat();break
                    left-=capacity
            scenarios.append({'scenario':name,'rate_per_hour':round(rate,2),'projected_sewn_by_due':round(projected,1),
                'meets_requirement':projected>=required,'shortfall_units':round(max(0,required-projected),1),'finish_by_due':finish})
        result.update(status='calculated',current_reported_sewn=current,total_required=required,remaining_units=remaining,
            available_work_hours=round(available,2),due_at=due.isoformat(),scenarios=scenarios,
            assessment='no_observed_pace_risk' if scenarios[0]['meets_requirement'] else 'delay_risk',
            required_rate_per_hour=round(remaining/available,2) if available>0 else None,
            evidence_ids=[plan.get('source_id'),po.get('source_id')])
    except (ValueError,KeyError,TypeError) as exc: result['reason']=str(exc)
    return result
