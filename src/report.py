"""One short, consistent report from the checked calculation and source limits."""
import datetime as dt
import math
import re
from zoneinfo import ZoneInfo


def number(value):
    return f'{value:,.0f}' if float(value).is_integer() else f'{value:,.1f}'


def short_time(value):
    if not value: return None
    parsed=dt.datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(ZoneInfo('Asia/Kolkata'))
    # A forecast should not imply sub-minute accuracy. Round up to a minute.
    parsed=dt.datetime.fromtimestamp(math.ceil(parsed.timestamp()/60)*60,parsed.tzinfo)
    return str(parsed.day)+' '+parsed.strftime('%b, %H:%M IST')


def short_date(value):
    try:
        parsed=dt.date.fromisoformat(value)
        return str(parsed.day)+' '+parsed.strftime('%b %Y')
    except ValueError: return value


def build_report(saved, context=None):
    result=saved.get('result') or {}; pace=result.get('pace_forecast') or {}
    context=context or saved.get('context') or {}
    flags=(saved.get('ocr') or {}).get('uncertain_fields') or []
    holds=result.get('validation_holds') or []
    dates=re.findall(r'20\d\d-\d\d-\d\d',saved.get('source_text') or '')
    planned=(context.get('as_of') or '')[:10]
    mismatch=(dates and planned and planned not in dates) or any('date differs' in str(f).lower() for f in flags)
    report={'version':'plain-report-v1','status':'unknown','title':'More information is needed.',
            'sentences':[],'check':None,'action':'Check the source details.',
            'scope':'This estimates sewing completion. It assumes the planned work hours and continued output.'}
    if pace.get('status')!='calculated':
        reason=pace.get('reason') or ''
        if pace.get('hourly_sum') is not None:
            report['sentences']=[f"{number(pace['hourly_sum'])} pieces are reported sewn in the hours shown."]
        if saved.get('status')=='failed_closed':
            report.update(title='The run could not finish.',check='Open run details to see the failed step.',action='Fix the failed step and try again.')
        elif not (context.get('po_details_api') or {}).get('due_at') or 'due' in reason.lower():
            report.update(check='The PO due date is missing or invalid.',action='Add the PO due date.')
        elif 'PO' in reason or 'line' in reason:
            report.update(check='The PO or line in the input does not match the source records.',action='Check the PO and line number.')
        elif 'sewn_before_shift' in reason:
            report.update(check='Output before this shift is not recorded.',action='Add the number sewn before this shift.')
        elif 'working_windows' in reason:
            report.update(check='The remaining work hours are not recorded.',action='Add the remaining shift dates and work hours.')
        else: report['check']='The available data does not support a finish estimate.'
        return report
    low=pace['scenarios'][0]; risk=pace['assessment']=='delay_risk'
    report.update(status='at_risk' if risk else 'on_track',
        title='Sewing may miss the due time.' if risk else 'Sewing can finish before the due time.',
        action='Ask the line supervisor for a recovery plan.' if risk else 'Keep checking the hourly output.')
    report['sentences']=[f"{number(pace['current_reported_sewn'])} of {number(pace['total_required'])} pieces are reported sewn. {number(pace['remaining_units'])} remain."]
    rate=number(low['rate_per_hour'])
    if low.get('finish_by_due'):
        report['sentences'].append(f"At the slowest recorded pace of {rate} pieces/hour, sewing could finish around {short_time(low['finish_by_due'])}.")
    else:
        report['sentences'].append(f"At the slowest recorded pace of {rate} pieces/hour, sewing would miss the due time.")
        average=next((s for s in pace['scenarios'] if s['scenario']=='average'),None)
        if average and not average['meets_requirement']:
            report['sentences'].append(f"At the average pace, about {number(average['shortfall_units'])} pieces would remain at the due time.")
    report['sentences'].append(f"The due time is {short_time(pace['due_at'])}.")
    if mismatch:
        read_date=short_date(dates[0]) if dates else None
        source_date=short_date(planned) if planned else None
        report.update(status='check_input',title='Check the date before confirming this forecast.',
                      check=(f"OCR read the date as {read_date}. The records use {source_date}." if read_date and source_date else 'The date read from the photo differs from the source records.'),
                      action='Check the date on the photo or correct the source date.')
        report['sentences'].insert(0,'The pace would be enough if the source date is correct.' if not risk else 'The recorded pace shows a risk, but the source date needs checking.')
    elif flags or holds:
        report.update(status='check_input',title='Check the input before confirming this forecast.',
                      check='Some source text could not be read or checked.',action='Check the marked text in the source details.')
    return report
