"""Render the executable graph, with a shared label/type for every node."""
import json
import live_workflow

VIEW_VERSION = "centered-v2"


def workflow_dot():
    graph = live_workflow.build_live_graph().get_graph()
    colors = {'CODE': '#edf7f1', 'IF / ELSE': '#fff3ce', 'TOOL CALL': '#e7f1ff', 'LLM CALL': '#f0eaff'}
    lines = ['digraph workflow {', 'rankdir=TB; center=true; bgcolor="white"; nodesep=0.25; ranksep=0.30;',
             'node [shape=box,style="rounded,filled",fontname="Helvetica",fontsize=13,fontcolor="#142231",color="#607080",margin="0.16,0.12"];',
             'edge [fontname="Helvetica",fontsize=11,color="#607080",fontcolor="#263342"];']
    for key in graph.nodes:
        if key in ('__start__', '__end__'):
            label = 'START - One incoming message' if key == '__start__' else 'END'
            lines.append(f'{json.dumps(key)} [label={json.dumps(label)},shape=oval,fillcolor="#f3f4f6"];')
        else:
            kind, label = live_workflow.NODE_LABELS[key]
            lines.append(f'{json.dumps(key)} [label={json.dumps(kind+chr(10)+label)},shape={"diamond" if kind=="IF / ELSE" else "box"},fillcolor="{colors[kind]}"];')
    for edge in graph.edges:
        label = str(edge.data or '')
        if label == 'next': label = 'ok'
        style = ',style=dashed,color="#ba453a",constraint=false,weight=0' if label == 'failed' else ',weight=20'
        lines.append(f'{json.dumps(edge.source)} -> {json.dumps(edge.target)} [label={json.dumps(label)}{style}];')
    # Balance the error lane so the main vertical path starts in the centre.
    # This invisible spacer controls layout only; it is not an executable node.
    lines.extend([
        'layout_balance [label="",style=invis,width=3.6,height=0.01,fixedsize=true];',
        '{rank=same; run_failed; prepare_report; layout_balance;}',
        'run_failed -> prepare_report [style=invis,weight=100];',
        'prepare_report -> layout_balance [style=invis,weight=100];',
    ])
    lines.append('}')
    return '\n'.join(lines)
