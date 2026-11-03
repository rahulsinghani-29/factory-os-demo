"""Render the executable graph, with a shared label/type for every node."""
import json
from live_workflow import build_live_graph, NODE_LABELS


def workflow_dot():
    graph = build_live_graph().get_graph()
    colors = {'CODE': '#edf7f1', 'IF / ELSE': '#fff3ce', 'TOOL CALL': '#e7f1ff', 'LLM CALL': '#f0eaff'}
    lines = ['digraph workflow {', 'rankdir=TB; bgcolor="white"; nodesep=0.25; ranksep=0.30;',
             'node [shape=box,style="rounded,filled",fontname="Helvetica",fontsize=13,fontcolor="#142231",color="#607080",margin="0.16,0.12"];',
             'edge [fontname="Helvetica",fontsize=11,color="#607080",fontcolor="#263342"];']
    for key in graph.nodes:
        if key in ('__start__', '__end__'):
            label = 'START - One incoming message' if key == '__start__' else 'END'
            lines.append(f'{json.dumps(key)} [label={json.dumps(label)},shape=oval,fillcolor="#f3f4f6"];')
        else:
            kind, label = NODE_LABELS[key]
            lines.append(f'{json.dumps(key)} [label={json.dumps(kind+chr(10)+label)},shape={"diamond" if kind=="IF / ELSE" else "box"},fillcolor="{colors[kind]}"];')
    for edge in graph.edges:
        label = str(edge.data or '')
        if label == 'next': label = 'ok'
        style = ',style=dashed,color="#ba453a"' if label == 'failed' else ''
        lines.append(f'{json.dumps(edge.source)} -> {json.dumps(edge.target)} [label={json.dumps(label)}{style}];')
    lines.append('}')
    return '\n'.join(lines)
