"""Presentation charts. Proposed message filters do not change the executable graph."""
import json
import live_workflow

VIEW_VERSION = "context-filters-final"


def _render(nodes, edges, proposed=()):
    colors = {'CODE': '#edf7f1', 'IF / ELSE': '#fff3ce', 'TOOL CALL': '#e7f1ff', 'LLM CALL': '#f0eaff', 'END': '#f3f4f6'}
    lines = ['digraph workflow {',
             'rankdir=TB; center=true; splines=ortho; bgcolor="white"; nodesep=0.65; ranksep=0.65; pad=0.25;',
             'node [shape=box,style="rounded,filled",fontname="Helvetica",fontsize=13,fontcolor="#142231",color="#607080",margin="0.16,0.12"];',
             'edge [fontname="Helvetica",fontsize=11,color="#607080",fontcolor="#263342",arrowsize=0.7];']
    for key, (kind, label) in nodes.items():
        prefix = 'PROPOSED · ' if key in proposed else ''
        shape = 'diamond' if kind == 'IF / ELSE' else ('oval' if kind == 'END' else 'box')
        style = 'rounded,filled,dashed' if key in proposed else 'rounded,filled'
        text = label if kind == 'END' else prefix + kind + chr(10) + label
        lines.append(f'{json.dumps(key)} [label={json.dumps(text, ensure_ascii=False)},shape={shape},style="{style}",fillcolor="{colors[kind]}"];')
    for source, target, label in edges:
        # Orthogonal Graphviz connectors use external labels to avoid diagonal routes.
        lines.append(f'{json.dumps(source)} -> {json.dumps(target)} [xlabel={json.dumps(label)},weight=10];')
    lines.append('}')
    return '\n'.join(lines)


def _configured_spec():
    graph = live_workflow.build_live_graph().get_graph()
    nodes = {'__start__': ('END', 'START · One incoming message'), **live_workflow.NODE_LABELS,
             '__end__': ('END', 'END')}
    edges = [(e.source, e.target, 'ok' if e.data == 'next' else str(e.data or '')) for e in graph.edges]
    return nodes, edges


def workflow_dot():
    """Exact executable topology, without the proposed message filters."""
    return _render(*_configured_spec())


def workflow_design_dot():
    """Design overlay: dashed message gates are proposed, not runtime features."""
    nodes, configured_edges = _configured_spec()
    proposed = {
        'understand_message': ('LLM CALL', 'Identify the message purpose and explicit PO'),
        'relevant_message': ('IF / ELSE', 'Is this a relevant factory update?'),
        'explicit_po': ('IF / ELSE', 'Does the message identify one PO?'),
        'recent_three': ('TOOL CALL', 'Read the previous three messages\nSame factory and conversation, before this message'),
        'context_po': ('IF / ELSE', 'Exactly one PO in those messages\nand a clear continuation of that discussion?'),
        'inherit_po': ('CODE', 'Use that PO and record its source\nNever replace an explicit or conflicting PO'),
        'ignore_message': ('END', 'END · Ignore this message\nNo response and no PO report'),
    }
    nodes.update(proposed)
    nodes['clear_result'] = ('CODE', 'Prepare clear result\nPass the checked sewing-risk answer to the report')
    nodes['supervisor_hold'] = ('CODE', 'Prepare clarification request\nPass missing facts to the report; no task is sent')
    # The overview groups error exits instead of drawing one overlapping lane per node.
    edges = [(a,b,label) for a,b,label in configured_edges
             if label != 'failed' and (a,b) != ('parse_document','get_po_details')]
    edges += [
        ('parse_document','understand_message',''),
        ('understand_message','relevant_message',''),
        ('understand_message','run_failed','Classifier failure'),
        ('relevant_message','ignore_message','No'),
        ('relevant_message','explicit_po','Yes'),
        ('explicit_po','get_po_details','Yes'),
        ('explicit_po','recent_three','Missing'),
        ('explicit_po','ignore_message','Multiple / conflicting POs'),
        ('recent_three','context_po',''),
        ('context_po','inherit_po','Yes'),
        ('context_po','ignore_message','No / uncertain'),
        ('inherit_po','get_po_details',''),
        ('ocr_photo','run_failed','Read failure'),
        ('qwen_inference','run_failed','Model failure'),
        ('validate_evidence','run_failed','Check failure'),
    ]
    return _render(nodes, edges, proposed)
