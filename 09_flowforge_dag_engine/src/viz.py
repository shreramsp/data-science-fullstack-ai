"""Matplotlib DAG rendering, laid out by topological level (no Graphviz binary needed)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from .engine import topological_levels
from .models import WorkflowDefinition

NODE_COLORS = {
    "source": "#4C9AFF",
    "map": "#57D9A3",
    "filter": "#FFAB00",
    "branch": "#FF7452",
    "aggregate": "#998DD9",
}


def draw_workflow(wf: WorkflowDefinition):
    graph = nx.DiGraph()
    for node in wf.nodes:
        graph.add_node(str(node.id), kind=node.type)
    for edge in wf.edges:
        graph.add_edge(str(edge.source), str(edge.target), label=edge.label or "")

    try:
        levels = topological_levels(wf.nodes, wf.edges)
    except Exception:
        levels = [[n.id] for n in wf.nodes]

    pos = {}
    for x, level in enumerate(levels):
        n = len(level)
        for y, nid in enumerate(level):
            pos[str(nid)] = (x, (n - 1) / 2 - y)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [NODE_COLORS.get(graph.nodes[n]["kind"], "#CCCCCC") for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=1800, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(graph, pos, arrowsize=15, ax=ax, connectionstyle="arc3,rad=0.05")
    edge_labels = {(u, v): d["label"] for u, v, d in graph.edges(data=True) if d["label"]}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    return fig
