"""Render the Bayesian Network structure as a hierarchical DAG diagram,
styled after the course's example figure (F2.png in the LaTeX template
folder): white ellipse nodes with a bold name and a short letter code,
arranged top-to-bottom by causal layer, connected by directed edges.

The network learned by hill-climb search is fairly dense (13 nodes, 29
edges), so nodes are laid out with longest-path-from-source layering
(root/source nodes at the top, sinks at the bottom) and a barycenter
crossing-reduction sweep, then drawn as properly sized ellipses so full
labels fit without truncation. Edges into the target node (`readmitted`)
are highlighted in blue, mirroring the reference figure's use of a second
edge color for emphasis; all other edges are light gray.

Usage:
    python3 render_bn_structure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import networkx as nx

HERE = Path(__file__).parent
TARGET = "readmitted"

# Edges learned by Hill-Climb search (K2 score, max in-degree 3) over the
# 12 selected features, as reported by the notebook
# (bayesian-network-cleaned-hill-climb.ipynb, "Inspect Learned Relations").
EDGES = [
    ("number_inpatient", "readmitted"),
    ("number_inpatient", "number_outpatient"),
    ("number_inpatient", "diag_1_group"),
    ("number_inpatient", "time_in_hospital"),
    ("discharge_disposition", "time_in_hospital"),
    ("discharge_disposition", "readmitted"),
    ("number_diagnoses", "admission_source"),
    ("time_in_hospital", "num_medications"),
    ("num_medications", "number_diagnoses"),
    ("admission_source", "medical_specialty"),
    ("admission_source", "number_outpatient"),
    ("age_group", "discharge_disposition"),
    ("age_group", "number_diagnoses"),
    ("number_emergency", "admission_source"),
    ("number_emergency", "discharge_disposition"),
    ("number_emergency", "number_outpatient"),
    ("number_emergency", "number_inpatient"),
    ("number_emergency", "readmitted"),
    ("number_emergency", "diag_1_group"),
    ("number_emergency", "time_in_hospital"),
    ("diag_1_group", "medical_specialty"),
    ("diag_1_group", "age_group"),
    ("diag_1_group", "discharge_disposition"),
    ("diag_1_group", "num_medications"),
    ("admission_type", "admission_source"),
    ("admission_type", "medical_specialty"),
    ("admission_type", "diag_1_group"),
    ("admission_type", "num_medications"),
    ("admission_type", "number_diagnoses"),
]

NAVY = "#1B2A4A"
BLUE = "#2F5CA6"
RED = "#B5292F"
GRAY_EDGE = "#B7BEC8"
BLACK = "#1A1A1A"
FONT = "DejaVu Sans"


def wrap_name(node):
    parts = node.split("_")
    if len(parts) == 1:
        return [node]
    lines, current = [], ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if current and len(candidate) > 12:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_layers(graph, target_rows=5):
    """Longest-path-from-source layering, then merged into a handful of
    visual rows. The learned network contains a genuine ~9-edge directed
    chain running through most nodes, so a strict one-rank-per-row layout
    would need 10 rows (far too tall/narrow for a report figure); merging
    consecutive ranks keeps every edge pointing the same way (row never
    decreases along an edge) while giving a much more compact, diamond-like
    shape similar to the reference figure."""
    order = list(nx.topological_sort(graph))
    layer = {}
    for node in order:
        preds = list(graph.predecessors(node))
        layer[node] = 0 if not preds else 1 + max(layer[p] for p in preds)
    max_layer = max(layer.values())
    merge = max(1, -(-(max_layer + 1) // target_rows))  # ceil division
    return {n: l // merge for n, l in layer.items()}


def order_layers(graph, layer):
    max_layer = max(layer.values())
    layers = {i: sorted(n for n, l in layer.items() if l == i) for i in range(max_layer + 1)}
    pos_in_layer = {n: idx for i, nodes in layers.items() for idx, n in enumerate(nodes)}

    def barycenter(node, neighbors):
        vals = [pos_in_layer[m] for m in neighbors if m in pos_in_layer]
        return sum(vals) / len(vals) if vals else pos_in_layer[node]

    for _ in range(4):
        for i in range(1, max_layer + 1):
            layers[i].sort(key=lambda n: barycenter(n, list(graph.predecessors(n))))
            for idx, n in enumerate(layers[i]):
                pos_in_layer[n] = idx
        for i in range(max_layer - 1, -1, -1):
            layers[i].sort(key=lambda n: barycenter(n, list(graph.successors(n))))
            for idx, n in enumerate(layers[i]):
                pos_in_layer[n] = idx
    return layers


def measure_lines(fig, lines, fontsize, fontweight):
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    max_w, total_h = 0.0, 0.0
    for line in lines:
        t = fig.text(0, 0, line, fontsize=fontsize, fontweight=fontweight, fontfamily=FONT)
        bbox = t.get_window_extent(renderer=renderer)
        t.remove()
        w, h = inv.transform((bbox.width, bbox.height))
        max_w = max(max_w, w)
        total_h += h
    return max_w, total_h


def render(output_path, dpi=200):
    graph = nx.DiGraph()
    graph.add_edges_from(EDGES)

    layer = build_layers(graph, target_rows=5)
    layers = order_layers(graph, layer)
    max_layer = max(layer.values())

    # Reading order (top layer first, left to right) assigns short letter
    # codes, mirroring the reference figure's A, B, C... labelling.
    code_order = [n for i in range(max_layer + 1) for n in layers[i]]
    code = {n: chr(ord("A") + i) for i, n in enumerate(code_order)}

    # --- sizing pass -----------------------------------------------------
    probe_fig = plt.figure()
    name_fontsize, code_fontsize = 11, 9.5
    line_gap = 0.045
    node_size = {}
    for node in graph.nodes():
        name_lines = wrap_name(node)
        name_w, name_h = measure_lines(probe_fig, name_lines, name_fontsize, "bold")
        code_w, code_h = measure_lines(probe_fig, [f"({code[node]})"], code_fontsize, "normal")
        text_w = max(name_w, code_w)
        text_h = name_h + code_h + line_gap * (len(name_lines))
        node_size[node] = (text_w * 1.28 + 0.14, text_h * 1.32 + 0.16)
    plt.close(probe_fig)

    # --- layout ------------------------------------------------------------
    col_gap, row_gap = 0.85, 0.4
    layer_width = {}
    layer_height = {}
    for i in range(max_layer + 1):
        nodes = layers[i]
        widths = [node_size[n][0] for n in nodes]
        layer_width[i] = sum(widths) + col_gap * max(len(nodes) - 1, 0)
        layer_height[i] = max((node_size[n][1] for n in nodes), default=1.0)

    max_row_width = max(layer_width.values())
    positions = {}
    y = 0.0
    for i in range(max_layer + 1):
        nodes = layers[i]
        row_w = layer_width[i]
        x = -row_w / 2
        y -= layer_height[i] / 2
        for n in nodes:
            w = node_size[n][0]
            positions[n] = (x + w / 2, y)
            x += w + col_gap
        y -= layer_height[i] / 2 + row_gap

    margin = 0.4
    all_x = [positions[n][0] for n in graph.nodes()]
    all_y = [positions[n][1] for n in graph.nodes()]
    max_half_w = max(node_size[n][0] for n in graph.nodes()) / 2
    max_half_h = max(node_size[n][1] for n in graph.nodes()) / 2
    xlim = (min(all_x) - max_half_w - margin, max(all_x) + max_half_w + margin)
    ylim = (min(all_y) - max_half_h - margin, max(all_y) + max_half_h + margin * 1.6)

    fig_w = xlim[1] - xlim[0]
    fig_h = ylim[1] - ylim[0]

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # Nodes first, so edges can clip cleanly to their boundaries.
    patches = {}
    for node in graph.nodes():
        x, y = positions[node]
        w, h = node_size[node]
        is_target = node == TARGET
        ell = Ellipse(
            (x, y), w, h,
            facecolor="white",
            edgecolor=(RED if is_target else BLACK),
            linewidth=2.2 if is_target else 1.5,
            zorder=3,
        )
        ax.add_patch(ell)
        patches[node] = ell

    for u, v in graph.edges():
        to_target = (v == TARGET)
        color = BLUE if to_target else GRAY_EDGE
        lw = 2.0 if to_target else 1.1
        ax.add_patch(FancyArrowPatch(
            positions[u], positions[v],
            patchA=patches[u], patchB=patches[v],
            arrowstyle="-|>",
            mutation_scale=15 if to_target else 11,
            connectionstyle="arc3,rad=0.0",
            linewidth=lw,
            color=color,
            alpha=1.0 if to_target else 0.85,
            zorder=2 if to_target else 1,
            shrinkA=0, shrinkB=0,
        ))

    for node in graph.nodes():
        x, y = positions[node]
        name_lines = wrap_name(node)
        is_target = node == TARGET
        n_lines = len(name_lines)
        top = y + (n_lines - 1) * (name_fontsize / 72 * 1.25) / 2 + (name_fontsize / 72 * 0.75)
        for i, line in enumerate(name_lines):
            ax.text(
                x, top - i * (name_fontsize / 72 * 1.25),
                line, ha="center", va="center",
                fontsize=name_fontsize, fontweight="bold",
                color=(RED if is_target else BLACK), fontfamily=FONT, zorder=4,
            )
        code_y = top - n_lines * (name_fontsize / 72 * 1.25) - (code_fontsize / 72 * 0.55)
        ax.text(
            x, code_y, f"({code[node]})", ha="center", va="center",
            fontsize=code_fontsize, fontweight="normal", color=(RED if is_target else "#3A3A3A"),
            fontfamily=FONT, zorder=4,
        )

    # Legend, top-left corner.
    lx, ly = xlim[0] + 0.25, ylim[1] - 0.35
    ax.add_patch(FancyArrowPatch((lx, ly), (lx + 0.55, ly), arrowstyle="-|>",
                                  mutation_scale=13, color=BLUE, linewidth=2.0, zorder=6))
    ax.text(lx + 0.72, ly, "edge into target", ha="left", va="center",
            fontsize=10.5, color=BLACK, fontfamily=FONT, zorder=6)
    ax.add_patch(FancyArrowPatch((lx + 3.3, ly), (lx + 3.85, ly), arrowstyle="-|>",
                                  mutation_scale=11, color=GRAY_EDGE, linewidth=1.1, zorder=6))
    ax.text(lx + 4.02, ly, "other learned edge", ha="left", va="center",
            fontsize=10.5, color=BLACK, fontfamily=FONT, zorder=6)

    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved {output_path} ({fig_w:.2f}in x {fig_h:.2f}in)")


if __name__ == "__main__":
    render(HERE / "bn_structure.png")
