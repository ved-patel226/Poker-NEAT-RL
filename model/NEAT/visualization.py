import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
import networkx as nx
import numpy as np
from collections import deque
import random

plt.ion()
_fig = None
_ax1 = None
_ax2 = None


def visualize_population_and_best(
    population, best_genome, generation=None, species_list=None
):
    global _fig, _ax1, _ax2

    if _fig is None:
        _fig = plt.figure(figsize=(16, 10))
        _ax1 = plt.subplot(2, 1, 1)
        _ax2 = plt.subplot(2, 1, 2)

    _ax1.clear()
    _ax2.clear()

    visualize_population_grid(population, best_genome, _ax1, generation, species_list)
    visualize_network(best_genome, _ax2)

    plt.tight_layout()
    plt.pause(0.01)

    return _fig


def visualize_population_grid(
    population, best_genome, ax, generation=None, species_list=None
):
    # sample at most 100 random genomes + best
    max_sample = 100
    if len(population) > max_sample:
        sampled = random.sample(population, max_sample - 1)
        # check if best_genome is already in sampled using identity check
        best_in_sample = any(genome is best_genome for genome in sampled)
        if not best_in_sample:
            sampled.append(best_genome)
        sampled_population = sampled
    else:
        sampled_population = population

    species_to_marker = {}
    if species_list:
        for i, species in enumerate(species_list):
            markers = ["o", "s", "^", "v", "D", "p", "*", "h"]
            species_to_marker[species.id] = markers[i % len(markers)]

    genome_to_species = {}
    if species_list:
        for species in species_list:
            for genome in species.members:
                genome_to_species[id(genome)] = species.id

    # create random grid positions
    cols = int(np.ceil(np.sqrt(len(sampled_population))))
    rows = int(np.ceil(len(sampled_population) / cols))

    x_positions = []
    y_positions = []
    colors = []
    markers = []
    fitnesses = []

    positions_used = set()
    for i, genome in enumerate(sampled_population):
        col = i % cols
        row = i // cols
        x = col + random.random() * 0.8
        y = rows - row + random.random() * 0.8

        x_positions.append(x)
        y_positions.append(y)

        fitness = genome.fitness_score.item()
        fitnesses.append(fitness)

        # color based on fitness (red=bad, green=good)
        if genome is best_genome:
            colors.append("darkgreen")
        else:
            norm_fitness = max(0, min(1, fitness))
            colors.append((1 - norm_fitness, norm_fitness, 0))

        species_id = genome_to_species.get(id(genome), 0)
        marker = species_to_marker.get(species_id, "o")
        markers.append(marker)

    for x, y, color, marker in zip(x_positions, y_positions, colors, markers):
        ax.scatter(
            x,
            y,
            c=[color],
            s=300,
            marker=marker,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

    best_idx = None
    for i, genome in enumerate(population):
        if genome is best_genome:
            best_idx = i
            break

    if best_idx is not None:
        best_col = best_idx % cols
        best_row = best_idx // cols
        best_x = best_col + random.random() * 0.8
        best_y = rows - best_row + random.random() * 0.8
        ax.scatter(
            best_x,
            best_y,
            c="darkred",
            s=1000,
            marker="*",
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )

    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 1)
    ax.set_aspect("equal")

    title = f"Population Grid (Showing: {len(sampled_population)}/{len(population)})"
    if generation is not None:
        title = f"Generation {generation} - {title}"
    if species_list:
        title += f" | Species: {len(species_list)}"

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    avg_fitness = np.mean(fitnesses)
    max_fitness = np.max(fitnesses)
    stats_text = f"Avg Fitness: {avg_fitness:.4f}\nMax Fitness: {max_fitness:.4f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )


def visualize_network(genome, ax):
    G = nx.DiGraph()

    node_ids = genome.nodes.node_ids.cpu().tolist()
    node_types = genome.nodes.node_types.cpu().tolist()

    for node_id, node_type in zip(node_ids, node_types):
        G.add_node(node_id, type=node_type)

    conn_indices = genome.connections.conn_indices.cpu().tolist()
    conn_enabled = genome.connections.conn_enabled.cpu().tolist()
    conn_weights = genome.connections.conn_weights.cpu().tolist()

    for (in_node, out_node), enabled, weight in zip(
        conn_indices, conn_enabled, conn_weights
    ):
        if enabled:
            G.add_edge(in_node, out_node, weight=weight)

    pos = _hierarchical_layout(genome, G)

    node_colors = []
    node_sizes = []
    for node_id in G.nodes():
        node_type = G.nodes[node_id]["type"]
        if node_type == 0:  # input
            node_colors.append("lightgreen")
            node_sizes.append(800)
        elif node_type == 1:  # hidden
            node_colors.append("lightyellow")
            node_sizes.append(600)
        elif node_type == 2:  # output
            node_colors.append("lightcoral")
            node_sizes.append(800)
        elif node_type == 3:  # bias
            node_colors.append("lightskyblue")
            node_sizes.append(700)
        else:
            node_colors.append("lightgray")
            node_sizes.append(600)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
        edgecolors="black",
        linewidths=2,
    )

    # draw edges with weights
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        width=1.5,
    )

    # draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    ax.set_title(
        f"Best Genome Network\n"
        f"Nodes: {len(node_ids)} | Enabled Connections: {sum(conn_enabled)} | Fitness: {genome.fitness_score.item():.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Input"),
        Patch(facecolor="lightskyblue", edgecolor="black", label="Bias"),
        Patch(facecolor="lightyellow", edgecolor="black", label="Hidden"),
        Patch(facecolor="lightcoral", edgecolor="black", label="Output"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)


def _hierarchical_layout(genome, G):
    pos = {}

    node_ids = genome.nodes.node_ids.cpu().tolist()
    node_types = genome.nodes.node_types.cpu().tolist()

    inputs = [nid for nid, ntype in zip(node_ids, node_types) if ntype in (0, 3)]
    outputs = [nid for nid, ntype in zip(node_ids, node_types) if ntype == 2]

    # BFS from inputs
    node_layers = {nid: 0 for nid in inputs}

    for _ in range(len(G.nodes())):
        changed = False
        for u, v in G.edges():
            if u in node_layers:
                new_depth = node_layers[u] + 1
                if v not in node_layers or new_depth > node_layers[v]:
                    node_layers[v] = new_depth
                    changed = True
        if not changed:
            break

    max_layer = max(node_layers.values()) if node_layers else 0
    final_layer = max_layer + 1
    for o in outputs:
        node_layers[o] = final_layer

    for nid in G.nodes():
        if nid not in node_layers:
            node_layers[nid] = -1

    layers = {}
    for nid, layer in node_layers.items():
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(nid)

    min_layer = min(node_layers.values())
    max_layer = max(node_layers.values())
    layer_range = max_layer - min_layer if max_layer > min_layer else 1

    for layer, nodes_in_layer in layers.items():
        if layer == -1:
            x = 0.5
        else:
            x = (layer - min_layer) / layer_range
        num_nodes = len(nodes_in_layer)
        for i, node_id in enumerate(sorted(nodes_in_layer)):
            if num_nodes == 1:
                y = 0.5
            else:
                y = 1.0 - (i / (num_nodes - 1))
            pos[node_id] = (x, y)

    return pos
