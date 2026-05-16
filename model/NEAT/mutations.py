import random
import torch

try:
    from .genome_operations_torch import compute_topological_order
except ImportError:
    from genome_operations_torch import compute_topological_order

# TODO: expirement with more interesting mutations


def mutate_weights(genome, perturb_rate=0.1, replace_rate=0.02):
    weights = genome.connections.conn_weights.clone()

    # 80% chance to mutate weights at all for this genome
    if random.random() < 0.8:
        for i in range(len(weights)):
            r = random.random()
            if r < perturb_rate:
                weights[i] += random.gauss(0, 0.1)  # small nudge
            elif r < perturb_rate + replace_rate:
                weights[i] = random.gauss(0, 1.0)  # full reset

    genome.connections.conn_weights = weights
    return genome


def add_connection(genome, tracker):
    device = genome.nodes.node_ids.device
    # find all pairs that don't already have a connection
    existing = set(map(tuple, genome.connections.conn_indices.tolist()))

    node_ids = genome.nodes.node_ids.tolist()

    candidates = [
        (a, b) for a in node_ids for b in node_ids if a != b and (a, b) not in existing
    ]

    if not candidates:
        return genome  # fully connected, nothing to add

    in_node, out_node = random.choice(candidates)
    innovation = tracker.get_innovation_number(in_node, out_node)
    weight = random.gauss(0, 1.0)

    # append to tensors
    new_conn = torch.tensor([[in_node, out_node]], device=device)
    genome.connections.conn_indices = torch.cat(
        [genome.connections.conn_indices, new_conn]
    )
    genome.connections.conn_weights = torch.cat(
        [genome.connections.conn_weights, torch.tensor([weight], device=device)]
    )
    genome.connections.conn_enabled = torch.cat(
        [genome.connections.conn_enabled, torch.tensor([True], device=device)]
    )
    genome.connections.conn_innovation = torch.cat(
        [genome.connections.conn_innovation, torch.tensor([innovation], device=device)]
    )

    genome.topological_order = compute_topological_order(genome)
    return genome


def add_node(genome, tracker):
    device = genome.nodes.node_ids.device
    enabled_mask = genome.connections.conn_enabled.bool()
    enabled_indices = torch.where(enabled_mask)[0].tolist()

    if not enabled_indices:
        return genome

    # pick a random enabled connection to split
    split_idx = random.choice(enabled_indices)
    in_node = genome.connections.conn_indices[split_idx, 0].item()
    out_node = genome.connections.conn_indices[split_idx, 1].item()
    old_weight = genome.connections.conn_weights[split_idx].item()

    # disable the old connection
    genome.connections.conn_enabled[split_idx] = False

    # add the new node
    new_node_id = genome.nodes.node_ids.max().item() + 1
    genome.nodes.node_ids = torch.cat(
        [genome.nodes.node_ids, torch.tensor([new_node_id], device=device)]
    )
    genome.nodes.node_types = torch.cat(
        [genome.nodes.node_types, torch.tensor([1], device=device)]  # 1 = hidden
    )
    genome.nodes.node_activations = torch.cat(
        [genome.nodes.node_activations, torch.tensor([0], device=device)]  # relu
    )

    # two new connections
    for a, b, w in [
        (in_node, new_node_id, 1.0),  # init weight with 1
        (new_node_id, out_node, old_weight),  # old weight preserves behavior
    ]:
        innov = tracker.get_innovation_number(a, b)
        genome.connections.conn_indices = torch.cat(
            [genome.connections.conn_indices, torch.tensor([[a, b]], device=device)]
        )
        genome.connections.conn_weights = torch.cat(
            [genome.connections.conn_weights, torch.tensor([w], device=device)]
        )
        genome.connections.conn_enabled = torch.cat(
            [genome.connections.conn_enabled, torch.tensor([True], device=device)]
        )
        genome.connections.conn_innovation = torch.cat(
            [genome.connections.conn_innovation, torch.tensor([innov], device=device)]
        )

    genome.topological_order = compute_topological_order(genome)
    return genome


def mutate(genome, tracker):
    genome = mutate_weights(genome)  # always

    if random.random() < 0.05:  # TODO: add config for this
        genome = add_connection(genome, tracker)

    if random.random() < 0.03:
        genome = add_node(genome, tracker)

    return genome
