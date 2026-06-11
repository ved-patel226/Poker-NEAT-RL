import os
import random
import torch

# TODO: expirement with more interesting mutations


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def re_enable_connection(genome, tracker):
    """Re-enable a random disabled connection or enable all if none are disabled."""
    device = genome.nodes.node_ids.device
    disabled_mask = ~genome.connections.conn_enabled.bool()
    disabled_indices = torch.where(disabled_mask)[0].tolist()

    if disabled_indices:
        # re-enable a random disabled connection
        idx = random.choice(disabled_indices)
        genome.connections.conn_enabled[idx] = True
        genome._topo_order = None
        return genome

    # If no disabled connections, force-enable some random ones to ensure diversity
    if len(genome.connections.conn_enabled) > 0 and random.random() < 0.3:
        idx = random.randint(0, len(genome.connections.conn_enabled) - 1)
        genome.connections.conn_enabled[idx] = True
        genome._topo_order = None

    return genome


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

    num_nodes = genome.nodes.node_ids.shape[0]
    node_ids = list(range(num_nodes))
    node_types = {
        idx: int(node_type)
        for idx, node_type in enumerate(genome.nodes.node_types.tolist())
    }

    def is_valid_source_target(in_node, out_node):
        in_type = node_types[int(in_node)]
        out_type = node_types[int(out_node)]

        # keep the graph feed-forward: inputs/bias/hidden can drive hidden/output,
        # hidden->hidden is only allowed in creation order.
        if in_type == 2 or out_type in (0, 3):
            return False
        if in_type == 1 and out_type == 1:
            return int(in_node) < int(out_node)
        return out_type in (1, 2)

    candidates = [
        (a, b)
        for a in node_ids
        for b in node_ids
        if a != b and (a, b) not in existing and is_valid_source_target(a, b)
    ]

    if not candidates:
        # Network is fully connected - try re-enabling a connection instead
        return re_enable_connection(genome, tracker)

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
    genome._topo_order = None

    return genome


def add_node(genome, tracker):
    device = genome.nodes.node_ids.device
    enabled_mask = genome.connections.conn_enabled.bool()
    enabled_indices = torch.where(enabled_mask)[0].tolist()

    if not enabled_indices:
        # No enabled connections to split - try re-enabling one first
        genome = re_enable_connection(genome, tracker)
        # Re-compute enabled indices after re-enabling
        enabled_mask = genome.connections.conn_enabled.bool()
        enabled_indices = torch.where(enabled_mask)[0].tolist()

        if not enabled_indices:
            return genome  # Still no enabled connections

    # pick a random enabled connection to split
    split_idx = random.choice(enabled_indices)
    in_node = genome.connections.conn_indices[split_idx, 0].item()
    out_node = genome.connections.conn_indices[split_idx, 1].item()
    old_weight = genome.connections.conn_weights[split_idx].item()

    # disable the old connection
    genome.connections.conn_enabled[split_idx] = False

    # add the new node
    new_node_id = genome.nodes.node_ids.shape[0]
    genome.nodes.node_ids = torch.arange(new_node_id + 1, device=device)
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

    genome._topo_order = None
    return genome


def mutate(genome, tracker):
    genome = mutate_weights(genome)  # always

    add_conn_prob = _env_float("NEAT_ADD_CONN_PROB", 0.12)
    add_node_prob = _env_float("NEAT_ADD_NODE_PROB", 0.08)
    add_node_no_hidden_prob = _env_float("NEAT_ADD_NODE_NO_HIDDEN_PROB", 0.35)
    reenable_prob = _env_float("NEAT_REENABLE_CONN_PROB", 0.05)

    has_hidden = bool((genome.nodes.node_types == 1).any().item())
    if not has_hidden and random.random() < add_node_no_hidden_prob:
        genome = add_node(genome, tracker)

    if random.random() < add_conn_prob:  # increased from 0.05
        genome = add_connection(genome, tracker)

    if has_hidden and random.random() < add_node_prob:  # increased from 0.03
        genome = add_node(genome, tracker)

    # Fallback mutation: ensure exploration by re-enabling connections sometimes
    # This helps escape the "fully connected + all disabled" state
    if random.random() < reenable_prob:
        genome = re_enable_connection(genome, tracker)

    return genome
