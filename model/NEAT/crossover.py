import random
import torch
import copy


def crossover(parent_a, parent_b):
    # make sure parent_a is the fitter one
    if parent_b.fitness_score.item() > parent_a.fitness_score.item():
        parent_a, parent_b = parent_b, parent_a

    innov_a = {
        parent_a.connections.conn_innovation[i].item(): i
        for i in range(len(parent_a.connections.conn_innovation))
    }
    innov_b = {
        parent_b.connections.conn_innovation[i].item(): i
        for i in range(len(parent_b.connections.conn_innovation))
    }

    all_innovations = set(innov_a) | set(innov_b)

    child_indices = []
    child_weights = []
    child_enabled = []
    child_innov = []

    for innov in sorted(all_innovations):
        in_a = innov in innov_a
        in_b = innov in innov_b

        if in_a and in_b:
            # matching
            idx = innov_a[innov] if random.random() < 0.5 else innov_b[innov]
            src = parent_a if idx == innov_a.get(innov) else parent_b

            # if either parent has it disabled, 25% chance child inherits disabled
            a_enabled = parent_a.connections.conn_enabled[innov_a[innov]].item()
            b_enabled = parent_b.connections.conn_enabled[innov_b[innov]].item()
            enabled = (a_enabled and b_enabled) or (random.random() < 0.75)

        elif in_a:
            # disjoint/excess: take from fitter parent (parent_a)
            src = parent_a
            idx = innov_a[innov]
            enabled = parent_a.connections.conn_enabled[idx].item()

        else:
            continue  # only in weaker parent, skip

        child_indices.append(
            parent_a.connections.conn_indices[innov_a.get(innov, idx)].tolist()
            if in_a
            else parent_b.connections.conn_indices[innov_b[innov]].tolist()
        )
        child_weights.append(src.connections.conn_weights[idx].item())
        child_enabled.append(enabled)
        child_innov.append(innov)

    # inherit structure from fitter parent
    child = copy.deepcopy(parent_a)
    device = parent_a.nodes.node_ids.device

    child.connections.conn_indices = torch.tensor(
        child_indices, dtype=torch.long, device=device
    )
    child.connections.conn_weights = torch.tensor(
        child_weights, dtype=torch.float32, device=device
    )
    child.connections.conn_enabled = torch.tensor(
        child_enabled, dtype=torch.bool, device=device
    )
    child.connections.conn_innovation = torch.tensor(
        child_innov, dtype=torch.long, device=device
    )

    child.fitness_score = torch.tensor(0.0, device=device)
    child._topo_order = None

    return child
