# if you're looking at this code for the first time, look at genome_operations_readable.py
# its much more readable cuz I didn't use random torch stuff to optimize for speed

import torch

try:
    from .data_structures import Genome, NodeGenes, ConnectionGenes
except ImportError:
    from data_structures import Genome, NodeGenes, ConnectionGenes


ACTIVATION_FN = {
    0: torch.relu,
    1: torch.tanh,
    2: torch.sigmoid,
}


def forward(genome: Genome, x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    device = (
        genome.nodes.node_ids.device
    )  # get device from genome TODO: future cpu + gpu batching?
    x = x.to(device)

    if x.dim() == 1:  # [batch, input]
        x = x.unsqueeze(0)

    batch_size = x.size(0)

    num_nodes = genome.nodes.node_ids.shape[0]

    values = torch.zeros((num_nodes, batch_size), dtype=torch.float32, device=device)

    input_mask = genome.nodes.node_types == 0
    bias_mask = genome.nodes.node_types == 3
    input_indices = torch.where(input_mask)[0]
    bias_indices = torch.where(bias_mask)[0]
    values[input_indices] = x.t()[: len(input_indices)]
    if bias_indices.numel() > 0:
        values[bias_indices] = 1.0

    order = compute_topological_order(genome)

    for node_idx in order:
        if input_mask[node_idx] or bias_mask[node_idx]:  # dont do input/bias nodes
            continue

        incoming_mask = genome.connections.conn_indices[:, 1] == node_idx
        incoming_mask &= genome.connections.conn_enabled.bool()  # AND

        if incoming_mask.sum() == 0:
            continue

        incoming_indices = genome.connections.conn_indices[incoming_mask, 0]
        incoming_weights = genome.connections.conn_weights[incoming_mask].unsqueeze(1)
        incoming_values = values[incoming_indices]

        total = torch.sum(
            incoming_values * incoming_weights, dim=0
        )  # sum all the inputs across the incoming nodes

        activation_fn = ACTIVATION_FN.get(
            genome.nodes.node_activations[node_idx].item(), torch.relu
        )  # default to relu TODO: more robust
        values[node_idx] = activation_fn(total)

    output_mask = genome.nodes.node_types == 2
    output_indices = torch.where(output_mask)[0]

    return values[output_indices].t()


def compute_topological_order(genome: Genome) -> list:
    num_nodes = genome.nodes.node_ids.shape[0]

    # build map on CPU
    conn_indices_cpu = genome.connections.conn_indices.cpu()
    conn_enabled_cpu = genome.connections.conn_enabled.cpu()

    # dependency map

    deps = [
        set() for _ in range(num_nodes)
    ]  # what value MUST i know before I can compute the next one?
    for i, (in_node, out_node) in enumerate(conn_indices_cpu.tolist()):
        if conn_enabled_cpu[i].item():
            deps[out_node].add(in_node)

    order = []  # the order in which I compute the ndoes
    visited = set()
    visiting = set()

    def visit(node_id):
        if node_id in visited:
            return
        if node_id in visiting:
            return  # cycle
        visiting.add(node_id)
        for dep in deps[node_id]:
            visit(dep)  # process dependencies first
        visiting.remove(node_id)
        visited.add(node_id)
        order.append(node_id)

    for i in range(
        num_nodes
    ):  # reverse doesn't do anything except add compute, it works without too
        visit(i)

    return order


def create_genome(
    input_nodes: int,
    hidden_nodes: int,
    output_nodes: int,
    connections: list = None,
    device: str = "cpu",
) -> Genome:
    total_nodes = input_nodes + hidden_nodes + output_nodes + 1

    bias_index = input_nodes
    hidden_start = bias_index + 1
    hidden_end = hidden_start + hidden_nodes
    output_start = hidden_end

    node_ids = torch.arange(total_nodes, device=device)

    # 0=input, 1=hidden, 2=output, 3=bias
    node_types = torch.zeros(total_nodes, dtype=torch.long, device=device)
    node_types[bias_index] = 3
    node_types[hidden_start:hidden_end] = 1
    node_types[output_start:] = 2

    # relu for input/hidden, tanh for output
    node_activations = torch.zeros(total_nodes, dtype=torch.long, device=device)
    node_activations[output_start:] = 1  # tanh for outputs

    if connections is None:
        conn_list = []
        innovation = 0

        if hidden_nodes > 0:
            for i in range(input_nodes):
                for j in range(hidden_start, hidden_end):
                    conn_list.append(
                        (i, j, float(torch.randn(1).item()), True, innovation)
                    )
                    innovation += 1

            for j in range(hidden_start, hidden_end):
                conn_list.append(
                    (bias_index, j, float(torch.randn(1).item()), True, innovation)
                )
                innovation += 1

            for i in range(hidden_start, hidden_end):
                for j in range(output_start, total_nodes):
                    conn_list.append(
                        (i, j, float(torch.randn(1).item()), True, innovation)
                    )
                    innovation += 1
        else:
            for i in range(input_nodes):
                for j in range(output_start, total_nodes):
                    conn_list.append(
                        (i, j, float(torch.randn(1).item()), True, innovation)
                    )
                    innovation += 1

        for j in range(output_start, total_nodes):
            conn_list.append(
                (bias_index, j, float(torch.randn(1).item()), True, innovation)
            )
            innovation += 1
    else:
        conn_list = [
            (in_idx, out_idx, w, True, i)
            for i, (in_idx, out_idx, w) in enumerate(connections)
        ]

    if conn_list:
        conn_indices = torch.tensor(
            [(c[0], c[1]) for c in conn_list],
            dtype=torch.long,
            device=device,
        )
        conn_weights = torch.tensor(
            [c[2] for c in conn_list],
            dtype=torch.float32,
            device=device,
        )
        conn_enabled = torch.tensor(
            [c[3] for c in conn_list],
            dtype=torch.bool,
            device=device,
        )
        conn_innovation = torch.tensor(
            [c[4] for c in conn_list],
            dtype=torch.long,
            device=device,
        )
    else:
        conn_indices = torch.zeros((0, 2), dtype=torch.long, device=device)
        conn_weights = torch.zeros(0, dtype=torch.float32, device=device)
        conn_enabled = torch.zeros(0, dtype=torch.bool, device=device)
        conn_innovation = torch.zeros(0, dtype=torch.long, device=device)

    nodes = NodeGenes(
        node_ids=node_ids,
        node_types=node_types,
        node_activations=node_activations,
    )

    connections = ConnectionGenes(
        conn_indices=conn_indices,
        conn_weights=conn_weights,
        conn_enabled=conn_enabled,
        conn_innovation=conn_innovation,
    )

    genome = Genome(
        nodes=nodes,
        connections=connections,
        fitness_score=torch.tensor(0.0, device=device),
    )
    return genome


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    genome = create_genome(
        input_nodes=2,
        hidden_nodes=2,
        output_nodes=3,
        device=device,
    )
    x = torch.tensor([0.7, 0.2], device=device)
    y = forward(genome, x)
    print(f"Inputs: {x}")
    print(f"Outputs: {y}")


if __name__ == "__main__":
    main()
