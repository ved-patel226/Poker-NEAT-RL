import torch

try:
    from .data_structures import Genome, NodeGene, ConnectionGene
except ImportError:
    from data_structures import Genome, NodeGene, ConnectionGene


def forward(genome, x):
    # ==== INPUT NODES FIRST =====

    values = {}
    input_nodes = [n for n in genome.node_genes if n.type == "input"]

    for node, val in zip(input_nodes, x):  # needs to be the same size
        values[node.id] = torch.tensor(val, dtype=torch.float32)

    order = sort_genes(genome)

    for id in order:

        if id in values:
            continue  # input/bias node

        incoming = [c for c in genome.connection_genes if c.N_out == id and c.enabled]

        total = torch.tensor(0.0)
        for conn in incoming:  # sum all the inputs
            total += values[conn.N_in] * torch.tensor(conn.weight)

        node = next(
            n for n in genome.node_genes if n.id == id
        )  # quick search for the node

        if node.activation == "sigmoid":
            values[id] = torch.sigmoid(total)
        elif node.activation == "tanh":
            values[id] = torch.tanh(total)
        elif node.activation == "relu":
            values[id] = torch.relu(total)
        else:
            raise NotImplementedError("unkown activation encountered")

    output_nodes = [n for n in genome.node_genes if n.type == "output"]
    if len(output_nodes) == 1:
        return values[output_nodes[0].id]
    return torch.stack([values[n.id] for n in output_nodes])


def sort_genes(genome):
    # dependency map
    deps = {
        n.id: set() for n in genome.node_genes
    }  # what value MUST i know before I can compute the next one?
    for conn in genome.connection_genes:
        if conn.enabled:
            deps[conn.N_out].add(conn.N_in)

    order = []  # the order in which to compute nodes
    visited = set()

    def visit(node_id):
        if node_id in visited:
            return
        for dep in deps[node_id]:
            visit(dep)  # process dependencies first
        visited.add(node_id)
        order.append(node_id)

    for node in genome.node_genes:  # reverse doesn't do anything except add compute
        visit(node.id)

    return order


def main() -> None:
    node_genes = [
        NodeGene(id=0, type="input", activation="relu"),
        NodeGene(id=1, type="input", activation="relu"),
        NodeGene(id=2, type="output", activation="sigmoid"),
    ]
    connection_genes = [
        ConnectionGene(N_in=0, N_out=2, weight=1.0, enabled=True, innovation=0),
        ConnectionGene(N_in=1, N_out=2, weight=-1.0, enabled=True, innovation=1),
    ]
    genome = Genome(
        node_genes=node_genes, connection_genes=connection_genes, fitness_score=0
    )

    inputs = [0.5, 1.0]
    output = forward(genome, inputs)
    print("Sample genome inputs:", inputs)
    print("Sample genome output:", output.item())


if __name__ == "__main__":
    main()
