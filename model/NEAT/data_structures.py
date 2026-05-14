import torch
from dataclasses import dataclass


@dataclass
class NodeGenes:
    node_ids: torch.Tensor  # (N,)
    node_types: torch.Tensor  # (N,) 0=input, 1=hidden, 2=output, 3=bias
    node_activations: torch.Tensor  # (N,) 0=relu, 1=tanh, 2=sigmoid


@dataclass
class ConnectionGenes:
    conn_indices: torch.Tensor  # (E, 2): (in_node, out_node)
    conn_weights: torch.Tensor  # (E,)
    conn_enabled: torch.Tensor  # (E,) bool or int
    conn_innovation: torch.Tensor  # (E,)


@dataclass
class Genome:
    nodes: NodeGenes
    connections: ConnectionGenes
    fitness_score: torch.Tensor  # scalar


class InnovationTracker:
    __slots__ = ("history", "next_innovation_number")

    def __init__(self):
        self.history = {}
        self.next_innovation_number = 0

    def get_innovation_number(self, in_node: int, out_node: int) -> int:
        if (in_node, out_node) in self.history:
            return self.history[(in_node, out_node)]
        else:
            self.history[(in_node, out_node)] = self.next_innovation_number
            self.next_innovation_number += 1
            return self.history[(in_node, out_node)]
