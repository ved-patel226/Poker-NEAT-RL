import sys
import os
import torch
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.NEAT.genome_operations_torch import forward
from model.checkpoint import CheckpointManager


def _ints_to_bits(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    shifts = torch.arange(num_bits, device=values.device)
    return ((values.unsqueeze(1) >> shifts) & 1).float()


def _bits_to_int(bits: torch.Tensor) -> int:
    weights = 2 ** torch.arange(bits.numel(), device=bits.device)
    return int((bits * weights).sum().item())


def test_genome(genome, num_tests: int = 10, device: str = "cpu"):
    test_a = torch.randint(0, 256, (num_tests,), device=device)
    test_b = torch.randint(0, 256, (num_tests,), device=device)
    test_sum = test_a + test_b

    test_a_bits = _ints_to_bits(test_a, 8)
    test_b_bits = _ints_to_bits(test_b, 8)
    test_inputs = torch.cat([test_a_bits, test_b_bits], dim=1)

    with torch.inference_mode():
        test_outputs = forward(genome, test_inputs)

    test_pred_bits = (test_outputs > 0).float()

    print("\nTest Results:")
    print("a + b = predicted (target)")
    print("-" * 50)

    correct = 0
    for i in range(num_tests):
        pred_val = _bits_to_int(test_pred_bits[i])
        target_val = int(test_sum[i])
        is_correct = pred_val == target_val
        correct += is_correct
        status = "✓" if is_correct else "✗"
        print(
            f"{status} {int(test_a[i])} + {int(test_b[i])} = {pred_val} ({target_val})"
        )

    print("-" * 50)
    print(f"Accuracy: {correct}/{num_tests} ({100*correct/num_tests:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Load and test saved NEAT genomes")
    parser.add_argument("checkpoint_dir", help="Directory containing checkpoints")
    parser.add_argument(
        "--genome",
        default=None,
        help="Specific genome file to load (default: latest best_genome)",
    )
    parser.add_argument("--state", default=None, help="Training state file to load")
    parser.add_argument(
        "--test", type=int, default=10, help="Number of test cases to run"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run on (cpu or cuda)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available checkpoints"
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {args.checkpoint_dir}")
        return

    checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)

    if args.list:
        checkpoints = checkpoint_manager.list_checkpoints()
        if not checkpoints:
            print("No checkpoints found.")
        else:
            print("Available checkpoints:")
            for ckpt in checkpoints:
                print(f"  - {ckpt}")
        return

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {device}")

    if args.genome:
        genome_file = args.genome
    else:
        # best genome file
        checkpoints = checkpoint_manager.list_checkpoints()
        best_genomes = [c for c in checkpoints if c.startswith("best_genome_")]
        if not best_genomes:
            print("Error: No best_genome checkpoints found")
            return
        genome_file = best_genomes[0]

    print(f"\nLoading genome: {genome_file}")
    try:
        genome, metadata = checkpoint_manager.load_genome(genome_file)
        print(f"Loaded genome metadata:")
        print(f"  Generation: {metadata['generation']}")
        print(f"  Timestamp: {metadata['timestamp']}")
        print(f"  Nodes: {genome.nodes.node_ids.shape[0]}")
        print(f"  Connections: {genome.connections.conn_indices.shape[0]}")
    except Exception as e:
        print(f"Error loading genome: {e}")
        return

    # TODO: make a func to do this
    genome.nodes.node_ids = genome.nodes.node_ids.to(device)
    genome.nodes.node_types = genome.nodes.node_types.to(device)
    genome.nodes.node_activations = genome.nodes.node_activations.to(device)
    genome.connections.conn_indices = genome.connections.conn_indices.to(device)
    genome.connections.conn_weights = genome.connections.conn_weights.to(device)
    genome.connections.conn_enabled = genome.connections.conn_enabled.to(device)
    genome.connections.conn_innovation = genome.connections.conn_innovation.to(device)
    genome.fitness_score = genome.fitness_score.to(device)

    test_genome(genome, num_tests=args.test, device=device)

    if args.state:
        print(f"\nLoading training state: {args.state}")
        try:
            state, state_metadata = checkpoint_manager.load_training_state(args.state)
            print(f"Training state loaded at generation {state['generation']}")
            print(f"Best fitness: {state['best_overall_fitness']:.6f}")
            print(f"Population size: {len(state['population'])}")
            print(f"Species count: {len(state['species_list'])}")
        except Exception as e:
            print(f"Error loading training state: {e}")


if __name__ == "__main__":
    main()
