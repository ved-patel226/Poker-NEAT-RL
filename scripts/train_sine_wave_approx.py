import sys
import os
import torch
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.NEAT.data_structures import Genome, InnovationTracker
from model.NEAT.genome_operations_torch import create_genome, forward
from model.NEAT.evolution import run_generation
from model.NEAT.visualization import visualize_population_and_best
from model.logger import TensorFlowLogger
from model.checkpoint import CheckpointManager


def _ints_to_bits(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    shifts = torch.arange(num_bits, device=values.device)
    return ((values.unsqueeze(1) >> shifts) & 1).float()


def _bits_to_int(bits: torch.Tensor) -> int:
    weights = 2 ** torch.arange(bits.numel(), device=bits.device)
    return int((bits * weights).sum().item())


def _make_adder_batch(
    device: str, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    a_vals = torch.randint(0, 256, (batch_size,), device=device)
    b_vals = torch.randint(0, 256, (batch_size,), device=device)
    sums = a_vals + b_vals

    a_bits = _ints_to_bits(a_vals, 8)
    b_bits = _ints_to_bits(b_vals, 8)
    x_vals = torch.cat([a_bits, b_bits], dim=1)
    target_bits = _ints_to_bits(sums, 9)

    return x_vals, target_bits


def evaluate_binary_adder(
    genome: Genome, x_vals: torch.Tensor, target_bits: torch.Tensor
) -> torch.Tensor:
    with torch.inference_mode():
        y_pred = forward(genome, x_vals)

        mse = torch.nn.functional.mse_loss(y_pred, target_bits)
        fitness = 1.0 / (1.0 + mse)

    return fitness


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    population_size = 500
    batch_size = 256
    num_generations = 10_000

    print("Initializing population...")
    population = [
        create_genome(input_nodes=16, hidden_nodes=0, output_nodes=9, device=device)
        for _ in range(population_size)
    ]

    tracker = InnovationTracker()
    species_list = []
    logger = TensorFlowLogger()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.join(logger.log_dir, "checkpoints")
    )

    print(f"Population size: {population_size}")
    print(f"Generations: {num_generations}\n")
    print("=" * 60)

    best_overall = None
    best_overall_fitness = 0

    generations = []
    best_fitnesses = []
    avg_fitnesses = []
    stdev_fitnesses = []
    species_sizes_history = []

    print("Generating static training dataset...")
    x_vals, target_bits = _make_adder_batch(
        device, 1024
    )  # fixed dataset of 1024 examples

    for gen in range(num_generations):
        print(f"\nGeneration {gen + 1}/{num_generations}")
        print("-" * 60)

        def evaluate_fn(genome: Genome) -> torch.Tensor:
            return evaluate_binary_adder(genome, x_vals, target_bits)

        population, species_list, best = run_generation(
            population, species_list, tracker, evaluate_fn
        )

        best_fitness = best.fitness_score.item()

        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall = best
            print(f"New best fitness: {best_overall_fitness:.6f}")

            checkpoint_path = checkpoint_manager.save_best_genome(
                best_overall, gen + 1, best_overall_fitness
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        print(f"   Current best: {best_fitness:.6f}")
        print(f"   Overall best: {best_overall_fitness:.6f}")

        avg_fitness = np.mean([g.fitness_score.item() for g in population])
        fitness_std = np.std([g.fitness_score.item() for g in population])

        logger.log_generation(
            gen + 1,
            best_overall_fitness,
            avg_fitness,
            len(species_list),
            len(population),
        )
        logger.log_network(
            gen + 1,
            best_overall.nodes.node_ids.shape[0],
            best_overall.connections.conn_indices.shape[0],
        )

        # per-species
        for species in species_list:
            species_fitnesses = [g.fitness_score.item() for g in species.members]
            species_best = max(species_fitnesses)
            species_avg = np.mean(species_fitnesses)
            logger.log_species_stats(
                gen + 1,
                species.id,
                len(species.members),
                species_best,
                species_avg,
            )

        # population diversity
        all_fitnesses = [g.fitness_score.item() for g in population]
        if species_list:
            avg_fitness_per_species = np.mean(
                [
                    np.mean([g.fitness_score.item() for g in s.members])
                    for s in species_list
                ]
            )
        else:
            avg_fitness_per_species = avg_fitness
        logger.log_population_diversity(gen + 1, fitness_std, avg_fitness_per_species)

        generations.append(gen + 1)
        best_fitnesses.append(best_overall_fitness)
        avg_fitnesses.append(avg_fitness)
        stdev_fitnesses.append(fitness_std)

        species_sizes = {s.id: len(s.members) for s in species_list}
        species_sizes_history.append(species_sizes)

        fig = None
        if (gen + 1) % 5 == 0 or gen == num_generations - 1:
            fig = visualize_population_and_best(
                population, best_overall, generation=gen + 1, species_list=species_list
            )
        if (gen + 1) % 100 == 0 and fig is not None:
            logger.log_figure("viz/population_and_best", fig, gen + 1)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final best fitness: {best_overall_fitness:.6f}")
    print(f"Number of species: {len(species_list)}")
    print(f"Network complexity:")
    print(f"  - Nodes: {best_overall.nodes.node_ids.shape[0]}")
    print(f"  - Connections: {best_overall.connections.conn_indices.shape[0]}")

    final_checkpoint_path = checkpoint_manager.save_best_genome(
        best_overall, num_generations, best_overall_fitness
    )
    print(f"Saved final best genome: {final_checkpoint_path}")

    logger.close()

    print("\nNetwork predictions:")
    print("a + b = predicted (target)")
    print("-" * 50)
    test_a = torch.randint(0, 256, (5,), device=device)
    test_b = torch.randint(0, 256, (5,), device=device)
    test_sum = test_a + test_b

    test_a_bits = _ints_to_bits(test_a, 8)
    test_b_bits = _ints_to_bits(test_b, 8)
    test_inputs = torch.cat([test_a_bits, test_b_bits], dim=1)
    test_outputs = forward(best_overall, test_inputs)
    test_pred_bits = (test_outputs > 0).float()

    for i in range(5):
        pred_val = _bits_to_int(test_pred_bits[i])
        print(f"{int(test_a[i])} + {int(test_b[i])} = {pred_val} ({int(test_sum[i])})")


if __name__ == "__main__":
    main()
