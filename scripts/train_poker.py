import warnings

warnings.filterwarnings("ignore", message="There is no reason for this player to fold.")

import sys
import os
import torch
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.NEAT.data_structures import Genome, InnovationTracker
from model.NEAT.genome_operations_torch import create_genome, forward
from model.NEAT.evolution import run_generation

# from model.NEAT.visualization import visualize_population_and_best
from model.logger import TensorFlowLogger
from model.checkpoint import CheckpointManager

from env.poker import Observation
from env.states import Action


def choose_action(
    genome: Genome,
    state_tensor: torch.Tensor,
    bounds: dict,
    state_dict: dict,
    device: str,
) -> Action:
    with torch.inference_mode():
        out = forward(genome, state_tensor.unsqueeze(0)).squeeze(0)

    logits = out[:3]

    # avoid CPU-GPU syncs from .item()
    valid_mask = torch.tensor(
        [
            bounds["can_fold"],
            bounds["can_call"],
            bounds["can_raise"],
        ],
        dtype=torch.bool,
        device=logits.device,
    )

    # mask invalid actions with negative logits
    masked_logits = torch.where(
        valid_mask,
        logits,
        torch.full_like(logits, -1e9),
    )

    type_picked = int(torch.argmax(masked_logits).item())

    amount = 0

    if type_picked == 2:  # raise
        size_out = torch.sigmoid(out[3]).item()

        min_r = bounds["raise_amount_min"]
        max_r = bounds["raise_amount_max"]

        amount = int(min_r + size_out * (max_r - min_r))

    return Action(
        player=state_dict["acting_idx"],
        street=state_dict["street"],
        type=type_picked,
        amount=amount,
    )


from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import copy


def evaluate_table(args):

    table_genomes, config_hands_per_table = args

    # CPU is usually faster for tiny networks due to CUDA overhead
    worker_device = "cpu"

    genomes = [copy.deepcopy(g) for g in table_genomes]

    obs = Observation()

    table_fitness = [0.0 for _ in genomes]

    for _ in range(config_hands_per_table):
        obs.reset()
        state_dict = obs.get_state()

        start_stacks = [p["stack"] for p in state_dict["players"]]

        while not state_dict["hand_over"]:
            player_idx = state_dict["acting_idx"]

            if player_idx is None:
                break

            genome = genomes[player_idx]

            tensor_input = obs.get_tensor_input(player_idx).to(worker_device)
            bounds = obs.get_action_bounds()

            action = choose_action(
                genome,
                tensor_input,
                bounds,
                state_dict,
                worker_device,
            )

            try:
                obs.send_action(action)

            except Exception:
                table_fitness[player_idx] -= 1000.0  #  penalty on invalid actions

                try:
                    if bounds["can_fold"]:
                        obs.send_action(
                            Action(
                                player=player_idx,
                                street=state_dict["street"],
                                type=0,
                                amount=0,
                            )
                        )

                    elif bounds["can_call"]:
                        obs.send_action(
                            Action(
                                player=player_idx,
                                street=state_dict["street"],
                                type=1,
                                amount=0,
                            )
                        )

                    else:
                        break

                except Exception:
                    break

            state_dict = obs.get_state()

        end_stacks = [p["stack"] for p in state_dict["players"]]

        for i in range(len(genomes)):
            delta = end_stacks[i] - start_stacks[i]
            table_fitness[i] += delta

    return table_fitness


def evaluate_poker_population(
    population: list[Genome],
    config_hands_per_table: int,
    device: str,
):
    for g in population:  # init fitness
        g.fitness_score = torch.tensor(0.0, device=device)

    table_size = 6

    # shuffle seating
    pop_indices = list(range(len(population)))
    random.shuffle(pop_indices)

    num_tables = len(population) // table_size

    table_jobs = []

    for t_idx in range(num_tables):
        table_members = pop_indices[t_idx * table_size : (t_idx + 1) * table_size]

        genomes = [population[i] for i in table_members]

        table_jobs.append((table_members, genomes))

    worker_args = [(genomes, config_hands_per_table) for _, genomes in table_jobs]

    max_workers = min(os.cpu_count() or 1, num_tables)

    # use for CUDA
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as executor:

        futures = {
            executor.submit(evaluate_table, args): idx
            for idx, args in enumerate(worker_args)
        }

        for future in as_completed(futures):
            table_idx = futures[future]

            table_members, _ = table_jobs[table_idx]

            try:
                deltas = future.result()

                for local_idx, genome_idx in enumerate(table_members):
                    population[genome_idx].fitness_score += deltas[local_idx]

            except Exception as e:
                print(f"Table worker failed: {e}")

                # penalize failed table
                for genome_idx in table_members:
                    population[genome_idx].fitness_score -= 1000.0

    min_fitness = min(g.fitness_score.item() for g in population)

    shift = abs(min_fitness) + 1.0 if min_fitness <= 0 else 0.0

    if shift > 0:
        for g in population:
            g.fitness_score += shift


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    population_size = 300  # divisible by 6
    num_generations = 10_000
    config_hands_per_table = 6

    print("Initializing poker population...")
    population = [
        create_genome(input_nodes=314, hidden_nodes=0, output_nodes=4, device=device)
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

    for gen in range(num_generations):
        print(f"\nGeneration {gen + 1}/{num_generations}")
        print("-" * 60)

        # eval entire population via self-play tables
        evaluate_poker_population(population, config_hands_per_table, device)

        def evaluate_fn(genome: Genome) -> torch.Tensor:
            return genome.fitness_score.clone().detach()

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

        # fig = None
        # if (gen + 1) % 5 == 0 or gen == num_generations - 1:
        #     fig = visualize_population_and_best(
        #         population, best_overall, generation=gen + 1, species_list=species_list
        #     )
        # if (gen + 1) % 100 == 0 and fig is not None:
        #     logger.log_figure("viz/population_and_best", fig, gen + 1)

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


if __name__ == "__main__":
    main()
