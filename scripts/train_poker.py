import warnings

warnings.filterwarnings("ignore", message="There is no reason for this player to fold.")

import sys
import os
import torch
import torch.multiprocessing as torch_mp
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

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import copy

from dataclasses import asdict

from envyaml import EnvYAML

torch_mp.set_sharing_strategy("file_system")


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


def evaluate_table(args):
    table_genomes, config_hands_per_table = args
    traces = []

    # CPU is faster for tiny nets
    # TODO: can we try to make cuda faster by batching or nah?
    worker_device = "cpu"

    genomes = [copy.deepcopy(g) for g in table_genomes]

    obs = Observation()

    table_fitness = [0.0 for _ in genomes]

    for _ in range(config_hands_per_table):
        hand_log = []
        obs.reset()
        state_dict = obs.get_state()

        start_stacks = [obs.nominal_stack for _ in state_dict["players"]]

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

            hand_log.append(
                {
                    "player": player_idx,
                    "state": state_dict,
                    "action": asdict(action),
                }
            )

            try:
                obs.send_action(action)
                state_dict = obs.get_state()

            except Exception:
                table_fitness[player_idx] -= 1000.0  # penalty on invalid actions

                try:
                    if bounds["can_fold"]:
                        action = Action(
                            player=player_idx,
                            street=state_dict["street"],
                            type=0,
                            amount=0,
                        )

                    elif bounds["can_call"]:
                        action = Action(
                            player=player_idx,
                            street=state_dict["street"],
                            type=1,
                            amount=0,
                        )

                    else:
                        print("INVALID ACTION: can't fold or call")
                        break

                except Exception:
                    break

                state_dict = obs.get_state()
                obs.send_action(action)

        traces.append(hand_log)

        end_stacks = [p["stack"] for p in state_dict["players"]]

        for i in range(len(genomes)):
            table_fitness[i] += float(
                end_stacks[i]
            )  # TODO: make sure ts works (if not, delta???)

    return table_fitness, traces


def evaluate_poker_population(
    population: list[Genome],
    config_hands_per_table: int,
    device: str,
    generation: int,
    logger: TensorFlowLogger,
    log_hands_every: int = 1,
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

    max_workers = max(1, min(8, num_tables))

    # CPU self-play is better served by threads here: it avoids shipping many
    # tensor-backed Genome objects through multiprocessing shared memory.
    if device == "cpu":
        executor_cls = ThreadPoolExecutor
        executor_kwargs = {}
    else:
        executor_cls = ProcessPoolExecutor
        executor_kwargs = {"mp_context": mp.get_context("spawn")}

    with executor_cls(max_workers=max_workers, **executor_kwargs) as executor:

        futures = {
            executor.submit(evaluate_table, args): idx
            for idx, args in enumerate(worker_args)
        }

        for future in as_completed(futures):
            table_idx = futures[future]

            table_members, _ = table_jobs[table_idx]

            try:
                deltas, traces = future.result()
                if log_hands_every > 0 and generation % log_hands_every == 0:
                    logger.log_hands(generation, traces)

                for local_idx, genome_idx in enumerate(table_members):
                    population[genome_idx].fitness_score += deltas[local_idx]

            except Exception as e:
                print(f"Table worker failed: {e}")

                # penalize failed table
                for genome_idx in table_members:
                    population[genome_idx].fitness_score -= 1000.0


def set_env_recursive(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}_{k.upper()}" if prefix else k.upper()
        if isinstance(v, dict):
            set_env_recursive(v, key)
        else:
            os.environ[key] = str(v)


def main():
    import os

    env = EnvYAML(yaml_file="config.yaml")
    set_env_recursive(dict(env))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using device: {device}\n")

    print("Initializing poker population...")
    population = [
        create_genome(input_nodes=39, hidden_nodes=0, output_nodes=4, device=device)
        for _ in range(int(os.environ["ENV_POPULATION_SIZE"]))
    ]

    tracker = InnovationTracker()
    species_list = []
    logger = TensorFlowLogger()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.join(logger.log_dir, "checkpoints")
    )
    log_hands_every = int(os.environ.get("ENV_LOG_HANDS_EVERY", 10))

    print(f"Population size: {int(os.environ['ENV_POPULATION_SIZE'])}")
    print(f"Generations: {int(os.environ['ENV_NUM_GENERATIONS'])}\n")
    print("=" * 60)

    best_overall = None
    best_overall_fitness = 0

    generations = []
    best_fitnesses = []
    avg_fitnesses = []
    stdev_fitnesses = []
    species_sizes_history = []

    for gen in range(int(os.environ["ENV_NUM_GENERATIONS"])):
        print(f"\nGeneration {gen + 1}/{int(os.environ['ENV_NUM_GENERATIONS'])}")
        print("-" * 60)

        # eval entire population via self-play tables
        evaluate_poker_population(
            population,
            int(os.environ["ENV_CONFIG_HANDS_PER_TABLE"]),
            device,
            gen,
            logger,
            log_hands_every=log_hands_every,
        )

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

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final best fitness: {best_overall_fitness:.6f}")
    print(f"Number of species: {len(species_list)}")
    print(f"Network complexity:")
    print(f"  - Nodes: {best_overall.nodes.node_ids.shape[0]}")
    print(f"  - Connections: {best_overall.connections.conn_indices.shape[0]}")

    final_checkpoint_path = checkpoint_manager.save_best_genome(
        best_overall, int(os.environ["ENV_NUM_GENERATIONS"]), best_overall_fitness
    )
    print(f"Saved final best genome: {final_checkpoint_path}")
    logger.close()


if __name__ == "__main__":
    main()
