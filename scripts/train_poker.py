import warnings

warnings.filterwarnings("ignore", message="There is no reason for this player to fold.")

import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as torch_mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict

from envyaml import EnvYAML

from env.poker import Observation
from env.states import Action
from model.checkpoint import CheckpointManager

# from model.NEAT.visualization import visualize_population_and_best
from model.logger import TensorFlowLogger
from model.NEAT.data_structures import Genome, InnovationTracker
from model.NEAT.evolution import run_generation
from model.NEAT.genome_operations_torch import create_genome, forward

torch_mp.set_sharing_strategy("file_system")


def choose_action(
    genome: Genome,
    state_tensor: torch.Tensor,
    bounds: dict,
    state_dict: dict,
    device: str,
    epsilon: float = 0.1,
    temperature: float = 1.0,
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

    if random.random() < epsilon:
        valid_indices = torch.where(valid_mask)[0]
        if valid_indices.numel() > 0:
            type_picked = int(
                valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
            )
        else:
            type_picked = 0
    else:
        probs = torch.softmax(masked_logits / temperature, dim=0)
        if torch.isfinite(probs).all() and probs.sum().item() > 0:
            type_picked = int(torch.multinomial(probs, 1).item())
        else:
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


def choose_action_baseline(strategy, bounds, state_dict) -> Action:
    player_idx = state_dict["acting_idx"]

    if strategy == "random":
        valid = [
            i
            for i, v in enumerate(
                [bounds["can_fold"], bounds["can_call"], bounds["can_raise"]]
            )
            if v
        ]
        type_picked = random.choice(valid) if valid else 0
    else:
        type_picked = 1 if bounds["can_call"] else 0

    amount = 0

    if type_picked == 2:  # raise
        # all this just makes it so small raises are more common than large ones
        min_raise = bounds["raise_amount_min"]
        max_raise = bounds["raise_amount_max"]

        x = random.expovariate(3)  # larger number = more small raises
        x = min(x / 3, 1.0)

        amount = int(min_raise + x * (max_raise - min_raise))

    return Action(
        player=player_idx,
        street=state_dict["street"],
        type=type_picked,
        amount=amount,
    )


def evaluate_vs_baseline(genome, strategy, n_hands, device) -> float:
    obs = Observation()
    big_blind = 400
    total_delta = 0.0

    for hand_num in range(n_hands):
        obs.reset()
        state_dict = obs.get_state()
        genome_seat = hand_num % 6
        start_stack = [p["stack"] for p in state_dict["players"]][genome_seat]

        while not state_dict["hand_over"]:
            player_idx = state_dict["acting_idx"]
            if player_idx is None:
                break
            bounds = obs.get_action_bounds()
            if player_idx == genome_seat:
                tensor_input = obs.get_tensor_input(player_idx).to(device)
                action = choose_action(
                    genome, tensor_input, bounds, state_dict, device, epsilon=0.0
                )
            else:
                action = choose_action_baseline(
                    strategy,
                    bounds,
                    state_dict,
                )

            try:
                obs.send_action(action)
                state_dict = obs.get_state()
            except ValueError as e:
                print(f"Invalid action: {e}")
                continue
        end_stack = [p["stack"] for p in state_dict["players"]][genome_seat]
        total_delta += float(end_stack - start_stack)

    return (total_delta / n_hands) / big_blind * 100


def evaluate_table(args):
    table_genomes, config_hands_per_table = args
    traces = []

    worker_device = "cpu"
    genomes = table_genomes  # forward() is read-only; no deepcopy needed

    epsilon = float(os.environ.get("ENV_ACTION_EPSILON", 0.1))
    temperature = float(os.environ.get("ENV_ACTION_TEMPERATURE", 1.0))
    if temperature <= 0:
        temperature = 1.0

    obs = Observation()

    table_fitness = [0.0 for _ in genomes]

    for _ in range(config_hands_per_table):
        hand_log = []
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
                epsilon,
                temperature,
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
                end_stacks[i] - start_stacks[i]
            )  # TODO: make sure ts works (if not, delta???)

    return table_fitness, traces


def evaluate_poker_population(
    population: list[Genome],
    config_hands_per_table: int,
    device: str,
    generation: int,
    logger: TensorFlowLogger,
    log_hands_every: int = 10_000,
    rounds: int = 5,
    baseline_mix: float = 0.7,
):
    for g in population:  # init fitness
        g.fitness_score = torch.tensor(0.0, device=device)

    # each genome plays against a fixed "call" baseline for a primary fitness signal.
    # baseline_mix controls what fraction of the fitness comes from this.
    baseline_hands = max(config_hands_per_table * 2, 20)

    def _baseline_job(args):
        genome, strategy, n, dev = args
        return evaluate_vs_baseline(genome, strategy, n, dev)

    baseline_args = [(g, "call", baseline_hands, device) for g in population]
    max_bl_workers = max(1, min(os.cpu_count() or 8, len(population)))
    with ThreadPoolExecutor(max_workers=max_bl_workers) as executor:
        bl_futures = {executor.submit(_baseline_job, args): i for i, args in enumerate(baseline_args)}
        for future in as_completed(bl_futures):
            idx = bl_futures[future]
            try:
                bb100 = future.result()
                population[idx].fitness_score += baseline_mix * bb100
            except Exception as e:
                print(f"Baseline eval failed for genome {idx}: {e}")

    # self-play, relative
    if baseline_mix < 1.0:
        table_size = 6
        num_tables = len(population) // table_size

        all_table_jobs = []
        for _ in range(rounds):
            pop_indices = list(range(len(population)))
            random.shuffle(pop_indices)
            for t_idx in range(num_tables):
                table_members = pop_indices[t_idx * table_size : (t_idx + 1) * table_size]
                genomes = [population[i] for i in table_members]
                all_table_jobs.append((table_members, genomes))

        max_workers = max(1, min(os.cpu_count() or 8, len(all_table_jobs)))
        worker_args = [(genomes, config_hands_per_table) for _, genomes in all_table_jobs]

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
                table_members, _ = all_table_jobs[table_idx]

                try:
                    deltas, traces = future.result()
                    if log_hands_every > 0 and generation % log_hands_every == 0:
                        logger.log_hands(generation, traces)

                    for local_idx, genome_idx in enumerate(table_members):
                        # scale chip delta to bb/100 for comparable units with baseline score
                        big_blind = 400
                        bb100_selfplay = (deltas[local_idx] / config_hands_per_table) / big_blind * 100
                        population[genome_idx].fitness_score += (1.0 - baseline_mix) * bb100_selfplay

                except Exception as e:
                    print(f"Table worker failed: {e}")
                    for genome_idx in table_members:
                        population[genome_idx].fitness_score -= 100.0


def set_env_recursive(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}_{k.upper()}" if prefix else k.upper()
        if isinstance(v, dict):
            set_env_recursive(v, key)
        else:
            os.environ[key] = str(v)


def main():
    env = EnvYAML(yaml_file="config.yaml")
    set_env_recursive(dict(env))

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
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

    baseline_eval_every = int(os.environ.get("ENV_BASELINE_EVAL_EVERY", 5))
    baseline_eval_hands = int(os.environ.get("ENV_BASELINE_EVAL_HANDS", 500))

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

        if (gen + 1) % baseline_eval_every == 0:
            print("   Running baseline evaluation...")

            bb100_random = evaluate_vs_baseline(
                best,
                "random",
                baseline_eval_hands,
                device,
            )

            bb100_call = evaluate_vs_baseline(
                best,
                "call",
                baseline_eval_hands,
                device,
            )

            logger.log_scalar(
                "baseline/vs_random_bb100",
                bb100_random,
                gen + 1,
            )

            logger.log_scalar(
                "baseline/vs_call_bb100",
                bb100_call,
                gen + 1,
            )

            print(
                f"   Baseline — "
                f"vs random: {bb100_random:+.1f} BB/100 | "
                f"vs call: {bb100_call:+.1f} BB/100"
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

        print(
            f"   Current best: {best_fitness:.6f} "
            f"(nodes: {best.nodes.node_ids.shape[0]}, "
            f"conns: {best.connections.conn_indices.shape[0]})"
        )
        print(f"   Overall best: {best_overall_fitness:.6f}")

        pop_nodes = [g.nodes.node_ids.shape[0] for g in population]
        pop_conns = [g.connections.conn_indices.shape[0] for g in population]
        print(
            "   Pop avg/max: "
            f"nodes {np.mean(pop_nodes):.1f}/{np.max(pop_nodes)}, "
            f"conns {np.mean(pop_conns):.1f}/{np.max(pop_conns)}"
        )

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
            best.nodes.node_ids.shape[0],
            best.connections.conn_indices.shape[0],
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

        last_model_path = os.path.join(logger.log_dir, "last_genome.pt")

        torch.save(
            {
                "generation": gen + 1,
                "fitness": best.fitness_score.item(),
                "genome": best,
            },
            last_model_path,
        )

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
