import argparse
import copy
import os
import pickle
import random
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch

from model.NEAT.data_structures import Genome


from env.poker import Observation
from env.states import Action
from model.logger import TensorFlowLogger
from model.NEAT.genome_operations_torch import forward


def load_genome(checkpoint_path: str):
    with open(checkpoint_path, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "genome" in payload:
        return payload["genome"], payload.get("generation", 0)

    return payload, 0


def move_genome_to_device(genome: Genome, device: str):
    genome.nodes.node_ids = genome.nodes.node_ids.to(device)
    genome.nodes.node_types = genome.nodes.node_types.to(device)
    genome.nodes.node_activations = genome.nodes.node_activations.to(device)
    genome.connections.conn_indices = genome.connections.conn_indices.to(device)
    genome.connections.conn_weights = genome.connections.conn_weights.to(device)
    genome.connections.conn_enabled = genome.connections.conn_enabled.to(device)
    genome.connections.conn_innovation = genome.connections.conn_innovation.to(device)
    genome.fitness_score = genome.fitness_score.to(device)
    return genome


def choose_action(genome, state_tensor, bounds, state_dict, device: str) -> Action:
    with torch.inference_mode():
        out = forward(genome, state_tensor.unsqueeze(0)).squeeze(0)

    logits = out[:3]

    valid_mask = torch.tensor(
        [
            bounds["can_fold"],
            bounds["can_call"],
            bounds["can_raise"],
        ],
        dtype=torch.bool,
        device=logits.device,
    )

    masked_logits = torch.where(
        valid_mask,
        logits,
        torch.full_like(logits, -1e9),
    )

    type_picked = int(torch.argmax(masked_logits).item())

    amount = 0
    if type_picked == 2:
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


def play_hands(genomes, num_hands: int, device: str):
    obs = Observation()
    traces = []

    for _ in range(num_hands):
        hand_log = []
        obs.reset()
        state_dict = obs.get_state()

        while not state_dict["hand_over"]:
            player_idx = state_dict["acting_idx"]
            if player_idx is None:
                break

            genome = genomes[player_idx]
            tensor_input = obs.get_tensor_input(player_idx).to(device)
            bounds = obs.get_action_bounds()

            action = choose_action(
                genome,
                tensor_input,
                bounds,
                state_dict,
                device,
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
                        break

                    obs.send_action(action)
                    state_dict = obs.get_state()
                except Exception:
                    break

        traces.append(hand_log)

    return traces


def main():
    parser = argparse.ArgumentParser(
        description="Play poker with 6 copies of a saved NEAT genome"
    )
    parser.add_argument(
        "--checkpoint",
        default="best_genome.pkl",
        help="Path to best_genome.pkl",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=20,
        help="Number of hands to simulate",
    )
    parser.add_argument(
        "--output",
        default="poker-ai-game.json",
        help="Output path for the game log JSON",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu or cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    genome, generation = load_genome(args.checkpoint)
    genomes = [copy.deepcopy(genome) for _ in range(6)]
    genomes = [move_genome_to_device(g, device) for g in genomes]

    traces = play_hands(genomes, args.hands, device)

    logger = TensorFlowLogger(game_log_path=args.output)
    logger.log_hands(generation, traces)
    logger.close()

    print(f"Saved {len(traces)} hands to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
