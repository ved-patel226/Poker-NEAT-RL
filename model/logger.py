import os
from datetime import datetime
from enum import Enum
from dataclasses import is_dataclass, asdict
from torch.utils.tensorboard import SummaryWriter
import json


class TensorFlowLogger:
    def __init__(
        self, log_dir: str = "runs", game_log_path: str = "poker-ai-game.json"
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"neat_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)
        self.game_log_path = os.path.abspath(game_log_path)
        print(f"Logging to: {self.log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        num_species: int,
        population_size: int,
    ):
        self.writer.add_scalar("fitness/best", best_fitness, generation)
        self.writer.add_scalar("fitness/average", avg_fitness, generation)
        self.writer.add_scalar("population/species_count", num_species, generation)
        self.writer.add_scalar("population/size", population_size, generation)

    def log_network(self, generation: int, num_nodes: int, num_connections: int):
        self.writer.add_scalar("network/num_nodes", num_nodes, generation)
        self.writer.add_scalar("network/num_connections", num_connections, generation)

    def log_species_stats(
        self,
        generation: int,
        species_id: int,
        species_size: int,
        species_best_fitness: float,
        species_avg_fitness: float,
    ):
        self.writer.add_scalar(f"species/{species_id}/size", species_size, generation)
        self.writer.add_scalar(
            f"species/{species_id}/best_fitness", species_best_fitness, generation
        )
        self.writer.add_scalar(
            f"species/{species_id}/avg_fitness", species_avg_fitness, generation
        )

    def log_population_diversity(
        self, generation: int, fitness_std: float, avg_fitness_per_species: float
    ):
        self.writer.add_scalar("population/fitness_std", fitness_std, generation)
        self.writer.add_scalar(
            "population/avg_fitness_per_species", avg_fitness_per_species, generation
        )

    def log_hands(self, generation: int, traces: list[dict]):
        if not traces:
            return

        # sample_count = min(3, len(traces))

        # for sample_idx, trace in enumerate(sample_traces):
        #     if not trace:
        #         continue

        #     # self.writer.add_text(
        #     #     f"hands/g{generation:05d}/sample_{sample_idx}",
        #     #     self._format_hand_trace(trace),
        #     #     generation,
        #     # )

        payload = {
            "generation": generation,
            "hand_count": len(traces),
            "games": [self._serialize_value(trace) for trace in traces if trace],
        }

        with open(self.game_log_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def _format_hand_trace(self, trace: list[dict]) -> str:
        lines = [f"# Hand trace ({len(trace)} actions)", ""]

        for step_idx, event in enumerate(trace, start=1):
            state = event["state"]
            action = event["action"]
            lines.append(
                f"{step_idx}. P{event['player']} -> action={self._action_name(action['type'])} "
                f"amount={action['amount']} | street={state['street']} | pot={state['pot']} | "
                f"current_bet={state['current_bet']} | stacks={self._stack_summary(state)}"
            )

        final_state = trace[-1]["state"]
        lines.extend(
            [
                "",
                "## Final State",
                f"- street: {final_state['street']}",
                f"- pot: {final_state['pot']}",
                f"- acting_idx: {final_state['acting_idx']}",
                f"- winner: {final_state['winner']}",
                f"- hand_over: {final_state['hand_over']}",
            ]
        )

        return "\n".join(lines)

    def _stack_summary(self, state: dict) -> str:
        players = state.get("players", [])
        return ", ".join(f"P{player['index']}:{player['stack']}" for player in players)

    def _action_name(self, action_type: int) -> str:
        return {0: "fold", 1: "call", 2: "raise"}.get(
            action_type, f"type_{action_type}"
        )

    def _serialize_value(self, value):
        if is_dataclass(value):
            value = asdict(value)

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, dict):
            return {key: self._serialize_value(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]

        if isinstance(value, tuple):
            return [self._serialize_value(item) for item in value]

        return value

    def log_figure(self, tag: str, figure, step: int):
        self.writer.add_figure(tag, figure, step, close=False)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
        print(f"Training logs saved to: {self.log_dir}")
