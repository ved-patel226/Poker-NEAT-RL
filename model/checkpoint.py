import os
import pickle
from datetime import datetime
from pathlib import Path


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save_genome(self, genome, filename: str, generation: int = None):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint_data = {
            "genome": genome,
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        return checkpoint_path

    def load_genome(self, filename: str):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)

        genome = checkpoint_data["genome"]
        metadata = {
            "timestamp": checkpoint_data.get("timestamp"),
            "generation": checkpoint_data.get("generation"),
        }

        return genome, metadata

    def save_best_genome(self, genome, generation: int, fitness: float):
        filename = "best_genome.pkl"
        path = self.save_genome(genome, filename, generation)
        return path

    def save_training_state(
        self, state_dict: dict, filename: str = "training_state.pkl"
    ):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint_data = {
            "state": state_dict,
            "timestamp": datetime.now().isoformat(),
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        return checkpoint_path

    def load_training_state(self, filename: str = "training_state.pkl"):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)

        state = checkpoint_data["state"]
        metadata = {
            "timestamp": checkpoint_data.get("timestamp"),
        }

        return state, metadata

    def list_checkpoints(self):
        if not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pkl")]
        return sorted(checkpoints, reverse=True)

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > keep_last_n:
            for old_checkpoint in checkpoints[keep_last_n:]:
                old_path = os.path.join(self.checkpoint_dir, old_checkpoint)
                os.remove(old_path)
                print(f"Removed old checkpoint: {old_checkpoint}")
