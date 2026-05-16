import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorFlowLogger:
    def __init__(self, log_dir: str = "runs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"neat_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)
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

    def log_figure(self, tag: str, figure, step: int):
        self.writer.add_figure(tag, figure, step, close=False)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
        print(f"Training logs saved to: {self.log_dir}")
