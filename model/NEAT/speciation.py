"""
speciation: δ = (c1 × E / N) + (c2 × D / N) + (c3 × W̄)
E = num of exess genes
D = num of disjoint genes
N = num of genes in the larger genome
W̄ = average weight diff in matching genes
c1, c2, c3 = constants, YOU HAVE TO TUNE TS (normally: 1.0, 1.0, 0.4)


If δ < threshold, its the same species
"""

import random
import os

try:
    from .data_structures import Species
except:
    from data_structures import Species


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def compatibility_distance(genome_a, genome_b):
    # looks complicated because of the tensor bs but it just gets all connections for both genomes
    genes_a = {
        int(innov): (int(in_node), int(out_node), float(weight))
        for (in_node, out_node), weight, innov in zip(
            genome_a.connections.conn_indices.tolist(),
            genome_a.connections.conn_weights.tolist(),
            genome_a.connections.conn_innovation.tolist(),
        )
    }
    genes_b = {
        int(innov): (int(in_node), int(out_node), float(weight))
        for (in_node, out_node), weight, innov in zip(
            genome_b.connections.conn_indices.tolist(),
            genome_b.connections.conn_weights.tolist(),
            genome_b.connections.conn_innovation.tolist(),
        )
    }

    all_innovations = set(genes_a) | set(genes_b)
    max_innovation_a = max(genes_a) if genes_a else 0
    max_innovation_b = max(genes_b) if genes_b else 0

    excess = disjoint = 0
    weight_diffs = []

    for innov in all_innovations:
        in_a = innov in genes_a
        in_b = innov in genes_b

        if in_a and in_b:  # both genomes have this node
            weight_diffs.append(abs(genes_a[innov][2] - genes_b[innov][2]))
        elif in_a and innov > max_innovation_b:
            excess += 1  # beyond b's range
        elif in_b and innov > max_innovation_a:
            excess += 1  # beyond a's range
        else:
            disjoint += 1  # gap within both ranges

    N = max(len(genes_a), len(genes_b), 1)
    W = sum(weight_diffs) / len(weight_diffs) if weight_diffs else 0.0

    return (
        (_env_float("NEAT_C1", 1.0) * excess / N)
        + (_env_float("NEAT_C2", 1.0) * disjoint / N)
        + (_env_float("NEAT_C3", 0.4) * W)
    )


CURRENT_THRESHOLD = None


def speciate(population, existing_species):
    global CURRENT_THRESHOLD

    if CURRENT_THRESHOLD is None:
        CURRENT_THRESHOLD = _env_float("NEAT_DELTA_THRESHOLD", 0.5)

    # clear members, keep representatives + history
    for s in existing_species:
        s.members = []

    species_list = existing_species
    next_species_id = max((s.id for s in species_list), default=0) + 1

    for genome in population:
        placed = False
        for species in species_list:
            if (
                compatibility_distance(genome, species.representative)
                < CURRENT_THRESHOLD
            ):
                species.members.append(genome)
                placed = True
                break

        if not placed:
            # create a new species with this genome as representative
            new_species = Species(
                id=next_species_id, representative=genome, members=[genome]
            )
            species_list.append(new_species)
            next_species_id += 1

    # remove empty species
    species_list = [s for s in species_list if len(s.members) > 0]

    # Dynamically adjust threshold to maintain target number of species (e.g. 5-15)
    target_species = 10
    if len(species_list) < target_species:
        CURRENT_THRESHOLD -= 0.05
    elif len(species_list) > target_species:
        CURRENT_THRESHOLD += 0.05

    CURRENT_THRESHOLD = max(0.1, CURRENT_THRESHOLD)  # prevent negative/zero threshold

    # update representatives for next generation
    for s in species_list:
        s.representative = random.choice(s.members)

    return species_list


# penalize genomes part of a large species (we want diversity)
def adjust_fitness(species_list):
    for species in species_list:
        if not species.members:
            continue

        fits = [g.fitness_score.item() for g in species.members]

        min_fit = min(fits)
        shift = -min_fit + 1e-6 if min_fit <= 0 else 0.0  # allows negative fitness

        size_penalty = len(species.members)

        for g, f in zip(species.members, fits):
            adjusted = (f + shift) / size_penalty
            g.adjusted_fitness = adjusted


# if the species isn't improving, remove them
def remove_stale_species(species_list):
    survivors = []
    stale_limit = _env_int("NEAT_STALE_LIMIT", 20)

    for species in species_list:
        best = max(g.fitness_score.item() for g in species.members)

        if best > species.best_fitness:
            species.best_fitness = best
            species.generations_stale = 0
        else:
            species.generations_stale += 1

        if species.generations_stale < stale_limit:
            survivors.append(species)

    return survivors
