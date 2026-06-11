import random
import copy

try:
    from .speciation import speciate, adjust_fitness, remove_stale_species
    from .crossover import crossover
    from .mutations import mutate
except:
    from speciation import speciate, adjust_fitness, remove_stale_species
    from crossover import crossover
    from mutations import mutate


def breed_next_generation(species_list, population_size, tracker):
    if not species_list:
        return []
    total_fitness = sum(
        g.adjusted_fitness for s in species_list for g in s.members
    )  # adjusted fitness takes account the species size

    next_generation = []

    for species in species_list:
        species_fitness = sum(g.adjusted_fitness for g in species.members)
        n_offspring = int((species_fitness / total_fitness) * population_size)

        if n_offspring == 0:
            continue

        members = sorted(
            species.members, key=lambda g: g.fitness_score.item(), reverse=True
        )  # sort by fitness score

        # best survives unchanged
        next_generation.append(copy.deepcopy(members[0]))

        for _ in range(n_offspring - 1):
            if len(members) == 1 or random.random() < 0.2:
                child = mutate(copy.deepcopy(members[0]), tracker)  # *need deepcopy
            else:
                a = random.choice(members)
                b = random.choice(members)
                child = crossover(a, b)
                child = mutate(child, tracker)

            next_generation.append(child)

    # fill rounding gaps
    # remainders to the best performing species ones
    best_species = max(
        species_list, key=lambda s: sum(g.adjusted_fitness for g in s.members)
    )

    while len(next_generation) < population_size:
        child = mutate(
            copy.deepcopy(best_species.members[0]), tracker
        )  # * need deepcopy here
        next_generation.append(child)

    return next_generation[:population_size]


def run_generation(population, species_list, tracker, evaluate_fn):
    # evaluate
    for genome in population:
        genome.fitness_score = evaluate_fn(genome)

    # speciate
    species_list = speciate(population, species_list)

    # adjust fitness
    adjust_fitness(species_list)

    # remove stale species
    species_list = remove_stale_species(species_list)

    # if all species were removed, re-speciate to avoid empty population
    if not species_list:
        species_list = speciate(population, [])

    # TODO: use wandb/tensorlflow
    best = max(population, key=lambda g: g.fitness_score.item())
    print(
        f"Species: {len(species_list)} | Best fitness: {best.fitness_score.item():.4f}"
    )

    # breed
    next_population = breed_next_generation(species_list, len(population), tracker)

    if not next_population:
        next_population = population

    # re-speciate the new generation to classify offspring into species
    species_list = speciate(next_population, species_list)

    return next_population, species_list, best
