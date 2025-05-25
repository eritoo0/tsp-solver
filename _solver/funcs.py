import random
import math
import time
import itertools
import numpy as np


# Distance calculation between two cities
def distance(i, j, city_coords):
    xi, yi = city_coords[i]
    xj, yj = city_coords[j]
    return int(round(math.hypot(xi - xj, yi - yj)))


# Total tour distance calculation


def total_distance(tour, distance_matrix, num_cities):
    return sum(
        distance_matrix[tour[i]][tour[(i + 1) % num_cities]] for i in range(num_cities)
    )


# Population generation


def create_population(num_cities, pop_size):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]


# Parent selection


def select_parents(pop, total_distance, distance_matrix, num_cities):
    return sorted(pop, key=lambda x: total_distance(x, distance_matrix, num_cities))[:2]


# Classical crossover operator


def crossover(parent1, parent2, num_cities):
    start, end = sorted(random.sample(range(num_cities), 2))
    child = [None] * num_cities
    child[start : end + 1] = parent1[start : end + 1]
    fill = [c for c in parent2 if c not in child]
    pointer = 0
    for i in range(num_cities):
        if child[i] is None:
            child[i] = fill[pointer]
            pointer += 1
    return child


# ACO-guided crossover


def aco_guided_crossover(
    parent1,
    parent2,
    num_cities,
    candidate_list,
    pheromone,
    alpha,
    beta,
    distance_matrix,
):
    """
    Builds a child tour by following pheromone+heuristic guidance (à la ACO).
    Guards against zero distances to avoid division-by-zero.
    """
    child = [-1] * num_cities
    visited = set()

    # 1) Start from a random city
    current = random.randrange(num_cities)
    child[0] = current
    visited.add(current)

    # 2) Build the rest of the tour
    for i in range(1, num_cities):
        # pick eligible candidates
        cand = [c for c in candidate_list[current] if c not in visited]
        if not cand:
            # fallback: any unvisited city
            cand = [c for c in range(num_cities) if c not in visited]

        # compute (tau * eta) weights
        weights = []
        for nxt in cand:
            tau = pheromone[current][nxt] ** alpha
            dist = distance_matrix[current][nxt]
            # guard against zero distance
            if dist <= 0:
                eta = 1e-9
            else:
                eta = (1.0 / dist) ** beta
            weights.append((nxt, tau * eta))

        total_w = sum(w for _, w in weights)
        if total_w <= 0:
            # if all weights zero, just pick random
            next_city = random.choice(cand)
        else:
            # roulette wheel
            r = random.random() * total_w
            acc = 0.0
            for city, w in weights:
                acc += w
                if acc >= r:
                    next_city = city
                    break

        child[i] = next_city
        visited.add(next_city)
        current = next_city

    # 3) Safety: fill any gaps
    for idx in range(num_cities):
        if child[idx] == -1:
            for c in range(num_cities):
                if c not in visited:
                    child[idx] = c
                    visited.add(c)
                    break

    return child




# Mutation operator


def mutate(tour, mutation_rate, num_cities):
    if random.random() < mutation_rate:
        i, j = random.sample(range(num_cities), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


# Pheromone update


def update_pheromones(
    best_tours, num_cities, pheromone, evaporation_rate, q, distance_matrix
):
    for i in range(num_cities):
        for j in range(num_cities):
            pheromone[i][j] *= 1 - evaporation_rate
    for tour in best_tours:
        dist = total_distance(tour, distance_matrix, num_cities)
        for i in range(num_cities):
            a, b = tour[i], tour[(i + 1) % num_cities]
            pheromone[a][b] += q / dist
            pheromone[b][a] += q / dist


# Simulated annealing optimization


def simulated_annealing(tour, distance_matrix, num_cities, T, cooling_rate):
    current = tour[:]
    best = tour[:]
    best_dist = total_distance(best, distance_matrix, num_cities)
    while T > 1:
        i, j = sorted(random.sample(range(num_cities), 2))
        new = current[:]
        new[i:j] = reversed(new[i:j])
        delta = total_distance(new, distance_matrix, num_cities) - total_distance(
            current, distance_matrix, num_cities
        )
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = new[:]
            if total_distance(current, distance_matrix, num_cities) < best_dist:
                best = current[:]
                best_dist = total_distance(current, distance_matrix, num_cities)
        T *= cooling_rate
    return best


# Lin-Kernighan heuristic


def lin_kernighan(tour, distance_matrix, num_cities, max_moves=50):
    best = tour[:]
    best_dist = total_distance(best, distance_matrix, num_cities)
    moves = 0
    while moves < max_moves:
        best_delta = 0
        best_move = None
        for i in range(1, num_cities - 2):
            a = best[i - 1]
            b = best[i]
            for j in range(i + 1, num_cities - (0 if i + 1 < num_cities else 1)):
                c = best[j]
                d = best[(j + 1) % num_cities]
                delta = (
                    -distance_matrix[a][b]
                    - distance_matrix[c][d]
                    + distance_matrix[a][c]
                    + distance_matrix[b][d]
                )
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
        if best_move is None:
            break
        i, j = best_move
        best[i : j + 1] = reversed(best[i : j + 1])
        best_dist += best_delta
        moves += 1
    return best


# Honey Bee Colony algorithm


def honey_bee_colony(
    population,
    distance_matrix,
    num_cities,
    pop_size,
    mutation_rate,
    num_bees=20,
    max_iter=100,
    exploration_rate=0.3,
):
    best_solution = min(
        population, key=lambda x: total_distance(x, distance_matrix, num_cities)
    )
    best_dist = total_distance(best_solution, distance_matrix, num_cities)
    for _ in range(max_iter):
        for _ in range(num_bees):
            bee = random.choice(population)
            if random.random() < exploration_rate:
                new_bee = large_perturbation(bee, num_cities)
            else:
                new_bee = mutate(bee, mutation_rate, num_cities)
            if total_distance(new_bee, distance_matrix, num_cities) < best_dist:
                best_solution = new_bee
                best_dist = total_distance(new_bee, distance_matrix, num_cities)
        population = sorted(
            population, key=lambda x: total_distance(x, distance_matrix, num_cities)
        )[:pop_size]
    return best_solution


# Large perturbation operator


def large_perturbation(tour, num_cities, intensity):
    tour = tour[:]
    for _ in range(intensity):
        i, j = sorted(random.sample(range(num_cities), 2))
        tour[i:j] = reversed(tour[i:j])
    return tour


# Population diversity measurement (no arguments needed)


def pop_diversity(pop):
    if len(pop) < 2:
        return 0.0
    n = len(pop[0])
    tot = 0
    for a, b in itertools.combinations(pop, 2):
        tot += sum(x != y for x, y in zip(a, b))
    pairs = len(pop) * (len(pop) - 1) // 2
    return tot / (pairs * n)


# Aggressive restart mechanism


def aggressive_restart(
    population,
    num_cities,
    pop_size,
    distance_matrix,
    elite_ratio=0.1,
    survivor_ratio=0.2,
    noise_intensity=10,
    sa_T=100.0,
    sa_cooling=0.995,
):
    n_elite = int(pop_size * elite_ratio)
    n_survivors = int(pop_size * survivor_ratio)
    sorted_pop = sorted(
        population, key=lambda x: total_distance(x, distance_matrix, num_cities)
    )
    elite = sorted_pop[:n_elite]
    survivors = sorted_pop[n_elite : n_elite + n_survivors]
    new_population = elite + survivors
    while len(new_population) < pop_size:
        base = random.sample(range(num_cities), num_cities)
        perturbed = large_perturbation(base, num_cities, noise_intensity)
        refined = simulated_annealing(
            perturbed, distance_matrix, num_cities, sa_T, sa_cooling
        )
        refined = lin_kernighan(refined, distance_matrix, num_cities)
        new_population.append(refined)
    return new_population


# Weighted scoring (no arguments needed)


def weighted_score(scores):
    n = len(scores)
    weights = [(i + 1) / n for i in range(n)]
    return sum(w * s for w, s in zip(weights, scores[-n:])) if n > 0 else 0
