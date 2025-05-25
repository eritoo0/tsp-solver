from .funcs import *
import random
import time
import numpy as np


def solve_tsp_ga(filename, callback, **params):
    # Parameters
    POP_SIZE = params.get("POP_SIZE", 50)
    NUM_GENERATIONS = params.get("NUM_GENERATIONS", 500)
    MUTATION_RATE = params.get("MUTATION_RATE", 0.2)

    # Load cities
    from pathlib import Path

    BASE = Path(__file__).resolve().parent.parent
    data_file = BASE / "data" / filename
    city_coords = np.loadtxt(data_file, delimiter=",")
    NUM_CITIES = len(city_coords)

    # Known optima
    if filename == "krA100_coords.txt":
        OPTIMAL_DISTANCE = 21282
    elif filename == "berlin52_coords.txt":
        OPTIMAL_DISTANCE = 7542
    elif filename == "ch150_coords.txt":
        OPTIMAL_DISTANCE = 6528
    elif filename == "st70_coords.txt":
        OPTIMAL_DISTANCE = 675
    elif filename == "eil101_coords.txt":
        OPTIMAL_DISTANCE = 629
    elif filename == "pr144_coords.txt":
        OPTIMAL_DISTANCE = 58537
    elif filename == "a280_coords.txt":
        OPTIMAL_DISTANCE = 2579
    elif filename == "pr107_coords.txt":
        OPTIMAL_DISTANCE = 44303
    elif filename == "pr152_coords.txt":
        OPTIMAL_DISTANCE = 73682
    elif filename == "pr299_coords.txt":
        OPTIMAL_DISTANCE = 48191
    elif filename == "rat99_coords.txt":
        OPTIMAL_DISTANCE = 1211
    elif filename == "rat195_coords.txt":
        OPTIMAL_DISTANCE = 2323
    else:
        OPTIMAL_DISTANCE = None

    # random.seed(time.time())
    random.seed(42)

    # Distance matrix
    distance_matrix = [
        [distance(i, j, city_coords) if i != j else 0 for j in range(NUM_CITIES)]
        for i in range(NUM_CITIES)
    ]

    # Initial population
    population = create_population(NUM_CITIES, POP_SIZE)

    best = min(population, key=lambda t: total_distance(t, distance_matrix, NUM_CITIES))
    best_score = total_distance(best, distance_matrix, NUM_CITIES)
    error_log = []
    logs = []
    distance_log = []

    error_rel = 100

    start_all = time.time()

    for gen in range(1, NUM_GENERATIONS + 1):
        start_gen = time.time()

        new_population = []

        # --- New population creation ---
        for _ in range(POP_SIZE):
            p1, p2 = select_parents(
                population, total_distance, distance_matrix, NUM_CITIES
            )
            child = crossover(p1, p2, NUM_CITIES)
            child = mutate(child, MUTATION_RATE, NUM_CITIES)
            new_population.append(child)

        # --- Elitism and best update ---
        population = sorted(
            population + new_population,
            key=lambda t: total_distance(t, distance_matrix, NUM_CITIES),
        )[:POP_SIZE]
        best = population[0]
        best_score = total_distance(best, distance_matrix, NUM_CITIES)

        # --- Relative error ---
        if OPTIMAL_DISTANCE:
            error_rel = 100 * (best_score - OPTIMAL_DISTANCE) / OPTIMAL_DISTANCE
        else:
            error_rel = 0
        elapsed_total = time.time() - start_all
        elapsed_gen = time.time() - start_gen

        distance_log.append(best_score)
        error_log.append(error_rel)
        logs.append(
            {
                "gen": gen,
                "distance": best_score,
                "error": round(error_rel, 2),
                "temps": round(elapsed_gen, 2),
                "temps_total": round(elapsed_total, 2),
            }
        )
        #
        callback("not done", best, city_coords.tolist(), best_score, error_log, logs)

        # --- Early stopping ---
        if NUM_CITIES > 100 and error_rel <= 1 :
            break
        elif OPTIMAL_DISTANCE == best_score:
            break

    # --- Final reporting ---
    total_seconds = time.time() - start_all
    minutes = int(total_seconds // 60)
    seconds = round(total_seconds % 60, 2)
    print("\n--- RESULTATS FINAUX ---")
    print("Meilleur tour:", best)
    print(f"Temps total: {minutes} minute(s) et {seconds} seconde(s)")
    print("Distance:", round(best_score, 2))
    print("Distance optimale:", OPTIMAL_DISTANCE)
    print(
        f"Erreur relative: {100 * (best_score - OPTIMAL_DISTANCE) /OPTIMAL_DISTANCE:.2f}%"
        if OPTIMAL_DISTANCE
        else ""
    )

    callback("done", best, city_coords.tolist(), best_score, error_log, logs)
    return best, best_score, error_log, logs, city_coords.tolist()
