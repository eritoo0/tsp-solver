import random
import time
import numpy as np
from .funcs import distance, mutate, large_perturbation


def solve_tsp_hbc(filename, callback, **params):
    """
    Résout le TSP uniquement avec la Honey-Bee Colony.

    - filename         : nom du fichier de coordonnées (dans data/)
    - callback(status, best_tour, city_coords, best_score, error_log, logs)
    - params peut contenir :
        POP_SIZE        : taille de la population (nombre d’ouvrières)
        NUM_ITERATIONS  : nombre d’itérations
        NUM_BEES        : nombre de butineuses par itération
        EXPLORATION     : probabilité d’exploration globale (0–1)
        INTENSITY       : intensité des perturbations lors de l’exploration
    """
    # ── Paramètres essentiels ────────────────────────────────────────────────
    POP_SIZE = params.get("POP_SIZE", 30)
    NUM_ITERATIONS = params.get("NUM_ITERATIONS", 200)
    NUM_BEES = params.get("NUM_BEES", 20)
    EXPLORATION = params.get("EXPLORATION", 0.3)
    INTENSITY = params.get("INTENSITY", 5)

    # ── Chargement des villes ────────────────────────────────────────────────
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
    # ── Matrice des distances euclidiennes arrondies ─────────────────────────
    distance_matrix = [
        [distance(i, j, city_coords) if i != j else 0 for j in range(NUM_CITIES)]
        for i in range(NUM_CITIES)
    ]

    def tour_length(tour):
        return sum(
            distance_matrix[tour[i]][tour[(i + 1) % NUM_CITIES]]
            for i in range(NUM_CITIES)
        )

    # ── Population initiale ──────────────────────────────────────────────────
    population = [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(POP_SIZE)]
    best = min(population, key=tour_length)
    best_score = tour_length(best)

    # ── Logs pour l’interface ────────────────────────────────────────────────
    error_log = []
    logs = []
    start_all = time.time()

    for it in range(1, NUM_ITERATIONS + 1):
        start_gen = time.time()
        # à chaque itération, chaque abeille tente une amélioration
        for _ in range(NUM_BEES):
            sol = random.choice(population)
            if random.random() < EXPLORATION:
                cand = large_perturbation(sol, NUM_CITIES, intensity=INTENSITY)
            else:
                cand = mutate(tour=sol, mutation_rate=0.1, num_cities=NUM_CITIES)
            length = tour_length(cand)
            if length < best_score:
                best, best_score = cand, length

        # on garde les meilleures solutions pour la prochaine itération
        population = sorted(population + [best], key=tour_length)[:POP_SIZE]

        # calcul de l’erreur relative si on connaît l’optimum
        if OPTIMAL_DISTANCE:
            error_rel = 100 * (best_score - OPTIMAL_DISTANCE) / OPTIMAL_DISTANCE
        else:
            error_rel = 0
        elapsed_total = time.time() - start_all
        elapsed_gen = time.time() - start_gen
        error_log.append(error_rel)
        logs.append(
            {
                "iteration": it,
                "distance": best_score,
                "error": round(error_rel, 2),
                "temps": round(elapsed_gen, 2),
                "temps_total": round(elapsed_total, 2),
            }
        )

        # callback vers l’UI
        callback("not done", best, city_coords.tolist(), best_score, error_log, logs)

        if NUM_CITIES > 100 and error_rel <= 1 :
            break
        elif OPTIMAL_DISTANCE == best_score:
            break

    # ── Fin et callback final ────────────────────────────────────────────────
    total_time = time.time() - start_all
    print(f"✅ HBC terminé en {round(total_time,2)} s – distance : {best_score}")

    callback("done", best, city_coords.tolist(), best_score, error_log, logs)
    return best, best_score, error_log, logs, city_coords.tolist()
