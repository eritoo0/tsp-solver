from .funcs import *
import random
import time
import numpy as np


def solve_tsp(filename, callback, **params):
    # Parameters
    # --- Paramètres récupérés depuis l’interface ---
    POP_SIZE = params.get("POP_SIZE", 50)
    NUM_GENERATIONS = params.get("NUM_GENERATIONS", 500)
    MUTATION_RATE = params.get("MUTATION_RATE", 0.2)

    # ACO
    ALPHA = params.get("ALPHA", 1.0)
    BETA = params.get("BETA", 3.0)
    EVAPORATION_RATE = params.get("EVAPORATION_RATE", 0.1)
    Q = params.get("Q", 100.0)

    # SA
    SA_T = params.get("SA_T", 50.0)  # température initiale
    SA_COOLING = params.get("SA_COOLING", 0.995)  # facteur de décroissance

    # HBC (pour éventuellement surcharger le taux de mutation dans HBC)
    HBC_MUTATION_RATE = params.get("HBC_MUTATION_RATE", 0.2)

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

    # Candidate list for each city
    CANDIDATE_SIZE = min(20, NUM_CITIES - 1)
    candidate_list = []
    for i in range(NUM_CITIES):
        neighbors = sorted(
            [j for j in range(NUM_CITIES) if j != i],
            key=lambda j: distance_matrix[i][j],
        )
        candidate_list.append(neighbors[:CANDIDATE_SIZE])

    # Pheromone matrix
    pheromone = [[1.0 for _ in range(NUM_CITIES)] for _ in range(NUM_CITIES)]

    # Adaptive parameters
    ADAPT_INTERVAL = 20
    ADAPT_THRESHOLD = 1.0
    improvement_log = []

    # Score logs
    ACO_scores = []
    SA_scores = []
    USE_DYNAMIC_CHOICE = True
    EVAL_WINDOW = 5

    # Initial population
    population = create_population(NUM_CITIES, POP_SIZE)

    best = min(population, key=lambda t: total_distance(t, distance_matrix, NUM_CITIES))
    best_score = total_distance(best, distance_matrix, NUM_CITIES)
    error_log = []
    logs = []
    distance_log = []
    improvement_log = []
    error_rel = 100
    stagnation_counter = 0
    stagnation_arreter = 0
    previous_best = best_score
    start_all = time.time()

    for gen in range(NUM_GENERATIONS):
        start_gen = time.time()
        ACO_ACTIVE = False
        SA_ACTIVE = False
        HBC_ACTIVE = False

        # --- Dynamic adaptation ---
        if gen > 0 and gen % ADAPT_INTERVAL == 0:
            recent = sum(improvement_log[-ADAPT_INTERVAL:])
            if recent < ADAPT_THRESHOLD:
                MUTATION_RATE = min(1.0, MUTATION_RATE + 0.02)
                EVAPORATION_RATE = min(1.0, EVAPORATION_RATE + 0.02)
                print(
                    f"🔧 Adaptation (stagnation) → MUT={MUTATION_RATE:.2f}, EVAP={EVAPORATION_RATE:.2f}"
                )
            else:
                MUTATION_RATE = max(0.01, MUTATION_RATE - 0.01)
                EVAPORATION_RATE = max(0.01, EVAPORATION_RATE - 0.005)
                print(
                    f"🔧 Adaptation (progrès) → MUT={MUTATION_RATE:.2f}, EVAP={EVAPORATION_RATE:.2f}"
                )

        # --- Stagnation detection ---
        if abs(best_score - previous_best) < 1e-4:
            stagnation_counter += 1
            stagnation_arreter += 1
        else:
            stagnation_counter = 0
            stagnation_arreter = 0
            previous_best = best_score

        # --- Moderate stagnation (10 ≤ n < 30) ---
        if stagnation_counter >= 10:
            if USE_DYNAMIC_CHOICE and (ACO_scores or SA_scores):
                recent_aco = weighted_score(ACO_scores[-EVAL_WINDOW:])
                recent_sa = weighted_score(SA_scores[-EVAL_WINDOW:])
                ACO_ACTIVE = recent_aco > recent_sa
                SA_ACTIVE = not ACO_ACTIVE
            else:
                ACO_ACTIVE = gen % 2 != 0
                SA_ACTIVE = not ACO_ACTIVE

        # --- Moderate stagnation (30 ≤ n < 60) ---
        if stagnation_counter >= 30:
            print("\n🧠 [STAGNATION > 30] Détection de stagnation prolongée.")
            diversity = pop_diversity(population)
            print(f"   Diversité ≈ {diversity:.2f}  |  erreur = {error_rel:.2f}%")
            if diversity > 0.20:
                n_refine = max(2, int(POP_SIZE * 0.25))
                top_k = sorted(
                    population,
                    key=lambda t: total_distance(t, distance_matrix, NUM_CITIES),
                )[:n_refine]
                refined = [
                    lin_kernighan(t, distance_matrix, NUM_CITIES, max_moves=50)
                    for t in top_k
                ]
                population = refined + population[n_refine:]
                print(f"🔍 LK appliqué sur {n_refine} individus (intensification)")
            elif error_rel > 5:
                population = [
                    honey_bee_colony(
                        population,
                        distance_matrix,
                        NUM_CITIES,
                        POP_SIZE,
                        mutation_rate=HBC_MUTATION_RATE,
                    )
                    for _ in range(POP_SIZE)
                ]
                print("🐝 Honey-Bee Colony (exploration globale)")
            else:
                population = [
                    large_perturbation(ind, NUM_CITIES, intensity=4)
                    for ind in population
                ]
                print("🔀 Perturbation aléatoire (mode shake-out)")
            stagnation_counter = 0
            EVAL_WINDOW = max(2, EVAL_WINDOW - 1)
            print(
                f"↪️ Population modifiée. Nouvelle taille de fenêtre = {EVAL_WINDOW}\n"
            )

        # --- Strong stagnation (n >= 60) ---
        if stagnation_counter >= 60:
            population = aggressive_restart(
                population,
                NUM_CITIES,
                POP_SIZE,
                distance_matrix,
                elite_ratio=0.1,
                survivor_ratio=0.2,
                noise_intensity=10,
                sa_T=120.0,
                sa_cooling=0.995,
            )
            stagnation_counter = 0
            print("↪️ Relance agressive partielle effectuée")
            EVAL_WINDOW = max(2, EVAL_WINDOW - 1)
            print(
                f"↪️ Population relancée. Nouvelle taille de fenêtre d'évaluation : {EVAL_WINDOW}\n"
            )

        # --- Early stop (n >= 300) ---
        if stagnation_arreter >= 300:
            print(
                "⛔ Arrêt anticipé : stagnation détectée pendant 250 générations consécutives."
            )
            break

        new_population = []

        # --- New population creation ---
        for _ in range(POP_SIZE):
            p1, p2 = select_parents(
                population, total_distance, distance_matrix, NUM_CITIES
            )
            if ACO_ACTIVE:
                child = aco_guided_crossover(
                    p1,
                    p2,
                    NUM_CITIES,
                    candidate_list,
                    pheromone,
                    ALPHA,
                    BETA,
                    distance_matrix,
                )
            else:
                child = crossover(p1, p2, NUM_CITIES)
            before = total_distance(child, distance_matrix, NUM_CITIES)
            child = mutate(child, MUTATION_RATE, NUM_CITIES)
            if SA_ACTIVE:
                child = simulated_annealing(
                    child, distance_matrix, NUM_CITIES, T=SA_T, cooling_rate=SA_COOLING
                )
            after = total_distance(child, distance_matrix, NUM_CITIES)
            improvement = before - after
            if gen % 20 == 0 and improvement < 5:
                child = lin_kernighan(child, distance_matrix, NUM_CITIES, max_moves=50)
            if ACO_ACTIVE:
                ACO_scores.append(improvement)
            elif SA_ACTIVE:
                SA_scores.append(improvement)
            new_population.append(child)

        # --- Elitism and best update ---
        population = sorted(
            population + new_population,
            key=lambda t: total_distance(t, distance_matrix, NUM_CITIES),
        )[:POP_SIZE]
        best = population[0]
        best_score = total_distance(best, distance_matrix, NUM_CITIES)

        # --- Pheromone update (ACO only) ---
        update_pheromones(
            population[: POP_SIZE // 4],
            NUM_CITIES,
            pheromone,
            EVAPORATION_RATE,
            Q,
            distance_matrix,
        )

        # --- Improvement log ---
        improvement = best_score - previous_best
        improvement_log.append(improvement)

        # --- Algorithm used ---
        algo_used = (
            "ACO"
            if ACO_ACTIVE
            else "SA" if SA_ACTIVE else "HBC" if HBC_ACTIVE else "GA pur"
        )

        # --- Relative error ---
        if OPTIMAL_DISTANCE:
            error_rel = 100 * (best_score - OPTIMAL_DISTANCE) / OPTIMAL_DISTANCE
        else:
            error_rel = 0
        elapsed_total = time.time() - start_all
        elapsed_gen = time.time() - start_gen

        # --- Console logging ---
        # print(
        #     f"Génération {gen+1} | Algorithme utilisé: {algo_used}"
        #     f" | Best: {best_score:.2f}"
        #     f" | Erreur relative: {error_rel:.2f}%"
        #     f" | Temps: {elapsed:.2f}s"
        # )
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
        if OPTIMAL_DISTANCE == best_score:
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
