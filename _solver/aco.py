from .funcs import distance, total_distance
import random
import time
import numpy as np

def solve_tsp_aco(filename, callback, **params):
    # Paramètres essentiels
    ANT_COUNT       = params.get('ANT_COUNT', 50)
    NUM_ITERATIONS  = params.get('NUM_ITERATIONS', 500)
    ALPHA           = params.get('ALPHA', 1.0)
    BETA            = params.get('BETA', 3.0)
    EVAPORATION     = params.get('EVAPORATION_RATE', 0.1)
    Q               = params.get('Q', 100.0)

    # Chargement des coordonnées
    from pathlib import Path
    BASE = Path(__file__).resolve().parent.parent
    data_file = BASE / 'data' / filename
    city_coords = np.loadtxt(data_file, delimiter=",")
    NUM_CITIES = len(city_coords)
    
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
        [distance(i, j, city_coords) if i !=
         j else 0 for j in range(NUM_CITIES)]
        for i in range(NUM_CITIES)
    ]

    # Initialisation des phéromones
    pheromone = [[1.0 for _ in range(NUM_CITIES)] for _ in range(NUM_CITIES)]

    # Meilleur tour global
    best  = None
    best_score = float('inf')
    error_log = []
    logs = []
    error_rel = 100
    start_all  = time.time()

    # Boucle principale ACO
    for it in range(1, NUM_ITERATIONS+1):
        start_gen = time.time()
        all_tours = []
        all_lengths = []

        # Chaque « fourmi » construit un tour complet
        for _ in range(ANT_COUNT):
            unvisited = set(range(NUM_CITIES))
            current   = random.choice(list(unvisited))
            tour      = [current]
            unvisited.remove(current)

            while unvisited:
                # calcule les probabilités de transition
                weights = []
                for j in unvisited:
                    tau = pheromone[current][j] ** ALPHA
                    eta = (1.0 / distance_matrix[current][j]) ** BETA
                    weights.append((j, tau * eta))

                # roulette
                total_w = sum(w for _,w in weights)
                r = random.random() * total_w
                accum = 0.0
                for j, w in weights:
                    accum += w
                    if accum >= r:
                        next_city = j
                        break

                tour.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            all_tours.append(tour)
            length = sum(distance_matrix[tour[i]][tour[(i+1)%NUM_CITIES]] for i in range(NUM_CITIES))
            all_lengths.append(length)

            # mise à jour du meilleur global
            if length < best_score:
                best_score = length
                best  = tour[:]

        # Évaporation globale
        for i in range(NUM_CITIES):
            for j in range(NUM_CITIES):
                pheromone[i][j] *= (1 - EVAPORATION)

        # Dépôt de phéromones par chaque fourmi
        for tour, length in zip(all_tours, all_lengths):
            deposit = Q / length
            for i in range(NUM_CITIES):
                a, b = tour[i], tour[(i+1)%NUM_CITIES]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit

        # Callback vers l’interface / logging
        elapsed_total = time.time() - start_all
        elapsed_gen = time.time() - start_gen
        
        if OPTIMAL_DISTANCE:
            error_rel = 100 * \
                (best_score - OPTIMAL_DISTANCE) / OPTIMAL_DISTANCE
        else:
            error_rel = 0
            
        error_log.append(error_rel)
        
        logs.append({
            'iteration': it,
            'distance': best_score,
            'error': round(error_rel, 2),
            'temps': round(elapsed_gen,2),
            'temps_total': round(elapsed_total, 2)
        })
        callback('not done', best, city_coords.tolist(),
                 best_score, error_log, logs)
        
        if NUM_CITIES > 100 and error_rel <= 1 :
            break
        elif OPTIMAL_DISTANCE == best_score:
            break

    
    # Fin de la résolution
    '''
    total_time = time.time() - start_all
    callback(
        status='done',
        tour=best,
        coords=city_coords.tolist(),
        length=best_score,
        total_time=total_time
    )
    '''    
    callback('done', best, city_coords.tolist(),
                 best_score, error_log, logs)
    return best, best_score, error_log, logs, city_coords.tolist()
