import random
import time
import numpy as np
from .funcs import distance


def solve_tsp_sa(filename, callback, **params):
    """
    SA temps-réel : on déclenche le callback à CHAQUE itération.
    """
    # ── Paramètres ───────────────────────────────────────────────────────────
    INITIAL_T = params.get("INITIAL_TEMPERATURE", 100.0)
    COOLING = params.get("COOLING_RATE", 0.995)
    MAX_ITER = params.get("MAX_ITER", 10000)
    RESTARTS = params.get("RESTARTS", 3)

    # ── Lecture des villes ────────────────────────────────────────────────────
    from pathlib import Path

    BASE = Path(__file__).resolve().parent.parent
    data_file = BASE / "data" / filename
    city_coords = np.loadtxt(data_file, delimiter=",")
    NUM_CITIES = len(city_coords)

    # ── Matrice de distance ───────────────────────────────────────────────────
    dist = [[0] * NUM_CITIES for _ in range(NUM_CITIES)]
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            if i != j:
                dist[i][j] = distance(i, j, city_coords)

    def tour_length(t):
        return sum(dist[t[i]][t[(i + 1) % NUM_CITIES]] for i in range(NUM_CITIES))

    # ── Optimum connu (pour erreur relative) ─────────────────────────────────
    if filename == "krA100_coords.txt":
        OPT = 21282
    elif filename == "berlin52_coords.txt":
        OPT = 7542
    elif filename == "ch150_coords.txt":
        OPT = 6528
    elif filename == "st70_coords.txt":
        OPT = 675
    elif filename == "eil101_coords.txt":
        OPT = 629
    elif filename == "pr144_coords.txt":
        OPT = 58537
    elif filename == "a280_coords.txt":
        OPT = 2579
    elif filename == "pr107_coords.txt":
        OPT = 44303
    elif filename == "pr152_coords.txt":
        OPT = 73682
    elif filename == "pr299_coords.txt":
        OPT = 48191
    elif filename == "rat99_coords.txt":
        OPT = 1211
    elif filename == "rat195_coords.txt":
        OPT = 2323
    else:
        OPT = None

    best = None
    best_score = float("inf")
    error_log = []
    logs = []
    t0 = time.time()

    # ── voisinage 2-opt ───────────────────────────────────────────────────────
    def neighbor(sol):
        i, j = sorted(random.sample(range(NUM_CITIES), 2))
        u = sol.copy()
        u[i : j + 1] = reversed(u[i : j + 1])
        return u

    start_all = time.time()
    # ── Redémarrages de SA ───────────────────────────────────────────────────
    for run in range(1, RESTARTS + 1):
        start_gen = time.time()
        # init aléatoire
        cur = list(range(NUM_CITIES))
        random.shuffle(cur)
        cur_score = tour_length(cur)
        T = INITIAL_T

        for it in range(1, MAX_ITER + 1):
            cand = neighbor(cur)
            cand_score = tour_length(cand)
            Δ = cand_score - cur_score
            if Δ < 0 or random.random() < np.exp(-Δ / T):
                cur, cur_score = cand, cand_score

            T *= COOLING

            if cur_score < best_score:
                best, best_score = cur.copy(), cur_score

            # ── **callback à chaque itération** ─────────────────────────────────
            # elapsed = time.time() - t0
            elapsed_total = time.time() - start_all
            elapsed_gen = time.time() - start_gen
            err = 0 if OPT is None else 100 * (best_score - OPT) / OPT
            error_log.append(round(err, 2))
            logs.append(
                {
                    "iteration": it + (run - 1) * MAX_ITER,
                    "distance": best_score,
                    "error": round(err, 2),
                    "temps": round(elapsed_gen, 2),
                    "temps_total": round(elapsed_total, 2),
                }
            )
            callback(
                "not done", best, city_coords.tolist(), best_score, error_log, logs
            )

        print(f"↪️ Run {run}/{RESTARTS} terminé — best={best_score}")
        if NUM_CITIES > 100 and err <= 1 :
            break
        elif OPT == best_score:
            break
    # ── Fin du solver ────────────────────────────────────────────────────────
    total = time.time() - t0
    
    
    
    print(f"✅ SA terminé en {total:.2f}s — distance={best_score}")
    callback("done", best, city_coords.tolist(), best_score, error_log, logs)
    return best, best_score, error_log, logs, city_coords.tolist()
