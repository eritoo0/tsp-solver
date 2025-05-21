import random
import time
import numpy as np
from .funcs import distance

def solve_tsp_sa(filename, callback, **params):
    """
    SA temps-réel : on déclenche le callback à CHAQUE itération.
    """
    # ── Paramètres ───────────────────────────────────────────────────────────
    INITIAL_T = params.get('INITIAL_TEMPERATURE', 100.0)
    COOLING   = params.get('COOLING_RATE', 0.995)
    MAX_ITER  = params.get('MAX_ITER', 10000)
    RESTARTS  = params.get('RESTARTS', 3)

    # ── Lecture des villes ────────────────────────────────────────────────────
    from pathlib import Path
    BASE = Path(__file__).resolve().parent.parent
    data_file = BASE / 'data' / filename
    city_coords = np.loadtxt(data_file, delimiter=",")
    N = len(city_coords)

    # ── Matrice de distance ───────────────────────────────────────────────────
    dist = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i!=j:
                dist[i][j] = distance(i, j, city_coords)

    def tour_length(t):
        return sum(dist[t[i]][t[(i+1)%N]] for i in range(N))

    # ── Optimum connu (pour erreur relative) ─────────────────────────────────
    OPT = None
    if filename=="berlin52_coords.txt": OPT=7542
    elif filename=="kr100_coords.txt":    OPT=21282
    elif filename=="ch150_coords.txt":    OPT=6528
    elif filename=="kr200_coords.txt":    OPT=29381

    best = None
    best_score  = float('inf')
    error_log   = []
    logs        = []
    t0 = time.time()

    # ── voisinage 2-opt ───────────────────────────────────────────────────────
    def neighbor(sol):
        i,j = sorted(random.sample(range(N),2))
        u = sol.copy()
        u[i:j+1] = reversed(u[i:j+1])
        return u

    # ── Redémarrages de SA ───────────────────────────────────────────────────
    for run in range(1, RESTARTS+1):
        # init aléatoire
        cur = list(range(N))
        random.shuffle(cur)
        cur_score = tour_length(cur)
        T = INITIAL_T

        for it in range(1, MAX_ITER+1):
            cand = neighbor(cur)
            cand_score = tour_length(cand)
            Δ = cand_score - cur_score
            if Δ<0 or random.random() < np.exp(-Δ/T):
                cur, cur_score = cand, cand_score

            T *= COOLING

            if cur_score<best_score:
                best, best_score = cur.copy(), cur_score

            # ── **callback à chaque itération** ─────────────────────────────────
            elapsed = time.time() - t0
            err = 0 if OPT is None else 100*(best_score-OPT)/OPT
            error_log.append(round(err,2))
            logs.append({
                'iteration': it + (run-1)*MAX_ITER,
                'distance': best_score,
                'error': round(err,2),
                'temps': round(elapsed,2)
            })
            callback('not done', best, city_coords.tolist(),
                 best_score, error_log, logs)

        print(f"↪️ Run {run}/{RESTARTS} terminé — best={best_score}")

    # ── Fin du solver ────────────────────────────────────────────────────────
    total = time.time() - t0
    print(f"✅ SA terminé en {total:.2f}s — distance={best_score}")
    callback('done', best, city_coords.tolist(),
                 best_score, error_log, logs)
    return best, best_score, error_log, logs, city_coords.tolist()
