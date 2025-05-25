import json
import uuid
import threading

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Import your solver functions
from _solver.tsp import solve_tsp
from _solver.ga import solve_tsp_ga
from _solver.aco import solve_tsp_aco
from _solver.sa import solve_tsp_sa
from _solver.hbc import solve_tsp_hbc

# In-memory store for jobs
JOBS = {}

# Definitions of available algorithms and their parameters
ALGORITHMS = {
    "tsp_algo": [
        # Contrôle de la population / itérations / mutation
        {
            "name": "POP_SIZE",
            "label": "Population Size",
            "type": "number",
            "default": 50,
        },
        {
            "name": "NUM_GENERATIONS",
            "label": "Number of Generations",
            "type": "number",
            "default": 500,
        },
        {
            "name": "MUTATION_RATE",
            "label": "Mutation Rate",
            "type": "number",
            "default": 0.2,
        },
        # Paramètres ACO
        {
            "name": "ALPHA",
            "label": "Alpha (ACO)",
            "type": "number",
            "default": 1,
        },
        {
            "name": "BETA",
            "label": "Beta (ACO)",
            "type": "number",
            "default": 3,
        },
        {
            "name": "EVAPORATION_RATE",
            "label": "Evaporation Rate (ACO)",
            "type": "number",
            "default": 0.1,
        },
        {
            "name": "Q",
            "label": "Q Value (ACO)",
            "type": "number",
            "default": 100,
        },
        # Paramètres SA
        {
            "name": "SA_T",
            "label": "Initial Temperature (SA)",
            "type": "number",
            "default": 50.0,
        },
        {
            "name": "SA_COOLING",
            "label": "Cooling Rate (SA)",
            "type": "number",
            "default": 0.995,
        },
        # Paramètre HBC (mutation spécifique)
        {
            "name": "HBC_MUTATION_RATE",
            "label": "Mutation Rate (HBC)",
            "type": "number",
            "default": 0.2,
        },
    ],
    "ga": [
        {
            "name": "POP_SIZE",
            "label": "Population Size",
            "type": "number",
            "default": 50,
        },
        {
            "name": "NUM_GENERATIONS",
            "label": "Number of Generations",
            "type": "number",
            "default": 500,
        },
        {
            "name": "MUTATION_RATE",
            "label": "Mutation Rate",
            "type": "number",
            "default": 0.2,
        },
    ],
    "aco": [
        {
            "name": "ANT_COUNT",
            "label": "Number of Ants",
            "type": "number",
            "default": 50,
        },
        {
            "name": "NUM_ITERATIONS",
            "label": "Number of Iterations",
            "type": "number",
            "default": 500,
        },
        {"name": "ALPHA", "label": "Alpha", "type": "number", "default": 1.0},
        {"name": "BETA", "label": "Beta", "type": "number", "default": 3.0},
        {
            "name": "EVAPORATION_RATE",
            "label": "Evaporation Rate",
            "type": "number",
            "default": 0.1,
        },
        {"name": "Q", "label": "Q Value", "type": "number", "default": 100.0},
    ],
    "hbc": [
        {
            "name": "POP_SIZE",
            "label": "Population Size",
            "type": "number",
            "default": 30,
        },
        {
            "name": "NUM_ITERATIONS",
            "label": "Number of Iterations",
            "type": "number",
            "default": 200,
        },
        {
            "name": "NUM_BEES",
            "label": "Number of Bees",
            "type": "number",
            "default": 20,
        },
        {
            "name": "EXPLORATION",
            "label": "Exploration Rate",
            "type": "number",
            "default": 0.3,
        },
        {"name": "INTENSITY", "label": "Intensity", "type": "number", "default": 5},
    ],
    "sa": [
        {
            "name": "INITIAL_TEMPERATURE",
            "label": "Initial Temperature",
            "type": "number",
            "default": 100.0,
        },
        {
            "name": "COOLING_RATE",
            "label": "Cooling Rate",
            "type": "number",
            "default": 0.995,
        },
        {
            "name": "MAX_ITER",
            "label": "Max Iterations",
            "type": "number",
            "default": 10000,
        },
        {"name": "RESTARTS", "label": "Restarts", "type": "number", "default": 3},
    ],
}


def _dispatch(algo, filename, params, callback):
    """
    Appelle la fonction solver appropriée en fonction de l'algorithme.
    """
    if algo == "tsp_algo":
        return solve_tsp(filename, callback, **params)
    if algo == "ga":
        return solve_tsp_ga(filename, callback, **params)
    if algo == "aco":
        return solve_tsp_aco(filename, callback, **params)
    if algo == "sa":
        return solve_tsp_sa(filename, callback, **params)
    if algo == "hbc":
        return solve_tsp_hbc(filename, callback, **params)
    raise ValueError(f"Unknown algorithm: {algo}")


def run_job(job_id, algo, filename, params):
    """
    Exécuté dans un thread : lance le solver et met à jour JOBS[job_id] via callback.
    """

    def callback(status, tour, city_coords, distance, error, logs):
        JOBS[job_id] = {
            "status": status,
            "tour": tour,
            "city_coords": city_coords,
            "distance": distance,
            "error": error,
            "logs": logs,
        }

    # état initial
    JOBS[job_id] = {"status": "running", "logs": []}
    _dispatch(algo, filename, params, callback)


@csrf_exempt
def solver_app(request):
    """
    POST: démarre un nouveau job, renvoie {'job_id': ...}.
    """
    if request.method == "POST":
        data = json.loads(request.body)
        algo = data.pop("algo")
        filename = data.pop("filename")
        job_id = str(uuid.uuid4())
        threading.Thread(
            target=run_job, args=(job_id, algo, filename, data), daemon=True
        ).start()
        return JsonResponse({"job_id": job_id})
    return JsonResponse({"error": "POST only"}, status=405)


def results_view(request):
    """
    GET: renvoie l'état courant du job {'status', 'logs', 'tour', ...}.
    """
    job_id = request.GET.get("job_id")
    result = JOBS.get(job_id)
    if not result:
        return JsonResponse(
            {
                "status": "not done",
                "tour": [],
                "city_coords": [],
                "distance": 0,
                "error": [],
                "logs": [],
            }
        )
    return JsonResponse(result)


def select_algorithms(request):
    """
    Page de sélection statique (optionnelle).
    """
    return render(
        request,
        "compare_app/select.html",
        {
            "algorithms": ALGORITHMS,
            "default_filename": "berlin52_coords.txt",
        },
    )


def compare_async(request):
    """
    Page principale de comparaison asynchrone.
    """
    return render(
        request,
        "compare_app/compare_async.html",
        {
            "algorithms": ALGORITHMS,
            "default_filename": "berlin52_coords.txt",
        },
    )


# ----- Synchronisation pour la page select.html -----


def compare_results(request):
    """
    Reçoit le POST depuis select.html, exécute synchroniquement les deux algos et rend results.html.
    """
    if request.method != "POST":
        return redirect("compare_app:select")

    filename = request.POST.get("filename")
    algo_a = request.POST.get("algoA")
    algo_b = request.POST.get("algoB")
    # récupérer les paramètres dynamiquement
    params_a = {
        p["name"]: request.POST.get(f"{algo_a}_{p['name']}") for p in ALGORITHMS[algo_a]
    }
    params_b = {
        p["name"]: request.POST.get(f"{algo_b}_{p['name']}") for p in ALGORITHMS[algo_b]
    }

    # callback dummy pour appels synchrones
    def _dummy_callback(status, tour, city_coords, distance, error, logs):
        return

    # exécuter les algos
    result_a = _dispatch(algo_a, filename, params_a, _dummy_callback)
    result_b = _dispatch(algo_b, filename, params_b, _dummy_callback)

    # préparer le contexte pour le template
    context = {
        "data": [
            {
                "name": algo_a,
                "distance": result_a["distance"],
                "time": result_a["time"],
            },
            {
                "name": algo_b,
                "distance": result_b["distance"],
                "time": result_b["time"],
            },
        ]
    }
    return render(request, "compare_app/results.html", context)
