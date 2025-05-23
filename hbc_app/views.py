from django.shortcuts import render
from django.http import JsonResponse
import json
import threading
import json
import uuid

from random import random
from time import time

from _solver import solve_tsp_hbc
from orjson import dumps, loads


from django.views.decorators.csrf import csrf_exempt


JOBS = {}


def run_job(job_id, filename, params):
    """
    Appelé dans un thread séparé :
    - lance le solveur avec les paramètres fournis
    - met à jour le dictionnaire `jobs` à la fin
    """

    def callback(status,
                 tour,
                 city_coords,
                 distance,
                 error,
                 logs):
    
        JOBS[job_id] = {
            "status": status,
            "tour": tour,
            "city_coords": city_coords,
            "distance": distance,
            "error": error,
            "logs": logs
        }

    #
    tour, distance, error, logs, city_coords = solve_tsp_hbc(
        filename, callback, **params)


jobs = []


@csrf_exempt
def solver_app(request):

    if request.method == 'POST':
        # reset the json
        # with open('solver/status2.json', 'wb') as f:
        #     f.write(dumps({
        #         "status": 'not done',
        #         "tour": [],
        #         "city_coords": [],
        #         "distance": 0,
        #         "error": [],
        #         "logs": []
        #     }))

        # launch job
        data = json.loads(request.body)
        filename = data.pop('filename')

        job_id = str(uuid.uuid4())
        while job_id in jobs:
            job_id = str(uuid.uuid4())
        jobs.append(job_id)
        # état initial du job
        # jobs[job_id] = {'status': 'running', 'logs': []}
        # on lance le thread qui exécute run_job()
        threading.Thread(
            target=run_job,
            args=(job_id, filename, data),
            daemon=True
        ).start()

        return JsonResponse({'job_id': job_id})

    return render(request, 'hbc_app/hbc_app.html')


def results_view(request):
    try:
        job_id = request.GET.get('job_id')
        return JsonResponse(JOBS[job_id])
    except:
        return JsonResponse(
            {
                "status": 'not done',
                "tour": [],
                "city_coords": [],
                "distance": 0,
                "error": [],
                "logs": []
            }
        )
