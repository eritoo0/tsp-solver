from django.urls import path
from .views import solver_view, results_view


app_name = 'tsp_app'

urlpatterns = [
    path('', solver_view, name='solver'),
    path('result', results_view, name='result'),
]
