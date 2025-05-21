from django.urls import path
from .views import solver_app, results_view


app_name = 'aco_app'

urlpatterns = [
    path('', solver_app, name='run_aco'),
    path('result', results_view, name='result'),
]
