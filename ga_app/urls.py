from django.urls import path
from .views import solver_app, results_view


app_name = 'ga_app'

urlpatterns = [
    path('', solver_app, name='run_ga'),
    path('result', results_view, name='result'),
]
