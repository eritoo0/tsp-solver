from django.urls import path
from .views import client_app, results_view


app_name = 'client_app'

urlpatterns = [
    path('', client_app, name='client_app'),
    path('result', results_view, name='result'),
]
