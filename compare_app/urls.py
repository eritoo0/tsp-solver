from django.urls import path
from . import views

app_name = 'compare_app'
urlpatterns = [
    path('',            views.select_algorithms, name='select'),
    path('solver/',     views.solver_app,       name='solver'),
    path('results/',    views.results_view,     name='results'),
    path('compare/',    views.compare_async,    name='compare_async'),
]
