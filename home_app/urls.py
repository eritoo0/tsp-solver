from django.urls import path
from .views import home_view ,meta_heuristics , compare


app_name = 'home_app'
urlpatterns = [
    path("", home_view, name = 'home_app'),
    path('meta_heuristics',meta_heuristics  , name='meta_heuristics'),
    path('compare',compare  , name='compare'),
]
