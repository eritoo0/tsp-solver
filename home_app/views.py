from django.shortcuts import render


# Create your views here.

def home_view(request):
    return render(request, 'home_app/home_app.html')

def meta_heuristics(request):
    """
    Affiche la page de sélection de la métaheuristique.
    """
    return render(request, 'home_app/meta_heuristics.html')



