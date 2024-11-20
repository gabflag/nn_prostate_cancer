from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import diagnostic_app.load_model as model

def index(request):
    return render(request, 'home.html')

@csrf_exempt
def process(request):
    if request.method == 'POST':
        radius = float(request.POST.get('radius', 0))
        texture = float(request.POST.get('texture', 0))
        perimeter = float(request.POST.get('perimeter', 0))
        area = float(request.POST.get('area', 0)) 
        smoothness = float(request.POST.get('smoothness', 0)) 
        compactness = float(request.POST.get('compactness', 0)) 
        symmetry = float(request.POST.get('symmetry', 0)) 
        fractal_dimension = float(request.POST.get('fractal_dimension', 0))  

        result = model.using_model(radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension)
        
        return JsonResponse({'status': 'ok', 'message': result})
    
    return JsonResponse({'status': 'error'}, status=400)