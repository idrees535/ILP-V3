from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .models import LiquidityPool, Transaction

def index(request):
    return render(request, 'myapp/index.html')
def liquidity_pools(request):
    pools = LiquidityPool.objects.all()
    return render(request, 'myapp/pools_list.html', {'pools': pools})
def transaction_history(request, pool_id):
    transactions = Transaction.objects.filter(pool_id=pool_id)
    return render(request, 'myapp/transactions.html', {'transactions': transactions})
def run_simulation(request):
    # Assuming you have a function to run the simulation
    results = run_ilp_simulation()
    return JsonResponse(results)
def execute_strategy(request, pool_id):
    # Implement the logic to execute your ILP strategy
    strategy_result = execute_ilp_strategy(pool_id)
    return JsonResponse({'result': strategy_result})

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

import mlflow.pyfunc

# Load your model (adjust the path as necessary)
model = mlflow.pyfunc.load_model("/mnt/c/Users/hijaz tr/Desktop/cadCADProject1/Intelligent-Liquidity-Provisioning-Framework-V1/model_storage/ddpg/ddpg_1")


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Assuming data is in the correct format for your model
            prediction = model.predict(data)
            return JsonResponse({'prediction': prediction.tolist()})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'message': 'This endpoint accepts only POST requests.'})

