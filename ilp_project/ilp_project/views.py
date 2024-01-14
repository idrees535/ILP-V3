from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .models import LiquidityPool, Transaction
quest, 'myapp/transactions.html', {'transactions': transactions})
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import mlflow.pyfunc
# Import the necessary functions from ilp_script.py
from .ilp_script import train_ddpg_agent, eval_ddpg_agent, train_ppo_agent, eval_ppo_agent, liquidity_strategy

# views.py in your Django app
from django.http import HttpResponse
from .tasks import train_model_task

def train_model(request, model_id):
    train_model_task.delay(model_id)
    return HttpResponse("Training started!")


def index(request):
    return render(request, 'myapp/index.html')
def liquidity_pools(request):
    pools = LiquidityPool.objects.all()
    return render(request, 'myapp/pools_list.html', {'pools': pools

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


# Define your Django view functions
def train_ilp_agents(request):
    # Assuming you have a function to run the DDPG and PPO training
    ddpg_train_data_log, ddpg_actor_model_path, ddpg_critic_model_path = train_ddpg_agent(
        max_steps=2, n_episodes=2, model_name='model_storage/ddpg/ddpg_fazool',
        alpha=0.001, beta=0.001,  tau=0.8,
        batch_size=50, training=True, agent_budget_usd=10000,
        use_running_statistics=False
    )

    # Your additional logic here

    # Execute PPO training
    ppo_train_data_log, ppo_actor_model_path, ppo_critic_model_path = train_ppo_agent(
        max_steps=10, n_episodes=2, model_name='model_storage/ppo/ppo1_fazool',
        buffer_size=5, n_epochs=20,
        gamma=0.5, alpha=0.001, gae_lambda=0.75, policy_clip=0.6, max_grad_norm=0.6,
        agent_budget_usd=10000, use_running_statistics=False, action_transform='linear'
    )

    # Your additional logic here

    # Return JsonResponse with results
    return JsonResponse({'ddpg_results': ddpg_train_data_log, 'ppo_results': ppo_train_data_log})

def execute_strategy(request, pool_id):
    # Implement the logic to execute your ILP strategy
    pool_state = {
        'current_profit': 500,
        'price_out_of_range': False,
        'time_since_last_adjustment': 40000,
        'pool_volatility': 0.2
    }

    user_preferences = {
        'risk_tolerance': {'profit_taking': 50, 'stop_loss': -500},
        'investment_horizon': 7,  # days
        'liquidity_preference': {'adjust_on_price_out_of_range': True},
        'risk_aversion_threshold': 0.1,
        'user_status': 'new_user'
    }
    
    # Call the liquidity_strategy function
    strategy_action, ddpg_action, ppo_action = liquidity_strategy(
        user_preferences, pool_state, pool_id=pool,
        ddpg_agent_path='model_storage/ddpg/ddpg_1', ppo_agent_path='model_storage/ppo/lstm_actor_critic_batch_norm'
    )

    # Return JsonResponse with the strategy results
    return JsonResponse({'strategy_action': strategy_action, 'ddpg_action': ddpg_action, 'ppo_action': ppo_action})
