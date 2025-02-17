import sys
import numpy as np
from datetime import datetime, timedelta,timezone
#sys.path.append('/mnt/d/Code/tempest/Intelligent-Liquidity-Provisioning-Framework-V2/model_notebooks')
import tensorflow as tf
import os
import pandas as pd
import threading
import threading
import time

# Determine the base directory of your project
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the module
module_path = os.path.join(base_dir, 'model_notebooks')

# Add the path to sys.path
if module_path not in sys.path:
    sys.path.append(module_path)

#from rl_ilp_script import train_ddpg_agent, train_ppo_agent, eval_ddpg_agent, eval_ppo_agent, liquidity_strategy
from .rl_ilp_script import env_setup,train_ddpg_agent, train_ppo_agent, eval_ddpg_agent, eval_ppo_agent, perform_inference,ddpg_training_vis,ppo_training_vis,ddpg_eval_vis,ppo_eval_vis,predict_action


import json
import pathlib
from django.shortcuts import render
from django.http import HttpResponse,HttpResponseBadRequest
from django.http import JsonResponse
# Import your training, evaluation, and inference functions
#from .ilp_script import train_ddpg_agent, train_ppo_agent,eval_ddpg_agent, eval_ppo_agent,liquidity_strategy
from django.views.decorators.csrf import csrf_exempt

train_ddpg_result = {
    'status': '',
    'ddpg_actor_model_path': '', 
    'ddpg_critic_model_path': '',
    'max_steps': 0,
    'time_consumed': 0
}

def finish_train_ddpg(status, ddpg_actor_model_path, ddpg_critic_model_path, max_steps, time_consumed):
    print("Finish train ddpg ----------------------------------------")
    train_ddpg_result['status'] = status
    train_ddpg_result['ddpg_actor_model_path'] = ddpg_actor_model_path
    train_ddpg_result['ddpg_critic_model_path'] = ddpg_critic_model_path
    train_ddpg_result['max_steps'] = max_steps
    train_ddpg_result['time_consumed'] = time_consumed
    print(train_ddpg_result)

@csrf_exempt
def home(request):
    return HttpResponse("Intelligent Liquidity Provisioning Framework")

@csrf_exempt
def download_file(request):
    base_path = pathlib.Path().resolve().as_posix()
    file_path = request.GET.get('file_path', '')
    file_name = request.GET.get('file_name', '')
    full_path = os.path.join(base_path, file_path, file_name)
    if os.path.exists(full_path):
        with open(full_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type='application/force-download')
            response['Content-Disposition'] = f'attachment; filename="{file_name}"'
            response['Access-Control-Expose-Headers'] = 'Content-Disposition'
            return response
    else:
        return HttpResponseBadRequest("File does not exist")

@csrf_exempt
def initialize_script(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            base_path = data.get('base_path')
            reset_env_var = data.get('reset_env_var')

            # Call the function to set the base path in rl_ilp_scripts.py
            env_setup(base_path,reset_env_var)

            return JsonResponse({'message': 'Base path and reset env var configured successfully'})
        except Exception as e:
            return JsonResponse({'error': str(e)})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})

@csrf_exempt
def get_train_ddpg_result(request):
    return JsonResponse(train_ddpg_result)

@csrf_exempt
def train_ddpg(request):
    if request.method == 'POST':
        try:
            if (train_ddpg_result['status'] == 'running'):
                return HttpResponseBadRequest('Another training is running')
            
            train_ddpg_result['status'] = 'running'
            train_ddpg_result['ddpg_actor_model_path'] = ''
            train_ddpg_result['ddpg_critic_model_path'] = ''
            train_ddpg_result['max_steps'] = 0
            train_ddpg_result['time_consumed'] = 0
            data = json.loads(request.body)
            
            # Extract arguments from data with default values
            max_steps = data.get('max_steps', 2)
            n_episodes = data.get('n_episodes', 2)
            model_name = data.get('model_name', 'model_storage/ddpg/ddpg_fazool')
            alpha = data.get('alpha', 0.001)
            beta = data.get('beta', 0.001)
            tau = data.get('tau', 0.8)
            batch_size = data.get('batch_size', 50)
            training = data.get('training', True)
            agent_budget_usd = data.get('agent_budget_usd', 10000)
            use_running_statistics = data.get('use_running_statistics', False)

            def train(max_steps, n_episodes, model_name, alpha, beta, tau, batch_size, training, agent_budget_usd, use_running_statistics):
                try:
                    start = time.time()
                    ddpg_train_data_log, ddpg_actor_model_path, ddpg_critic_model_path = train_ddpg_agent(
                        max_steps=max_steps, 
                        n_episodes=n_episodes, 
                        model_name=model_name, 
                        alpha=alpha, 
                        beta=beta, 
                        tau=tau, 
                        batch_size=batch_size, 
                        training=training, 
                        agent_budget_usd=agent_budget_usd, 
                        use_running_statistics=use_running_statistics)
                    finish_train_ddpg('done', ddpg_actor_model_path, ddpg_critic_model_path, max_steps, time.time() - start)
                except Exception as e:
                    print('Train ddpg error ---------------------------------')
                    print(e)
                    finish_train_ddpg('error', '', '', 0, 0)

            threading.Thread(target=train, args=(max_steps, n_episodes, model_name, alpha, beta, tau, batch_size, training, agent_budget_usd, use_running_statistics)).start()
            return JsonResponse(train_ddpg_result)

        except Exception as e:
             print(e)
             return HttpResponse({'error': str(e)})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})


@csrf_exempt
def evaluate_ddpg(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Extract parameters from data with default values
            eval_steps = data.get('eval_steps', 2)
            eval_episodes = data.get('eval_episodes', 2)
            model_name = data.get('model_name', 'model_storage/ddpg/ddpg_fazool')
            percentage_range = data.get('percentage_range', 0.6)
            agent_budget_usd = data.get('agent_budget_usd', 10000)
            use_running_statistics = data.get('use_running_statistics', False)

            # Call the evaluation function
            ddpg_eval_data_log = eval_ddpg_agent(
                eval_steps=eval_steps,
                eval_episodes=eval_episodes,
                model_name=model_name,
                percentage_range=percentage_range,
                agent_budget_usd=agent_budget_usd,
                use_running_statistics=use_running_statistics
            )


            # Prepare and return the response
            return JsonResponse({
                'message': f'Evaluation peformed'
                
            })
        except Exception as e:
            return JsonResponse({'error': str(e)})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})


@csrf_exempt
def train_ppo(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Extract parameters from data with default values
            max_steps = data.get('max_steps', 2)
            n_episodes = data.get('n_episodes', 2)
            model_name = data.get('model_name', 'model_storage/ppo/ppo2_fazool')
            buffer_size = data.get('buffer_size', 5)
            n_epochs = data.get('n_epochs', 20)
            gamma = data.get('gamma', 0.5)
            alpha = data.get('alpha', 0.001)
            gae_lambda = data.get('gae_lambda', 0.75)
            policy_clip = data.get('policy_clip', 0.6)
            max_grad_norm = data.get('max_grad_norm', 0.6)
            agent_budget_usd = data.get('agent_budget_usd', 10000)
            use_running_statistics = data.get('use_running_statistics', False)
            action_transform = data.get('action_transform', 'linear')

            # Call the PPO training function
            ppo_train_data_log_path, ppo_actor_model_path, ppo_critic_model_path = train_ppo_agent(
                max_steps=max_steps, n_episodes=n_episodes, model_name=model_name, 
                buffer_size=buffer_size, n_epochs=n_epochs, gamma=gamma, alpha=alpha, 
                gae_lambda=gae_lambda, policy_clip=policy_clip, max_grad_norm=max_grad_norm, agent_budget_usd=agent_budget_usd,
                use_running_statistics=use_running_statistics, action_transform=action_transform
                )
            
            
            response_data = {
                'ppo_actor_model_path': ppo_actor_model_path,
                'ppo_critic_model_path': ppo_critic_model_path
            }
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({'error': str(e)})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})

@csrf_exempt
def evaluate_ppo(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Extract parameters from data with default values
            eval_steps = data.get('eval_steps', 2)
            eval_episodes = data.get('eval_episodes', 2)
            model_name = data.get('model_name', 'model_storage/ppo/ppo2_fazool')
            percentage_range = data.get('percentage_range', 0.5)
            agent_budget_usd = data.get('agent_budget_usd', 10000)
            use_running_statistics = data.get('use_running_statistics', False)
            action_transform = data.get('action_transform', 'linear')

            # Call the PPO evaluation function
            ppo_eval_data_log = eval_ppo_agent(
                eval_steps=eval_steps, eval_episodes=eval_episodes, model_name=model_name, 
                percentage_range=percentage_range, agent_budget_usd=agent_budget_usd, 
                use_running_statistics=use_running_statistics, action_transform=action_transform
            )

            # Prepare and return the response
            return JsonResponse({
                'message': f'Evaluation peformed'
                
                })
        except Exception as e:
            return JsonResponse({'error': str(e)})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})

@csrf_exempt
def inference(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Extract pool_state and user_preferences from data
            pool_state = data.get('pool_state', {
                'current_profit': 500,
                'price_out_of_range': False,
                'time_since_last_adjustment': 40000,
                'pool_volatility': 0.2
            })
            user_preferences = data.get('user_preferences', {
                'risk_tolerance': {'profit_taking': 50, 'stop_loss': -500},
                'investment_horizon': 7,
                'liquidity_preference': {'adjust_on_price_out_of_range': True},
                'risk_aversion_threshold': 0.1,
                'user_status': 'new_user'
            })
            pool_id = data.get('pool_id', '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36')  # Default to ETH/USDT
            ddpg_agent_path = data.get('ddpg_agent_path', 'model_storage/ddpg/ddpg_1')
            ppo_agent_path = data.get('ppo_agent_path', 'model_storage/ppo/lstm_actor_critic_batch_norm')
            date_str = data.get('date_str', '2024-05-05')

            # Call the perform_inference function
            strategy_action, ddpg_action, ppo_action = perform_inference(
                user_preferences, pool_state, pool_id=pool_id, 
                ddpg_agent_path=ddpg_agent_path, ppo_agent_path=ppo_agent_path, date_str=date_str
                )
            response_data = {
            'strategy_action': strategy_action, 
            'ddpg_action': ddpg_action, 
            'ppo_action': ppo_action
        }
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({'error': str(e)})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})

#pool_id="0x3416cf6c708da44db2624d63ea0aaef7113527c6",ddpg_agent_path='model_storage/ddpg/ddpg_1',ppo_agent_path='model_storage/ppo/lstm_actor_critic_batch_norm'
@csrf_exempt
def predict_action(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            pool_id = data.get('pool_id', '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36')  # Default to ETH/USDT
            ddpg_agent_path = data.get('ddpg_agent_path', 'model_storage/ddpg/ddpg_1')
            ppo_agent_path = data.get('ppo_agent_path', 'model_storage/ppo/lstm_actor_critic_batch_norm')
            date_str = data.get('date_str', '2024-05-05')

            # Call the perform_inference function
            ddpg_action,ddpg_action_dict,ddpg_action_ticks,ppo_action, ppo_action_dict,ppo_action_ticks = predict_action(
                pool_id=pool_id, 
                ddpg_agent_path=ddpg_agent_path, 
                ppo_agent_path=ppo_agent_path,
                date_str=date_str
                )
            response_data = { 
            'ddpg_action': ddpg_action, 
            'ppo_action': ppo_action
        }
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({'error': str(e)})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'})