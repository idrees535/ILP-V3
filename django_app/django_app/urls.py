from django.contrib import admin
from django.urls import path, include
from . import views
from .views import train_ddpg,get_train_ddpg_result,train_ppo, evaluate_ddpg,evaluate_ppo, inference,initialize_script,predict_action,download_file

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('ilp/', include('ilp_app.urls')),
    path('train_ddpg/', train_ddpg, name='train_ddpg'),
    path('train_ddpg_result/', get_train_ddpg_result, name='get_train_ddpg_result'),
    path('evaluate_ddpg/', evaluate_ddpg, name='evaluate_ddpg'),
    path('train_ppo/', train_ppo, name='train_ppo'),
    path('evaluate_ppo/', evaluate_ppo, name='evaluate_ppo'),
    path('inference/', inference, name='inference'),
    path('predict_action/', inference, name='predict_action'),
    path('initialize_script/', initialize_script, name='initialize_script'), 
    path('download_file/', download_file, name='download_file'), 
]


