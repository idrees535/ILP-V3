# tasks.py in your Django app
from celery import shared_task
from .models import YourModel
from .your_script import your_training_function

@shared_task
def train_model_task(model_id):
    model_instance = YourModel.objects.get(id=model_id)
    your_training_function(model_instance)
