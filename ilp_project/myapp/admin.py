from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import LiquidityPool, Transaction, User

admin.site.register(LiquidityPool)
admin.site.register(Transaction)
admin.site.register(User)
