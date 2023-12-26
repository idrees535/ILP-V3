from django.db import models

# Create your models here.
from django.db import models

class LiquidityPool(models.Model):
    pool_id = models.CharField(max_length=100, unique=True)
    token0 = models.CharField(max_length=50)
    token1 = models.CharField(max_length=50)
    fee_tier = models.IntegerField()
    total_liquidity = models.DecimalField(max_digits=20, decimal_places=6)
    # Other relevant fields...

    def __str__(self):
        return f"{self.token0}/{self.token1} Pool"
class Transaction(models.Model):
    transaction_id = models.CharField(max_length=100, unique=True)
    pool = models.ForeignKey(LiquidityPool, on_delete=models.CASCADE)
    user = models.ForeignKey('User', on_delete=models.CASCADE)  # Assuming you have a User model
    amount = models.DecimalField(max_digits=20, decimal_places=6)
    transaction_type = models.CharField(max_length=50)  # e.g., 'Add Liquidity', 'Remove Liquidity'
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.transaction_id
    

from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission

class User(AbstractUser):
    groups = models.ManyToManyField(
        Group,
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_name="myapp_user_set",
        related_query_name="myapp_user",
    )
    user_permissions = models.ManyToManyField(
        Permission,
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name="myapp_user_set",
    )
