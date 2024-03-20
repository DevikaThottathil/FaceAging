from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class UserReg(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE,null=True)
    name=models.CharField(max_length=250,null=True)
    email=models.EmailField(max_length=250,null=True)
    address=models.TextField(max_length=100,null=True)
    phone_no=models.IntegerField(null=True)