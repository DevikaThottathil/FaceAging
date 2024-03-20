from django.urls import path
from app import views

urlpatterns=[
    path('', views.userhome),
   # path('login', views.login),
   # path('reg', views.reg),
    path('user_home', views.userhome),

]