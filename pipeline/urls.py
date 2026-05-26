from django.urls import path
from pipeline import views

urlpatterns = [
    path("",              views.upload, name="upload"),
    path("result/<str:apk_hash>/", views.result,  name="result"),
]
