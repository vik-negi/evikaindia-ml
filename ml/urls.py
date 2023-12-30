from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/keywords", views.get_keywords, name="key"),
]