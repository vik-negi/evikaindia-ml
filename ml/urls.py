from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/keywords", views.get_keywords, name="key"),
    path("api/posts-by-user-preferences", views.get_posts_by_user_preferences, name="get_posts_by_user_preferences"),
]