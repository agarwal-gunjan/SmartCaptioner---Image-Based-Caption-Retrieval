from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse

def test_ping(request):
    return JsonResponse({"message": "Django is working"})
urlpatterns = [
    path('', views.home, name='home'),  # website
    path('api/caption/', views.caption_api, name='caption_api'),
    path('ping/', test_ping),
    path('app/', views.app, name='app'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
