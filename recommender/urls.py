from django.urls import path
from django.views.generic import RedirectView
from . import views

urlpatterns = [
    path('api/recommendations/', views.get_recommendations, name='get_recommendations'),
    path('api/train/', views.train_model, name='train_model'),
    path('api/upload-csv/', views.upload_csv_data, name='upload_csv_data'),
    path('', RedirectView.as_view(url='/swagger/', permanent=False), name='home'),
] 