from .views.ExtractImagesView import ExtractImagesView
from .views.MicrostructureAnalyserView import MicrostructureAnalysisView
from .views import views

from django.views.decorators.csrf import csrf_exempt
from django.urls import path


urlpatterns = [
     path('', views.index, name='index'),
     path('liveness/', views.liveness, name='liveness'),
     
     path('publications', views.PublicationsView.as_view()),
     path('publications/<int:pk>', views.PublicationView.as_view()),
     path('images', views.ImagesView.as_view()),
     path('images/<int:pk>', views.ImageView.as_view()),
     path('preprocessed_images', views.PreprocessedImageView.as_view()),
          
     path('extract_images/', 
          csrf_exempt(ExtractImagesView.as_view()), 
          name='extract_images'),
     #path('extract_images/progress/', views.ProgressView.as_view(), name='extract_images_progress'),
     path('extract_images/process_form/', 
          ExtractImagesView.process_form, 
          name='extract_images_process_form'),
     path('extract_images/get_cookie/<str:cookie_name>/', 
          ExtractImagesView.get_cookie, 
          name='get_cookie'),
     path('extract_images/get_package_response/<str:package_path>', 
          ExtractImagesView.get_package_response,
          name='extract_images_get_package_response'),
     
     path('microstructure_analysis/', 
          csrf_exempt(MicrostructureAnalysisView.as_view()), 
          name='microstructure_analysis'),
     path('microstructure_analysis/process_form/', 
          MicrostructureAnalysisView.process_form, 
          name='microstructure_analysis_process_form'),
     path('microstructure_analysis/get_package_response/<str:package_path>', 
          MicrostructureAnalysisView.get_package_response,
          name='microstructure_analysis_get_package_response'),
]