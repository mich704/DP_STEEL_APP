from ..models import Publication, Image, PreprocessedImage
from ..serializers import PublicationSerializer, ImageSerializer, PreprocessedImageSerializer
from ..storage import MediaStorage

import os
import logging
logger = logging.getLogger(__name__)

from django.http import JsonResponse, FileResponse
from django.views import View
from django.shortcuts import render
from django.contrib import messages
from rest_framework import generics


# Create your views here.
def index(response):
    return render(response, 'index.html',{})

def liveness(response):
    return JsonResponse({'status': 'working'})

def get_package_response(package_path, storage_subdir):
    '''Get the response for the extracted images package.'''
    if storage_subdir is None:
        return JsonResponse({'error': 'Storage subdir is not provided'}, status=500)
    if storage_subdir == 'extracted_images' or storage_subdir == 'microstructure_analysis':
        storage_location = MediaStorage(subdir=storage_subdir).location
        file_path =  os.path.join(storage_location, package_path)
        file = open(file_path, 'rb')
        logger.info(f"Sending package {file_path} to the user.")
        return FileResponse(file, as_attachment=True, filename=package_path)
    return JsonResponse({'error': 'Invalid storage subdir'}, status=500)

###### API views ######
class PublicationView(generics.RetrieveAPIView):
    queryset = Publication.objects.all()
    serializer_class = PublicationSerializer
    lookup_field = 'pk'


class PublicationsView(generics.ListAPIView):
    queryset = Publication.objects.all()
    serializer_class = PublicationSerializer
    
    
class ImageView(generics.RetrieveAPIView):
    #render first 100 images
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
    lookup_field = 'pk'


class ImagesView(generics.ListAPIView):
    queryset = Image.objects.all()[:100]
    serializer_class = ImageSerializer
    
    
class PreprocessedImageView(generics.ListAPIView):
    queryset = PreprocessedImage.objects.all()
    serializer_class = PreprocessedImageSerializer
    
    
# class ProgressView(View):
#     '''View for showing progress of extracting images from a PDF file.'''
#     def get(self, request):
#         # Render the progress page
#         content = render(request, 'extract_images_progress.html').content
#         view = ExtractImagesView()
#         response = HttpResponse(content)

#         def close():
#             super(HttpResponse, response).close()
#             view.process_form(request)
#             package_path = request.session.get('package_path')
#             package_response = reverse('get_package_response', args=[package_path])
#             response.content = render(request, 
#                                 'extract_images_success.html', 
#                                 {'package_response': package_response}).content

#         #TODO !!update response_package and pass it to template as response

     
#         return response
        