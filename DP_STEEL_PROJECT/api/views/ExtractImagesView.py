from ..forms import ExtractImagesForm
from ..storage import MediaStorage, PDFStorage
from ..utils.PDFImgExtractor import PDFImgExtractor

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from datetime import datetime
import zipfile
import os
import logging

from django.views import View
from django.contrib import messages
from django.http import FileResponse, JsonResponse, HttpResponse
from django.core.files.uploadedfile import TemporaryUploadedFile
from django.core.cache import cache
from django.urls import reverse

logger = logging.getLogger('api.views')


EXTRACTED_IMAGES_STORAGE = MediaStorage(subdir='extracted_images')
UPLOAD_STORAGE = MediaStorage(subdir='uploads')
PDF_UPLOAD_STORAGE =  PDFStorage()
EXTRACTED_IMG_ZIP_FILENAME = os.path.join(EXTRACTED_IMAGES_STORAGE.location, 'extracted_images_tmp.zip')

class ExtractImagesView(View):
    '''View for extracting images from a PDF file.'''
    def get(self, request):
        return JsonResponse({'status': 'ok'})
    
    def post(self, request):
        # if request.session.get('processing_form') is True:
        #     return HttpResponse('Service temporarily unavailable', status=503)

        form = ExtractImagesForm(request_POST=request.POST, request_FILES=request.FILES)
        #request.session['form'] = form
        if form.is_valid():
            #upload form files to the PDF_UPLOAD_STORAGE and get the paths
            publications = request.FILES.getlist('publications')
            request.session['filenames'] = []
            
            for publication in publications:
                filename = PDF_UPLOAD_STORAGE.save(publication.name, publication)
                request.session['filenames'].append(filename)
            # how form.clead
            request.session['classification_type'] = form.cleaned_data['classification_type']
            #print(request.session.get('processing_form'))
            if request.session.get('processing_form') is True:
                return HttpResponse('Service temporarily unavailable', status=503)  
         
            self.process_form(request)            
            request.session.save()
            # FIXME add status 500 response if error
            #request.session.clear()
            return self.get_package_response_wrapper(request, request.session['package_path'])
        else:
            messages.error(request, 'Form is not valid.')
                
    def get_cookie(request, cookie_name):
        '''Get the value of the session variable.'''
        logger.info(f"get_cookie on frontend {cookie_name}, {request.session.get(cookie_name)}")
        print(f"get_cookie on frontend {cookie_name}, {request.session.get(cookie_name)}")       
        response = JsonResponse({cookie_name: request.session.get(cookie_name)})
        return response
    
    def process_form(self, request):
        '''
        Process the form and extract images from the PDF files.
        Save the extracted images to a zip and save the zip file to the storage.
        '''
        # Generate a unique cache key using the session ID
        session_id = request.headers.get('session-id')
        cache_key = f'is_processing_form_{session_id}'
       
        if cache.get(cache_key):
            return JsonResponse({'error': 'Process is already running in this session'}, status=429)
        # Set the flag in the session to True to indicate the process has started
        cache.set(cache_key, True, timeout=300)

        try:
            classification_type = request.session.get('classification_type')
            filenames = request.session.get('filenames')
            publications = [PDF_UPLOAD_STORAGE.open(filename) for filename in filenames]
            response_images_paths = self.get_response_images_paths(request, publications, classification_type)
            for publication in publications:
                publication.close()
            self.write_images_zip(response_images_paths, EXTRACTED_IMG_ZIP_FILENAME, classification_type)
            filename = self.generate_package_name(classification_type)
            with open(EXTRACTED_IMG_ZIP_FILENAME, 'rb') as f:
                package_path = EXTRACTED_IMAGES_STORAGE.save(filename, f)
            request.session['package_path'] = package_path
            logger.info(f"Package saved to {package_path}")

            # Assuming a successful process, return an appropriate response
           
        except Exception as e:
            # Handle any exceptions, log errors, etc.
            return JsonResponse({'error': str(e)}, status=500)
        
        finally:
            cache.delete(cache_key)
            return JsonResponse({'success': 'Form processed successfully'}, status=200)
            
    def get_response_images_paths(self, request, publications: list[TemporaryUploadedFile],
                                  classification_type: str):
        '''
        Args:
            publications: List of publication form files.
            microstructure_classification: (bool) - if True, images will be classified as microstructure or rest.
        Returns:
            list: List of paths to the response images.
        '''
        if classification_type not in ['image', 'microstructure', 'none']:
            raise ValueError('Invalid classification type.')
        response_images_paths = []
        session_id = request.headers.get('session-id')
        channel_layer = get_channel_layer()
        for i in range(len(publications)):
            publication = publications[i]
            #log the publication
            async_to_sync(channel_layer.group_send)(
                session_id,
                {
                    'type': 'progress_update',
                    'message': f'Processing file {i+1}/{len(publications)}'
                }
            )
            logger.info(publication)
            request.session['extractor_status'] = f'processing {publication.name} '
            request.session.save()
            try:
                extractor = PDFImgExtractor(publication, classification_type, request)
                if classification_type == 'image' or classification_type == 'microstructure':
                    response_images_paths.extend(extractor.classifier.get_images_paths())
                elif classification_type == 'none':
                    response_images_paths.extend(extractor.get_images_paths())
            except Exception as e:
                logger.debug(e, exc_info=True)
                return JsonResponse({'error': str(e)}, status=500)
            #logger.info(f"\033[92m{i+1}/{len(publications)} publications processed.\033[0m")
            logger.info(f"{i+1}/{len(publications)} publications processed.")
        return response_images_paths
    
    def generate_package_name(self, classification_type: str):
        '''Generate a name for the zip file based on classification type.'''
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if classification_type == 'image':
            return f'{timestamp}_classified_images.zip'
        elif classification_type == 'none':
            return f'{timestamp}_extracted_images.zip'
        elif classification_type == 'microstructure':
            return f'{timestamp}_classified_microstructure_images.zip'
        
    def write_images_zip(self, response_images_paths: list, zip_file_path: str, classification_type: str):
        '''Write images to a zip file.'''
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for img in response_images_paths:
                if classification_type == 'image':
                    if 'microstructure' in img:
                        path =  f'microstructure/{os.path.basename(img)}'
                    elif 'rest' in img:
                        path =  f'rest/{os.path.basename(img)}'
                    zipf.write(img, path)
                elif classification_type == 'microstructure':
                    if 'microstructure_rest' in img:
                        path =  f'microstructure_rest/{os.path.basename(img)}'
                    elif 'dp_steel' in img:
                        path =  f'dp_steel/{os.path.basename(img)}'
                    zipf.write(img, path)
                else:
                    zipf.write(img, os.path.basename(img))
        logger.info(f"Files in result package:\n {zipf.namelist()}")
                    
    def get_package_response_wrapper(self, request, package_path):
        '''View for showing the success page after extracting images from a PDF file.'''
        package_response = reverse('extract_images_get_package_response', args=[package_path])
        # delete all request session variables
        if os.path.exists(EXTRACTED_IMG_ZIP_FILENAME):
            os.remove(EXTRACTED_IMG_ZIP_FILENAME)
        logger.info(f"Processing form success!, processing_form = {request.session.get('processing_form')}")
        logger.info("----------------Form processing finished----------------\n\n")
        package_response = {
            "download_url": package_response
        }
        request.session.flush()
        return JsonResponse(package_response)

    def get_package_response(self, package_path):
        '''Get the response for the extracted images package.'''
        file_path =  os.path.join(EXTRACTED_IMAGES_STORAGE.location, package_path)
        file = open(file_path, 'rb')
        logger.info(f"Sending package {file_path} to the user.")
        return FileResponse(file, as_attachment=True, filename=package_path)
    