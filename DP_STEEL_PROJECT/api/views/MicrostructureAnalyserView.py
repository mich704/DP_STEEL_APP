from ..forms import AnalyseMicrostructureForm
from ..storage import MediaStorage
from ..utils.MicrostructureAnalyser import MicrostructureAnalyser

import zipfile
import os
import logging
from datetime import datetime

from django.http import FileResponse, JsonResponse, HttpResponse
from django.contrib import messages
from django.core.files.uploadedfile import TemporaryUploadedFile
from django.views import View
from django.core.cache import cache
from django.urls import reverse

logger = logging.getLogger(__name__)

IMAGE_UPLOAD_STORAGE = MediaStorage(subdir='uploads/images/dp_steel')
MICROSTRUCTURE_ANALYSIS_STORAGE = MediaStorage(subdir='microstructure_analysis')
EXTRACTED_IMG_ZIP_FILENAME = os.path.join(MICROSTRUCTURE_ANALYSIS_STORAGE.location, 'analyser_output_tmp.zip')


class MicrostructureAnalysisView(View):
    '''View for microstructure images analysis.'''
    def get(self, request):
        return JsonResponse({'status': 'ok'})
    
    def post(self, request):
        form = AnalyseMicrostructureForm(request.POST, request.FILES)
        if form.is_valid():
            images = request.FILES.getlist('images')
            request.session['filenames'] = []
            for img in images:
                filename = IMAGE_UPLOAD_STORAGE.save(img.name, img)
                request.session['filenames'].append(filename)    
            if request.session.get('processing_form') is True:
                return HttpResponse('Service temporarily unavailable', status=503)    
            try: 
                self.process_form(request)
                return self.get_package_response_wrapper(request, request.session['package_path'])     
            except Exception as e:
                logger.debug(e, exc_info=True)
                return JsonResponse({'error': 'An error occurred while processing the form'}, status=500)   
            request.session.save()   
        else:
            messages.error(request, 'Form is not valid.')
        return JsonResponse({'status': 'ok'})
    
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
            files = [IMAGE_UPLOAD_STORAGE.open(filename) for filename in filenames]
            response_files_paths = self.get_response_files_paths(request, files)
            for f in files:
                f.close()
            self.write_images_zip(response_files_paths, EXTRACTED_IMG_ZIP_FILENAME)
            filename = self.generate_package_name(classification_type)
            with open(EXTRACTED_IMG_ZIP_FILENAME, 'rb') as f:
                package_path = MICROSTRUCTURE_ANALYSIS_STORAGE.save(filename, f)
            request.session['package_path'] = package_path            # Assuming a successful process, return an appropriate response
            return response_files_paths
        except Exception as e:
            raise e
        finally:
            cache.delete(cache_key)
            
    def get_response_files_paths(self, request, files: list[TemporaryUploadedFile]):
        '''
        Args:
            images: List of DP steel images.
        Returns:
            list: List of paths to the response images.
        '''
        response_images_paths = []
      
        request.session['analyser_status'] = f'analysing images'
        request.session.save()
        session_id = request.headers.get('session-id')
        try:
            analyser = MicrostructureAnalyser(files, session_id)
            response_files_paths = analyser.get_output_files()
            response = response_files_paths
        except Exception as e:
            #logger.debug(e, exc_info=True)
            raise e
        #logger.info(f"\033[92m{i+1}/{len(publications)} publications processed.\033[0m")
        return response
    
    def generate_package_name(self, classification_type: str):
        '''Generate a name for the zip file based on classification type.'''
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        return f'{timestamp}_analyser_output.zip'
    
    def write_images_zip(self, response_images_paths: list, zip_file_path: str):
        '''Write images to a zip file.'''
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for img in response_images_paths:
                zipf.write(img, os.path.basename(img))
        logger.info(f"Files in result package:\n {zipf.namelist()}")
        
    def get_package_response_wrapper(self, request, package_path):
        '''View for showing the success page after extracting images from a PDF file.'''
        package_response = reverse('microstructure_analysis_get_package_response', args=[package_path])
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
        file_path =  os.path.join(MICROSTRUCTURE_ANALYSIS_STORAGE.location, package_path)
        file = open(file_path, 'rb')
        logger.info(f"Sending package {file_path} to the user.")
        return FileResponse(file, as_attachment=True, filename=package_path)