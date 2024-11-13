import argparse
import os
import io
import fitz
import shutil
import PIL

from PIL import Image as PILImage
from .helpers import is_image, create_if_not_exists
from .PDFHelpers import get_publication_document_data
from .classifiers.ClassifierStrategy import ClassifierStrategy
from .classifiers.ClassifierFactory import ClassifierFactory
from .classifiers.ImageClassifier import ImageClassifier
from .classifiers.MicrostructureClassifier import MicrostructureClassifier
from ..storage import StaticStorage
from api.models import ExtractedImage, Publication
from tqdm import tqdm
from django.db import IntegrityError
from django.core.exceptions import ObjectDoesNotExist
from .SimplePreprocessor import SimplePreprocessor
from django.core.files.uploadedfile import TemporaryUploadedFile
from api.models_typing import PublicationType

from .ThrowError import ThrowError


PDFS_STORAGE = StaticStorage(subdir='pdfs')
IMAGES_STORAGE = StaticStorage(subdir='images')

class PDFImgExtractor:
    '''
    Class representing a PDF image extractor.
    
    This class provides a basic structure for extraction of all images in a PDF file.
    It can extract images from a single PDF file or from a folder of scientific publication files.
    PDFImgExtractor also provides interface for database operations and is a base for furhter image processing and classification.
    '''
    
    def __init__(self, publication_form_file_obj: TemporaryUploadedFile,
                 classification_type: str, request=None) -> None:
        '''
        Initialises the PDFImgExtractor object.
        Params:
            publication_form_file_obj(TemporaryUploadedFile): Publication temporaty file object.
        '''
        #init publication model get or create
        self.extracted_pub_path = os.path.join(PDFS_STORAGE.base_location, 'unlabelled',
                                               os.path.splitext(os.path.basename(publication_form_file_obj.name))[0])
        self.extracted_images = []
        self.publication = publication_form_file_obj
        self.request = request
        self.output_folder = os.path.join(IMAGES_STORAGE.base_location, 'extracted',
                                          os.path.splitext(self.publication.filename)[0])
        create_if_not_exists(self.output_folder)
        self.pdf_extract_images()
        self.preprocessor = SimplePreprocessor(self.get_preprocessor_args(), self.publication)
        self.preprocessed_images = self.preprocessor.get_images_paths()
        
        self.classifier = classification_type
        if self.classifier is not None:
            self.classifier.classify_images()
        if not self.extracted_images:
            self.extracted_images = self.preprocessed_images
        
    @property
    def publication(self) -> Publication:
        return self.__publication
    
    @publication.setter
    def publication(self, publication_form_file_obj) -> None:
        doc, publication_metadata = get_publication_document_data(publication_form_file_obj.file,
                                                                      publication_form_file_obj.name)
        self.doc = doc
        self.publication_metadata = publication_metadata
        try:
            self.__publication =  Publication.objects.get(filename=publication_metadata['filename'])
        except ObjectDoesNotExist:
            source_tmp = publication_metadata['path']
            self.__publication = Publication.objects.create(**publication_metadata)
            self.__publication.save()
            shutil.copy(source_tmp, self.extracted_pub_path)
            print(f"Publication {publication_metadata['title']} by {publication_metadata['author']} added to the database.")
        
    @property      
    def classifier(self) -> ClassifierStrategy:
        return self.__classifier
    
    @classifier.setter
    def classifier(self, classification_type: str) -> None:
        if classification_type not in ['image', 'microstructure', 'none']:
            raise ValueError('Invalid classification type.')
        if classification_type == 'none':
            self.__classifier = None    
        else:
            ClassifierFactory.register_strategy('image', ImageClassifier)
            ClassifierFactory.register_strategy('microstructure', MicrostructureClassifier)
            self.__classifier = ClassifierFactory.get_classifier(classification_type,
                                                                 classifier_input=self.preprocessor.output)  
           
    def pdf_extract_images(self) -> None:
        ''' Extracts images from a PDF file and saves them to the output folder.'''
        print(f"Extracting images from {self.publication.title}...")
        #send status to view here
    
        image_xrefs = {}
        publication_filename = os.path.splitext(self.publication.filename)[0]
   
        #generate image xrefs and filenames
        for i, page in enumerate(self.doc):
            for image in page.get_images():
                image_xrefs[image[0]] = os.path.join(self.output_folder,
                                        f'{publication_filename}_page{i+1}_img{image[0]}')
        
        for index, xref in enumerate(image_xrefs):
            img = self.doc.extract_image(xref)
            if img:
                img_out_path = os.path.join(image_xrefs[xref]+'.'+img["ext"])
                try:
                    pil_image = PILImage.open(io.BytesIO(img['image']))
                    width, height = pil_image.size
                    if width > 20 and height > 50 and is_image(img_out_path):
                        try:
                            with open(img_out_path, 'wb') as image:
                                if not ExtractedImage.objects.filter(path=img_out_path,
                                                                    publication=self.publication).exists():
                                    ex_img = ExtractedImage.objects.create(path=img_out_path, 
                                                                            publication=self.publication)
                                    ex_img.save()
                                image.write(img['image'])
                            image.close()
                        except:
                            a = len(img_out_path)
                            pass
                except PIL.UnidentifiedImageError as e:
                    if os.path.splitext(img_out_path)[1] == '.jb2':
                        print(f"\033[91mError: {e}, {img_out_path}\033[0m")
                    continue
                #print(img_out_path)
        print('DONE, Images extracted successfully!')
    
    def get_preprocessor_args(self) -> argparse.Namespace:
        '''Returns the preprocessor parsed arguments.'''
        preprocessor_args = SimplePreprocessor.parse_arguments()
        preprocessor_args.input = self.output_folder
        return preprocessor_args
    
    def get_images_paths(self) -> list:
        '''Returns the list of extracted images.'''
        return [img.path for img in ExtractedImage.objects.filter(publication=self.publication)]
    
    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        '''
        Parses only known command line arguments.
        Returns:
            argparse.Namespace: Parsed arguments.
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input_file", required=False, help="path to PDF document")
        parser.add_argument("-in_dir", "--input_dir", required=False, help="path to folder with PDFs")
        parser.add_argument("-o", "--output", required=True, help="path to images output folder")
        parser.add_argument("--preprocess", required=False, action="store_true", help="preprocessing flag")
        parser.add_argument("-m", "--model", required=False, help="classification model")
        parser.add_argument('--data_type', type=PublicationType, choices=list(PublicationType),
                            default=PublicationType.unlabelled,
                            required=False, help="data type ( unlabelled)")
        args, unknown = parser.parse_known_args()
        return argparse.Namespace(**vars(args))