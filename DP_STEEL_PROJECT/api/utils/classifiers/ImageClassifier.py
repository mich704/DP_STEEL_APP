import cv2
import numpy as np
import argparse
import os
import shutil

from tqdm import tqdm
from ..ThrowError import ThrowError
from ..helpers import list_all_images, create_if_not_exists
from api.models import Publication, ClassifiedImage, PreprocessedImage, AIModel
from .ClassifierStrategy import ClassifierStrategy
from django.conf import settings as PROJECT_SETTINGS

import logging
logger = logging.getLogger(__name__)
class ImageClassifier(ClassifierStrategy):
    #FIXME for multiple publications
    def __init__(self, args: argparse.Namespace) -> None:
        '''Initializes the ImageClassifier class with the input path and model path.'''
        if args.model is None:
            args.model =  os.path.join(PROJECT_SETTINGS.AI_MODELS_DIR,
                                                  'image_classifier',
                                                  'image_classifier_CNN.keras')
        super().__init__(args)

    def get_img_class(self, path: str) -> str:
        '''
        Classifies the image based on the model provided in args.
        Args:
            path: (str) - path to the image.
        Returns:
            str: classification of the image.
        '''
        img = cv2.imread(path)
    
        if img is not None:
            resize = cv2.resize(img, (256, 256))
            yhat = self.model.predict(np.expand_dims(resize / 255, 0), verbose=None)
            classification = "rest" if yhat > 0.5 else "microstructure"
            return classification
        else:
            print(f"{path} file extension is not supported.")
            return None

    @property
    def output(self) -> str:
        '''Returns the output path for the classified images.'''
        return self._output
    
    @output.setter
    def output(self, args: argparse.Namespace) -> None:
        '''Sets the output path for the classified images.'''
        #FIXME shorten output path
        model_filename = os.path.splitext(os.path.basename(args.model))[0]
        self._output = (
                args.output
                if args.output is not None
                else os.path.join(self.input, 'classification', f'{model_filename}')
            )
        a = len(self.output)
        if a > 180:
            pass
        create_if_not_exists(self.output)
        create_if_not_exists(self.output + os.sep + "microstructure")
        create_if_not_exists(self.output + os.sep + "rest")
        args.output = self.output
        
    def classify_images(self) -> None:
        print('Regular image classification...')
        super().classify_images()
        
    def save_to_database(self, label: str = None, parent_img_path: str = None) -> None:
        '''Save given image with filename to self.output folder and into database.'''
        filename = os.path.basename(parent_img_path)
        path = os.path.join(self.output, label, filename)
        img = cv2.imread(parent_img_path)
        try:
            preprocessed_img_parent_id = PreprocessedImage.objects.get(path=parent_img_path).id
            if not ClassifiedImage.objects.filter(path=path, label=label,
                                                image_parent_id=preprocessed_img_parent_id,
                                                ai_model=self.ai_model.id).exists():
                ClassifiedImage.objects.create(path=path, 
                                            label=label,
                                            image_parent_id=preprocessed_img_parent_id,
                                            ai_model_id=self.ai_model.id)
                
            len_dst = len(os.path.join(path))
            try:
                a = shutil.copy(parent_img_path, path)
                if len_dst > 256:
                    logger.error(f"Path too long: {len_dst}")
            except Exception as e: 
                print(f"Error saving image to {path}: {e}")
                pass
        except ClassifiedImage.DoesNotExist:
            raise ValueError("Parent image does not exist in the database.")
        
    def get_images_paths(self) -> list:
        '''Returns the list of classified images paths.'''
        #get path from keras.src.engine.sequential.Sequential
        preprocessed_images = PreprocessedImage.objects.filter(path__in = self.input_img_list)
        return [img.path for img in ClassifiedImage.objects.filter(image_parent__in=preprocessed_images,  
                                                                   ai_model_id=self.ai_model.id)]
    