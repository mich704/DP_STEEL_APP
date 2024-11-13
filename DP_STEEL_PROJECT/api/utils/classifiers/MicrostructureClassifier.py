import cv2
import numpy as np
import argparse
import os
import shutil
import traceback

from ..helpers import list_all_images, create_if_not_exists
from api.models import Publication, ClassifiedImage, PreprocessedImage, AIModel
from .ClassifierStrategy import ClassifierStrategy
from .ImageClassifier import ImageClassifier
from django.conf import settings as PROJECT_SETTINGS

class MicrostructureClassifier(ClassifierStrategy):
    def __init__(self, args: argparse.Namespace) -> None:
        #TODO improve classifying od dp steel framework
        # 1. Train CNN or some other arch
        # 2. Run ImageClassifier then run MicrostructureClassifier 
        # on ImageClassifier.output.microstructure folder path
        '''Initializes the ImageClassifier class with the input path and model path.'''
        if args.model is None:
            args.model =  os.path.join(PROJECT_SETTINGS.AI_MODELS_DIR,
                                        'microstructure_classifier',
                                        'microstructure_classifier_CNN.keras')
        img_classifier_args = ImageClassifier.parse_arguments()
        img_classifier_args.input = args.input
  
        self.img_classifier = ImageClassifier(img_classifier_args)
        self.img_classifier.classify_images()
        args.input = self.img_classifier.output + os.sep + "microstructure"
        self.classified_images_paths = []
        super().__init__(args)
        
        
    @property
    def output(self) -> str:
        '''Returns the output path for the classified images.'''
        return self._output
    
    @output.setter
    def output(self, args: argparse.Namespace) -> None:
        '''Sets the output path for the classified images.'''
        model_filename = os.path.splitext(os.path.basename(args.model))[0]
        classification_subfolder = os.path.dirname(os.path.dirname(args.input))
        self._output = (
                args.output
                if args.output is not None
                else os.path.join(classification_subfolder, f'{model_filename}')
            )
        create_if_not_exists(self.output)
        create_if_not_exists(self.output + os.sep + "dp_steel")
        create_if_not_exists(self.output + os.sep + "microstructure_rest")
        args.output = self.output
        
    def classify_images(self) -> None:
        '''Classifies the images based on the model provided in args.'''
        print('Microstructure image classification...')
        super().classify_images()
    
    def get_images_paths(self) -> list:
        '''Returns the list of classified images paths.'''
        #get path from keras.src.engine.sequential.Sequential
        return self.classified_images_paths
        # preprocessed_images = PreprocessedImage.objects.filter(path__in = self.input_img_list)
        # return [img.path for img in ClassifiedImage.objects.filter(image_parent__in=preprocessed_images,  
        #                                                            ai_model_id=self.ai_model.id)]
    
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
            classification = "dp_steel" if yhat > 0.5 else "microstructure_rest"
            if classification == 'dp_steel':
                pass
            return classification
        else:
            print(f"{path} file extension is not supported.")
            return None

    def save_to_database(self, label: str = None, parent_img_path: str = None) -> None:
        '''Save given image with filename to self.output folder and into database.'''
        filename = os.path.basename(parent_img_path)
        dst_dir = os.path.join(self.output, label)
        dst_path = os.path.join(dst_dir, filename)
        try:
            classified_microstructure_img = ClassifiedImage.objects.get(path=parent_img_path)
            if classified_microstructure_img:
                classified_microstructure_img.label = label
                classified_microstructure_img.path = dst_path
                classified_microstructure_img.ai_model = self.ai_model  
                classified_microstructure_img.save()
                self.classified_images_paths.append(classified_microstructure_img.path)
            try:
                shutil.copyfile(parent_img_path, dst_path)
            except Exception as e:
                print("Error traceback:")
                traceback.print_exc()
                raise e
        except ClassifiedImage.DoesNotExist:
            raise ValueError("Parent image does not exist in the database.")