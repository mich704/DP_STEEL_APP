import argparse
import os

from tqdm import tqdm
from ..helpers import list_all_images, create_if_not_exists

from api.models import Publication, ClassifiedImage, PreprocessedImage, AIModel
from abc import ABC, abstractmethod

class ClassifierStrategy(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        from keras.models import load_model
        '''Initializes the ClassifierStrategy class with the input path and model path.'''
        if args.input is None:
            raise ValueError("Input path is required.")
        if args.model is None:
            raise ValueError("Model path is required.")
        self.input = args.input
        self.ai_model = args.model
        self.model = load_model(self.ai_model.path)
        self.input_img_list =  list_all_images(self.input)
        self.output = args
        self.logfile = args
        
    @property
    def ai_model(self):
        '''Returns the AI model.'''
        return self.__ai_model
    
    @ai_model.setter
    def ai_model(self, model_path: str):
        '''Sets the AI model.'''
        self.__ai_model, _ = AIModel.objects.get_or_create(path=model_path, name=os.path.basename(model_path))
        
    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        '''
        Parses only known command line arguments.
        Returns:
            argparse.Namespace: Parsed arguments.
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-i", "--input", required=False, help="path to folder with images to classify"
        )
        parser.add_argument("-m", "--model", required=False, help="path to classification model")
        parser.add_argument(
            "-o",
            "--output",
            required=False,
            help="path to output folder with classification subfolders",
        )
        args, unknown = parser.parse_known_args()
        return argparse.Namespace(**vars(args))
    
    @property
    def logfile(self) -> str:
        '''Returns the path to the logfile.'''
        return self._logfile

    @logfile.setter
    def logfile(self, args_value: argparse.Namespace) -> None:
        '''Sets the path to the logfile and contents of logfile.'''
        path = os.path.join(self.output, "logs.txt")
        with open(path, "w") as f:
            for key, value in vars(args_value).items():
                f.write(f"--{key} '{value}'\n")
        self._logfile = path
    
   
    def classify_images(self) -> None:
        print(f"\nImages classification from {self.input}")
        for i, img_path in tqdm(enumerate(self.input_img_list), total=len(self.input_img_list)):
            prediction = self.get_img_class(img_path)
            if prediction is not None:
                self.save_to_database(prediction, img_path)
        print(f"DONE, Images classified into classes saved to {self.output}")
    
    @abstractmethod
    def get_img_class(self, path: str) -> str:
        pass    
    
    @abstractmethod
    def save_to_database(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod 
    def get_images_paths(self) -> list:
        pass
    
    
class ClassifierContext:
    def __init__(self, classifier: ClassifierStrategy):
        self._classifier = classifier

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier: ClassifierStrategy):
        self._classifier = classifier