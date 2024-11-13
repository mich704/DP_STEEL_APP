import os
from django import setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DP_STEEL_PROJECT.settings')
setup()

from django.conf import settings as PROJECT_SETTINGS
from api.models import Publication, PreprocessedImage, Image, ClassifiedImage, AIModel
from api.utils.PDFHelpers import get_publication_document_data
from tqdm import tqdm
from django.db import IntegrityError
from django.core.exceptions import ValidationError

    
class DatabaseManager():
    '''Class to manage and seed the database with labelled data.'''
    def __init__(self):
        self.api_settings_defaults = PROJECT_SETTINGS
        self.publications_dir = PROJECT_SETTINGS.PDFS_DIR
        self.labelled_dataset_dir = PROJECT_SETTINGS.LABELLED_DATASET_DIR
        self.DB_NAME = PROJECT_SETTINGS.DATABASES['default']['NAME']
        self.ai_models_dir = PROJECT_SETTINGS.AI_MODELS_DIR
       
    
    def insert_publications(self, label: str = 'labelled') -> None:
        '''Insert publications with given label into the database'''
        print(f'Inserting {label} publications...')
        target_dir = os.path.join(self.publications_dir, label)
        for dirpath, subdirs, filenames in os.walk(target_dir):
            for filename in tqdm(filenames):
                try:
                    path = os.path.join(dirpath, filename)
                    file_obj = open(path, "rb")
                    doc, publication_metadata = get_publication_document_data(path, filename, label)
                    file_obj.close()
                    #check if the publication already exists in the database
                    try:
                        publication = Publication.objects.create(**publication_metadata)
                    except IntegrityError as e:
                        print(f'\033[91mPublication {publication_metadata["title"]} by {publication_metadata["author"]} already exists in the database.\033[0m')
                except ValueError as e:
                    print(f'Omitting file {path} due to error: {e}')
                    os.remove(path)
               
                
    
    def insert_labelled_images(self) -> None:
        '''Insert labelled images into the database.'''
        print('Inserting labelled images...')
        for dirpath, subdirs, filenames in os.walk(self.labelled_dataset_dir):
            for subdir in subdirs:
                label = subdir
                print(f'label: {label}')
                for filename in tqdm(os.listdir(os.path.join(dirpath, subdir))):
                    path = os.path.join(dirpath, subdir, filename)
                    #check if image does not exist, is unique
                    if not Image.objects.filter(path=path).exists():
                        img_filename = os.path.basename(path)
                        publication_filename = img_filename.split('_page')[0] + '.pdf'
                        try:
                            publication = Publication.objects.get(filename=publication_filename)
                            image = Image(path=path, label=label, publication=publication)
                        except:
                            image = Image(path=path, label=label)
                        #print(f'Inserting image {path}...')
                        image.save()
                    
    def delete_all_rows(self) -> None:
        '''Delete all rows from the database.'''
        PreprocessedImage.objects.all().delete()
        Image.objects.all().delete()
        Publication.objects.all().delete()
        AIModel.objects.all().delete()
       
        print('All rows deleted.')
        
    def insert_ai_models(self) -> None:
        '''Insert AI models into the database.'''
        print('Inserting AI models...')
        for dirpath, subdirs, filenames in os.walk(self.ai_models_dir):
            for filename in filenames:
                if filename.endswith('.h5') or filename.endswith('.keras'):
                    path = os.path.join(dirpath, filename)
                    try:
                        model = AIModel(name=filename, path=path)
                        model.save()
                        print(f'AI model {filename} inserted.')
                    except ValidationError as e:
                        print(e)
        
def main():
    
    db_manager = DatabaseManager()
    db_manager.delete_all_rows()
    db_manager.insert_ai_models()
    db_manager.insert_publications('labelled')
    db_manager.insert_publications('unlabelled')
    db_manager.insert_labelled_images()
    
    

if __name__ == "__main__":
    main()
    