from django.db import models
import os
import sys
from django.core.exceptions import ValidationError
from .models_typing import PublicationType, ImageLabel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Create your models here.
class Publication(models.Model):
    '''Model representing a scientific publication.'''
    author = models.CharField(max_length=255, null=True)
    path = models.CharField(max_length=400, unique=True)
    filename = models.CharField(max_length=300, unique=True)
    type = models.CharField(max_length=255, choices=PublicationType.choices(), null=True)
    creation_date_raw = models.CharField(max_length=255, null=True)
    title = models.CharField(max_length=255, unique=False, null=True)
    keywords = models.CharField(max_length=255, null=True)
    
    def publication_images_paths(self):
        return [ex_img.path for ex_img in ExtractedImage.objects.filter(publication=self)]
    
    # def save(self, *args, **kwargs) -> None:
    #     filename, ext = os.path.splitext(self.filename)
    #     self.filename = filename[:50] + ext
    #     super().save(*args, **kwargs)
       
    def __str__(self) -> str:
        return f'{self.title} by {self.author}'
    class Meta:
        db_table = 'publications'
        

class AIModel(models.Model):
    '''Model representing an AI model.'''
    name = models.CharField(max_length=255, unique=False)
    path = models.CharField(max_length=400, unique=True)
    class Meta:
        db_table = 'ai_models'
    
    def save(self, *args, **kwargs):
        if '_epoch_' in self.path and 'keras_callback' in self.path:
            raise ValidationError(f'{self.path} is most likely a model checkpoint, not a final model.')
        super().save(*args, **kwargs)
    


class Image(models.Model):
    '''Model representing an image.'''
    path = models.CharField(max_length=400, unique=True)
    label = models.CharField(max_length=255, choices=ImageLabel.choices(), null=True)
    publication = models.ForeignKey(Publication, on_delete=models.CASCADE, null=True)
    class Meta:
        db_table = 'images'
        

class ExtractedImage(models.Model):
    '''Model representing an image extracted from publication.'''
    path = models.CharField(max_length=400, unique=True)
    publication = models.ForeignKey(Publication, on_delete=models.CASCADE, null=False)
    class Meta:
        db_table = 'extracted_images'


class PreprocessedImage(models.Model):
    '''Model representing a preprocessed image.'''
    path = models.CharField(max_length=400, unique=True)
    extracted_image_parent = models.ForeignKey(ExtractedImage, on_delete=models.CASCADE, null=True)
    image_parent = models.ForeignKey(Image, on_delete=models.CASCADE, null=True)
    
    def save(self, *args, **kwargs):
        if not self.extracted_image_parent and not self.image_parent:
            raise ValidationError("At least one of 'extracted_image_parent' or 'image_parent' must be set.")
        elif self.extracted_image_parent and self.image_parent:
            raise ValidationError("Only one of 'extracted_image_parent' or 'image_parent' should be set, not both.")
        if ClassifiedImage.objects.filter(path__endswith=os.path.basename(self.path)).exists():
            raise ValidationError(f"Image with {self.path} already exists in the database as ClassifiedImage.")
        
        super().save(*args, **kwargs)
    class Meta:
        db_table = 'preprocessed_images'

class ClassifiedImage(models.Model):
    '''Model representing a classified image.'''
    path = models.CharField(max_length=400, unique=True)
    label = models.CharField(max_length=255, choices=ImageLabel.choices(), null=False)
    image_parent = models.ForeignKey(PreprocessedImage, on_delete=models.CASCADE)
    ai_model = models.ForeignKey(AIModel, on_delete=models.CASCADE, null=False)
    
    class Meta:
        db_table = 'classified_images'