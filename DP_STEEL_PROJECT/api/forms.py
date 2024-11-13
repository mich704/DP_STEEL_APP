from typing import Any, Mapping
from django import forms
from django.core.exceptions import ValidationError
from django.forms.renderers import BaseRenderer
from django.forms.utils import ErrorList
from .models import Publication, Image, PreprocessedImage

class ExtractImagesForm(forms.Form):
    '''Form for extracting images from a PDF file.'''
    publications = forms.FileField(
        label='Select a publication file',
        required=False,
        widget = forms.TextInput(attrs={
            "name": "publications",
            "type": "File",
            "class": "form-control",
            "multiple": "True",
            "accept": ".pdf",
            "required": "True"
        })
    )
    classification_type = forms.ChoiceField(
        choices=[('none', 'None'), ('image', 'Image Classification'), ('microstructure', 'Microstructure Classification')],
        widget=forms.Select,
        label='Choose classification type',
        initial='none',
        required=False,
    ) 
    
    def __init__(self, request_POST=None, request_FILES=None) -> None:
        if request_FILES is not None:
            for pub in request_FILES.getlist('publications'):
                if pub.content_type != 'application/pdf':
                    raise ValidationError('Only PDF files are allowed.')
        super().__init__(request_POST, request_FILES)
       
       
class MultiImageInput(forms.FileInput):
    def render(self, name, value, attrs=None, renderer=None):
        if attrs is None:
            attrs = {}
        attrs['multiple'] = 'multiple'
        attrs['accept'] = 'image/*'
        return super().render(name, value, attrs)


class AnalyseMicrostructureForm(forms.Form):
    '''Form for analysis of microstructure images.'''
    images = forms.FileField(widget=MultiImageInput(), required=False)
    
    def __init__(self, request_POST=None, request_FILES=None) -> None:
        if request_FILES is not None:
            for img in request_FILES.getlist('images'):
                if 'image' not in img.content_type:
                    raise ValidationError('Only image files are allowed.')
        super().__init__(request_POST, request_FILES)