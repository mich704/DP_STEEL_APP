from rest_framework import serializers
from .models import Publication, Image, PreprocessedImage

class PublicationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Publication
        fields = '__all__'
        
class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = '__all__'
        
class PreprocessedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = PreprocessedImage
        fields = '__all__'