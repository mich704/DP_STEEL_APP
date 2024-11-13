from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from pathlib import Path

class MediaStorage(FileSystemStorage):
    def __init__(self, subdir='', mode='DEFAULT', *args, **kwargs):
        target_dir = os.path.join(settings.MEDIA_ROOT, subdir)
        os.makedirs(target_dir, exist_ok=True)
        kwargs['location'] = target_dir
        super(MediaStorage, self).__init__(*args, **kwargs)
        
        
class PDFStorage(MediaStorage):
    def __init__(self, *args, **kwargs):
        super().__init__('uploads/pdfs', *args, **kwargs)
        
    def save(self, name: str | None, content, **kwargs) -> str:
        filename, ext = os.path.splitext(name)
        return super().save(filename[:50]+ext, content, **kwargs)
    
    
class StaticStorage(FileSystemStorage):
    def __init__(self, subdir='', *args, **kwargs):
        target_dir = os.path.join(settings.STORAGE_DIR, subdir)
        os.makedirs(target_dir, exist_ok=True)
        kwargs['location'] = target_dir
        super(StaticStorage, self).__init__(*args, **kwargs)
