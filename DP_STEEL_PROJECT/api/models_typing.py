from enum import Enum

class GeneralType(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
    
    @classmethod
    def choices(cls):
        return [(c.value, c.name) for c in cls]
    
    def __str__(self):
        return self.value

class PublicationType(GeneralType):
    unlabelled = 'unlabelled'
    labelled = 'labelled'
    
class ImageLabel(GeneralType):
    microstructure = 'microstructure'
    rest = 'rest'
    
class MicrostructureImageLabel(GeneralType):
    dp_steel = 'dp_steel'
    microstructure_rest = 'rest'

