from enum import Enum,auto

class ImageType(Enum):
    Image = auto()
    Segmentation = auto()

class ProjectionType(Enum):
    ap = auto()
    lat = auto()
