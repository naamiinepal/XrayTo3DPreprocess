from enum import Enum,auto

class ImagePixelType(Enum):
    ImageType = auto()
    SegmentationType = auto()

class ProjectionType(Enum):
    ap = auto()
    lat = auto()
