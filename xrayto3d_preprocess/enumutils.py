"""Enums"""
from enum import Enum, auto


class ImageType(Enum):
    """volume type"""

    IMAGE = auto()
    SEGMENTATION = auto()


class ProjectionType(Enum):
    """x-ray projection type"""

    AP = auto()
    LAT = auto()
