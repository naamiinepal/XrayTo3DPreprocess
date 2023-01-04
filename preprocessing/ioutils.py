import SimpleITK as sitk
from pathlib import Path
from .enumutils import ImagePixelType
import json
import yaml
import logging
from logging.handlers import RotatingFileHandler
import nibabel as nib

def read_nibabel(image_path):
    return nib.load(image_path)
    
def read_yaml(yaml_path):
    stream = open(yaml_path,'r')
    return yaml.safe_load(stream)

def strip_item(item,strip='\n'):
    return item.strip(strip)

def get_logger(name,level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(f'logs/{name}.log','w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    return logger


def load_centroids(ctd_path):
    # from https://github.com/anjany/verse/blob/main/utils/data_utilities.py
    """loads the json centroid file

    ctd_path: full path to the json file

    Returns:
    --------
    ctd_list: a list containing the orientation and coordinates of the centroids"""

    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    direction = []
    for d in dict_list:
        if 'direction' in d:
            direction = d['direction']
        else:
            ctd_list.append((d['label'], d['X'], d['Y'], d['Z']))
    return direction, ctd_list

def read_image(img_path):
    """returns the SimpleITK image read from given path

    Parameters:
    -----------
    pixeltype (ImagePixelType):
    """

    if isinstance(img_path, Path):
        img_path = str(img_path)

    # if pixeltype == ImagePixelType.ImageType:
    #     pixeltype = sitk.sitkUInt16
    #     return sitk.ReadImage(img_path,pixeltype)

    # elif pixeltype == ImagePixelType.SegmentationType:
    #     pixeltype = sitk.sitkUInt8
    #     return sitk.ReadImage(img_path,pixeltype)

    # else:
    #     raise ValueError(f'ImagePixelType cannot be {pixeltype}')

    return sitk.ReadImage(img_path)



def write_image(img, out_path,pixeltype=None):
    if isinstance(out_path, Path):
        out_path = str(out_path)
    if pixeltype:
        img = sitk.Cast(img,pixeltype)
    sitk.WriteImage(img, out_path)