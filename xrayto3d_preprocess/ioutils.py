"""i/o utils
- read/write volume
- read centroid
"""
import json
import logging
from pathlib import Path
from typing import Tuple, Sequence
from typing import Dict, List, Optional


import nibabel as nib
import pandas as pd
import numpy as np
import SimpleITK as sitk
import yaml


def read_nibabel(image_path):
    """read volume using nibabel"""
    return nib.load(image_path)


def read_yaml(yaml_path):
    """read configuration file"""
    stream = open(yaml_path, "r")
    return yaml.safe_load(stream)


def strip_newline(item, strip="\n"):
    """strip newline"""
    return item.strip(strip)


def read_subject_list(subject_list_path) -> np.ndarray:
    """read csv file and return numpy array"""
    return pd.read_csv(subject_list_path, header=None).to_numpy()


def get_logger(name, dirpath="logs", level=logging.DEBUG):
    """return logger
    create `dirpath` if it does not exist
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    filepath = f"{dirpath}/{name}.log"
    Path(filepath).parent.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(filepath, "w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    return logger


def load_centroids(ctd_path) -> Tuple[Sequence, Sequence]:
    """loads the json centroid file

    ctd_path: full path to the json file
    # from https://github.com/anjany/verse/blob/main/utils/data_utilities.py

    Returns:
    --------
    direction: Sequence representing orientation e.g. PIR, RAI
    ctd_list: a list containing the orientation and coordinates of the centroids"""

    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    direction = []
    for d in dict_list:
        if "direction" in d:
            direction = d["direction"]
        else:
            ctd_list.append((d["label"], d["X"], d["Y"], d["Z"]))
    return direction, ctd_list


def read_image(img_path) -> sitk.Image:
    """returns the SimpleITK image read from given path

    Parameters:
    -----------
    pixeltype (ImagePixelType):
    """
    img_path = Path(img_path).resolve()
    img_path = str(img_path)

    # sitk.ReadImage itself is robust
    # specifying pixelType resulted in more trouble
    # if pixeltype == ImagePixelType.ImageType:
    #     pixeltype = sitk.sitkUInt16
    #     return sitk.ReadImage(img_path,pixeltype)

    # elif pixeltype == ImagePixelType.SegmentationType:
    #     pixeltype = sitk.sitkUInt8
    #     return sitk.ReadImage(img_path,pixeltype)

    # else:
    #     raise ValueError(f'ImagePixelType cannot be {pixeltype}')

    return sitk.ReadImage(img_path)


def write_image(img, out_path, pixeltype=None):
    """save image"""
    if isinstance(out_path, Path):
        out_path = str(out_path)
    if pixeltype:
        img = sitk.Cast(img, pixeltype)
    sitk.WriteImage(img, out_path)


def read_dicom(dicom_dir_path) -> sitk.Image:
    """read dicom series"""
    if isinstance(dicom_dir_path, Path):
        dicom_dir_path = str(dicom_dir_path)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir_path)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()
    return image


def write_csv(data: Dict[str, List], column_names: Optional[List[str]], file_path):
    """
    write data into csv

    The dictionary should consist of key:List as key value pairs representing a single row.
    Optional name of each columns may also be provided
    """
    df = pd.DataFrame.from_dict(data, orient="index", columns=column_names)
    df.to_csv(file_path)
