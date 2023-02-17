import requests
import numpy as np
from pathlib import Path
import shutil
import tempfile
from typing import List, Optional, Tuple, Union
import os
import zipfile
from tqdm import tqdm
import SimpleITK as sitk

from monai.apps.utils import extractall, check_hash
from .ioutils import read_dicom

def download_synapse(synapse_id: str, filename: str, output_dir: Union[Path, str], hash_val: Optional[str], hash_type='md5'):
    """    use synapse client to download data from synapse.org

    Args:
        synapse_id (str): example: "syn3379050"
        filename (str): name of the file to be downloaded
        output_dir (Union[Path, str]): where to download the file
        hash_val (Optional[str]): hash string for verification
        hash_type (str, optional): _description_. Defaults to 'md5'.

    Returns:
        _type_: _description_
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    out_path = output_dir/filename
    if out_path.exists():
        print(f'{out_path} exists. verifying integrity of the file... ')
        if check_hash(out_path, val=hash_val, hash_type=hash_type):
            return True
        else:
            # delete the unverified download
            os.system(f'rm {out_path}')
    else:
        command = f'synapse get -r {synapse_id} --downloadLocation {output_dir}'
        os.system(command)
        return check_hash(out_path, val=hash_val, hash_type=hash_type)


def download_gdown(file_id: str, filename: str, output_dir: Union[Path, str], hash_val=None, hash_type='md5'):
    """use gdown to download files from google drive. using monai.apps.util.download_url which recognises google drive link could not find gdown module even when installed. Hence, this functions

    Args:
        file_id (str): google drive file id 
        filename (str): target filename to save the downloaded file 
        output_dir (Union[Path,str]): filepath to save the downloaded file to
        hash_val (bool): _description_
        hash_type (str, optional): _description_. Defaults to 'md5'.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    out_path = output_dir/filename
    if out_path.exists():
        print(f'{out_path} exists.')
    else:
        command = f'gdown --id {file_id}  -O {str(out_path)}'
        os.system(command)

def download_wget(url: str, filename: str, output_dir: Union[Path, str], hash_val: Optional[str] = None, hash_type='md5'):
    """
    use wget to download files (tested on zenodo).
    tools such as curl and urllib resulted in "SSL: CERTIFICATE_VERIFY_FAILED" error. Hence, the choice to use `wget`.

    Args:
        url (str): source URL link to download file
        filename (str): target filename to save the downloaded file
        output_dir (Path): filepath to save the downloaded file to
        hash_val (str,optional): hash
        hash_type (str,optional): Defaults to 'md5'.

    Returns:
        bool: True if file was succesfully downloaded and verified
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    url_format = 'wget {input_url} -O {save_path}'
    out_path = output_dir/filename
    if out_path.exists():
        print(f'{out_path} exists. verifying integrity of the file... ')
        if check_hash(out_path, val=hash_val, hash_type=hash_type):
            return True
        else:
            # delete the unverified download
            os.system(f'rm {out_path}')
    os.system(url_format.format(input_url=url,
              save_path=str(out_path)))
    return check_hash(out_path, val=hash_val, hash_type=hash_type)


def getImage_TCIA_restAPI_URL(Series_UID: str):
    """return url to the set of images in a ZIP file"""
    return f'https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={Series_UID}'


def getImageWithMD5Hash_TCIA_restAPI_URL(Series_UID: str):
    """return url to the set of images in a ZIP file"""
    return f'https://services.cancerimagingarchive.net/nbia-api/services/v1/getImageWithMD5Hash?SeriesInstanceUID={Series_UID}'


def getImageMetaData_TCIA_restAPI_URL(Series_UID: str):
    """return url for obtaining the metadata of a series from TCIA Rest API"""
    return f'https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeriesMetaData?SeriesInstanceUID={Series_UID}'


def call_rest_api(url: str):
    response = requests.get(url)
    return response.json()


def zipDICOMtoNifti(zip_filepath: Union[Path, str], output_dir:Union[Path,str]='.'):
    """convert zip DICOM to .nii.gz format

    Args:
        zip_filepath (Union[Path, str]): _description_
    """
    if isinstance(output_dir,str):
        output_dir = Path(output_dir)
        
    if not output_dir.exists():
        output_dir.mkdir(parents=True)


    with tempfile.TemporaryDirectory() as defaultTempDir:
        f = zipfile.ZipFile(zip_filepath, 'r')
        try:
            f.extractall(defaultTempDir)
        except BaseException:
            print('Zip extract failed')

        for path_, _, file_ in os.walk(defaultTempDir):
            dcm_images = read_dicom(path_)
            zip_filepath = Path(zip_filepath)
            nifti_filepath = zip_filepath.with_suffix('.nii.gz')
            sitk.WriteImage(
                dcm_images, f'{output_dir}/{str(nifti_filepath.name)}')

        shutil.rmtree(defaultTempDir)