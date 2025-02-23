"""utils to download dataset from varied sources such as 
TCIA, google drive, zenodo, synapese
"""
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Union

import requests
import SimpleITK as sitk
from monai.apps.utils import check_hash

from .ioutils import read_dicom


def download_synapse(
    synapse_id: str,
    filename: str,
    output_dir: Union[Path, str],
    hash_val: Optional[str],
    hash_type="md5",
):
    """use synapse client to download data from synapse.org

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

    out_path = output_dir / filename
    if out_path.exists():
        print(f"{out_path} exists. verifying integrity of the file... ")
        if check_hash(out_path, val=hash_val, hash_type=hash_type):
            return True
        else:
            # delete the unverified download
            os.system(f"rm {out_path}")
    else:
        command = f"synapse get -r {synapse_id} --downloadLocation {output_dir}"
        os.system(command)
        return check_hash(out_path, val=hash_val, hash_type=hash_type)


def download_gdown(file_id: str, filename: str, output_dir: Union[Path, str]):
    """
    use gdown to download files from google drive.
    using monai.apps.util.download_url
    which recognises google drive link could not find gdown module even when installed.

    Args:
        file_id (str): google drive file id
        filename (str): target filename to save the downloaded file
        output_dir (Union[Path,str]): filepath to save the downloaded file to
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    out_path = output_dir / filename
    if out_path.exists():
        print(f"{out_path} exists.")
    else:
        command = f"gdown --id {file_id}  -O {str(out_path)}"
        os.system(command)


def download_wget(
    url: str,
    filename: str,
    output_dir: Union[Path, str],
    hash_val: Optional[str] = None,
    hash_type="md5",
):
    """
    use wget to download files (tested on zenodo).
    tools such as curl and urllib resulted in "SSL: CERTIFICATE_VERIFY_FAILED" error.
    Hence, the choice to use `wget`.

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

    url_format = "wget --no-check-certificate {input_url} -O {save_path}"
    out_path = output_dir / filename
    if out_path.exists():
        print(f"{out_path} exists. verifying integrity of the file... ")
        if check_hash(out_path, val=hash_val, hash_type=hash_type):
            return True
        else:
            # delete the unverified download
            os.system(f"rm {out_path}")
    os.system(url_format.format(input_url=url, save_path=str(out_path)))
    return check_hash(out_path, val=hash_val, hash_type=hash_type)


def get_image_tcia_restapi_url(series_uid: str) -> str:
    """return url to the set of images in a ZIP file"""
    return f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"


def get_image_with_md5hash_tcia_restapi_url(series_uid: str) -> str:
    """return url to the set of images in a ZIP file"""
    return f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImageWithMD5Hash?SeriesInstanceUID={series_uid}"


def get_image_metadata_tcia_restapi_url(series_uid: str) -> str:
    """return url for obtaining the metadata of a series from TCIA Rest API"""
    return f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeriesMetaData?SeriesInstanceUID={series_uid}"


def call_rest_api(url: str, timeout=10000):
    """used to call TCIA rest api and obtain a json response"""
    response = requests.get(url, timeout=timeout)
    return response.json()


def zip_dicom_to_nifti(
    zip_filepath: Union[Path, str], output_dir: Union[Path, str] = "."
):
    """convert zip DICOM to .nii.gz format

    Args:
        zip_filepath (Union[Path, str]): _description_
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as default_temp_dir:
        with zipfile.ZipFile(zip_filepath, "r") as zip_file:
            try:
                zip_file.extractall(default_temp_dir)
            except Exception:
                print(f"Zip extract failed {zip_filepath}")

        for path_, _, file_ in os.walk(default_temp_dir):
            dcm_images = read_dicom(path_)
            zip_filepath = Path(zip_filepath)
            nifti_filepath = zip_filepath.with_suffix(".nii.gz")
            sitk.WriteImage(dcm_images, f"{output_dir}/{str(nifti_filepath.name)}")

        shutil.rmtree(default_temp_dir)
