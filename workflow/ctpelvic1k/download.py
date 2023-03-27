"""download ctpelvic1k"""
from pathlib import Path
import shutil
import tempfile
from tqdm import tqdm
import pandas as pd
from xrayto3d_preprocess import mkdir_or_exist
from monai.apps.utils import extractall, download_url

from xrayto3d_preprocess import (
    call_rest_api,
    download_synapse,
    download_wget,
    get_image_tcia_restapi_url,
    get_image_metadata_tcia_restapi_url,
    zip_dicom_to_nifti,
)


BASE_PATH = "2D-3D-Reconstruction-Datasets"
CTPELVIC1K_PATH = Path(BASE_PATH) / "ctpelvic1k"

SEG_URL_PATH = "external/XrayTo3DPreprocess/workflow/ctpelvic1k/download_links/segmentation_metadata.csv"
CLINIC_RAW_PATH = (
    "external/XrayTo3DPreprocess/workflow/ctpelvic1k/download_links/clinic_raw.csv"
)
KITS_RAW_PATH = (
    "external/XrayTo3DPreprocess/workflow/ctpelvic1k/download_links/kits_raw.csv"
)
COLONOG_RAW_PATH = (
    "external/XrayTo3DPreprocess/workflow/ctpelvic1k/CTColonography_MetaData.csv"
)


def get_series_ID(patient_id: str):
    df = pd.read_csv(COLONOG_RAW_PATH)
    subdf = df[df["Patient Id"].str.contains(patient_id)]
    subdf = subdf[~subdf["Series Description"].str.contains("Topo|topo")][
        "Series UID"
    ]  # avoid topograms
    return subdf.to_numpy()


def get_segmentation_series_number(filename: str):
    """dataset2_1.3.6.1.4.1.9328.50.4.0001_3_325_mask_4label.nii.gz -> 3"""
    return int(filename.split("_")[2])


def get_value_from_tcia_json_metadata(metadata_json, key):
    return metadata_json[0][key]


def makedirs(base_dir: Path):
    """
    create directories and subdirectories to store zips,raw and processed image-segmentation pairs
    Three subdirectories are created for each subdataset to store zips, raw and processed files.
    there are separate directories for storing images and segmentations.

    Args:
        base_dir (Path): base directory where the dataset is to be stored.
    """

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    datasets = [
        "ABDOMEN",
        "CERVIX",
        "CLINIC",
        "CLINIC-METAL",
        "COLONOG",
        "KITS19",
        "MSD-T10",
    ]

    for dir in datasets:
        mkdir_or_exist(str(base_dir / "zips" / dir))
        mkdir_or_exist(str(base_dir / "zips" / dir / "img"))  # store ct zips
        mkdir_or_exist(str(base_dir / "zips" / dir / "seg"))  # store seg zips

        mkdir_or_exist(str(base_dir / "raw" / dir))
        # store extracted ct scans
        mkdir_or_exist(str(base_dir / "raw" / dir / "img"))
        mkdir_or_exist(str(base_dir / "raw" / dir / "seg"))  # store extracted seg
        mkdir_or_exist(str(base_dir / "raw" / dir / "BIDS"))  # store extracted seg


def download_segmentations():
    """download and extract segmentations from urls stored in file. each segmentations are extracted to segmentation directory of the  raw subdataset dir."""
    df = pd.read_csv(SEG_URL_PATH).to_numpy()

    DOWNLOAD_ZIP_DIR = CTPELVIC1K_PATH / "zips"
    SEG_OUT_DIR = CTPELVIC1K_PATH / "raw"

    for dataset_name, seg_url, md5, zip_filename in df:
        zip_filename = dataset_name + "_" + zip_filename
        download_file_path = DOWNLOAD_ZIP_DIR / dataset_name / "seg"
        zip_extract_dir = str(SEG_OUT_DIR / dataset_name / "seg")

        download_wget(seg_url, zip_filename, download_file_path, md5)

        extractall(str(download_file_path / zip_filename), output_dir=zip_extract_dir)


def download_msdt10_raw():
    """download images from MSD-T10 dataset from google drive."""

    url = "https://drive.google.com/uc?id=1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y"
    DOWNLOAD_ZIP_DIR = CTPELVIC1K_PATH / "zips" / "MSD-T10" / "img"
    RAW_OUT_DIR = CTPELVIC1K_PATH / "raw" / "MSD-T10" / "img"

    md5_hash = "bad7a188931dc2f6acf72b08eb6202d0"
    zip_filename = "Task10_Colon.tar"
    download_file_path = str(DOWNLOAD_ZIP_DIR / zip_filename)
    download_url(url, download_file_path, hash_val=md5_hash)
    extractall(download_file_path, output_dir=RAW_OUT_DIR)


def download_clinic_raw():
    """download images from CLINIC and CLINIC_METAL dataset. extract the zip into
    each subdirectory of the corresponding ct scan dir. MD5 hash for zip file available.

    known issues: if the raw zip has already been downloaded and extracted,
    this code will try and verify the zip file twice. Verifying can take >1 mins for large files.
    """
    df = pd.read_csv(CLINIC_RAW_PATH).to_numpy()
    ZIP_DOWNLOAD_DIR = CTPELVIC1K_PATH / "zips"
    mkdir_or_exist(ZIP_DOWNLOAD_DIR)
    RAW_OUT_DIR = CTPELVIC1K_PATH / "raw"

    for dataset_name, seg_url, md5, zip_filename in df:
        zip_filename = dataset_name + "_" + zip_filename
        zip_extract_dir = str(RAW_OUT_DIR / dataset_name / "img")
        download_file_path = ZIP_DOWNLOAD_DIR / dataset_name / "img"
        download_wget(seg_url, zip_filename, download_file_path, md5)

        extractall(str(download_file_path / zip_filename), output_dir=zip_extract_dir)


def download_cervix_raw():
    """download images from cervix dataset from synapse.org using synapseclient"""
    DOWNLOAD_ZIP_DIR = CTPELVIC1K_PATH / "zips" / "CERVIX" / "img"
    zip_filename = "CervixRawData.zip"
    md5_hash = "4ef26f4bc567dbb3041e955d18cd36a5"
    cervix_synapse_id = "syn3546986"
    download_synapse(cervix_synapse_id, zip_filename, DOWNLOAD_ZIP_DIR, md5_hash)

    RAW_OUT_DIR = CTPELVIC1K_PATH / "raw" / "CERVIX" / "img"
    extractall(str(DOWNLOAD_ZIP_DIR / zip_filename), str(RAW_OUT_DIR))


def download_abdomen_raw():
    """download images from abdomen dataset from synapse.org using synapseclient"""
    DOWNLOAD_ZIP_DIR = CTPELVIC1K_PATH / "zips" / "ABDOMEN" / "img"
    zip_filename = "RawData.zip"
    md5_hash = "e8e3fc9604eadc34c47067e2332f8ea1"
    cervix_synapse_id = "syn3379050"
    download_synapse(cervix_synapse_id, zip_filename, DOWNLOAD_ZIP_DIR, md5_hash)

    RAW_OUT_DIR = CTPELVIC1K_PATH / "raw" / "ABDOMEN" / "img"
    extractall(str(DOWNLOAD_ZIP_DIR / zip_filename), str(RAW_OUT_DIR))


def download_kits_raw():
    """download images from KITS19 dataset. URL for each images is stored in csv. MD5 hash is not available."""
    df = pd.read_csv(KITS_RAW_PATH).to_numpy()
    RAW_OUT_DIR = CTPELVIC1K_PATH / "raw" / "KITS19" / "img"
    mkdir_or_exist(RAW_OUT_DIR)
    for case_id, url, md5, filename in tqdm(df):
        # md5 does not exist for kits19 raw images
        download_wget(url, filename, output_dir=RAW_OUT_DIR)


def download_colonog_raw():
    """download images from COLONOG dataset."""
    RAW_OUT_DIR = CTPELVIC1K_PATH / "raw" / "COLONOG" / "img"
    mkdir_or_exist(RAW_OUT_DIR)
    seg_csv = "tools/convert_datasets/ctpelvic1k/colonog_seg.csv"
    df = pd.read_csv(seg_csv)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = str(row["Patient Id"])
        segmentation_filename = str(row["segmentation-filename"])
        for sid in get_series_ID(patient_id):
            url = get_image_metadata_tcia_restapi_url(sid)
            metadata_json = call_rest_api(url)
            series_number = int(
                float(get_value_from_tcia_json_metadata(metadata_json, "Series Number"))
            )
            if series_number == get_segmentation_series_number(segmentation_filename):
                with tempfile.TemporaryDirectory() as defaultTempDir:
                    print(defaultTempDir)

                    image_url = get_image_tcia_restapi_url(sid)
                    dicom_filepath = f"{defaultTempDir}/{patient_id}.zip"
                    download_wget(image_url, dicom_filepath, ".")
                    zip_dicom_to_nifti(dicom_filepath, output_dir=str(RAW_OUT_DIR))

                    # remove the temporary downloaded DICOM
                    shutil.rmtree(defaultTempDir)


def main():
    makedirs(CTPELVIC1K_PATH)
    download_abdomen_raw()

    download_kits_raw()
    download_cervix_raw()

    # download_clinic_raw()
    # download_segmentations()
    # download_msdt10_raw()


if __name__ == "__main__":
    main()
