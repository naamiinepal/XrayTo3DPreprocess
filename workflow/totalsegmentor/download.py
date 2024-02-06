"""download totalsegmentor"""
from pathlib import Path
import shutil
import tempfile
from tqdm import tqdm
import pandas as pd
from monai.apps.utils import extractall, download_url

from xrayto3d_preprocess import (
    call_rest_api,
    download_synapse,
    download_wget,
    get_image_tcia_restapi_url,
    get_image_metadata_tcia_restapi_url,
    zip_dicom_to_nifti,
    mkdir_or_exist,
)


BASE_PATH = "2D-3D-Reconstruction-Datasets"
TOTALSEGMENTOR_PATH = Path(BASE_PATH) / "totalsegmentor"

SEG_URL_PATH = "external/XrayTo3DPreprocess/workflow/totalsegmentor/download_links/segmentation_metadata.csv"
CLINIC_RAW_PATH = (
    "external/XrayTo3DPreprocess/workflow/totalsegmentor/download_links/clinic_raw.csv"
)
KITS_RAW_PATH = (
    "external/XrayTo3DPreprocess/workflow/totalsegmentor/download_links/kits_raw.csv"
)
COLONOG_RAW_PATH = (
    "external/XrayTo3DPreprocess/workflow/totalsegmentor/CTColonography_MetaData.csv"
)


def get_series_id(patient_id: str):
    """find series id"""
    df = pd.read_csv(COLONOG_RAW_PATH)
    subdf = df[df["Patient Id"].str.contains(patient_id)]
    subdf = subdf[~subdf["Series Description"].str.contains("Topo|topo")][
        "Series UID"
    ]  # avoid topograms
    return subdf.to_numpy()


def get_segmentation_series_number(filename: str):
    """dataset2_1.3.6.1.4.1.9328.50.4.0001_3_325_mask_4label.nii.gz -> 3
    There can be multiple series for same id.choose one corresponding to the
    available segmentation"""
    return int(filename.split("_")[2])


def get_value_from_tcia_json_metadata(metadata_json, key):
    """return value from metadata"""
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

    for d in datasets:
        mkdir_or_exist(base_dir / "zips" / d)
        mkdir_or_exist(base_dir / "zips" / d / "img")  # store ct zips
        mkdir_or_exist(base_dir / "zips" / d / "seg")  # store seg zips

        mkdir_or_exist(base_dir / "raw" / d)
        # store extracted ct scans
        mkdir_or_exist(base_dir / "raw" / d / "img")
        mkdir_or_exist(base_dir / "raw" / d / "seg")  # store extracted seg
        mkdir_or_exist(base_dir / "raw" / d / "BIDS")  # store extracted seg


def download_segmentations():
    """download and extract segmentations from urls stored in file.
    each segmentations are extracted to segmentation dir of the raw subdir."""
    df = pd.read_csv(SEG_URL_PATH).to_numpy()

    DOWNLOAD_ZIP_DIR = TOTALSEGMENTOR_PATH / "zips"
    SEG_OUT_DIR = TOTALSEGMENTOR_PATH / "raw"

    for dataset_name, seg_url, md5, zip_filename in df:
        zip_filename = dataset_name + "_" + zip_filename
        zip_dest = DOWNLOAD_ZIP_DIR / dataset_name / "seg"
        extract_dest = str(SEG_OUT_DIR / dataset_name / "seg")

        download_wget(seg_url, zip_filename, zip_dest, md5)

        extractall(zip_dest / zip_filename, output_dir=extract_dest)


def download_msdt10_raw():
    """download images from MSD-T10 dataset from google drive."""

    url = "https://drive.google.com/uc?id=1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y"
    zip_dest = TOTALSEGMENTOR_PATH / "zips" / "MSD-T10" / "img"
    extract_dest = TOTALSEGMENTOR_PATH / "raw" / "MSD-T10" / "img"

    md5_hash = "bad7a188931dc2f6acf72b08eb6202d0"
    zip_filename = "Task10_Colon.tar"
    download_file_path = str(zip_dest / zip_filename)
    download_url(url, download_file_path, hash_val=md5_hash)
    extractall(download_file_path, output_dir=extract_dest)


def download_clinic_raw():
    """download images from CLINIC and CLINIC_METAL dataset. extract the zip into
    each subdirectory of the corresponding ct scan dir. MD5 hash for zip file available.

    known issues: if the raw zip has already been downloaded and extracted,
    this code will try and verify the zip file twice. Verifying can take >1 mins for large files.
    """
    df = pd.read_csv(CLINIC_RAW_PATH).to_numpy()
    zip_dest = TOTALSEGMENTOR_PATH / "zips"
    mkdir_or_exist(zip_dest)
    extract_dest = TOTALSEGMENTOR_PATH / "raw"

    for dataset_name, url, md5, zip_filename in df:
        zip_filename = dataset_name + "_" + zip_filename
        download_file_path = zip_dest / dataset_name / "img"
        download_wget(url, zip_filename, download_file_path, md5)

        extract_dest = extract_dest / dataset_name / "img"
        extractall(download_file_path / zip_filename, output_dir=extract_dest)


def download_cervix_raw():
    """download images from cervix dataset from synapse.org using synapseclient"""
    zip_filename = "CervixRawData.zip"
    md5_hash = "4ef26f4bc567dbb3041e955d18cd36a5"
    cervix_synapse_id = "syn3546986"
    zip_dest = TOTALSEGMENTOR_PATH / "zips" / "CERVIX" / "img"
    download_synapse(cervix_synapse_id, zip_filename, zip_dest, md5_hash)

    extract_dest = TOTALSEGMENTOR_PATH / "raw" / "CERVIX" / "img"
    extractall(zip_dest / zip_filename, extract_dest)


def download_abdomen_raw():
    """download images from abdomen dataset from synapse.org using synapseclient"""
    zip_filename = "RawData.zip"
    md5_hash = "e8e3fc9604eadc34c47067e2332f8ea1"
    cervix_synapse_id = "syn3379050"
    zip_dest = TOTALSEGMENTOR_PATH / "zips" / "ABDOMEN" / "img"
    download_synapse(cervix_synapse_id, zip_filename, zip_dest, md5_hash)

    extract_dest = TOTALSEGMENTOR_PATH / "raw" / "ABDOMEN" / "img"
    extractall(zip_dest / zip_filename, extract_dest)


def download_kits_raw():
    """download images from KITS19 dataset. URL for each images is stored in csv.
    MD5 hash is not available."""
    df = pd.read_csv(KITS_RAW_PATH).to_numpy()
    extract_dest = TOTALSEGMENTOR_PATH / "raw" / "KITS19" / "img"
    mkdir_or_exist(extract_dest)
    for _, url, _, filename in tqdm(df):
        # md5 does not exist for kits19 raw images
        download_wget(url, filename, output_dir=extract_dest)


def download_colonog_raw():
    """download images from COLONOG dataset."""
    extract_dest = TOTALSEGMENTOR_PATH / "raw" / "COLONOG" / "img"
    mkdir_or_exist(extract_dest)
    seg_csv = "tools/convert_datasets/totalsegmentor/colonog_seg.csv"
    df = pd.read_csv(seg_csv)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = str(row["Patient Id"])
        segmentation_filename = str(row["segmentation-filename"])
        for sid in get_series_id(patient_id):
            url = get_image_metadata_tcia_restapi_url(sid)
            metadata_json = call_rest_api(url)
            series_number = int(
                float(get_value_from_tcia_json_metadata(metadata_json, "Series Number"))
            )
            if series_number == get_segmentation_series_number(segmentation_filename):
                with tempfile.TemporaryDirectory() as default_temp_dir:
                    print(default_temp_dir)

                    image_url = get_image_tcia_restapi_url(sid)
                    dicom_filepath = f"{default_temp_dir}/{patient_id}.zip"
                    download_wget(image_url, dicom_filepath, ".")
                    zip_dicom_to_nifti(dicom_filepath, output_dir=str(extract_dest))

                    # remove the temporary downloaded DICOM
                    shutil.rmtree(default_temp_dir)


def main():
    """entry point"""
    makedirs(TOTALSEGMENTOR_PATH)
    download_abdomen_raw()

    download_kits_raw()
    download_cervix_raw()

    # download_clinic_raw()
    # download_segmentations()
    # download_msdt10_raw()


if __name__ == "__main__":
    main()
