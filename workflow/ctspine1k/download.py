"""download ctspine1k dataset"""
from pathlib import Path

import pandas as pd
from monai.apps.utils import extractall
from xrayto3d_preprocess import download_gdown, mkdir_or_exist

BASE_PATH = "2D-3D-Reconstruction-Datasets"
CTSPINE1K_PATH = Path(BASE_PATH) / "ctspine1k"

RAW_COLONOG_URL_PATH = "external/XrayTo3DPreprocess/workflow/ctspine1k/colonog_Path.csv"
METADATA_PATH = (
    "external/XrayTo3DPreprocess/workflow/ctspine1k/CTColonography_MetaData.csv"
)


def makedirs(base_dir: Path):
    """
    create directories and subdirectories to store zips,raw and processed image-segmentation pairs
    Three subdirectories are created for each subdataset to store zips, raw and processed files.
    there are separate directories for
    storing images and segmentations.

    Args:
        base_dir (Path): base directory where the dataset is to be stored.
    """

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    datasets = ["COLONOG", "COVID-19", "HNSCC-3DCT-RT"]

    for d in datasets:
        mkdir_or_exist(base_dir / "zips" / d)
        mkdir_or_exist(base_dir / "zips" / d / "img")  # store ct zips
        mkdir_or_exist(base_dir / "zips" / d / "seg")  # store seg zips

        mkdir_or_exist(base_dir / "raw" / d)
        # store extracted ct scans
        mkdir_or_exist(base_dir / "raw" / d / "img")
        mkdir_or_exist(base_dir / "raw" / d / "seg")  # store extracted seg
        mkdir_or_exist(base_dir / "raw" / d / "BIDS")  # store extracted seg


def download_colonog_raw():
    """donwload images from COLONOG dataset"""
    pass


def download_segmentation():
    """download and extract segmentations from urls stored in file.
    each segmentations are extracted to segmentation dir of the raw subdataset dir"""

    file_id = "19jUYA7qy19YLLp_12T8UbZTCGO-h4wN_"

    zip_dest = CTSPINE1K_PATH / "zips" / "seg"
    extract_dest = CTSPINE1K_PATH / "raw" / "seg"
    zip_filename = "ctspine1k.zip"
    # download_file_path = str(DOWNLOAD_ZIP_DIR/zip_filename)
    mkdir_or_exist(zip_dest)
    mkdir_or_exist(extract_dest)
    download_gdown(file_id, output_dir=zip_dest, filename=zip_filename)
    extractall(zip_dest / zip_filename, output_dir=extract_dest)


def parse_row(row):
    """each row has three parts separated by slash(/).
    patient_id/modalities etc/last 5characters of Series UID
    """
    patient_id, _, end = row.split("/")
    partial_series_uid = end.split("-")[-1]
    return patient_id, partial_series_uid


def read_csv():
    """
    read patient_id csv and metadata csv. Find the Series UID corresponding to
    patient_id in the metadata csv and return (patient_id,Series_UID)


    Returns:
        List: each item in the list consists of tuple(patient_id,Series_UID)
    """
    df = pd.read_csv(METADATA_PATH)
    i = 0
    num_samples_to_process = 5
    pivot_row = []
    with open(RAW_COLONOG_URL_PATH) as f:
        while line := f.readline().rstrip():
            row = parse_row(line)
            patient_row = df.loc[
                (df["Patient Id"] == row[0])
                & (df["Series UID"].str.endswith(str(row[1])))
            ]
            assert len(patient_row) == 1, "Ambiguous row"
            i += 1
            print(i, row[0], patient_row["Series UID"].values[0])
            pivot_row.append((row[0], patient_row["Series UID"].values[0]))

            if i >= num_samples_to_process:  # why?
                break
    return pivot_row


def main():
    """entry point"""
    makedirs(CTSPINE1K_PATH)
    download_segmentation()


if __name__ == "__main__":
    main()
