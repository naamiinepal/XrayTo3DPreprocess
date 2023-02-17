import os
from pathlib import Path
import pandas as pd
from xrayto3d_preprocess import mkdir_or_exist
from xrayto3d_preprocess import download_gdown
from monai.apps.utils import extractall

BASE_PATH = '2D-3D-Reconstruction-Datasets'
CTSPINE1K_PATH = Path(BASE_PATH)/'ctspine1k'

RAW_COLONOG_URL_PATH = 'external/XrayTo3DPreprocess/workflow/ctspine1k/colonog_Path.csv'
METADATA_PATH = 'external/XrayTo3DPreprocess/workflow/ctspine1k/CTColonography_MetaData.csv'


def makedirs(base_dir: Path):
    """
    create directories and subdirectories to store zips,raw and processed image-segmentation pairs
    Three subdirectories are created for each subdataset to store zips, raw and processed files. there are separate directories for
    storing images and segmentations.

    Args:
        base_dir (Path): base directory where the dataset is to be stored.
    """

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    datasets = [
        'COLONOG',
        'COVID-19',
        'HNSCC-3DCT-RT']

    for dir in datasets:
        mkdir_or_exist(str(base_dir/'zips'/dir))
        mkdir_or_exist(str(base_dir/'zips'/dir/'img'))  # store ct zips
        mkdir_or_exist(str(base_dir/'zips'/dir/'seg'))  # store seg zips

        mkdir_or_exist(str(base_dir/'raw'/dir))
        # store extracted ct scans
        mkdir_or_exist(str(base_dir/'raw'/dir/'img'))
        mkdir_or_exist(str(base_dir/'raw'/dir/'seg'))  # store extracted seg
        mkdir_or_exist(str(base_dir/'raw'/dir/'BIDS'))  # store extracted seg

def download_colonog_raw():
    """donwload images from COLONOG dataset"""
    pass

def download_segmentation():
    """download and extract segmentations from urls stored in file. each segmentations are extracted to segmentation dir of the raw subdataset dir"""

    file_id = '19jUYA7qy19YLLp_12T8UbZTCGO-h4wN_'
    
    DOWNLOAD_ZIP_DIR = CTSPINE1K_PATH/'zips'/'seg'
    RAW_OUT_DIR = CTSPINE1K_PATH/'raw'/'seg'
    zip_filename = 'ctspine1k.zip'
    # download_file_path = str(DOWNLOAD_ZIP_DIR/zip_filename)
    mkdir_or_exist(DOWNLOAD_ZIP_DIR)
    mkdir_or_exist(RAW_OUT_DIR)
    download_gdown(file_id, output_dir=DOWNLOAD_ZIP_DIR,filename=zip_filename)
    extractall(DOWNLOAD_ZIP_DIR/zip_filename, output_dir=RAW_OUT_DIR)

def parse_row(row):
    """each row has three parts separated by slash(/). patient_id/modalities etc/last 5characters of Series UID"""
    patient_id, _, end = row.split('/')
    partial_series_UID = end.split('-')[-1]
    return patient_id, partial_series_UID


def read_csv():
    """
    read patient_id csv and metadata csv. Find the Series UID corresponding to patient_id in the metadata csv and return (patient_id,Series_UID)


    Returns:
        List: each item in the list consists of tuple(patient_id,Series_UID)
    """
    df = pd.read_csv(METADATA_PATH)
    i = 0
    num_samples_to_process = 5
    pivot_row = []
    with open(RAW_COLONOG_URL_PATH) as f:
        while(line := f.readline().rstrip()):
            row = parse_row(line)
            patient_row = df.loc[(df['Patient Id'] == row[0]) & (
                df['Series UID'].str.endswith(str(row[1])))]
            assert len(patient_row) == 1, 'Ambiguous row'
            i += 1
            print(i, row[0], patient_row['Series UID'].values[0])
            pivot_row.append((row[0], patient_row['Series UID'].values[0]))

            if i >= num_samples_to_process:
                break
    return pivot_row


def main():
    makedirs(CTSPINE1K_PATH)
    download_segmentation()

if __name__ == '__main__':
    main()
