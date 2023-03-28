"""download Ribfrac dataset"""
from pathlib import Path

import pandas as pd
from monai.apps.utils import extractall
from xrayto3d_preprocess import download_wget, mkdir_or_exist

BASE_PATH = "2D-3D-Reconstruction-Datasets"
RIBFRAC_PATH = Path(BASE_PATH) / "ribfrac"
RIBFRAC_RAW_PATH = (
    "external/XrayTo3DPreprocess/workflow/ribfrac/download_links/ribfrac_raw.csv"
)
RIBFRAC_SEG_PATH = (
    "external/XrayTo3DPreprocess/workflow/ribfrac/download_links/ribfrac_seg.csv"
)


def makedirs(base_dir: Path):
    """create directories and subdirectories to store
    zip, raw and processed image-segmentation pairs."""
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    mkdir_or_exist(base_dir / "zips")
    mkdir_or_exist(base_dir / "raw")
    mkdir_or_exist(base_dir / "raw" / "img")
    mkdir_or_exist(base_dir / "raw" / "seg")
    mkdir_or_exist(base_dir / "subjectwise")


def download_ribfrac_raw():
    """download images from Ribfrac dataset.
    extract zip and verify MD5"""
    df = pd.read_csv(RIBFRAC_RAW_PATH).to_numpy()
    for _, url, md5, zip_filename in df:
        download_dir = RIBFRAC_PATH / "zips"
        download_wget(url, zip_filename, download_dir, md5)
        extract_dest = RIBFRAC_PATH / "raw" / "img"
        extractall(download_dir / zip_filename, output_dir=extract_dest)


def download_segmentations():
    """download segmentations from Ribfrac dataset"""

    df = pd.read_csv(RIBFRAC_SEG_PATH).to_numpy()
    for _, url, md5, zip_filename in df:
        download_dir = RIBFRAC_PATH / "zips"
        download_wget(url, zip_filename, download_dir, md5)
        extract_dest = RIBFRAC_PATH / "raw" / "seg"
        extractall(download_dir / zip_filename, output_dir=extract_dest)


def main():
    """entry point"""
    makedirs(RIBFRAC_PATH)
    download_ribfrac_raw()
    # download_segmentations()


if __name__ == "__main__":
    main()
