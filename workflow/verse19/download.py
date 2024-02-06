"""download verse19 dataset"""
from pathlib import Path

import pandas as pd
from monai.apps.utils import extractall
from xrayto3d_preprocess import download_wget, mkdir_or_exist

BASE_PATH = "2D-3D-Reconstruction-Datasets"
verse19_PATH = Path(BASE_PATH) / "verse19"
verse19_RAW_PATH = (
    "workflow/verse19/download_links/verse19_raw.csv"
)


def makedirs(base_dir: Path):
    """create directories and subdirectories to store
    zip, raw and processed image-segmentation pairs."""
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    mkdir_or_exist(base_dir / "zips")
    mkdir_or_exist(base_dir / "raw")
    mkdir_or_exist(base_dir / "subjectwise")


def download_verse19_raw():
    """download images from verse19 dataset.
    extract zip and verify MD5"""
    df = pd.read_csv(verse19_RAW_PATH).to_numpy()
    for _, url, md5, zip_filename in df:
        download_dir = verse19_PATH / "zips"
        download_wget(url, zip_filename, download_dir, md5)
        extract_dest = verse19_PATH / "raw" 
        extractall(download_dir / zip_filename, output_dir=extract_dest)


def main():
    """entry point"""
    makedirs(verse19_PATH)
    download_verse19_raw()


if __name__ == "__main__":
    main()
