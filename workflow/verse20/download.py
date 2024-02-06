"""download verse20 dataset"""
from pathlib import Path

import pandas as pd
from monai.apps.utils import extractall
from xrayto3d_preprocess import download_wget, mkdir_or_exist

BASE_PATH = "2D-3D-Reconstruction-Datasets"
VERSE20_PATH = Path(BASE_PATH) / "verse20"
verse20_RAW_PATH = (
    "workflow/verse20/download_links/verse20_raw.csv"
)


def makedirs(base_dir: Path):
    """create directories and subdirectories to store
    zip, raw and processed image-segmentation pairs."""
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    mkdir_or_exist(base_dir / "zips")
    mkdir_or_exist(base_dir / "raw")
    mkdir_or_exist(base_dir / "subjectwise")


def download_verse20_raw():
    """download images from verse20 dataset.
    extract zip and verify MD5"""
    df = pd.read_csv(verse20_RAW_PATH).to_numpy()
    for _, url, md5, zip_filename in df:
        download_dir = VERSE20_PATH / "zips"
        download_wget(url, zip_filename, download_dir, md5)
        extract_dest = VERSE20_PATH / "raw" 
        extractall(download_dir / zip_filename, output_dir=extract_dest)


def main():
    """entry point"""
    makedirs(VERSE20_PATH)
    download_verse20_raw()


if __name__ == "__main__":
    main()
