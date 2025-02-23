"""Download verse19 dataset (Supports Windows & Linux)"""
import os
from pathlib import Path
import platform
import pandas as pd
from monai.apps.utils import extractall
from xrayto3d_preprocess import download_wget, mkdir_or_exist

BASE_PATH = Path("2D-3D-Reconstruction-Datasets")
verse19_PATH = BASE_PATH / "verse19"
verse19_RAW_PATH = Path("workflow") / "verse19" / "download_links" / "verse19_raw.csv"

print("verse19_PATH:", verse19_PATH)
print("verse19_RAW_PATH:", verse19_RAW_PATH)


def makedirs(base_dir: Path):
    """Create directories and subdirectories to store
    zip, raw, and processed image-segmentation pairs."""
    mkdir_or_exist(base_dir / "zips")
    mkdir_or_exist(base_dir / "raw")
    mkdir_or_exist(base_dir / "subjectwise")


def download_verse19_raw():
    """Download images from the verse19 dataset.
    Extract zip and verify MD5 checksum."""
    df = pd.read_csv(verse19_RAW_PATH).to_numpy()
    
    for _, url, md5, zip_filename in df:
        download_dir = verse19_PATH / "zips"
        zip_path = download_dir / zip_filename

        if platform.system() == "Windows":
            # Use curl or requests instead of wget
            print(f"Downloading {zip_filename} using curl (Windows)...")
            os.system(f'curl  -o "{zip_path}" {url}')
        else:
            # Use wget for Linux/macOS
            download_wget(url, zip_filename, download_dir, md5)
        print("here i am ")
        extract_dest = verse19_PATH / "raw"
        extractall(zip_path, output_dir=extract_dest)


def main():
    """Entry point."""
    makedirs(verse19_PATH)
    download_verse19_raw()


if __name__ == "__main__":
    main()
