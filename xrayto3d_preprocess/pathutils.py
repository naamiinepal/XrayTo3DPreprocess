"""path utils"""
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def mkdir_or_exist(out_dir):
    """wrap os.makedirs"""
    os.makedirs(out_dir, exist_ok=True)


def get_nifti_stem(path) -> str:
    """
    '/home/user/image.nii.gz' -> 'image'
    1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235.nii.gz ->1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235
    """

    def _get_stem(path_string) -> str:
        name_subparts = Path(path_string).name.split(".")
        return ".".join(name_subparts[:-2])  # get rid of [nii, gz]

    return _get_stem(path)


def get_stem(path):
    """wrap Path.stem"""
    return Path(path).stem


def get_file_format_suffix(path) -> str:
    """
    '/home/user/image.nii.gz' -> 'nii.gz'
    """

    def _get_stem(path_string) -> str:
        return ".".join(Path(path_string).name.split(".")[1:])

    return _get_stem(path)


def dest_path(source_path, dest_dir, dest_format=None):
    """
    '/home/user/image.nii.gz' , /home/dest -> /home/dest/image.nii.gz
    """
    filename = get_nifti_stem(source_path)
    if dest_format is None:
        dest_format = get_file_format_suffix(source_path)
    return Path(dest_dir) / f"{filename}.{dest_format}"


def copy_subjects_to_individual_dir(
    subjects_csv,
    src_img_basepath,
    src_seg_basepath,
    dest_basepath,
    dest_img_file_pattern="{subject_id}_img.nii.gz",
    dest_seg_file_pattern="{subject_id}_seg.nii.gz",
    subject_id_header_name="subject-id",
    image_filename_header_name="image-filename",
    segmentation_filename_header_name="segmentation-filename",
):
    """copy source and segmentation into its own subject directory, rename to consistent format subject_suffix
    default csv column header: [subject-id, image-filename, segmentation-filename]
    """
    if isinstance(src_img_basepath, str):
        src_img_basepath = Path(src_img_basepath)

    if isinstance(src_seg_basepath, str):
        src_seg_basepath = Path(src_seg_basepath)

    if isinstance(dest_basepath, str):
        dest_basepath = Path(dest_basepath)

    df = pd.read_csv(subjects_csv)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        subject_id, img_filename, seg_filename = (
            row[subject_id_header_name],
            row[image_filename_header_name],
            row[segmentation_filename_header_name],
        )
        subject_path = dest_basepath / str(subject_id)

        src_path = src_img_basepath / img_filename
        target_path = subject_path / dest_img_file_pattern.format(subject_id=subject_id)

        target_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(src_path, target_path)

        src_path = src_seg_basepath / seg_filename
        target_path = subject_path / dest_seg_file_pattern.format(subject_id=subject_id)

        # directory where AP, LAT and segmentations are stored
        derivatives_path = subject_path / "derivatives"
        derivatives_path.mkdir(exist_ok=True, parents=True)

        target_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(src_path, target_path)


if __name__ == "__main__":
    print(get_file_format_suffix("/home/user/image.nii.gz"))
    print(dest_path("/home/user/image.nii.gz", "/home/dest", "png"))
