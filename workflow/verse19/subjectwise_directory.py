"""create subjectwise subdirectory containing ct-segmentation for each patient"""
import shutil
from pathlib import Path

from tqdm import tqdm
from xrayto3d_preprocess import get_verse_subject_id

if __name__ == "__main__":
    base_dirs = [
        "2D-3D-Reconstruction-Datasets/verse19/raw/dataset-verse19validation",
        "2D-3D-Reconstruction-Datasets/verse19/raw/dataset-verse19training",
        "2D-3D-Reconstruction-Datasets/verse19/raw/dataset-verse19test",
    ]

    dest_dir = "2D-3D-Reconstruction-Datasets/verse19/subjectwise"

    # create destination directory if it does not exist
    Path(dest_dir).mkdir(exist_ok=True, parents=True)

    for base_dir in base_dirs:
        # find all files with a given suffix in the base directory
        image_files = sorted(Path(base_dir).rglob("*_ct.nii.gz"))
        seg_files = sorted(Path(base_dir).rglob("*_seg-vert_msk.nii.gz"))
        json_files = sorted(Path(base_dir).rglob("*_ctd.json"))

        for image_file, seg_file, json_file in tqdm(
            zip(image_files, seg_files, json_files), total=len(seg_files)
        ):
            # extract subject id from file path
            # print(get_nifti_stem(image_file),get_verse_subject_id(image_file))

            subject_id = get_verse_subject_id(image_file)
            print(subject_id)

            subject_dir = Path(dest_dir) / subject_id
            # create the subject directory if it does not exist
            subject_dir.mkdir(exist_ok=True, parents=True)

            # copy the image, seg and json files to destination directory
            shutil.copy(image_file, subject_dir / "ct.nii.gz")
            shutil.copy(seg_file, subject_dir / "seg.nii.gz")
            shutil.copy(json_file, subject_dir/'ctd.json')
