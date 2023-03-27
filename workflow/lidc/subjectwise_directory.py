import shutil
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    base_dir = "2D-3D-Reconstruction-Datasets/LIDC-test/raw/subset0"
    dest_dir = "2D-3D-Reconstruction-Datasets/LIDC-test/subjectwise"

    # find all files with a given suffix in the base dir
    image_dir = Path(base_dir) / "data_preprocessed"
    image_files = sorted(image_dir.rglob("*.nii.gz"))

    seg_dir = Path(base_dir) / "vertebrae_segmentation"
    seg_files = sorted(seg_dir.rglob("*_seg.nii.gz"))

    json_dir = Path(base_dir) / "vertebrae_localization"
    json_files = sorted(json_dir.rglob("*_ctd.json"))

    print(
        f"found {len(image_files)} image {len(seg_files)} seg {len(json_files)} jsons "
    )

    subject_list = "workflow/data/lidc/subjects/subset0.lst"
    df = pd.read_csv(subject_list, header=None)

    image_name = base_dir + "/data_preprocessed" + "/{lidc_id}.nii.gz"
    seg_name = base_dir + "/vertebrae_segmentation" + "/{lidc_id}_seg.nii.gz"
    json_name = base_dir + "/vertebrae_localization" + "/{lidc_id}_ctd.json"

    def create_subjectwise_dir(subject_list):
        for subject_id, lidc_id in subject_list:
            subject_dir = f"{dest_dir}/LIDC-{subject_id:04d}"
            print(subject_dir)
            # create the subject dir if it does not exist
            Path(subject_dir).mkdir(exist_ok=True, parents=True)

            # copy image, seg and json files to destination dir
            shutil.copy(
                image_name.format(lidc_id=lidc_id),
                Path(subject_dir) / f"LIDC-{subject_id:04d}.nii.gz",
            )
            shutil.copy(
                seg_name.format(lidc_id=lidc_id),
                Path(subject_dir) / f"LIDC-{subject_id:04d}_seg.nii.gz",
            )
            shutil.copy(
                json_name.format(lidc_id=lidc_id),
                Path(subject_dir) / f"LIDC-{subject_id:04d}_ctd.json",
            )

    create_subjectwise_dir(df.to_numpy())
