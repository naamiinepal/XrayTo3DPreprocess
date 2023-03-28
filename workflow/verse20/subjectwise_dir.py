from pathlib import Path
from xrayto3d_preprocess import get_nifti_stem, get_verse_subject_id
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    base_dir_template = "2D-3D-Reconstruction-Datasets/verse20/raw/{subset}"

    dest_dir = "2D-3D-Reconstruction-Datasets/verse20/subjectwise"

    # create destination directory if it does not exist
    Path(dest_dir).mkdir(exist_ok=True, parents=True)

    # find all files with a given suffix in the base directory

    image_files = []
    seg_files = []
    json_files = []
    for dir in ["dataset-01training", "dataset-02validation", "dataset-03test"]:
        base_dir = base_dir_template.format(subset=dir)

        image_files.extend(sorted(Path(base_dir).rglob("*_ct.nii.gz")))

        seg_files.extend(sorted(Path(base_dir).rglob("*_seg-vert_msk.nii.gz")))

        json_files.extend(sorted(Path(base_dir).rglob("*_ctd.json")))

    print(len(image_files), len(seg_files), len(json_files))

    for image_file, seg_file, json_file in tqdm(
        zip(image_files, seg_files, json_files), total=len(seg_files)
    ):
        # extract subject id from file path
        print(get_nifti_stem(image_file), get_verse_subject_id(image_file))

        subject_id = get_verse_subject_id(image_file)

        subject_dir = Path(dest_dir) / subject_id
        # create the subject directory if it does not exist
        subject_dir.mkdir(exist_ok=True, parents=True)

        # copy the image, seg and json files to destination directory
        shutil.copy(image_file, subject_dir)
        shutil.copy(seg_file, subject_dir)
        shutil.copy(json_file, subject_dir)
