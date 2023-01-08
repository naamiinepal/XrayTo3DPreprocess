from pathlib import Path
from xrayto3d_preprocess import get_nifti_stem, get_verse_subject_id
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    base_dir = 'VERSE2019/raw/test/dataset-verse19test'

    dest_dir = 'VERSE2019/Verse2019-DRR/subjectwise'

    # create destination directory if it does not exist
    Path(dest_dir).mkdir(exist_ok=True,parents=True)

    # find all files with a given suffix in the base directory
    image_files = sorted(Path(base_dir).rglob('*_ct.nii.gz'))
    seg_files = sorted(Path(base_dir).rglob('*_seg-vert_msk.nii.gz')) 
    json_files = sorted(Path(base_dir).rglob('*_seg-vb_ctd.json'))

    print(image_files)
    for image_file, seg_file, json_file in tqdm(zip(image_files,seg_files,json_files),total=len(seg_files)):
        # extract subject id from file path
        print(get_nifti_stem(image_file),get_verse_subject_id(image_file))

        subject_id = get_verse_subject_id(image_file)

        subject_dir = Path(dest_dir)/subject_id
        # create the subject directory if it does not exist
        subject_dir.mkdir(exist_ok=True,parents=True)

        # copy the image, seg and json files to destination directory
        shutil.copy(image_file,subject_dir)
        shutil.copy(seg_file,subject_dir)
        shutil.copy(json_file,subject_dir)