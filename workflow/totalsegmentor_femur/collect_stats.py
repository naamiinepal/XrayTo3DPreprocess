import os
from typing import List
import pandas as pd
from tqdm import tqdm
import pathlib
from pathlib import Path

import numpy as np
from xrayto3d_preprocess import get_largest_connected_component, write_csv,read_image,get_segmentation_stats,get_nifti_stem

def get_totalsegmentor_subjects(base_path:str)-> List[str]:
    subfolders = [ f.path for f in os.scandir(base_path) if f.is_dir() ]
    return subfolders

def get_totalsegmentor_femur_filenames() -> List[str]:
    return ['femur_left.nii.gz', 'femur_right.nii.gz']

def get_femur_position_from_filename(filepath) -> str:
    """2D-3D-Reconstruction-Datasets/TotalSegmentor-full/s0671/femur_left.nii.gz"""
    anatomy, position =  get_nifti_stem(filepath).split('_')
    return position

def get_subject_femur_stats(subject_seg_dir,femur_filenames):
    full_paths = [str(Path(subject_seg_dir)/p) for p in femur_filenames]

    voxel_count = {}
    for sample_femur_path in full_paths:
        sample_femur = read_image(sample_femur_path)
        sample_femur = get_largest_connected_component(sample_femur) # some segmentations have islands of spurious segmentations
        stats_obj = get_segmentation_stats(sample_femur)
        label_voxels = [ stats_obj.GetNumberOfPixels(l) for l in stats_obj.GetLabels()]
        femur_position = get_femur_position_from_filename(sample_femur_path)
        voxel_count[femur_position] = np.sum(label_voxels,dtype=np.int32)
    return voxel_count

def save_total_voxel_stats_for_whole_dataset(subjects_path, femur_filenames,stats_out_path):
    femur_meta_dict = {}
    header = ['left','right']
    
    for sample_subject_path in tqdm(subjects_path,total=len(subjects_path)):
        ct_path = f'{sample_subject_path}/ct.nii.gz'
        subject_seg_dir = f'{sample_subject_path}/segmentations'

        subject_id = str(Path(sample_subject_path).name)
        voxel_count = get_subject_femur_stats(subject_seg_dir,femur_filenames)
        
        femur_meta_dict[subject_id] = [voxel_count['left'],voxel_count['right']]

        # overwrite csv once a new row of data is available 
        write_csv(femur_meta_dict,header,stats_out_path)

def analyze_stats(metadata_path):
    df = pd.read_csv(metadata_path)

    # how many ct scans contain neither of both left and right femur?
    print(f'Total rows {len(df)} Empty rows',len(df[(df['left'] == 0) &(df['right'] == 0)]))


    full_femur_df = df[df.transpose().all()].copy() # find ct scans with both left and right femur annotated
    full_femur_df.rename( columns={'Unnamed: 0' :'subject_id'}, inplace=True )
    # median volume of the left and right femurs
    print(f'Median right {full_femur_df["right"].median()} left {full_femur_df["left"].median()}')

    # left_p = full_femur_df['left'].quantile(0.4)
    threshold_50k = 50000
    full_femur_left_nonpartial = full_femur_df[(full_femur_df['left'] > threshold_50k)]
    full_femur_right_nonpartial = full_femur_df[(full_femur_df['right'] > threshold_50k)]
    print(f'left {len(full_femur_left_nonpartial)} right {len(full_femur_right_nonpartial)}')


    # save subjects with both left and right femur annotated
    full_femur_df.to_csv(Path(__file__).with_name('subjects_with_both_femur.csv'),columns=['subject_id'],index=False,header=False)    
    
    # save subjects with sizable volume of left femur 
    full_femur_left_nonpartial.to_csv(Path(__file__).with_name('subjects_with_left_femur.csv'),columns=['subject_id'],index=False, header=False)    

    full_femur_right_nonpartial.to_csv(Path(__file__).with_name('subjects_with_right_femur.csv'),columns=['subject_id'],index=False,header=False)

if __name__ == '__main__':
        # base_path = '2D-3D-Reconstruction-Datasets/TotalSegmentor-full/'
        dataset_base_path = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset'
        subjects_path = get_totalsegmentor_subjects(dataset_base_path)
        femur_filenames = get_totalsegmentor_femur_filenames()

        print(f'subjects {len(subjects_path)}')
        total_subjects = len(subjects_path)

        OBTAIN_STATS = True
        ANALYZE_STATS = True

        stats_save_dir = 'external/XrayTo3DPreprocess/workflow/data/totalsegmentor_femur'
        if OBTAIN_STATS:
            stats_out_path = Path(stats_save_dir)/'femur_stats.csv'
            save_total_voxel_stats_for_whole_dataset(subjects_path, femur_filenames,stats_out_path)
        
        if ANALYZE_STATS:
            analyze_stats(Path(stats_save_dir)/'femur_stats.csv')
        