import os
from typing import List
from tqdm import tqdm
from pathlib import Path
import numpy as np
from multiprocessing import Pool

from xrayto3d_preprocess import read_image,combine_segmentations,write_image,get_segmentation_stats,write_csv,get_nifti_stem

def get_totalsegmentor_subjects(base_path:str)-> List[str]:
    subfolders = [ f.path for f in os.scandir(base_path) if f.is_dir() ]
    return subfolders

hip_bones = ['hip_left.nii.gz', 'hip_right.nii.gz']

def get_hip_position_from_filename(filepath):
    anatomy,position = get_nifti_stem(filepath).split('_')
    return position

def get_hip_stats(seg_dir,file_patterns = hip_bones):
    full_paths = [str(Path(seg_dir)/p) for p in file_patterns]

    labels = []
    voxels = []
    for sample_path in full_paths:
        sample = read_image(sample_path)
        stats_obj = get_segmentation_stats(sample)
        labelwise_voxels = [stats_obj.GetNumberOfPixels(l) for l in stats_obj.GetLabels() ]
        voxels.append(np.sum(labelwise_voxels,dtype=np.int32))

        labels.append('_'.join(get_hip_position_from_filename(sample_path)))
    return labels,voxels

def process_totalsegmentor_subject(ct_path, seg_dir,file_patterns=hip_bones,out_name='hip.nii.gz'):
    full_paths = [str(Path(seg_dir)/p) for p in hip_bones]
    
    seg_sitk = [read_image(p) for p in full_paths]

    complete_seg_sitk = combine_segmentations(seg_sitk,seg_sitk[0],fill_label=1)

    out_path = Path(ct_path).with_name(out_name)

    write_image(complete_seg_sitk,str(out_path))


def process_totalsegmentor_subject_helper(subject_path):
    subject_id = str(Path(subject_path).name)

    print(f'Subject {subject_id}')

    ct_path = f'{subject_path}/ct.nii.gz'
    seg_dir = f'{subject_path}/segmentations'

    process_totalsegmentor_subject(ct_path,seg_dir)

def analyze_stats(stats_path):
    import pandas as pd
    df = pd.read_csv(stats_path)
    # how many ct scans contain neither of both left and right femur?
    print(f'Total rows {len(df)} Empty rows',len(df[(df['total_voxels'] == 0)]))
    print(f'Non Empty subjects {len(df[df["total_voxels"] != 0])}')
    non_empty_df = df[df['total_voxels'] != 0]

    non_empty_df.to_csv('hip_nonempty.csv',index=False)

if __name__ == '__main__':
    import argparse

    parser   = argparse.ArgumentParser()
    base_path = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset'
    parser.add_argument('--base_path',default=base_path)
    parser.add_argument('--generate_stats',default=False,action='store_true')    
    parser.add_argument('--generate_hips',default=False,action='store_true')
    parser.add_argument('--analyze_stats',default=False,action='store_true')

    args = parser.parse_args()

    subjects_path = get_totalsegmentor_subjects(args.base_path)

    total_subjects = len(subjects_path)    
    print(f'subjects {total_subjects}')

    GENERATE_STATS = args.generate_stats
    GENERATE_HIPS = args.generate_hips
    ANALYZE_STATS = args.analyze_stats

    if GENERATE_HIPS:
        with Pool(processes=os.cpu_count()) as p:
            tqdm(p.map(process_totalsegmentor_subject_helper,sorted(subjects_path)),total=total_subjects)

    if GENERATE_STATS:
        hip_meta_dict = {}
        for sample_subject_path in tqdm(subjects_path,total=len(subjects_path)):
            ct_path = f'{sample_subject_path}/ct.nii.gz'
            seg_dir = f'{sample_subject_path}/segmentations'

            subject_id = str(Path(sample_subject_path).name)        
            labels, voxels = get_hip_stats(seg_dir)
            total_voxels = np.sum(voxels, dtype=np.int32)

            hip_meta_dict[subject_id] = [total_voxels,*voxels]

            hip_meta_path = Path(__file__).parent/'hip_stats.csv'
            write_csv(hip_meta_dict,['total_voxels',*labels],hip_meta_path)
    
    if ANALYZE_STATS:
        analyze_stats(Path(__file__).parent/'hip_stats.csv')