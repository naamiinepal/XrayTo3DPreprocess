from xrayto3d_preprocess import get_nifti_stem,range_inclusive,read_image,combine_segmentations,write_image,get_segmentation_stats,write_csv
from typing import List,Tuple
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def get_totasegmentor_ribs_filenames() -> List[str]:
    """['rib_left_1.nii.gz',...,'rib_right_12.nii.gz]"""
    ribs_filename_template ='rib_{position}_{rib_number}.nii.gz'
    position = ['left','right']
    rib_number = list(range_inclusive(1,12))

    ribs_filenames = []
    for rib_num in rib_number:
        for pos in position:
            ribs_filenames.append(ribs_filename_template.format(position = pos, rib_number = rib_num))
    return ribs_filenames    

def get_rib_number_from_filename(filepath) -> Tuple[str,str]:
    """2D-3D-Reconstruction-Datasets/TotalSegmentor-full/s0671/rib_left_11.nii.gz"""
    anatomy, position, number = get_nifti_stem(filepath).split('_')
    return position, number

def get_totalsegmentor_subjects(base_path:str)-> List[str]:
    subfolders = [ f.path for f in os.scandir(base_path) if f.is_dir() ]
    return subfolders

def process_totalsegmentor_subject(ct_path, seg_dir, ribs_filenames):
    full_paths = [str(Path(seg_dir)/p) for p in ribs_filenames]
    
    rib_seg_sitk = [read_image(p) for p in full_paths]

    rib_complete_seg_sitk = combine_segmentations(rib_seg_sitk,rib_seg_sitk[0],fill_label=1)

    stats_obj = get_segmentation_stats(rib_complete_seg_sitk)
    
    total_voxels = np.sum([stats_obj.GetNumberOfPixels(l) for l in stats_obj.GetLabels()],dtype=np.int32)

    rib_out_path = Path(ct_path).with_name('rib.nii.gz')

    write_image(rib_complete_seg_sitk,str(rib_out_path))
    return total_voxels

def process_totalsegmentor_subject_helper(subject_path):
    subject_id = str(Path(subject_path).name)

    print(f'Subject {subject_id}')

    ct_path = f'{subject_path}/ct.nii.gz'
    seg_dir = f'{subject_path}/segmentations'
    rib_filenames = get_totasegmentor_ribs_filenames()
    total_voxels = process_totalsegmentor_subject(ct_path, seg_dir, rib_filenames)
    return total_voxels

def get_ribs_stats(seg_dir):
    full_paths = [str(Path(seg_dir)/p) for p in ribs_filenames]

    rib_labels = []
    voxels = []
    for sample_rib_path in full_paths:
        sample_rib = read_image(sample_rib_path)
        stats_obj = get_segmentation_stats(sample_rib)
        label_voxels = [stats_obj.GetNumberOfPixels(l) for l in stats_obj.GetLabels()]

        rib_labels.append('_'.join(get_rib_number_from_filename(sample_rib_path)))
        voxels.append(np.sum(label_voxels,dtype=np.int32))
    return rib_labels,voxels

if __name__ == '__main__':

    base_path = '2D-3D-Reconstruction-Datasets/TotalSegmentor-full/'
    subjects_path = get_totalsegmentor_subjects(base_path)
    ribs_filenames = get_totasegmentor_ribs_filenames()
    rib_meta_dict = {}
    
    print(f'subjects {len(subjects_path)}')
    total_subjects = len(subjects_path)

    GENERATE_RIB_SEG = False
    OBTAIN_STATS = False

    if GENERATE_RIB_SEG:
        num_workers = os.cpu_count()
        with Pool(processes=num_workers) as p:
            results = tqdm(p.map(process_totalsegmentor_subject_helper,sorted(subjects_path)),total=total_subjects)
            print('done')


    if OBTAIN_STATS:
        for sample_subject_path in tqdm(subjects_path,total=len(subjects_path)):
            ct_path = f'{sample_subject_path}/ct.nii.gz'
            seg_dir = f'{sample_subject_path}/segmentations'

            subject_id = str(Path(sample_subject_path).name)
            labels,voxels = get_ribs_stats(seg_dir)
            total_voxels = np.sum(voxels,dtype=np.int32)

            rib_meta_dict[subject_id] = [total_voxels,*voxels]

            rib_meta_path = Path(base_path)/'rib_meta.csv'    
            write_csv(rib_meta_dict,['total_voxels',*labels],rib_meta_path)