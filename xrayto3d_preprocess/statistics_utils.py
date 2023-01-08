from pathlib import Path
import SimpleITK as sitk
from xrayto3d_preprocess import get_orientation_code_itk, get_orientation_code_nifti,read_nibabel,get_verse_subject_id
import pandas as pd

def get_statistics_for_entire_dir(ct_file):
    if isinstance(ct_file,Path):
        ct_file = str(ct_file)
    
    reader = sitk.ImageFileReader()
    reader.SetFileName(ct_file)
    reader.LoadPrivateTagsOff()
    reader.ReadImageInformation()

    return {'subject_id':get_verse_subject_id(ct_file),'voxel_sz_0':reader.GetSpacing()[0],'voxel_sz_1':reader.GetSpacing()[1],'voxel_sz_2':reader.GetSpacing()[2], 
    'direction(itk)':get_orientation_code_itk(reader.GetDirection()),
    'direction(nifti)':get_orientation_code_nifti(read_nibabel(ct_file))}

if __name__ == '__main__':
    
    base_dir = 'VERSE2019/Verse2019-DRR/subjectwise'

    ct_paths = Path(base_dir).rglob('*_ct.nii.gz')

    records = []
    for p in ct_paths:
        stats = get_statistics_for_entire_dir(p)
        records.append(stats)
    
    # save csv
    df = pd.DataFrame.from_records(records)
    df.to_csv('data_stats.csv')
