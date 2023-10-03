from xrayto3d_preprocess import read_image
import SimpleITK as sitk
from pathlib import Path

PATCH_SZ = 128
NUM_PATCH = 2

def segregate_by_subject_id(patch_paths):
    path_dict = {}
    for gt_path in patch_paths:
        subject_id, patch_id = get_subjectid_patch_pos(gt_path)
        if subject_id not in path_dict.keys():
            path_dict[subject_id] = []
        path_dict[subject_id].append(gt_path)
    return path_dict

def get_subjectid_patch_pos(filename):
    PREFIX_LEN = len('s0046')
    subject_id = filename.name[:PREFIX_LEN]
    PATCH_ID_POS = 1
    patch_id = filename.name.split('_')[PATCH_ID_POS]
    return subject_id, patch_id

def process_subject(subject_id, subject_paths,out_filename_template):
    patches = [read_image(p) for p in sorted(subject_paths)]
    patch_start_position_list = [int(i*PATCH_SZ) for i in range(NUM_PATCH)] # [0,40, 80, 120, ...]
    start_pos_list = [ (PA,IS,RL) for PA in patch_start_position_list for IS in patch_start_position_list for RL in patch_start_position_list ]

    combined_size = (PATCH_SZ *2, ) * 3
    combined_image = sitk.Image(*combined_size,sitk.sitkFloat32)
    for sample_patch, start_pos in zip(patches, start_pos_list):
        combined_image = sitk.Paste(combined_image, sample_patch,sample_patch.GetSize(),(0,0,0),start_pos)
    
    out_path = subject_paths[0].parent.parent/'combined_patches'
    out_path.mkdir(exist_ok=True, parents=False)

    out_file = out_path/out_filename_template.format(subject_id = subject_id)
    sitk.WriteImage(combined_image,str(out_file))

patch_start_position_list = [int(i*PATCH_SZ) for i in range(NUM_PATCH)] # [0,40, 80, 120, ...]
start_pos_list = [ (PA,IS,RL) for PA in patch_start_position_list for IS in patch_start_position_list for RL in patch_start_position_list ]
if __name__ == '__main__':
    from pathlib import Path
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    patch_dir = f'{args.path}/evaluation'

    gt_patches = sorted(list(Path(patch_dir).glob('*rib_msk_gt.nii.gz')))
    pred_patches = sorted(list(Path(patch_dir).glob('*rib_msk_pred.nii.gz')))

    print(f'gt {len(gt_patches)} pred {len(pred_patches)}')

    
    subject_wise_gt_path_dict = segregate_by_subject_id(gt_patches)

    for subject_id, subject_paths in tqdm(subject_wise_gt_path_dict.items(),total =len( subject_wise_gt_path_dict.keys())):
        process_subject(subject_id,subject_paths,'{subject_id}_rib_msk_gt.nii.gz')
        # break

    subject_wise_pred_path_dict = segregate_by_subject_id(pred_patches)
    for subject_id, subject_paths in tqdm(subject_wise_pred_path_dict.items(),total=len(subject_wise_pred_path_dict.keys())):
        process_subject(subject_id, subject_paths,'{subject_id}_rib_msk_pred.nii.gz')
        # break