from pathlib import Path
from xrayto3d_preprocess import read_image
import SimpleITK as sitk
base_dir = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives'
expected_seg_filepath = Path(base_dir)/'seg_roi'/'s0004_rib_msk.nii.gz'
patches = Path(base_dir)/'seg_roi_patch'
patches = sorted(patches.glob('*msk.nii.gz'))

expected_seg = read_image(str(expected_seg_filepath))
print(f'Expected origin {expected_seg.GetOrigin()} seg size {expected_seg.GetSize()} spacing {expected_seg.GetSpacing()}')
print(f'Patches {len(patches)}')
for p in patches:
    img_p = read_image(str(p))
    print(f'{img_p.GetOrigin()} {img_p.GetSize()} {img_p.GetSpacing()}')

patches = [read_image(p) for p in patches]

new_combined_seg = sitk.Image(*expected_seg.GetSize(),expected_seg.GetPixelIDValue())
new_combined_seg.SetDirection(expected_seg.GetDirection())
new_combined_seg.SetSpacing(expected_seg.GetSpacing())
new_combined_seg.SetOrigin(expected_seg.GetOrigin())

PATCH_SZ = 128
NUM_PATCH = 2
patch_start_position_list = [int(i*PATCH_SZ) for i in range(NUM_PATCH)] # [0,40, 80, 120, ...]
start_pos_list = [ (PA,IS,RL) for PA in patch_start_position_list for IS in patch_start_position_list for RL in patch_start_position_list ]

print(start_pos_list)

sample_id = 0
sample_patch = patches[sample_id]
for sample_patch, start_pos in zip(patches,start_pos_list):
    new_combined_seg = sitk.Paste(new_combined_seg, sample_patch, sample_patch.GetSize(), (0,0,0),start_pos)
sitk.WriteImage(new_combined_seg,'test_data/s0004_rib_msk_combined.nii.gz')