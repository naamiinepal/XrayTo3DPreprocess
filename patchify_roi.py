import numpy as np
import SimpleITK as sitk
from pathlib import Path
from xrayto3d_preprocess import read_image, get_orientation_code_itk, resample_isotropic,subtract_tuple, add_tuple, generate_xray, ProjectionType
from monai.transforms import Compose, LoadImage, EnsureType, Resize
from monai.data import PatchIter
from monai.data.image_writer import PILWriter

# extract ROI
def extract_roi(img, PATCH_SZ, idx, patch_roi_start_index,output_path,type='ct') :
    if type not in ['ct','seg']:
        raise ValueError(f'type should be one of [ct,seg]. got {type}')

    patch_roi = sitk.RegionOfInterest(img, (PATCH_SZ,) * img.GetDimension(), patch_roi_start_index)

    if type == 'ct':
        patch_roi = sitk.Cast(patch_roi,sitk.sitkInt16)
    elif type == 'seg':
        patch_roi = sitk.Cast(patch_roi, sitk.sitkUInt8)

    output_path = Path(output_path.format(type=type,idx=idx))
    output_path.parent.mkdir(exist_ok=True, parents=True)    
    # write roi
    sitk.WriteImage(patch_roi, str(output_path))

    return output_path

sample_ap = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/xray_from_ct/s0004_rib-ap.png'
sample_lat = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/xray_from_ct/s0004_rib-lat.png'
sample_ct_roi = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/ct_roi/s0004_rib-ct.nii.gz'
sample_seg = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/seg_roi/s0004_rib_msk.nii.gz'



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('patch_sz',type=int)
args = parser.parse_args()

# read inputs

ct_img = read_image(sample_ct_roi)
seg_img = read_image(sample_seg)

PATCH_SZ = args.patch_sz
PATCH_RES = 1
ORIG_RES_SZ = np.ceil(ct_img.GetSize()[0])
ORIG_RES = np.around(ct_img.GetSpacing()[0],3)
EXPECTED_SZ = np.ceil(ORIG_RES_SZ * 1.5)
CT_PAD_VAL = -0
SEG_PAD_VAL = 0
NUM_PATCH = int(EXPECTED_SZ / (PATCH_RES * PATCH_SZ))

output_path = f'test_data/test_patches_v4_{PATCH_SZ}/test_patch_{{type}}_{{idx}}.nii.gz'


print(f'Generating {NUM_PATCH} patches with size {PATCH_SZ} and resolution {PATCH_RES}')

print(f'CT Image size {ct_img.GetSize()} Spacing {np.around(ct_img.GetSpacing(),3)} Orientation {get_orientation_code_itk(ct_img)} Origin {ct_img.GetOrigin()}')
print(f'Seg Image size {seg_img.GetSize()} Spacing {np.around(seg_img.GetSpacing(),3)} Orientation {get_orientation_code_itk(seg_img)} Origin {seg_img.GetOrigin()}')


# bring to PATCH RES resolution
ct_img = resample_isotropic(ct_img, PATCH_RES, interpolator='linear')
seg_img = resample_isotropic(seg_img, PATCH_RES, interpolator='nearest')

expected_size = (EXPECTED_SZ,)*3
required_padding = subtract_tuple(expected_size,ct_img.GetSize())
required_padding = [ int(p) for p in required_padding]

# pad to take fraction error in voxel spacing
ct_img = sitk.ConstantPad(ct_img, (0,0,0), required_padding, CT_PAD_VAL)
seg_img = sitk.ConstantPad(seg_img, (0,0,0), required_padding, SEG_PAD_VAL)

print(f'After resampling to resolution {PATCH_RES}')
print(f'CT Image size {ct_img.GetSize()} Spacing {np.around(ct_img.GetSpacing(),3)} Orientation {get_orientation_code_itk(ct_img)} Origin {ct_img.GetOrigin()}')
print(f'Seg Image size {seg_img.GetSize()} Spacing {np.around(seg_img.GetSpacing(),3)} Orientation {get_orientation_code_itk(seg_img)} Origin {seg_img.GetOrigin()}')
print(f'Required Padding {required_padding} Expected size {expected_size}')

idx = 0
patch_roi_start_index = (0,0,0)

patch_start_pos = [ int(i*PATCH_SZ) for i in range(int(NUM_PATCH))]

start_pos_list = [] 
for pa_start_pos in patch_start_pos:
    for is_start_pos in patch_start_pos:
        for rl_start_pos in patch_start_pos:
            start_pos_list.append(add_tuple(patch_roi_start_index,(pa_start_pos, is_start_pos, rl_start_pos)))        


print(start_pos_list)
xray_pose_dict = {'ap':{'rx':-90, 'ry':0, 'rz': 90}, 'lat': {'rx':-90,'ry':0,'rz':0},'drr_from_mask':False, 'size':PATCH_SZ, 'res': PATCH_RES}

# generate xray patch
img_trans = Compose([LoadImage(image_only=True, ensure_channel_first=True),EnsureType(),
                    Resize(spatial_size=(EXPECTED_SZ,EXPECTED_SZ),size_mode='all', mode='bilinear', align_corners=True)])
xray_ap = img_trans(sample_ap)
xray_lat = img_trans(sample_lat)
xray_patch_generator = PatchIter(patch_size=(PATCH_SZ, PATCH_SZ))
img_ap_patches = list(xray_patch_generator(np.swapaxes(xray_ap,1,2)))
img_ap_patches = list(np.swapaxes(patch,1,2) for patch,coord in img_ap_patches)
# AP: now reverse the images in a single row
reverse_ordering = np.flip(np.array(list(range(len(img_ap_patches)))).reshape(NUM_PATCH,NUM_PATCH),axis=1).flatten()
img_ap_patches = np.array(img_ap_patches)[reverse_ordering]

img_lat_patches = [patch for patch, coord in xray_patch_generator(xray_lat)]

print(f'ap shape {img_ap_patches[0].shape} dtype {img_ap_patches[0].dtype} min/max {np.min(img_ap_patches[0]), np.max(img_lat_patches[0])}')
for idx, roi_idx in enumerate(start_pos_list):
    ct_roi_path = extract_roi(ct_img, PATCH_SZ, idx, roi_idx, output_path, type='ct')
    seg_roi_path = extract_roi(seg_img, PATCH_SZ, idx, roi_idx, output_path, type='seg')

    ap_path = ct_roi_path.with_name(f'test_patch_ap_{idx}.png')
    lat_path = ct_roi_path.with_name(f'test_patch_lat_{idx}.png')


    lat_xray_writer = PILWriter(output_dtype=np.uint8, scale=None)
    lat_xray_writer.set_data_array(img_lat_patches[idx // NUM_PATCH])
    lat_xray_writer.write(lat_path)

    ap_xray_writer = PILWriter(output_dtype=np.uint8, scale=None)
    ap_xray_writer.set_data_array(img_ap_patches[idx % (NUM_PATCH * NUM_PATCH)])
    ap_xray_writer.write(ap_path)
