from pathlib import Path
from monai.data import PatchIter
import numpy as np
from monai.transforms import Compose, LoadImage,  Resize, EnsureType
from monai.data.nifti_writer import write_nifti
from monai.data.image_writer import PILWriter

from XrayTo3DShape import get_nonkasten_transforms

sample_ap = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/xray_from_ct/s0004_rib-ap.png'
sample_lat = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/xray_from_ct/s0004_rib-lat.png'
sample_seg = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/seg_roi/s0004_rib_msk.nii.gz'

PATCH_SZ = 160
FULLRES_SZ = 320
RES = 1

xray_patch_generator = PatchIter(patch_size=(PATCH_SZ, PATCH_SZ))

seg_patch_generator = PatchIter(patch_size=(PATCH_SZ, PATCH_SZ, PATCH_SZ), start_pos=(0,0,0))

img_trans = Compose([LoadImage(image_only=True, ensure_channel_first=True),EnsureType(),
                     Resize(spatial_size=(FULLRES_SZ,FULLRES_SZ),size_mode='all', mode='bilinear', align_corners=True)])

trans_dict = get_nonkasten_transforms(size=FULLRES_SZ, resolution=RES)
seg_trans = trans_dict['seg']

img_ap = img_trans(sample_ap)
img_lat = img_trans(sample_lat)
seg_msk_dict = seg_trans({'seg': sample_seg})
seg_msk = seg_msk_dict['seg']
seg_patches = list(seg_patch_generator(seg_msk))

print(img_ap.shape, img_lat.shape, type(img_ap), seg_msk[0].shape, type(seg_msk))

img_ap_patches = list(xray_patch_generator(np.swapaxes(img_ap,1,2)))
img_ap_patches = [(np.swapaxes(patch[0],1,2),patch[1]) for patch in img_ap_patches]

img_lat_patches =list( xray_patch_generator(img_lat))


# print(len(seg_patches), len(img_ap_patches), len(img_lat_patches))

xray_writer = PILWriter(output_dtype=np.uint8,scale=None)
for idx, (ap_patch, lat_patch) in enumerate(zip(img_ap_patches,img_lat_patches)):
    # write ap patch
    xray_writer.set_data_array(ap_patch[0])
    output_path = Path(f'test_data/test_patches/test_patch_ap_{idx}.png')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    xray_writer.write(output_path)

    # write lat patch
    xray_writer.set_data_array(lat_patch[0])
    output_path = Path(f'test_data/test_patches/test_patch_lat_{idx}.png')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    xray_writer.write(output_path)


# extract original metadata
seg_msk_metadata = seg_msk_dict['seg_meta_dict']

for idx, (seg_patch,coord) in enumerate(seg_patches):
    # generate metadata
    metadict = {}
    metadict["spatial_shape"] = np.asarray(
            [
                [PATCH_SZ, PATCH_SZ, PATCH_SZ],
            ]
        )

    # metadict["affine"] = seg_msk_metadata['affine']
    # metadict['affine'][0,2] = RES
    # metadict['affine'][1,0] = -RES
    # metadict['affine'][2,1] = -RES
    metadict['affine'] = np.asarray(
                [
                    [0, 0, RES, 0],
                    [-RES, 0, 0, 0],
                    [0, -RES, 0, 0],
                    [0, 0, 0, 1],
                ],
        )

    output_path = Path(f'test_data/test_patches/test_patch_seg_{idx}.nii.gz')
    output_path.parent.mkdir(exist_ok=True, parents=True)    
    write_nifti(data=seg_patch[0], file_name = str(output_path), affine=metadict['affine'],resample=False, output_spatial_shape=metadict['spatial_shape'],mode='nearest', align_corners=True,output_dtype=np.uint8)