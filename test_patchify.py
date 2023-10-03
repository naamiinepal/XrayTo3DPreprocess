import matplotlib.pyplot as plt
import SimpleITK as sitk
from XrayTo3DShape import get_nonkasten_patch_transforms, get_kasten_transforms, get_nonkasten_transforms, get_projectionslices_from_3d, create_figure
from xrayto3d_preprocess import write_image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ap')
parser.add_argument('lat')
parser.add_argument('seg')
parser.add_argument('--size',type=int)
parser.add_argument('--res',type=float)

args = parser.parse_args()

transforms = get_nonkasten_patch_transforms(size=args.size,resolution=args.res)
ap_transform, lat_transform, seg_transform = (
    transforms["ap"],
    transforms["lat"],
    transforms["seg"],
)


# sample_ap = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/xray_from_ct/s0004_rib-ap.png'
# sample_lat = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/xray_from_ct/s0004_rib-lat.png'
# sample_seg = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset/s0004/derivatives/seg_roi/s0004_rib_msk.nii.gz'

sample_ap, sample_lat, sample_seg = args.ap, args.lat, args.seg

ap_dict = ap_transform({"ap": sample_ap})
lat_dict = lat_transform({"lat": sample_lat})
seg_dict = seg_transform({"seg": sample_seg})

print(ap_dict["ap_meta_dict"])
ap_img = ap_dict["ap"]
lat_img = lat_dict["lat"]
seg_img = seg_dict["seg"]

print(ap_img.shape, lat_img.shape, seg_img.shape)

# fig = plt.figure(figsize=(4, 4))
# plt.imshow(ap_img[0], cmap="gray")
# plt.axis("off")
# fig = plt.figure(figsize=(4, 4))
# plt.axis("off")
# plt.imshow(lat_img[0], cmap="gray")

seg_slices = get_projectionslices_from_3d(seg_img.squeeze())
fig, axes = create_figure(*[*seg_slices,ap_img[0], lat_img[0]])
for ax, img in zip(axes, [*seg_slices,ap_img[0],lat_img[0]]):
    ax.imshow(img, cmap=plt.cm.gray)

write_image(sitk.GetImageFromArray(seg_img),'tests/patchify_seg_roi_patch.nii.gz')
plt.savefig('tests/patchify_alignment_test.png')
