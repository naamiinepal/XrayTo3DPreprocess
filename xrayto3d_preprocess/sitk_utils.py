"""simpleitk utils"""
import math
from typing import Union, List

import nibabel.orientations as nio
import numpy as np
import SimpleITK as sitk

from .enumutils import ProjectionType
from .ioutils import read_image, write_image
from .metadata_utils import get_orientation_code_itk
from .tuple_ops import all_elements_equal

# relax global tolerance (by default ~1e-7) to avoid this error:
# itk::ERROR: itk::ERROR:  Inputs do not occupy the same physical space!
sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(0.01)
sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(0.01)


def save_overlays(img_path, in_overlay_path, out_path, threshold=0.5):
    """save img-seg overlay image"""
    img = read_image(img_path)
    overlay_seg = read_image(in_overlay_path)
    overlay = overlay_seg > threshold
    overlayed_img = sitk.LabelOverlay(img, overlay)
    write_image(overlayed_img, out_path)


def get_largest_connected_component(binary_image: sitk.Image):
    """get largest connected components
    - sideffect: the largest component gets relabeled
    """
    component_image = sitk.ConnectedComponent(sitk.Cast(binary_image, sitk.sitkUInt8))
    sorted_component_image = sitk.RelabelComponent(
        component_image, sortByObjectSize=True
    )
    largest_component_binary_image = sorted_component_image == 1
    return largest_component_binary_image


def combine_segmentations(imgs: List[sitk.Image], ref_img: sitk.Image, fill_label=1):
    """Combine multiple segmentation images into a single segmentation image.

    Precondition:
        segmentation masks do not overlap

    Postcondition:
        a new segmentation images is returned where voxels are filled with fill_label
        if the voxel position is labelled in one of the segmentation image
    """
    new_seg = np.zeros_like(sitk.GetArrayFromImage(ref_img))

    for seg in imgs:
        single_seg = sitk.GetArrayViewFromImage(seg)
        new_seg[single_seg > 0.5] = fill_label

    img_out = sitk.GetImageFromArray(new_seg)
    img_out.CopyInformation(ref_img)
    return img_out


def mask_ct_with_seg(img: sitk.Image, seg: sitk.Image):
    """wrap sitk.LabelMapMask"""
    return sitk.LabelMapMask(sitk.Cast(seg, sitk.sitkLabelUInt8), img)


def change_label(img: sitk.Image, mapping_dict) -> sitk.Image:
    """
    use SimplITK AggregateLabelMapFilter to merge all segmentation labels to first label.
    This is used to obtain the bounding box of all the labels
    """
    fltr = sitk.ChangeLabelImageFilter()
    fltr.SetChangeMap(mapping_dict)
    return fltr.Execute(sitk.Cast(img, sitk.sitkUInt8))


def keep_only_label(segmentation: sitk.Image, label_id) -> sitk.Image:
    """If the segmentation contains more than one labels, keep only label_id"""
    return sitk.Threshold(segmentation, label_id, label_id, 0)


def get_segmentation_labels(segmentation: sitk.Image):
    """return segmentation labels"""
    fltr = sitk.LabelShapeStatisticsImageFilter()
    fltr.Execute(sitk.Cast(segmentation, sitk.sitkUInt8))
    return fltr.GetLabels()


def get_segmentation_stats(
    segmentation: sitk.Image,
) -> sitk.LabelShapeStatisticsImageFilter:
    """return sitk filter obj containing segmentation stats"""
    fltr = sitk.LabelShapeStatisticsImageFilter()
    fltr.Execute(sitk.Cast(segmentation, sitk.sitkUInt8))
    return fltr


def flip_image(img: sitk.Image, flip_axes) -> sitk.Image:
    """
    flip along an axis, but do not update the metadata
    - may be used for RSNA dataset
    - use with caution, since you are deliberately changing the metadata
    one use case for this:
     -  used to mirror the proximal femur ct so that the right femur
        looks similar in orientation to the left femur, for model training
    """

    img_arr = sitk.GetArrayFromImage(img)
    flipped_img_arr = np.flip(img_arr, axis=flip_axes)
    flipped_img: sitk.Image = sitk.GetImageFromArray(flipped_img_arr)
    flipped_img.CopyInformation(img)
    return flipped_img


mirror_image = flip_image  # for legacy reasons


def get_interpolator(interpolator: str):
    """Utility function to convert string representation of interpolator to sitk.sitkInterpolator type"""
    if interpolator == "nearest":
        return sitk.sitkNearestNeighbor
    if interpolator == "linear":
        return sitk.sitkLinear

def resample_isotropic(img: sitk.Image, spacing, interpolator):
    'wrapper around make_isotropic'
    return make_isotropic(img, spacing, interpolator)

def make_isotropic(img: sitk.Image, spacing=None, interpolator="linear"):
    """
    Resample `img` so that the voxel is isotropic with given physical spacing
    The image volume is shrunk or expanded as necessary to represent the same physical space.
    Use sitk.sitkNearestNeighbour while resampling Label images,
    when spacing is not supplied by the user, the highest resolution axis spacing is used
    """
    # keep the same physical space, size may shrink or expand

    if spacing is None:
        spacing = min(list(img.GetSpacing()))

    original_spacing = list(img.GetSpacing())
    if all_elements_equal(original_spacing) and original_spacing[0] == spacing:
        return img

    resampler = sitk.ResampleImageFilter()
    new_size = [
        round(old_size * old_spacing / spacing)
        for old_size, old_spacing in zip(img.GetSize(), img.GetSpacing())
    ]
    output_spacing = [spacing] * len(img.GetSpacing())

    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(get_interpolator(interpolator))
    # resampler.SetDefaultPixelValue(img.GetPixelIDValue())

    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(img)


def simulate_parallel_projection(
    segmentation: sitk.Image, projectiontype: ProjectionType
):
    """return a mean projection"""
    segmentation = make_isotropic(segmentation, 1.0, "nearest")
    orientation = get_orientation_code_itk(segmentation)
    orientation = list(orientation)

    if projectiontype == ProjectionType.AP:
        if "P" in orientation:
            dim = orientation.index("P")
        elif "A" in orientation:
            dim = orientation.index("A")
    elif projectiontype == ProjectionType.LAT:
        if "L" in orientation:
            dim = orientation.index("L")
        elif "R" in orientation:
            dim = orientation.index("R")
    else:
        raise ValueError(
            f"Projection type should be one of {ProjectionType.AP} or {ProjectionType.LAT}"
        )

    projection = sitk.MeanProjection(segmentation, projectionDimension=dim)
    projection = sitk.Cast(sitk.RescaleIntensity(projection), sitk.sitkUInt8)
    if dim == 0:
        return projection[0, :, :]
    elif dim == 1:
        return projection[:, 0, :]
    elif dim == 2:
        return projection[:, :, 0]


def rotate_about_image_center(img: sitk.Image, rx, ry, rz) -> sitk.Image:
    """rotation angles are assumed to be given in degrees"""

    transform = sitk.Euler3DTransform()
    transform.SetComputeZYX(True)

    # constant for converting degrees to radians
    dtr = math.atan(1.0) * 4.0 / 180.0
    transform.SetRotation(dtr * rx, dtr * ry, dtr * rz)

    im_sz = img.GetSize()
    center_index = list(sz // 2 for sz in im_sz)
    print(center_index)
    isocenter = img.TransformIndexToPhysicalPoint(center_index)

    transform.SetCenter(isocenter)

    return sitk.Resample(img, transform=transform)


def reorient_to(img, axcodes_to: Union[sitk.Image, str] = "PIR", verb=False):
    """Reorients the Image from its original orientation to another specified orientation

    adapted from https://github.com/anjany/verse/blob/main/utils/data_utilities.py

    Parameters:
    ----------
    img: SimpleITK image
    axcodes_to: a string of 3 characters specifying the desired orientation

    Returns:
    ----------
    new_img: The reoriented SimpleITK image

    """
    if isinstance(axcodes_to, sitk.Image):
        axcodes_to = get_orientation_code_itk(axcodes_to)

    new_img = sitk.DICOMOrient(img, axcodes_to)

    if verb:
        print(
            "[*] Image reoriented from", get_orientation_code_itk(img), "to", axcodes_to
        )
    return new_img


def reorient_centroids_to(
    ctd_list, target_axcodes, image_shape, decimals=1, verb=False
):
    """reorient centroids to image orientation

    adapted from https://github.com/anjany/verse/blob/main/utils/data_utilities.py

    Parameters:
    ----------
    ctd_list: list of centroids [ the first element of the list should be axcode ]
    target_axcodes: for example: 'PIR'
    image_shape: (sx,sy,sz) This is required when inverting axis
    decimals: rounding decimal digits

    Returns:
    ----------
    out_list: reoriented list of centroids

    """

    if isinstance(target_axcodes, str):
        target_axcodes = list(target_axcodes)  # 'PIR' -> ['P', 'I', 'R']

    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present")
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    ornt_to = nio.axcodes2ornt(target_axcodes)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(image_shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [target_axcodes]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    if verb:
        print(
            "[*] Centroids reoriented from",
            nio.ornt2axcodes(ornt_fr),
            "to",
            target_axcodes,
        )
    return out_list
