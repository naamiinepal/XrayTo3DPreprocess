"""SimpleITK metadata related utils"""
from typing import Sequence, Union, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk


def get_metadata(img_path) -> sitk.ImageFileReader:
    """read metadata without loading the image array"""
    reader = sitk.ImageFileReader()

    reader.SetFileName(img_path)
    reader.LoadPrivateTagsOn()

    reader.ReadImageInformation()
    return reader


def set_image_metadata(img: sitk.Image, origin, direction, spacing):
    """update image metadata
    modify image in-place
    """
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)


def get_opposite_axis(orientation_axis):
    """S -> I, L-> R, A -> P"""
    if orientation_axis == "L":
        return "R"
    if orientation_axis == "R":
        return "L"
    if orientation_axis == "S":
        return "I"
    if orientation_axis == "I":
        return "S"
    if orientation_axis == "A":
        return "P"
    if orientation_axis == "P":
        return "A"


def voxel_size_to_physical_size(img, voxel_size) -> Tuple:
    """Given an img, how much physical space does a volume of given voxel size occupy?"""
    return tuple([v * sp for v, sp in zip(voxel_size, img.GetSpacing())])


def physical_size_to_voxel_size(img, physical_size) -> Tuple:
    """Given an img, how many voxels(rounded evenly) does it require to represent a given physical size?"""
    return tuple(
        [int(np.around(p / sp)) for p, sp in zip(physical_size, img.GetSpacing())]
    )


def get_superior_inferior_axis(orientation: str) -> int:
    """which dimension represents the Superior-Inferior axis?"""
    assert len(list(orientation)) == 3, f"invalid orientation string {orientation}"
    if "I" in list(orientation):
        return list(orientation).index("I")
    elif "S" in list(orientation):
        return list(orientation).index("S")
    else:
        raise ValueError(f"invalid orientation string {orientation}")


def is_superior_to_inferior(orientation: str) -> bool:
    """RAS -> True, RAI ->False, PSR->True, PIR ->False"""
    axis_index = get_superior_inferior_axis(orientation)
    return list(orientation)[axis_index] == "S"


def get_orientation_code_itk(img_or_affine_mtrx: Union[sitk.Image, Sequence]) -> str:
    """Orientation is a tricky topic:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained

    """
    if isinstance(img_or_affine_mtrx, sitk.Image):
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            img_or_affine_mtrx.GetDirection()
        )
    else:
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            img_or_affine_mtrx
        )


def get_orientation_code_nifti(img: nib.Nifti1Image) -> str:
    """get nibabel image and return nifti orientation"""
    return "".join(nib.aff2axcodes(img.affine))
