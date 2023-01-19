import SimpleITK as sitk
from typing import Union,Sequence
import nibabel as nib

def set_image_metadata(img: sitk.Image, origin, direction, spacing):
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)

def get_opposite_axis(orientation_axis):
    if orientation_axis == 'L':
        return 'R'
    if orientation_axis == 'R':
        return 'L'
    if orientation_axis == 'S':
        return 'I'
    if orientation_axis == 'I':
        return 'S'
    if orientation_axis == 'A':
        return 'P'
    if orientation_axis == 'P':
        return 'A'

def voxel_size_to_physical_size(img, voxel_size):
    """Given an img, how much physical space does a volume of given voxel size occupy?"""
    return tuple([v*sp for v, sp in zip(voxel_size, img.GetSpacing())])


def physical_size_to_voxel_size(img, physical_size):
    """Given an img, how many voxels(rounded up) does it require to represent a given physical size?"""
    return tuple([int(p / sp) for p, sp in zip(physical_size, img.GetSpacing())])


def get_orientation_code_itk(img_or_affine_mtrx: Union[sitk.Image,Sequence]):
    """Orientation is a tricky topic:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained
    
    """
    if isinstance(img_or_affine_mtrx,sitk.Image):
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_or_affine_mtrx.GetDirection())
    else:
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_or_affine_mtrx)

def get_orientation_code_nifti(img:nib.Nifti1Image):
    """get nibabel image and return nifti orientation"""
    return ''.join(nib.aff2axcodes(img.affine))