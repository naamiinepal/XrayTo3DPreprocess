import SimpleITK as sitk

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


def get_orientation_code(img: sitk.Image):
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())
