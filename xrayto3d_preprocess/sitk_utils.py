import numpy as np
import SimpleITK as sitk
from typing import Optional,Union
from .enumutils import ProjectionType
from .tuple_ops import all_elements_equal
from .metadata_utils import get_orientation_code_itk

def change_label(img: sitk.Image, mapping_dict) -> sitk.Image:
    """use SimplITK AggregateLabelMapFilter to merge all segmentation labels to first label. This is used to obtain the bounding box of all the labels """
    fltr = sitk.ChangeLabelImageFilter()
    fltr.SetChangeMap(mapping_dict)
    return fltr.Execute(sitk.Cast(img,sitk.sitkUInt8))

def keep_only_label(segmentation:sitk.Image, label_id) -> sitk.Image:
    """If the segmentation contains more than one labels, keep only label_id"""
    return sitk.Threshold(segmentation, label_id, label_id, 0)

def get_segmentation_labels(segmentation:sitk.Image):
    fltr = sitk.LabelShapeStatisticsImageFilter()
    fltr.Execute(sitk.Cast(segmentation,sitk.sitkUInt8))
    return fltr.GetLabels()


    
def get_segmentation_stats(segmentation: sitk.Image)->sitk.LabelShapeStatisticsImageFilter:
    fltr = sitk.LabelShapeStatisticsImageFilter()
    fltr.Execute(sitk.Cast(segmentation,sitk.sitkUInt8))
    return fltr

def rsna_flip_mirror(img:sitk.Image,flip_axes)->sitk.Image:
    """flip in Z axis and mirror in X axis"""
    
    img_arr = sitk.GetArrayFromImage(img)
    flipped_img_arr = np.flip(img_arr, axis=flip_axes)
    flipped_img:sitk.Image = sitk.GetImageFromArray(flipped_img_arr)
    flipped_img.CopyInformation(img)
    return flipped_img
    
def get_interpolator(interpolator: str):
    """Utility function to converte string representation of interpolator to sitk.sitkInterpolator type"""
    if interpolator == 'nearest':
        return sitk.sitkNearestNeighbor
    if interpolator == 'linear':
        return sitk.sitkLinear

def make_isotropic(img: sitk.Image, spacing=None, interpolator='linear'):
    '''
    Resample `img` so that the voxel is isotropic with given physical spacing
    The image volume is shrunk or expanded as necessary to represent the same physical space. 
    Use sitk.sitkNearestNeighbour while resampling Label images,
    when spacing is not supplied by the user, the highest resolution axis spacing is used
    '''
    # keep the same physical space, size may shrink or expand

    if spacing is None:
        spacing = min(list(img.GetSpacing()))

    original_spacing = list(img.GetSpacing())
    if all_elements_equal(original_spacing) and original_spacing[0] == spacing:
        return img

    resampler = sitk.ResampleImageFilter()
    new_size = [round(old_size * old_spacing / spacing)
                for old_size, old_spacing in zip(img.GetSize(), img.GetSpacing())]
    output_spacing = [spacing]*len(img.GetSpacing())

    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(get_interpolator(interpolator))
    resampler.SetDefaultPixelValue(
        img.GetPixelIDValue())

    return resampler.Execute(img)

def simulate_projection(segmentation:sitk.Image,projectiontype:ProjectionType) -> sitk.Image:
    segmentation = make_isotropic(segmentation,1.0,"nearest")
    orientation = get_orientation_code_itk(segmentation)
    orientation = list(orientation)

    if projectiontype == ProjectionType.ap:
        if 'P' in orientation:
            projectionDimension = orientation.index('P')
        elif 'A' in orientation:
            projectionDimension = orientation.index('A')
    elif projectiontype == ProjectionType.lat:
        if 'L' in orientation:
            projectionDimension = orientation.index('L')
        elif 'R' in orientation:
            projectionDimension = orientation.index('R')
    else:
        raise ValueError(f'Projection type should be one of {ProjectionType.ap} or {ProjectionType.lat}')

    projection = sitk.MeanProjection(segmentation,projectionDimension=projectionDimension)
    projection = sitk.Cast(sitk.RescaleIntensity(projection),sitk.sitkUInt8)
    if projectionDimension == 0:
        return projection[0,:,:]
    elif projectionDimension == 1:
        return projection[:,0,:]
    elif projectionDimension == 2:
        return projection[:,:,0]

def rotate_about_image_center(img: sitk.Image, rx,ry,rz):
    """rotation angles are assumed to be given in degrees"""

    import math
    transform = sitk.Euler3DTransform()
    transform.SetComputeZYX(True)
    
    # constant for converting degrees to radians
    dtr = math.atan(1.0) * 4.0 / 180.0
    transform.SetRotation(dtr*rx, dtr*ry, dtr*rz)

    im_sz = img.GetSize()
    center_index = list(sz//2 for sz in im_sz)
    print(center_index)
    isocenter = img.TransformIndexToPhysicalPoint(center_index)

    transform.SetCenter(isocenter)

    return sitk.Resample(img,transform=transform)


def reorient_to(img, axcodes_to:Union[sitk.Image,str]='PIR', verb=False):
    """Reorients the Image from its original orientation to another specified orientation
    
    Parameters:
    ----------
    img: SimpleITK image
    axcodes_to: a string of 3 characters specifying the desired orientation
    
    Returns:
    ----------
    new_img: The reoriented SimpleITK image 
    
    """
    if isinstance(axcodes_to,sitk.Image):
        axcodes_to = get_orientation_code_itk(axcodes_to)
        
    new_img = sitk.DICOMOrient(img,axcodes_to)

    if verb:
        print("[*] Image reoriented from", get_orientation_code_itk(img), "to", axcodes_to)
    return new_img

def reorient_centroids_to(ctd_list, target_axcodes,image_shape, decimals=1, verb=False):
    # adapted from github.com/anjany/verse
    """reorient centroids to image orientation
    
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
    import numpy as np
    import nibabel.orientations as nio

    if isinstance(target_axcodes,str):
        target_axcodes = list(target_axcodes) #'PIR' -> ['P', 'I', 'R']

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
        print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", target_axcodes)
    return out_list

def mirror_image(image:sitk.Image,flip_along=0):
    """used to mirror the proximal femur ct so that the right femur looks similar in orientation to the left femur"""
    array_image = sitk.GetArrayFromImage(image)
    flipped_array_image = np.flip(array_image, flip_along)
    flippped_image = sitk.GetImageFromArray(flipped_array_image)
    flippped_image.CopyInformation(image)
    return flippped_image