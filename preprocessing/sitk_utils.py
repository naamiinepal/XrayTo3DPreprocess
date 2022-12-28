import SimpleITK as sitk
from .enumutils import ProjectionType

def change_label(img: sitk.Image, mapping_dict) -> sitk.Image:
    """use SimplITK AggregateLabelMapFilter to merge all segmentation labels to first label. This is used to obtain the bounding box of all the labels """
    fltr = sitk.ChangeLabelImageFilter()
    fltr.SetChangeMap(mapping_dict)
    return fltr.Execute(img)

def keep_only_label(segmentation:sitk.Image, label_id) -> sitk.Image:
    """If the segmentation contains more than one labels, keep only label_id"""
    return sitk.Threshold(segmentation, label_id, label_id, 0)

def simulate_projection(segmentation:sitk.Image,projectiontype:ProjectionType) -> sitk.Image:
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(segmentation.GetDirection())
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

    prj = sitk.MeanProjection(segmentation,projectionDimension=projectionDimension)
    prj = sitk.Cast(sitk.RescaleIntensity(prj),sitk.sitkUInt8)
    if projectionDimension == 0:
        return prj[0,:,:]
    elif projectionDimension == 1:
        return prj[:,0,:]
    elif projectionDimension == 2:
        return prj[:,:,0]

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
