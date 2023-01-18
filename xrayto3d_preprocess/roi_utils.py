import math
import SimpleITK as sitk
import numpy as np

from .tuple_ops import *
from .metadata_utils import *
from typing import List,Optional
from logging import Logger

def ROI_centroid_index_to_start_index(centroid_index, ROI_voxel_size):
    """Given a ROI of size `ROI_voxel_size` in voxel units and whose centroid index is centroid_index,
    return the corresponding starting index of the ROI in the image  """

    roi_start_index = tuple([int(c - s // 2)
                            for c, s in zip(centroid_index, ROI_voxel_size)])
    return roi_start_index


def ROI_centroid_index_to_start_index_v2(centroid_index, ROI_voxel_size, extraction_ratio: tuple):
    """Given a ROI of size ROI_voxel_size in voxel units and whose centroid index is centroid_index,
    return the corresponding starting index of the ROI in the image.
    We want to extract ROI_voxel_size at given extraction_ration wr.t. the centroid index"""

    roi_start_index = tuple(
        [(c - s * r) for c, s, r in zip(centroid_index, ROI_voxel_size, extraction_ratio)])
    roi_start_index = list(map(int, roi_start_index))
    return roi_start_index




def required_padding(img, voxel_size, centroid_index, verbose=True):
    """how much padding is required to be able to extract ROI of voxel_size whose centroid index is at centroid_index?"""
    upperbound_index: tuple = add_tuple(
        centroid_index, divide_tuple_scalar(voxel_size, 2))
    lowerbound_index: tuple = subtract_tuple(
        centroid_index, divide_tuple_scalar(voxel_size, 2))

    upperbound_pad = tuple([int(max(0, ub_idx - im_idx))
                           for im_idx, ub_idx in zip(img.GetSize(), upperbound_index)])
    lowerbound_pad = tuple([int(max(0, -lb_idx))
                           for lb_idx in lowerbound_index])

    if verbose:
        print(
            f'target voxel {voxel_size} lowerbound {lowerbound_pad} upperbound {upperbound_pad}')
    np.testing.assert_array_equal(np.array(voxel_size), np.array(
        subtract_tuple(upperbound_index, lowerbound_index)))

    return lowerbound_pad, upperbound_pad




def dict_to_tuple(orientation_code, orientation_dict: dict):
    """
    LAS,{'L':0.5,'A':0.5,'S':0.5} -> (0.5,0.5,0.5)
    LPS,{'L':0.5,'A':0.33,'S':0.5} -> (0.5,0.67,0.5) """
    new_orientation_dict = {}
    for plane in orientation_code:
        if plane in orientation_dict.keys():
            new_orientation_dict[plane] = orientation_dict[plane]
        else:
            new_orientation_dict[plane] = 1.0 - \
                orientation_dict[get_opposite_axis(plane)]

    return tuple(new_orientation_dict[plane] for plane in orientation_code)


def required_padding_v2(img, voxel_size, centroid_index, extraction_ratio: dict, verbose=True):
    extraction_ratio_tuple = dict_to_tuple(
        get_orientation_code_itk(img), extraction_ratio)
    upperbound_index = add_tuple(centroid_index, multiply_tuple(
        voxel_size, extraction_ratio_tuple))

    extraction_ratio_tuple = subtract_tuple(
        (1,)*img.GetDimension(), extraction_ratio_tuple)
    lowerbound_index = subtract_tuple(
        centroid_index, multiply_tuple(voxel_size, extraction_ratio_tuple))

    lowerbound_index = list(map(int, lowerbound_index))
    upperbound_index = list(map(int, upperbound_index))

    # handle off-by-one error due to integer truncation
    # since we truncate the decimal part of the index, the volume may not be exact
    # we will handle this by Padding 1 px later on

    TRUNCATION_ERROR = 1 # add 1 voxel to account for truncation error         
    upperbound_pad = tuple([TRUNCATION_ERROR + int(max(0, ub_idx - im_idx))
                           for im_idx, ub_idx in zip(img.GetSize(), upperbound_index)])
    lowerbound_pad = tuple([TRUNCATION_ERROR + int(max(0, -lb_idx))
                           for lb_idx in lowerbound_index])
    
    if verbose:
        if np.any(upperbound_pad) or np.any(lowerbound_pad):
            print(f'Padding Required to extract {voxel_size} lower index padding {lowerbound_pad} upper index padding {upperbound_pad}')
            extracted_voxel_size = subtract_tuple(upperbound_index,lowerbound_index)
            print(f'The extracted voxel size is going to be {extracted_voxel_size}')

    # np.testing.assert_array_equal(np.array(voxel_size), np.array(
    #     subtract_tuple(upperbound_index, lowerbound_index)))

    return lowerbound_pad, upperbound_pad


def extract_bbox(img, seg, label_id, physical_size, padding_value, verbose=True):
    """extract ROI from img of given physical size by finding the bounding box with label id from seg image

    Args:
        img (sitk.Image): 
        seg (sitk.Image):
        physical_size (tuple): _description_
        padding_value (scalar): value to fill in for region outside of the image
        verbose (bool, optional): print additional information. Defaults to True.
    """
    assert isinstance(img, sitk.Image)
    assert isinstance(seg, sitk.Image)

    voxel_size = physical_size_to_voxel_size(img, physical_size)

    # execute filter to obtain bounding box and centroid of given segmentation label
    fltr = sitk.LabelShapeStatisticsImageFilter()
    fltr.Execute(seg)

    labels = fltr.GetLabels()

    # make sure the label being asked for exists in the segmentation map
    assert label_id in labels

    # pad around Bounding Box centroid to attain given voxel size
    centroid_index = img.TransformPhysicalPointToIndex(
        fltr.GetCentroid(label_id))
    lb, ub = required_padding(img, voxel_size, centroid_index, verbose=True)
    padded_img: sitk.Image = sitk.ConstantPad(img, lb, ub, padding_value)

    # find the index of the centroid in the padded image
    padded_centroid_index = padded_img.TransformPhysicalPointToIndex(
        fltr.GetCentroid(label_id))

    # get the start of the ROI
    roi_start_index = ROI_centroid_index_to_start_index(
        padded_centroid_index, voxel_size)
    ROI: sitk.Image = sitk.RegionOfInterest(
        padded_img, voxel_size, roi_start_index)

    if verbose:
        print(f'Label Bounding Box: {fltr.GetBoundingBox(label_id)}')
        print(
            f'Coordinates of Segmentation Centroid {img.TransformPhysicalPointToIndex(fltr.GetCentroid(label_id))}')

    return ROI


def extract_around_centroid_v2(img, physical_size, centroid_index, extraction_ratio: dict, padding_value, verbose=True,logger:Optional[Logger]=None):
    """extract ROI from img of given physical size at a given ratio w.r.t the centroid_index

    Args:
        img (sitk.Image): Image
        physical_size (tuple): size of volume in mm
        centroid_index (tuple): index of vertebral body centroid
        extraction_ratio (dict): {'P':0.33, 'L':0.5,'S': 0.5}
        padding_value (scalar): value to fill in for region outside of img
        verbose (bool, optional): print diagnostic info. Defaults to True.
    
    postcondition:
        The actual extracted voxel tuple can be less by 1 voxel due to truncation error.
        This is due to transformation back and forth between physical coordinates and index coordinates.
    """
    assert isinstance(img, sitk.Image)
    voxel_size = physical_size_to_voxel_size(img, physical_size)


    lb, ub = required_padding_v2(
        img, voxel_size, centroid_index, extraction_ratio,verbose=verbose)
    lb_padded = add_tuple(lb,(50,)*3)
    ub_padded = add_tuple(ub,(50,)*3) # the exact padding can be off due to floating point ops, hence add safety padding
    padded_img: sitk.Image = sitk.ConstantPad(img, lb_padded, ub_padded, padding_value)


    # find the index of the centrod in the padded image
    original_centroid_coords = img.TransformContinuousIndexToPhysicalPoint(
        centroid_index)
    padded_centroid_index = padded_img.TransformPhysicalPointToIndex(
        original_centroid_coords)

    # get the start of the ROI
    extraction_tuple = dict_to_tuple(
        get_orientation_code_itk(img), extraction_ratio)

    roi_start_index = ROI_centroid_index_to_start_index_v2(
        padded_centroid_index, voxel_size, extraction_tuple)

    if logger:
        # check ROI outside the largest possible region
        roi_outside = [True if elem < 0 else False for elem in subtract_tuple(padded_img.GetSize(), add_tuple(roi_start_index,voxel_size))]
        if np.any(roi_outside):
            logger.debug(f'possibly requested ROI outside of region: origin {roi_start_index} to extract {voxel_size} from padded_img size {padded_img.GetSize()} by {subtract_tuple(padded_img.GetSize(), add_tuple(roi_start_index,voxel_size))}')

    ROI: sitk.Image = sitk.RegionOfInterest(
        padded_img, voxel_size, roi_start_index)

    ROI_vertebra_centroid_index = ROI.TransformPhysicalPointToContinuousIndex(
        original_centroid_coords)

    heatmap = generate_gaussian_heatmap(ROI_vertebra_centroid_index, ROI)
    if verbose:
        print(f'Vertebra centroid in ROI Index{ROI_vertebra_centroid_index}')

    return ROI, heatmap


def generate_gaussian_heatmap(centroid_index, reference_image, sigma=5):
    """Generate a Centroid Landmark Image represented by a Gaussian at the centroid index with same physical attributes as the reference image

    from https://github.com/christianpayer/MedicalDataAugmentationTool/blob/34e3723397ac5b343f14ec0a8ee49f792e13aeca/utils/landmark/heatmap_image_generator.py

    Args:
        centroid_index (tuple:int): 
        volume_size (tuple): _description_
        reference_image (sitk.Image): _description_
    """
    img_sz = reference_image.GetSize()
    img_thickness = reference_image.GetSpacing()
    iso_img_sz = multiply_tuple(img_sz, img_thickness)
    iso_img_sz = list(map(int, iso_img_sz))

    # flip point from [x,y,z] to [z,y,x]
    centroid_index = list(map(int, centroid_index))
    flipped_coords = np.flip(centroid_index, 0)
    flipped_image_thickness = np.flip(img_thickness, 0)
    dy, dx, dz = np.meshgrid(range(iso_img_sz[1]), range(
        iso_img_sz[0]), range(iso_img_sz[2]))

    x_diff = dx - flipped_coords[0] * flipped_image_thickness[0]
    y_diff = dy - flipped_coords[1] * flipped_image_thickness[1]
    z_diff = dz - flipped_coords[2] * flipped_image_thickness[2]

    squared_distances = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
    heatmap = min(iso_img_sz) * \
        np.exp(-squared_distances / (2*math.pow(sigma, 2)))

    heatmap_sitk = sitk.GetImageFromArray(heatmap)
    set_image_metadata(heatmap_sitk, origin=reference_image.GetOrigin(
    ), direction=reference_image.GetDirection(), spacing=(1, 1, 1))
    heatmap_sitk = sitk.Resample(heatmap_sitk, reference_image)

    return heatmap_sitk





def extract_around_centroid(img, physical_size, centroid_index, padding_value, verbose=True):
    """extracts a simpleitk image of a given voxel size around centroid

    Args:
        img (SimpleITK.Image): image
        voxel_size (tuple): size of the volume in voxel units
        centroid (tuple): coordinates of centroid of the volume to be extracted in voxel units
        padding (tuple): add padding around img to avoid 
    """
    assert isinstance(img, sitk.Image)
    voxel_size = physical_size_to_voxel_size(img, physical_size)
    lb, ub = required_padding(img, voxel_size, centroid_index)

    # we add padding equal to half the size of physical volume to be extracted
    # so that we do not get RuntimeError: Exception thrown in SimpleITK RegionOfInterest
    # Requested region is (at least partially) outside the largest possible region.
    padded_img: sitk.Image = sitk.ConstantPad(img, lb, ub, padding_value)

    # find the index of the centroid in the padded image
    original_centroid_coords = img.TransformContinuousIndexToPhysicalPoint(
        centroid_index)
    padded_centroid_index = padded_img.TransformPhysicalPointToIndex(
        original_centroid_coords)

    # get the start of the ROI
    roi_start_index = ROI_centroid_index_to_start_index(
        padded_centroid_index, voxel_size)

    ROI = sitk.RegionOfInterest(padded_img, voxel_size, roi_start_index)

    if verbose:
        print(f'Centroid (in world coordinates {original_centroid_coords}')
        print(
            f'extracted {ROI.GetSize()} voxels starting at index {roi_start_index}')

    return ROI


def combine_segmentations(imgs: List[sitk.Image], ref_img: sitk.Image, fill_label=1):
    """Combine multiple segmentation images into a single segmentation image.

    Precondition:
        segmentation masks do not overlap
        
    Postcondition:
        a new segmentation images is returned where voxels are filled with fill_label
        if the voxel position is labelled in one of the segmentation image
    """
    # Staying within simpleitk was not always possible
    #box = seg[:,:,:] > 0
    #new_seg =  sitk.Paste(new_seg,seg,sourceIndex=bbox.TransformPhysicalToIndex(bbox.GetOrigin()),
    #           destIndex = new_seg.TransformPhysicalToIndex(bbox.GetOrigin()),
    #           sourceSize=bbox.GetSize())
    #

    new_seg = np.zeros_like(sitk.GetArrayFromImage(ref_img))

    for seg in imgs:
        single_seg = sitk.GetArrayViewFromImage(seg)
        new_seg[single_seg > 0.5] = fill_label
    
    img_out = sitk.GetImageFromArray(new_seg)
    img_out.CopyInformation(ref_img)
    return img_out


if __name__ == '__main__':
    from xrayto3d_preprocess import read_image, write_image, load_centroids

    centroid_jsonpath = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse835/sub-verse835_dir-iso_seg-subreg_ctd.json'
    ct_path = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse835/sub-verse835_dir-iso_ct.nii.gz'
    seg_path = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse835/sub-verse835_dir-iso_seg-vert_msk.nii.gz'
    out_img_path = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse835/vertebra/sub-verse835_dir-iso_seg-subreg_vertebra_5_ctv2.nii.gz'
    centroid_heatmap_path = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse835/vertebra/sub-verse835_dir-iso_seg-subreg_vertebra_5_ct-heatmap.nii.gz'

    # centroid_jsonpath = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse572/sub-verse572_dir-sag_seg-subreg_ctd.json'
    # ct_path = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse572/sub-verse572_dir-sag_ct.nii.gz'
    # out_path = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse572/vertebra/sub-verse572_dir-sag_vertebra-5_ct.nii.gz'
    # centroid_heatmap_path = '2D-3D-Reconstruction-Datasets/verse20/BIDS/sub-verse572/vertebra/sub-verse572_dir-sag_vertebra-5_ct-heatmap.nii.gz'

    ct_path = '2D-3D-Reconstruction-Datasets/totalsegmentor/BIDS/example_ct.nii.gz'
    seg_path = '2D-3D-Reconstruction-Datasets/totalsegmentor/BIDS/example_seg_fast/vertebrae_L4.nii.gz'
    out_img_path = '2D-3D-Reconstruction-Datasets/totalsegmentor/BIDS/vertebra/example_vertebra-l4_ct.nii.gz'
    out_seg_path = '2D-3D-Reconstruction-Datasets/totalsegmentor/BIDS/vertebra/example_vertebra-l4_seg-vert_msk.nii.gz'

    img = read_image(ct_path)
    seg = read_image(seg_path)
    ctd = load_centroids(centroid_jsonpath)

    vb_id, *centroid = ctd[5]

    # required_padding_v2(img, (100,100,100),centroid,{'L': 0.5, 'A': 0.5, 'S' :0.5})
    # ROI,centroid_heatmap = extract_around_centroid_v2(img, (96,96,96),centroid,{'L': 0.5, 'A': 0.7, 'S' :0.5},-1024)
    # write_image(centroid_heatmap, centroid_heatmap_path)

    ROI = extract_bbox(img, seg, label_id=1, physical_size=(
        96, 96, 96), padding_value=-1024, verbose=True)
    write_image(ROI, out_img_path)
    ROI = extract_bbox(seg, seg, label_id=1, physical_size=(
        96, 96, 96), padding_value=0, verbose=True)
    write_image(ROI, out_seg_path)

    rib_paths = '2D-3D-Reconstruction-Datasets/totalsegmentor/BIDS/example_seg_fast/rib_{rib_side}_{rib_id}.nii.gz'
    rib_out_path = '2D-3D-Reconstruction-Datasets/totalsegmentor/BIDS/vertebra/example_rib_seg-mask.nii.gz'

    rib_filenames = list(rib_paths.format(rib_side=j, rib_id=i)
                         for i in range(1, 12) for j in ('left', 'right'))
    images = list(map(read_image, rib_filenames))
    fused_seg = combine_segmentations(images, ref_img=images[0])
    write_image(fused_seg, rib_out_path)
