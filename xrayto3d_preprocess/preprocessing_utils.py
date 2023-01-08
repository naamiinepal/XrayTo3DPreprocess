from .ioutils import read_image, write_image
import SimpleITK as sitk
from typing  import Sequence, Dict
import os
from .sitk_utils import reorient_to,simulate_projection,keep_only_label
from .enumutils import ImageType, ProjectionType
from pathlib import Path
from .metadata_utils import get_orientation_code_itk
from .roi_utils import extract_around_centroid_v2
from shutil import which

def command_exists(command_name):
    """check if a command exists or is in path"""
    return which(command_name)

def get_DRR_command():
    possible_commands = ['TwoProjectionRegistrationTestDriver GetDRRSiddonJacobsRayTracing','DRRSiddonJacobs']
    for cmd in possible_commands:
        if command_exists(cmd.split(' ')[0]):
            return cmd

def get_DRRSiddonJacobs_Command_string(input_filepath, output_filepath, orientation, config):
    # DRRSiddonJacobs has to be in path
    res = config['res']
    size = config['size']
    drr_command_executable = get_DRR_command()
    rx, ry, rz = config[orientation]['rx'], config[orientation]['ry'], config[orientation]['rz']
    command = f'{drr_command_executable} {input_filepath} -o {output_filepath} -rx {rx} -ry {ry} -rz {rz} -res {res} {res} -size {size} {size} > /dev/null 2>&1'

    return command


def save_overlays(img_path, in_overlay_path, out_path):
    img = read_image(img_path)
    overlay_seg = read_image(in_overlay_path)
    overlay = overlay_seg > 0.5  # threshold
    overlayed_img = sitk.LabelOverlay(img, overlay)
    write_image(overlayed_img, out_path)

def spatialnet_reorient(img: sitk.Image, saved_landmark: Sequence):
    # the landmark is supposed to be in physical coordinates
    # take a single centroid and do some quirky stuff
    # see https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe/blob/51569f9680e34b0e6dd74bc33587204cf4b2afdf/verse2020/inference/main_vertebrae_localization.py#L118
    # verse_coords = np.array([coords[1], size[2] * spacing[2] - coords[2], size[0] * spacing[0] - coords[0]])
    # we need to reverse this now to
    size, spacing = img.GetSize(), img.GetSpacing()
    oriented_centroids = (size[0] * spacing[0] - saved_landmark[2],
                          saved_landmark[0],
                          spacing[2] * size[2] - saved_landmark[1])
    return oriented_centroids

def extract_ROI(config: Dict, img: sitk.Image, vertebra_level, vertebra_centroid, imageType: ImageType):
    ROI_physical_size = (config['size'],)*img.GetDimension()
    padding_intensity_value = config['seg_padding'] if imageType == ImageType.Segmentation else config['ct_padding']

    seg_roi, centroid_heatmap = extract_around_centroid_v2(img=img, physical_size=ROI_physical_size,
                                                           centroid_index=vertebra_centroid, extraction_ratio=config['extraction_ratio'], padding_value=padding_intensity_value, verbose=False)
    if imageType == ImageType.Segmentation:
        seg_roi = keep_only_label(seg_roi, vertebra_level)

    # reorient ROI if required
    if get_orientation_code_itk(seg_roi) != config['axcode']:
        seg_roi = reorient_to(seg_roi, axcodes_to=config['axcode'])
        centroid_heatmap = reorient_to(
            centroid_heatmap, axcodes_to=config['axcode'])

    return seg_roi, centroid_heatmap


def generate_xray(input_image_path, projection_type: ProjectionType, mask_roi, config, out_xray_path):
    drr_from_mask = config['drr_from_mask']
    orientation = 'ap' if projection_type == ProjectionType.ap else 'lat'

    if drr_from_mask:
        lat_img = simulate_projection(
            mask_roi, projectiontype=projection_type)
        write_image(lat_img, out_xray_path)
    else:
        lat_command = get_DRRSiddonJacobs_Command_string(
            input_image_path, out_xray_path, orientation=orientation, config=config)
        os.system(lat_command)