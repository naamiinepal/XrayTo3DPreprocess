import os
from typing import Dict, Sequence, Tuple

import SimpleITK as sitk

from .enumutils import ImageType, ProjectionType
from .ioutils import write_image
from .metadata_utils import get_orientation_code_itk
from .misc import get_drrsiddonjacobs_command_string
from .roi_utils import extract_around_centroid_v2
from .sitk_utils import keep_only_label, reorient_to, simulate_parallel_projection


def extract_vertebra_around_vbcentroid(
    config: Dict,
    img: sitk.Image,
    vertebra_level,
    vertebra_centroid,
    image_type: ImageType,
) -> Tuple[sitk.Image, sitk.Image]:
    """return region of interest defined by vertebra centroid and bounding box size (defined in config)
    also, return centroid heatmap"""
    roi_physical_size = (config["size"],) * img.GetDimension()
    padding_intensity_value = (
        config["seg_padding"]
        if image_type == ImageType.SEGMENTATION
        else config["ct_padding"]
    )

    region_of_interest, centroid_heatmap = extract_around_centroid_v2(
        img=img,
        physical_size=roi_physical_size,
        centroid_index=vertebra_centroid,
        extraction_ratio=config["extraction_ratio"],
        padding_value=padding_intensity_value,
        verbose=False,
    )
    if image_type == ImageType.SEGMENTATION:
        region_of_interest = keep_only_label(region_of_interest, vertebra_level)

    # reorient ROI if required
    if get_orientation_code_itk(region_of_interest) != config["axcode"]:
        region_of_interest = reorient_to(
            region_of_interest, axcodes_to=config["axcode"]
        )
        centroid_heatmap = reorient_to(centroid_heatmap, axcodes_to=config["axcode"])

    return region_of_interest, centroid_heatmap


def generate_xray(
    input_image_path, projection_type: ProjectionType, mask_roi, config, out_xray_path
):
    """Generate X-ray from CT
    - use DRRSiddonJacobs if CT
    - simulate projection if Segmentation (DRRSiddonJacobs does not work well for segmentation)
    """
    drr_from_mask = config["drr_from_mask"]
    orientation = "ap" if projection_type == ProjectionType.AP else "lat"

    if drr_from_mask:
        lat_img = simulate_parallel_projection(mask_roi, projectiontype=projection_type)
        write_image(lat_img, out_xray_path)
    else:
        lat_command = get_drrsiddonjacobs_command_string(
            input_image_path, out_xray_path, orientation=orientation, config=config
        )
        os.system(lat_command)


def spatialnet_reorient(img: sitk.Image, saved_landmark: Sequence):
    """
    Do some quirky stuff to bring the output from Vertebra Localization tool (spatialnet) into proper orientation
    the landmark is supposed to be in physical coordinates
    see https://github.com/christianpayer/MedicalDataAugmentationTool-VerSe/blob/51569f9680e34b0e6dd74bc33587204cf4b2afdf/verse2020/inference/main_vertebrae_localization.py#L118
    The code there does:
        verse_coords = np.array([coords[1], size[2] * spacing[2] - coords[2], size[0] * spacing[0] - coords[0]])
    To bring them into our pipeline, we inverse transform
    """
    size, spacing = img.GetSize(), img.GetSpacing()
    oriented_centroids = (
        size[0] * spacing[0] - saved_landmark[2],
        saved_landmark[0],
        spacing[2] * size[2] - saved_landmark[1],
    )
    return oriented_centroids
