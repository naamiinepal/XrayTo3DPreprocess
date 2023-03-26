from multiprocessing import Pool
from pathlib import Path

import numpy as np
from xrayto3d_preprocess import (
    ProjectionType,
    generate_xray,
    get_logger,
    get_orientation_code_itk,
    get_stem,
    read_config_and_load_components,
    read_image,
    reorient_to,
    write_image,
    get_largest_connected_component,
    extract_bbox_topleft,
    mirror_image,
    mask_ct_with_seg,
)


def process_subject(subject_id, ct_path, seg_path, config, output_path_template):
    ct = read_image(ct_path)
    seg = read_image(seg_path)
    seg = get_largest_connected_component(
        seg
    )  # some of the segmentations have islands in irrelevant places

    logger.debug(
        f" {subject_id} Image Size {ct.GetSize()} Spacing {np.around(ct.GetSpacing(),3)}"
    )

    if config["ROI_properties"].is_left == False:
        # flip image
        ct = mirror_image(ct, flip_axes=2)
        seg = mirror_image(seg, flip_axes=2)

    ct_mask = mask_ct_with_seg(ct, seg)

    # extract ROI and orient to particular orientation
    roi_properties = config["ROI_properties"]
    size = (roi_properties["size"],) * ct.GetDimension()

    ct_roi = extract_bbox_topleft(
        ct,
        seg,
        label_id=1,
        physical_size=size,
        padding_value=roi_properties["ct_padding"],
        verbose=False,
    )

    logger.debug(
        f" CT ROI {ct_roi.GetSize()} Spacing {np.around(ct_roi.GetSpacing(),3)}"
    )

    if get_orientation_code_itk(ct_roi) != roi_properties["axcode"]:
        ct_roi = reorient_to(ct_roi, axcodes_to=roi_properties["axcode"])

    seg_roi = extract_bbox_topleft(
        seg,
        seg,
        label_id=1,
        physical_size=size,
        padding_value=roi_properties["seg_padding"],
        verbose=False,
    )
    if get_orientation_code_itk(seg_roi) != roi_properties["axcode"]:
        seg_roi = reorient_to(seg_roi, axcodes_to=roi_properties["axcode"])
    logger.debug(
        f" Seg ROI {seg_roi.GetSize()} Spacing {np.around(seg_roi.GetSpacing(),3)}"
    )

    out_ct_path = generate_path(
        "ct_roi", "ct_roi", subject_id, output_path_template, config
    )
    logger.debug(f"writing ct roi to {out_ct_path}")
    write_image(ct_roi, out_ct_path)

    out_seg_path = generate_path(
        "seg_roi", "seg_roi", subject_id, output_path_template, config
    )
    write_image(seg_roi, out_seg_path)

    ct_mask_roi = extract_bbox_topleft(
        ct_mask,
        seg,
        label_id=1,
        physical_size=size,
        padding_value=roi_properties["ct_padding"],
        verbose=False,
    )
    if get_orientation_code_itk(ct_mask_roi) != roi_properties["axcode"]:
        ct_mask_roi = reorient_to(ct_mask_roi, axcodes_to=roi_properties["axcode"])
    out_ct_mask_path = generate_path(
        "ct_mask_roi", "ct_mask_roi", subject_id, output_path_template, config
    )
    write_image(ct_mask_roi, out_ct_mask_path)

    out_xray_ap_path = generate_path(
        "xray_from_ct", "xray_ap", subject_id, output_path_template, config
    )
    generate_xray(
        out_ct_path, ProjectionType.AP, seg_roi, config["xray_pose"], out_xray_ap_path
    )

    out_xray_lat_path = generate_path(
        "xray_from_ct", "xray_lat", subject_id, output_path_template, config
    )
    generate_xray(
        out_ct_path, ProjectionType.LAT, seg_roi, config["xray_pose"], out_xray_lat_path
    )

    out_xray_ap_path = generate_path(
        "xray_from_ctmask", "xray_mask_ap", subject_id, output_path_template, config
    )
    generate_xray(
        out_ct_mask_path,
        ProjectionType.AP,
        seg_roi,
        config["xray_pose"],
        out_xray_ap_path,
    )

    out_xray_lat_path = generate_path(
        "xray_from_ctmask", "xray_mask_lat", subject_id, output_path_template, config
    )
    generate_xray(
        out_ct_mask_path,
        ProjectionType.LAT,
        seg_roi,
        config["xray_pose"],
        out_xray_lat_path,
    )


def create_directories(out_path_template, config):
    for key, out_dir in config["out_directories"].items():
        Path(out_path_template.format(output_type=out_dir)).mkdir(
            exist_ok=True, parents=True
        )


def process_total_segmentor_subject_helper(subject_id: str, verbose=False):
    # define paths
    input_fileformat = config["filename_convention"]["input"]

    subject_basepath = config["subjects"]["subject_basepath"]

    ct_path = Path(subject_basepath) / subject_id / input_fileformat["ct"]
    seg_path = Path(subject_basepath) / subject_id / input_fileformat["seg"]

    logger.debug(f"reading ct and seg from {ct_path} {seg_path}")
    OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'
    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    create_directories(OUT_DIR_TEMPLATE, config)
    process_subject(subject_id, ct_path, seg_path, config, OUT_PATH_TEMPLATE)


def generate_path(sub_dir: str, name: str, subject_id, output_path_template, config):
    output_fileformat = config["filename_convention"]["output"]
    out_dirs = config["out_directories"]
    filename = output_fileformat[name].format(id=subject_id)
    out_path = output_path_template.format(
        output_type=out_dirs[sub_dir], output_name=filename
    )
    return out_path


if __name__ == "__main__":
    import argparse

    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--num_workers", default=4, type=int)

    args = parser.parse_args()
    config = read_config_and_load_components(args.config_file)

    # create logger
    dataset_name = get_stem(args.config_file)
    logger = get_logger(dataset_name)

    logger.debug(f"Generating dataset {dataset_name}")
    logger.debug(f"Configuration {config}")

    subject_list = (
        pd.read_csv(config["subjects"]["subject_list"], header=None)
        .to_numpy()
        .flatten()
    )

    logger.debug(f"found {len(subject_list)} subjects")
    logger.debug(subject_list)

    num_workers = args.num_workers

    def initialize_config_for_all_workers():
        global config
        config = read_config_and_load_components(args.config_file)

    with Pool(
        processes=num_workers, initializer=initialize_config_for_all_workers
    ) as p:
        results = tqdm(
            p.map(process_total_segmentor_subject_helper, sorted(subject_list))
        )
        print("done")
