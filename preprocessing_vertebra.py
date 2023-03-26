import numpy as np
import SimpleITK as sitk
from xrayto3d_preprocess import (
    ImageType,
    ProjectionType,
    extract_vertebra_around_vbcentroid,
    generate_xray,
    get_logger,
    get_stem,
    load_centroids,
    read_config_and_load_components,
    read_image,
    save_overlays,
    spatialnet_reorient,
    write_image,
)


def process_subject(
    subject_id,
    ct_volume_path,
    seg_volume_path,
    dataset_name,
    centroid_path,
    config,
    output_path_template,
):
    """process vertebra dataset with vertebra centroid annotation"""
    # read inputs
    ct_img = read_image(ct_volume_path)
    seg_img = read_image(seg_volume_path)
    _, centroids = load_centroids(centroid_path)

    logger.debug(
        f"Image Size {ct_img.GetSize()} Spacing {np.around(ct_img.GetSpacing(),3)}"
    )

    for vb_id, *ctd in centroids:
        logger.debug(f"Vertebra {vb_id}")

        if dataset_name == "lidc":
            ctd_physical = spatialnet_reorient(seg_img, ctd)
            ctd = seg_img.TransformPhysicalPointToIndex(ctd_physical)

        # extract ROI and orient to particular orientation
        seg_roi, _ = extract_vertebra_around_vbcentroid(
            config["ROI_properties"], seg_img, vb_id, ctd, ImageType.SEGMENTATION
        )
        out_seg_path = generate_path(
            "seg", "vert_seg", vb_id, subject_id, output_path_template, config
        )
        write_image(seg_roi, out_seg_path)

        ct_roi, centroid_heatmap = extract_vertebra_around_vbcentroid(
            config["ROI_properties"], ct_img, vb_id, ctd, ImageType.IMAGE
        )
        if config["ROI_properties"]["drr_from_ct_mask"]:
            ct_roi = sitk.Mask(ct_roi, seg_roi > 0.5)
        out_ct_path = generate_path(
            "ct", "vert_ct", vb_id, subject_id, output_path_template, config
        )
        write_image(ct_roi, out_ct_path)

        out_centroid_path = generate_path(
            "centroid", "vert_centroid", vb_id, subject_id, output_path_template, config
        )
        write_image(centroid_heatmap, out_centroid_path)

        if config["ROI_properties"]["drr_from_ct_mask"]:
            out_dir = "xray_from_ctmask"
        elif config["ROI_properties"]["drr_from_mask"]:
            out_dir = "xray_from_mask"
        else:
            out_dir = "xray_from_ct"
        out_xray_ap_path = generate_path(
            out_dir, "vert_xray_ap", vb_id, subject_id, output_path_template, config
        )
        generate_xray(
            out_ct_path,
            ProjectionType.AP,
            seg_roi,
            config["xray_pose"],
            out_xray_ap_path,
        )

        out_xray_lat_path = generate_path(
            out_dir, "vert_xray_lat", vb_id, subject_id, output_path_template, config
        )
        generate_xray(
            out_ct_path,
            ProjectionType.LAT,
            seg_roi,
            config["xray_pose"],
            out_xray_lat_path,
        )

        out_ctd_xray_ap_path = generate_path(
            out_dir,
            "vert_centroid_xray_ap",
            vb_id,
            subject_id,
            output_path_template,
            config,
        )
        generate_xray(
            out_centroid_path,
            ProjectionType.AP,
            centroid_heatmap,
            config["xray_pose"],
            out_ctd_xray_ap_path,
        )

        out_ctd_xray_lat_path = generate_path(
            out_dir,
            "vert_centroid_xray_lat",
            vb_id,
            subject_id,
            output_path_template,
            config,
        )
        generate_xray(
            out_centroid_path,
            ProjectionType.LAT,
            centroid_heatmap,
            config["xray_pose"],
            out_ctd_xray_lat_path,
        )

        # generate visualization overlays
        out_overlay_ap_path = generate_path(
            "overlay",
            "vert_overlay_ap",
            vb_id,
            subject_id,
            output_path_template,
            config,
        )
        save_overlays(out_xray_ap_path, out_ctd_xray_ap_path, out_overlay_ap_path)
        out_overlay_lat_path = generate_path(
            "overlay",
            "vert_overlay_lat",
            vb_id,
            subject_id,
            output_path_template,
            config,
        )
        save_overlays(out_xray_lat_path, out_ctd_xray_lat_path, out_overlay_lat_path)


def generate_path(
    sub_dir: str, name: str, vb_id, subject_id, output_path_template, config
):
    output_fileformat = config["filename_convention"]["output"]
    out_dirs = config["out_directories"]
    filename = output_fileformat[name].format(id=subject_id, vert=vb_id)
    out_path = output_path_template.format(
        output_type=out_dirs[sub_dir], output_name=filename
    )
    return out_path


def create_directories(out_path_template, config):
    for key, out_dir in config["out_directories"].items():
        Path(out_path_template.format(output_type=out_dir)).mkdir(
            exist_ok=True, parents=True
        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--dataset")

    args = parser.parse_args()

    config = read_config_and_load_components(args.config_file)

    #  create logger
    dataset_name = get_stem(args.config_file)
    logger = get_logger(dataset_name)

    logger.debug(f"Generating dataset {dataset_name}")
    if args.dataset:
        logger.debug(f"args.dataset {args.dataset}")
    logger.debug(f"Configuration {config}")

    # define paths
    input_fileformat = config["filename_convention"]["input"]
    output_fileformat = config["filename_convention"]["output"]

    subject_basepath = config["subjects"]["subject_basepath"]

    subject_list = (
        pd.read_csv(config["subjects"]["subject_list"], header=None)
        .to_numpy()
        .flatten()
    )

    logger.debug(f"found {len(subject_list)} subjects")
    logger.debug(subject_list)

    for subject_id in tqdm(subject_list, total=len(subject_list)):
        logger.debug(f"subject {subject_id}")
        if args.dataset == "verse2020":
            subject_id, input_filename_prefix = subject_id
            ct_path = (
                Path(subject_basepath)
                / subject_id
                / input_fileformat["ct"].format(id=input_filename_prefix)
            )
            seg_path = (
                Path(subject_basepath)
                / subject_id
                / input_fileformat["seg"].format(id=input_filename_prefix)
            )
            centroid_path = (
                Path(subject_basepath)
                / subject_id
                / input_fileformat["ctd"].format(id=input_filename_prefix)
            )
        else:
            ct_path = (
                Path(subject_basepath)
                / subject_id
                / input_fileformat["ct"].format(id=subject_id)
            )
            seg_path = (
                Path(subject_basepath)
                / subject_id
                / input_fileformat["seg"].format(id=subject_id)
            )
            centroid_path = (
                Path(subject_basepath)
                / subject_id
                / input_fileformat["ctd"].format(id=subject_id)
            )

        OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'
        OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

        create_directories(OUT_DIR_TEMPLATE, config)
        process_subject(
            subject_id,
            ct_path,
            seg_path,
            args.dataset,
            centroid_path,
            config,
            OUT_PATH_TEMPLATE,
        )
