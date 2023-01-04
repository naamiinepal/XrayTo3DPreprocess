from pathlib import Path
import os
from typing import Dict
from preprocessing import read_yaml, strip_item, get_logger, get_stem, load_centroids, read_image, write_image, extract_around_centroid_v2, keep_only_label, change_label, get_orientation_code_itk, reorient_centroids_to, reorient_to,simulate_projection,ProjectionType
import SimpleITK as sitk


def get_DRRSiddonJacobs_Command_string(input_filepath, output_filepath, orientation, config):
    # DRRSiddonJacobs has to be in path
    res = config['res']
    size = config['size']
    rx, ry, rz = config[orientation]['rx'], config[orientation]['ry'], config[orientation]['rz']
    command = f'DRRSiddonJacobs {input_filepath} -o {output_filepath} -rx {rx} -ry {ry} -rz {rz} -res {res} {res} -size {size} {size}'
    return command


def save_overlays(img_path, in_overlay_path, out_path):
    img = read_image(img_path)
    overlay_seg = read_image(in_overlay_path)
    overlay = overlay_seg > 0.5  # threshold
    overlayed_img = sitk.LabelOverlay(img, overlay)
    write_image(overlayed_img, out_path)


def process_subject(ct_path, seg_path, json_path, config: Dict,
                    OUT_VERTEBRA_CT_PATH, OUT_VERTEBRA_SEG_PATH, OUT_VERTEBRA_CENTROID_PATH, OUT_XRAY_PATH, OUT_CENTROID_XRAY_PATH, OUT_CENTROID_XRAY_OVERLAY_PATH):
    centroid_orientation, centroids = load_centroids(json_path)
    centroid_list = [centroid_orientation, *centroids]
    logger.debug(
        f'Source axcode {centroid_orientation} target axcode {config["axcode"]}')

    if isinstance(centroid_orientation, list):
        centroid_orientation = ''.join(centroid_orientation)

    ct = read_image(ct_path)
    seg = read_image(seg_path)

    drr_from_mask = config['drr_from_mask']
    drr_from_ct_mask = config['drr_from_ct_mask']

    logger.debug(
        f'{c} {id} Image Spacing {ct.GetSpacing()} Voxel Size {ct.GetSize()}')
    for vb_id, *ctd in centroids:
        logger.debug(f'Extract vertebra {vb_id}')

        seg_roi, centroid_heatmap = extract_around_centroid_v2(img=seg, physical_size=(config['size'],)*ct.GetDimension(),
                                                               centroid_index=ctd, extraction_ratio=config['extraction_ratio'], padding_value=config['seg_padding'], verbose=False)
        seg_roi = keep_only_label(seg_roi, vb_id)
        if centroid_orientation != config['axcode']:
            seg_roi = reorient_to(seg_roi, axcodes_to=config['axcode'])
        out_seg_path = OUT_VERTEBRA_SEG_PATH.format(id=id, vert=vb_id)
        Path(out_seg_path).parent.mkdir(exist_ok=True, parents=True)
        write_image(seg_roi, out_seg_path)

        ct_roi, centroid_heatmap = extract_around_centroid_v2(img=ct, physical_size=(config['size'],)*ct.GetDimension(),
                                                              centroid_index=ctd, extraction_ratio=config['extraction_ratio'], padding_value=config['ct_padding'], verbose=False, logger=logger)

        out_ct_path = OUT_VERTEBRA_CT_PATH.format(id=id, vert=vb_id)
        Path(out_ct_path).parent.mkdir(exist_ok=True, parents=True)

        if centroid_orientation != config['axcode']:
            ct_roi = reorient_to(ct_roi, axcodes_to=config['axcode'])
        if drr_from_ct_mask:
            ct_roi = sitk.Mask(ct_roi, seg_roi > 0.5)
        write_image(ct_roi, out_ct_path)

        out_centroid_path = OUT_VERTEBRA_CENTROID_PATH.format(
            id=id, vert=vb_id)
        Path(out_centroid_path).parent.mkdir(exist_ok=True, parents=True)
        if centroid_orientation != config['axcode']:
            centroid_heatmap = reorient_to(
                centroid_heatmap, axcodes_to=config['axcode'])
        write_image(centroid_heatmap, out_centroid_path)

        # generate drr
        out_xray_ap_path = OUT_XRAY_PATH.format(
            id=id, vert=vb_id, xray_orientation='ap')
        Path(out_xray_ap_path).parent.mkdir(exist_ok=True, parents=True)
        if drr_from_mask:
            ap_img = simulate_projection(seg_roi,projectiontype=ProjectionType.ap)
            write_image(ap_img, out_xray_ap_path)
        else:
            ap_command = get_DRRSiddonJacobs_Command_string(
                out_ct_path, out_xray_ap_path, orientation='ap', config=config)
            os.system(ap_command)

        out_xray_lat_path = OUT_XRAY_PATH.format(
            id=id, vert=vb_id, xray_orientation='lat')
        Path(out_xray_lat_path).parent.mkdir(exist_ok=True, parents=True)
        if drr_from_mask:
            lat_img = simulate_projection(seg_roi,projectiontype=ProjectionType.lat)
            write_image(lat_img,out_xray_lat_path)
        else:
            lat_command = get_DRRSiddonJacobs_Command_string(
                out_ct_path, out_xray_lat_path, orientation='lat', config=config)
            os.system(lat_command)

        out_ctd_xray_ap_path = OUT_CENTROID_XRAY_PATH.format(
            id=id, vert=vb_id, xray_orientation='ap')
        if drr_from_mask:
            ctd_ap_img = simulate_projection(centroid_heatmap,projectiontype=ProjectionType.ap)
            write_image(ctd_ap_img, out_ctd_xray_ap_path)
        else:
            ctd_ap_command = get_DRRSiddonJacobs_Command_string(
                out_centroid_path, out_ctd_xray_ap_path, orientation='ap', config=config)
            os.system(ctd_ap_command)

        out_ctd_xray_lat_path = OUT_CENTROID_XRAY_PATH.format(
            id=id, vert=vb_id, xray_orientation='lat')
        if drr_from_mask:
            ctd_lat_img = simulate_projection(centroid_heatmap,projectiontype=ProjectionType.lat)
            write_image(ctd_lat_img, out_ctd_xray_lat_path)
        else:
            ctd_lat_command = get_DRRSiddonJacobs_Command_string(
                out_centroid_path, out_ctd_xray_lat_path, orientation='lat', config=config)
            os.system(ctd_lat_command)

        # generate visualization overlays
        # save X-ray centroid overlays
        save_overlays(out_xray_ap_path, out_ctd_xray_ap_path, OUT_CENTROID_XRAY_OVERLAY_PATH.format(
            id=id, vert=vb_id, xray_orientation='ap'))
        save_overlays(out_xray_lat_path, out_ctd_xray_lat_path, OUT_CENTROID_XRAY_OVERLAY_PATH.format(
            id=id, vert=vb_id, xray_orientation='lat'))


if __name__ == '__main__':
    import argparse
    import sys
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')

    args = parser.parse_args()

    config = read_yaml(args.config_file)

    # create logger
    logger_name = get_stem(args.config_file)
    logger = get_logger(logger_name)
    logger.debug(f'Generating dataset {logger_name}')
    logger.debug(f'Configuration {config}')

    # define paths
    INPUT_FILEFORMAT = config['filename_convention']['input']
    SUBJECT_BASEPATH = f'{config["subject_basepath"]}/'+'{id}'
    vertebra_dir_name = config['vertebra']
    ct_roi_dir_name = config['ct_roi']
    seg_roi_dir_name = config['seg_roi']
    centroid_roi_dirname = config['centroid_roi']
    xray_roi_dirname = config['xray_roi']

    OUT_VERTEBRA_CT_PATH = SUBJECT_BASEPATH + \
        f'/{vertebra_dir_name}' + f'/{ct_roi_dir_name}' + \
        '/{id}_vert-{vert}_ct.nii.gz'
    OUT_VERTEBRA_SEG_PATH = SUBJECT_BASEPATH + \
        f'/{vertebra_dir_name}' + f'/{seg_roi_dir_name}' + \
        '/{id}_vert-{vert}-seg-vert_msk.nii.gz'
    OUT_VERTEBRA_CENTROID_PATH = SUBJECT_BASEPATH + \
        f'/{vertebra_dir_name}' + f'/{centroid_roi_dirname}' + \
        '/{id}_vert-{vert}_centroid.nii.gz'
    OUT_XRAY_PATH = SUBJECT_BASEPATH + \
        f'/{vertebra_dir_name}' + f'/{xray_roi_dirname}' + \
        '/{id}_vert-{vert}_{xray_orientation}.png'
    OUT_CENTROID_XRAY_PATH = SUBJECT_BASEPATH + \
        f'/{vertebra_dir_name}' + f'/{xray_roi_dirname}' + \
        '/{id}_vert-{vert}_{xray_orientation}_centroid.png'
    OUT_CENTROID_XRAY_OVERLAY_PATH = SUBJECT_BASEPATH + \
        f'/{vertebra_dir_name}' + f'/{xray_roi_dirname}' + \
        '/{id}_vert-{vert}_{xray_orientation}_centroidoverlay.png'

    with open(config['subject_list'], 'r') as f:
        subject_list = list(map(strip_item, f.readlines()))
        logger.debug(f'found {len(subject_list)} subjects')
        logger.debug(subject_list)

        for c, id in enumerate(tqdm(subject_list, total=len(subject_list))):
            ct_path = Path(SUBJECT_BASEPATH.format(id=id)) / \
                INPUT_FILEFORMAT['ct'].format(id=id)
            seg_path = Path(SUBJECT_BASEPATH.format(id=id)) / \
                INPUT_FILEFORMAT['seg'].format(id=id)
            json_path = Path(SUBJECT_BASEPATH.format(id=id)) / \
                INPUT_FILEFORMAT['ctd'].format(id=id)

            process_subject(ct_path=ct_path, seg_path=seg_path,
                            json_path=json_path, config=config, OUT_VERTEBRA_CT_PATH=OUT_VERTEBRA_CT_PATH, OUT_VERTEBRA_SEG_PATH=OUT_VERTEBRA_SEG_PATH, OUT_VERTEBRA_CENTROID_PATH=OUT_VERTEBRA_CENTROID_PATH, OUT_XRAY_PATH=OUT_XRAY_PATH, OUT_CENTROID_XRAY_PATH=OUT_CENTROID_XRAY_PATH, OUT_CENTROID_XRAY_OVERLAY_PATH=OUT_CENTROID_XRAY_OVERLAY_PATH)
