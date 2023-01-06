from preprocessing import *
import numpy as np

def process_subject(subject_id, ct_path, seg_path, dataset_name, centroid_path, config,output_path_template):

    # read inputs
    ct = read_image(ct_path)
    seg = read_image(seg_path)
    centroid_orientation, centroids = load_centroids(centroid_path)

    logger.debug(f'Image Size {ct.GetSize()} Spacing {np.around(ct.GetSpacing())}')

    for vb_id, *ctd in centroids:
        logger.debug(f'Vertebra {vb_id}')

        if dataset_name == 'lidc':
            ctd_physical = spatialnet_reorient(seg, ctd)
            ctd = seg.TransformPhysicalPointToIndex(ctd_physical)
        
        # extract ROI and orient to particular orientation
        seg_roi, _ = extract_ROI(config['ROI_properties'],seg,vb_id,ctd,ImageType.Segmentation)
        out_seg_path = generate_path('seg','vert_seg',vb_id, subject_id, output_path_template,config)
        write_image(seg_roi, out_seg_path)

        ct_roi, centroid_heatmap = extract_ROI(config['ROI_properties'],ct,vb_id, ctd, ImageType.Image)
        if config['ROI_properties']['drr_from_ct_mask']:
            ct_roi = sitk.Mask(ct_roi, seg_roi > 0.5)
        out_ct_path = generate_path('ct','vert_ct',vb_id,subject_id,output_path_template,config)
        write_image(ct_roi, out_ct_path)

        out_centroid_path = generate_path('centroid','vert_centroid',vb_id,subject_id,output_path_template,config)
        write_image(centroid_heatmap,out_centroid_path)
        
        out_xray_ap_path = generate_path('xray_from_ct','vert_xray_ap',vb_id,subject_id,output_path_template,config)
        generate_xray(out_ct_path, ProjectionType.ap, seg_roi, config['xray_pose'], out_xray_ap_path)

        out_xray_lat_path = generate_path('xray_from_ct','vert_xray_lat',vb_id,subject_id,output_path_template,config)
        generate_xray(out_ct_path, ProjectionType.lat, seg_roi, config['xray_pose'], out_xray_lat_path)

        out_ctd_xray_ap_path = generate_path('xray_from_ct','vert_centroid_xray_ap',vb_id,subject_id,output_path_template,config) 
        generate_xray(out_centroid_path, ProjectionType.ap, centroid_heatmap, config['xray_pose'], out_ctd_xray_ap_path)

        out_ctd_xray_lat_path = generate_path('xray_from_ct','vert_centroid_xray_lat',vb_id,subject_id,output_path_template,config)
        generate_xray(out_centroid_path, ProjectionType.lat, centroid_heatmap, config['xray_pose'],out_ctd_xray_lat_path)

        # generate visualization overlays
        out_overlay_ap_path = generate_path('overlay','vert_overlay_ap',vb_id,subject_id,output_path_template,config)
        save_overlays(out_xray_ap_path, out_ctd_xray_ap_path,out_overlay_ap_path)
        out_overlay_lat_path = generate_path('overlay','vert_overlay_lat',vb_id,subject_id,output_path_template,config)
        save_overlays(out_xray_lat_path, out_ctd_xray_lat_path, out_overlay_lat_path)

def generate_path(sub_dir:str, name:str, vb_id, subject_id, output_path_template, config):
    output_fileformat = config['filename_convention']['output']
    out_dirs = config['out_directories']    
    filename = output_fileformat[name].format(id=subject_id,vert=vb_id)
    out_seg_path = output_path_template.format(output_type=out_dirs[sub_dir],output_name=filename)
    return out_seg_path

def create_directories(out_path_template, config):
    for key, out_dir in config['out_directories'].items():
        Path(out_path_template.format(output_type=out_dir)).mkdir(exist_ok=True,parents=True)

if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--dataset')

    args = parser.parse_args()

    config = read_config_and_load_components(args.config_file)

    #  create logger
    dataset_name = get_stem(args.config_file)
    logger = get_logger(dataset_name)

    logger.debug(f'Generating dataset {dataset_name}')
    logger.debug(f'Configuration {config}')

    # define paths
    input_fileformat = config['filename_convention']['input']
    output_fileformat = config['filename_convention']['output']

    subject_basepath = config['subjects']['subject_basepath']

    subject_list = pd.read_csv(config['subjects']['subject_list'],header=None).to_numpy().flatten()
    
    logger.debug(f'found {len(subject_list)} subjects')
    logger.debug(subject_list)


    for subject_id in tqdm(subject_list, total=len(subject_list)):
        ct_path = Path(subject_basepath)/subject_id/input_fileformat['ct'].format(id=subject_id)
        seg_path = Path(subject_basepath)/subject_id/input_fileformat['seg'].format(id=subject_id)
        centroid_path = Path(subject_basepath)/subject_id/input_fileformat['ctd'].format(id=subject_id)
        
        OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'
        OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

        logger.debug(OUT_PATH_TEMPLATE)
        create_directories(OUT_DIR_TEMPLATE, config)
        process_subject(subject_id, ct_path, seg_path, args.dataset, centroid_path, config, OUT_PATH_TEMPLATE)