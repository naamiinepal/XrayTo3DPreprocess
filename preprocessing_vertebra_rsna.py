from xrayto3d_preprocess import *
import numpy as np
from multiprocessing import Pool
import os

def process_subject(subject_id, ct_path, seg_path, config,output_path_template):
    # read inputs
    ct = read_image(ct_path)
    seg = read_image(seg_path)



    logger.debug(f'Image Size {ct.GetSize()} Spacing {np.around(ct.GetSpacing(),3)}')

    # extract ROI and orient to particular orientation
    roi_properties = config['ROI_properties']
    size = (roi_properties['size'],)*ct.GetDimension()

    if get_orientation_code_itk(ct) != roi_properties['axcode']:
        ct = reorient_to(ct,axcodes_to=roi_properties['axcode'])

    if get_orientation_code_itk(seg) != roi_properties['axcode']:
        seg = reorient_to(seg,axcodes_to=roi_properties['axcode'])

    stats = get_segmentation_stats(seg)

    vb_labels = get_segmentation_labels(seg)
    for vb_id in vb_labels:
        logger.debug(f'Vertebra {vb_id}')
        if stats.GetNumberOfPixelsOnBorder(vb_id) > 0:
            logger.debug(f'{vb_id} Pixels on border {stats.GetNumberOfPixelsOnBorder(vb_id)}')
            continue
        # extract ROI and orient to particular orientation
        # ct_roi = extract_bbox(ct,seg,vb_id,physical_size=size,padding_value=roi_properties['ct_padding'])
        centroid_index = ct.TransformPhysicalPointToIndex(stats.GetCentroid(vb_id))
        logger.debug(f'Extraction ratio {config["ROI_properties"]["extraction_ratio"]}')
        ct_roi, centroid_heatmap = extract_ROI(config['ROI_properties'],ct,vb_id,centroid_index,ImageType.Image)
        out_ct_path = generate_path('ct_roi','vert_ct',vb_id,subject_id,output_path_template,config)
        write_image(ct_roi,out_ct_path)

        
        # seg_roi = extract_bbox(seg,seg,vb_id,physical_size=size,padding_value=roi_properties['seg_padding'])
        seg_roi, _  = extract_ROI(config['ROI_properties'],seg,vb_id,centroid_index,ImageType.Segmentation)
        seg_roi = keep_only_label(seg_roi,vb_id)
        seg_roi = get_largest_connected_component(seg_roi)


        out_seg_path = generate_path('seg_roi','vert_seg',vb_id, subject_id, output_path_template,config)
        write_image(seg_roi, out_seg_path)

        out_centroid_path = generate_path('centroid','vert_centroid',vb_id,subject_id,output_path_template,config)
        write_image(centroid_heatmap,out_centroid_path)

        if config['ROI_properties']['drr_from_ct_mask']:
            out_dir = 'xray_from_ctmask'
        elif config['ROI_properties']['drr_from_mask']:
            out_dir = 'xray_from_mask'
        else:
            out_dir = 'xray_from_ct'

        out_xray_ap_path = generate_path(out_dir,'vert_xray_ap',vb_id,subject_id,output_path_template,config)
        generate_xray(out_ct_path, ProjectionType.ap, seg_roi, config['xray_pose'], out_xray_ap_path)

        out_xray_lat_path = generate_path(out_dir,'vert_xray_lat',vb_id,subject_id,output_path_template,config)
        generate_xray(out_ct_path, ProjectionType.lat, seg_roi, config['xray_pose'], out_xray_lat_path)


        out_ctd_xray_ap_path = generate_path(out_dir,'vert_centroid_xray_ap',vb_id,subject_id,output_path_template,config) 
        generate_xray(out_centroid_path, ProjectionType.ap, centroid_heatmap, config['xray_pose'], out_ctd_xray_ap_path)

        out_ctd_xray_lat_path = generate_path(out_dir,'vert_centroid_xray_lat',vb_id,subject_id,output_path_template,config)
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
    out_path = output_path_template.format(output_type=out_dirs[sub_dir],output_name=filename)
    return out_path

def create_directories(out_path_template, config):
    for key, out_dir in config['out_directories'].items():
        Path(out_path_template.format(output_type=out_dir)).mkdir(exist_ok=True,parents=True)

def process_vertebra_subject_helper(subject_id:str):
    logger.debug(f'{subject_id}')
    # define paths
    input_fileformat = config['filename_convention']['input']

    subject_basepath = config['subjects']['subject_basepath']

    ct_path = Path(subject_basepath)/subject_id/input_fileformat['ct']
    seg_path = Path(subject_basepath)/subject_id/input_fileformat['seg']

    OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'
    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    create_directories(OUT_DIR_TEMPLATE,config)
    process_subject(subject_id,ct_path,seg_path,config,OUT_PATH_TEMPLATE)

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
    if args.dataset:
        logger.debug(f'args.dataset {args.dataset}')
    logger.debug(f'Configuration {config}')

    # define paths
    input_fileformat = config['filename_convention']['input']
    output_fileformat = config['filename_convention']['output']

    subject_basepath = config['subjects']['subject_basepath']

    subject_list = pd.read_csv(config['subjects']['subject_list'],header=None).to_numpy().flatten()
    
    

    logger.debug(f'found {len(subject_list)} subjects')
    logger.debug(subject_list)

    num_workers = os.cpu_count()
    def initialize_config_for_all_workers():
        global config
        config = read_config_and_load_components(args.config_file)

    with Pool(processes=num_workers, initializer=initialize_config_for_all_workers) as p:
        results = tqdm(p.map(process_vertebra_subject_helper,sorted(subject_list)),total=len(subject_list))
