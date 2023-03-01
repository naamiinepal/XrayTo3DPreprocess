from xrayto3d_preprocess import *
import numpy as np
from multiprocessing import Pool
import os


def process_subject(subject_id, ct_path, seg_path, config, output_path_template):
    
    ct = read_image(ct_path)
    seg = read_image(seg_path)

    seg = change_label(seg,{1:1,2:1,3:1}) # combine left and right hip and sacrum into same label
    seg = keep_only_label(seg,label_id=1) # remove vertebra

    ct_mask = mask_ct_with_seg(ct,seg)

    logger.debug(f'Image Size {ct.GetSize()} Spacing {np.around(ct.GetSpacing(),3)}')    

    # extract ROI and orient to particular orientation
    roi_properties = config['ROI_properties']
    size = (roi_properties['size'],)*ct.GetDimension()
    
    labels = get_segmentation_labels(seg)
    # some scans may not have required anatomy labels
    if 1 not in labels:
        return 
    ct_roi = extract_bbox(ct,seg,label_id=1,physical_size=size,padding_value=roi_properties['ct_padding'])

    if get_orientation_code_itk(ct_roi) != roi_properties['axcode']:
        ct_roi = reorient_to(ct_roi,axcodes_to=roi_properties['axcode'])
    out_ct_path = generate_path('ct_roi','ct_roi',subject_id,output_path_template,config)
    write_image(ct_roi,out_ct_path)

    seg_roi = extract_bbox(seg,seg,label_id=1,physical_size=size,padding_value=roi_properties['seg_padding'])
    if get_orientation_code_itk(seg_roi) != roi_properties['axcode']:
        seg_roi = reorient_to(seg_roi,axcodes_to=roi_properties['axcode'])

    
    out_seg_path = generate_path('seg_roi','seg_roi',subject_id,output_path_template,config)
    write_image(seg_roi,out_seg_path)

    ct_mask_roi = extract_bbox(ct_mask,seg,label_id=1,physical_size=size,padding_value=roi_properties['seg_padding'])
    if get_orientation_code_itk(ct_mask_roi) != roi_properties['axcode']:
        ct_mask_roi = reorient_to(ct_mask_roi,axcodes_to=roi_properties['axcode'])    
    out_ct_mask_path = generate_path('ct_mask_roi','ct_mask_roi',subject_id,output_path_template,config)
    write_image(ct_mask_roi,out_ct_mask_path)

    out_xray_ap_path = generate_path('xray_from_ct','xray_ap',subject_id,output_path_template,config)
    generate_xray(out_ct_path, ProjectionType.ap, seg_roi, config['xray_pose'], out_xray_ap_path)

    out_xray_lat_path = generate_path('xray_from_ct','xray_lat',subject_id,output_path_template,config)
    generate_xray(out_ct_path, ProjectionType.lat, seg_roi, config['xray_pose'], out_xray_lat_path)

    out_xray_ap_path = generate_path('xray_from_ctmask','xray_mask_ap',subject_id,output_path_template,config)
    generate_xray(out_ct_mask_path, ProjectionType.ap, seg_roi, config['xray_pose'], out_xray_ap_path)

    out_xray_lat_path = generate_path('xray_from_ctmask','xray_mask_lat',subject_id,output_path_template,config)
    generate_xray(out_ct_mask_path, ProjectionType.lat, seg_roi, config['xray_pose'], out_xray_lat_path)
    
def create_directories(out_path_template, config):
    for key, out_dir in config['out_directories'].items():
        Path(out_path_template.format(output_type=out_dir)).mkdir(exist_ok=True,parents=True)


def generate_path(sub_dir:str, name:str, subject_id, output_path_template, config):
    output_fileformat = config['filename_convention']['output']
    out_dirs = config['out_directories']    
    filename = output_fileformat[name].format(id=subject_id)
    logger.debug(filename)
    out_path = output_path_template.format(output_type=out_dirs[sub_dir],output_name=filename)
    return out_path

def process_totalsegmentor_subject_helper(subject_id:str):
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
    
    args  = parser.parse_args()
    config = read_config_and_load_components(args.config_file)

    # create logger
    dataset_name = get_stem(args.config_file)
    logger = get_logger(dataset_name)

    logger.debug(f'Generating dataset {dataset_name}')
    logger.debug(f'Configuration {config}')


    subject_list = pd.read_csv(config['subjects']['subject_list'],header=None).to_numpy().flatten()

    logger.debug(f'found {len(subject_list)} subjects')
    logger.debug(subject_list)

    num_workers = os.cpu_count()
    # num_workers = 1
    def initialize_config_for_all_workers():
        global config
        config = read_config_and_load_components(args.config_file)

    with Pool(processes=num_workers,initializer=initialize_config_for_all_workers) as p:
        results = tqdm(p.map(process_totalsegmentor_subject_helper,sorted(subject_list)),total=len(subject_list))
        print('done')    