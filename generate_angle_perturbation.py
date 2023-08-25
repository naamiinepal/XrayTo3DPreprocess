from multiprocessing import Pool
from pathlib import Path
import numpy as np
from xrayto3d_preprocess import read_config_and_load_components, get_stem, get_logger, generate_xray, generate_perturbed_xray, ProjectionType


def generate_angle_perturbation(subject_id:str):
    "create required subdirs and generate perturbed xrays"
    logger.debug(f'{subject_id}')
    # define paths
    input_fileformat = config['filename_convention']['input']
    subject_basepath = config['subjects']['subject_basepath']

    ct_path = Path(subject_basepath)/subject_id/input_fileformat['ct']

    seg_path = Path(subject_basepath)/subject_id/input_fileformat['seg']
    OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'

    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    OUT_LAT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{perturbation_angle}}/{{output_name}}'

    Path(OUT_DIR_TEMPLATE.format(output_type='xray_from_ct_angle_perturbation')).mkdir(exist_ok=True,parents=True)

    process_subject(subject_id, config,OUT_PATH_TEMPLATE, OUT_LAT_PATH_TEMPLATE,perturbation_angles=[1,2,5,10])

def process_subject(subject_id, config, out_path_template, out_xray_path_template,perturbation_angles):
    """generate perturbed x-ray for corresponding ct"""
    out_ct_path = generate_path('ct_roi','ct_roi',subject_id,out_path_template,config)
    


     
    for ptb_angle in perturbation_angles:
        out_xray_ap_path = generate_perturbation_path('xray_from_ct_angle_perturbation','xray_ap',subject_id,ptb_angle,out_path_template, config)

        logger.debug(f'xray_ap {out_xray_ap_path}')
        # AP view is generated as is
        generate_xray(out_ct_path,ProjectionType.AP,None,config['xray_pose'],out_xray_ap_path )

        out_xray_lat_path = generate_perturbation_path("xray_from_ct_angle_perturbation","xray_lat",subject_id,ptb_angle,out_xray_path_template,config)
        logger.debug(f'xray_lat {out_xray_lat_path}')
        # LAT view is perturbed by various angles
        generate_perturbed_xray(out_ct_path,ProjectionType.LAT,config['xray_pose'],out_xray_lat_path,ptb_angle)

def generate_path(sub_dir: str, name: str, subject_id, output_path_template, config):
    """xray_ap:"{id}_hip-ap.png -> img0001_hip-ap.png"""
    output_fileformat = config["filename_convention"]["output"]
    out_dirs = config["out_directories"]
    filename = output_fileformat[name].format(id=subject_id)
    out_path = output_path_template.format(
        output_type=out_dirs[sub_dir], output_name=filename
    )
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    return out_path

def generate_perturbation_path(sub_dir:str, name:str, subject_id, perturbation_angle, output_path_template, config):
    """xray_ap:"{id_hip-ap.png} -> img001_hip-ap.png"""
    output_fileformat = config['filename_convention']['output']
    out_dirs = config['out_directories']
    filename = output_fileformat[name].format(id=subject_id)
    out_path = output_path_template.format(output_type=out_dirs[sub_dir],perturbation_angle=perturbation_angle,output_name=filename)
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    return out_path

if __name__ == '__main__':
    import argparse

    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument('--num_workers',default=4,type=int)

    args = parser.parse_args()
    config = read_config_and_load_components(args.config_file)
    print(config)

    subject_list = pd.read_csv(config['subjects']['subject_list'],header=None).to_numpy().flatten()

    # create logger
    dataset_name = get_stem(args.config_file)+'_'+'angle_perturbation'
    logger = get_logger(dataset_name)

    logger.debug(f'Generating dataset {dataset_name}')
    logger.debug(f'Configuration {config}')
    logger.debug(f'found {len(subject_list)} subjects')
    logger.debug(subject_list)

    num_workers = args.num_workers
    def initialize_config_for_all_workers():
        """
        passed to multiprocessing threads for all of them to
        be able to access global configuration
        """
        global config
        config = read_config_and_load_components(args.config_file)

    with Pool(processes=num_workers,initializer=initialize_config_for_all_workers) as p:
        results = tqdm(p.map(generate_angle_perturbation, sorted(subject_list)),
                       total=len(subject_list))

