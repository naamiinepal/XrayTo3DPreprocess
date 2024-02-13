from xrayto3d_preprocess.enumutils import ProjectionType
from xrayto3d_preprocess.ioutils import get_logger, load_centroids
from xrayto3d_preprocess.pathutils import get_stem
from pathlib import Path

from xrayto3d_preprocess.preprocessing_utils import generate_perturbed_xray
generate_augmented_xray = generate_perturbed_xray
from xrayto3d_preprocess.ioutils import read_image,write_image
from xrayto3d_preprocess.sitk_utils import rotate_about_image_center,get_interpolator
from xrayto3d_preprocess.enumutils import ImageType

def generate_angle_augmented_image(z_angle,in_image_path,image_type:ImageType,out_path):
    image = read_image(in_image_path)
    interpolator = get_interpolator('nearest') if image_type == ImageType.SEGMENTATION else get_interpolator('linear')
    rotated_image = rotate_about_image_center(image,rx=0,ry=0,rz=360-z_angle,interpolator=interpolator)
    write_image(rotated_image,out_path,pixeltype=image.GetPixelID())

def generate_360_augmentation(subject_id:str):
    "create required subdirs and generate rotated augmentations"
    logger.debug(f'{subject_id}')

    # define paths
    input_fileformat = config['filename_convention']['input']
    subject_basepath = config['subjects']['subject_basepath']

    # ct_path = Path(subject_basepath)/subject_id/input_fileformat['ct']
    # seg_path = Path(subject_basepath)/subject_id/input_fileformat['seg']
    centroid_path = Path(subject_basepath)/subject_id/input_fileformat['ctd'].format(id=subject_id)

    OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'

    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    # create augment subdirectories
    Path(OUT_DIR_TEMPLATE.format(output_type='xray_from_ct_augment')).mkdir(exist_ok=True,parents=True)

    Path(OUT_DIR_TEMPLATE.format(output_type='seg_roi_augment')).mkdir(exist_ok=True,parents=True)

    process_subject(subject_id, centroid_path, config, OUT_PATH_TEMPLATE)

def process_subject(subject_id, centroid_path, config, out_path_template):
    "generate augmented x-ray from ct"

    _, centroids = load_centroids(centroid_path)

    for vert_id, *ctd in centroids:
        logger.debug(f'Vertebra {vert_id}')

        out_ct_path = generate_path('ct_roi', 'vert_ct',subject_id,vert_id,out_path_template,config)
        out_seg_path = generate_path('seg_roi','vert_seg',subject_id,vert_id,out_path_template,config)

        logger.debug(out_ct_path)
        logger.debug(out_seg_path)

        for augment_angle in range(0,360,5):
            out_seg_aug_path = generate_augment_path('seg_roi_augment','vert_seg_aug',subject_id,vert_id,augment_angle,out_path_template,config)
            logger.debug(out_seg_aug_path)
            generate_angle_augmented_image(augment_angle,out_seg_path,ImageType.SEGMENTATION,out_seg_aug_path)

            out_xray_ap_path = generate_augment_path('xray_from_ct_augment','vert_xray_aug_ap',subject_id,vert_id,augment_angle,out_path_template,config)
            logger.debug(f'xray_ap {out_xray_ap_path}')
            generate_augmented_xray(out_ct_path,ProjectionType.AP,config['xray_pose'],out_xray_ap_path,augment_angle)

            out_xray_lat_path = generate_augment_path("xray_from_ct_augment","vert_xray_aug_lat",subject_id, vert_id, augment_angle,out_path_template,config)
            logger.debug(f'xray_lat {out_xray_lat_path}')
            generate_augmented_xray(out_ct_path,ProjectionType.LAT,config['xray_pose'],out_xray_lat_path,augment_angle)


def generate_augment_path(sub_dir:str, name:str, subject_id, vert_id, augment_angle, output_path_template, config):
    """xray_ap:"{id_hip-ap.png} -> img001_hip-ap.png"""
    output_fileformat = config['filename_convention']['output']
    out_dirs = config['out_directories']
    filename = output_fileformat[name].format(id=subject_id,vert=vert_id,angle=augment_angle)
    out_path = output_path_template.format(output_type=out_dirs[sub_dir],output_name=filename)
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    return out_path

def generate_path(sub_dir: str, name: str, subject_id, vert_id, output_path_template, config):
    """xray_ap:"{id}_hip-ap.png -> img0001_hip-ap.png"""
    output_fileformat = config["filename_convention"]["output"]
    out_dirs = config["out_directories"]
    filename = output_fileformat[name].format(id=subject_id,vert=vert_id)
    out_path = output_path_template.format(
        output_type=out_dirs[sub_dir], output_name=filename
    )
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    return out_path



if __name__ == '__main__':
    import argparse
    import pandas as pd
    from tqdm import tqdm
    from multiprocessing import Pool

    from xrayto3d_preprocess import read_config_and_load_components
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--num_workers',default=4,type=int)
    parser.add_argument('--debug',default=False,action='store_true')

    args = parser.parse_args()

    config = read_config_and_load_components(args.config_file)
    print(config)

    subject_list = pd.read_csv(config['subjects']['subject_list'],header=None).to_numpy().flatten()
    if args.debug:
        subject_list = subject_list[:args.num_workers]
    # create logger
    dataset_name = get_stem(args.config_file)+'_'+'360_augmentation'
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
        results = tqdm(p.map(generate_360_augmentation, sorted(subject_list)),
                       total=len(subject_list))

