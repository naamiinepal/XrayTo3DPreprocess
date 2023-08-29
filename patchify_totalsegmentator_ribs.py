from concurrent.futures import ThreadPoolExecutor
import concurrent.futures 
import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from monai.data import PatchIter, PILWriter
from monai.transforms import Compose, LoadImage, EnsureType, Resize
from xrayto3d_preprocess import read_config_and_load_components, get_stem, get_logger, read_image, resample_isotropic



def process_total_segmentator_subject_helper(subject_id, config, patch_sz, patch_res):
    # define paths
    subject_basepath = config['subjects']['subject_basepath']

    DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'
    INPUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'
    
    output_fileformats = config['filename_convention']['output']

    xray_ap_name = output_fileformats['xray_ap'].format(id=subject_id)
    xray_ap_path = INPUT_PATH_TEMPLATE.format(output_type='xray_from_ct',output_name=xray_ap_name)
    xray_lat_name = output_fileformats['xray_lat'].format(id=subject_id)
    xray_lat_path = INPUT_PATH_TEMPLATE.format(output_type='xray_from_ct',output_name=xray_lat_name)
    seg_roi_name = output_fileformats['seg_roi'].format(id=subject_id)
    seg_path = INPUT_PATH_TEMPLATE.format(output_type='seg_roi', output_name=seg_roi_name)

    logger.debug(f'{xray_ap_path}, {xray_lat_path},{seg_path}')


    # read inputs
    seg_img = read_image(seg_path)
    ORIG_SZ = np.ceil(seg_img.GetSize()[0]) # using cubic volume
    ORIG_RES = np.around(seg_img.GetSpacing()[0],3)
    ORIG_RES_IN_MM = np.ceil(ORIG_SZ * ORIG_RES)
    CT_PAD_VAL = -1024
    SEG_PAD_VAL = 0
    NUM_PATCH = int(ORIG_RES_IN_MM / (patch_res * patch_sz))
    logger.debug(f'Original 3D Volume size {seg_img.GetSize()} res {seg_img.GetSpacing()}')


    # make patches
    ## bring to PATCH_RES resolution
    seg_img = resample_isotropic(seg_img,patch_res,interpolator='linear')

    logger.debug(f'{subject_id} After resampling: size {seg_img.GetSize()} spacing {np.around(seg_img.GetSpacing(),3)}')

    logger.debug(f'Generating {NUM_PATCH**3} patches with size {patch_sz} and resolution {patch_res}')

    patch_start_position_list = [int(i*patch_sz) for i in range(NUM_PATCH)] # [0,40, 80, 120, ...]
    start_pos_list = [ (PA,IS,RL) for PA in patch_start_position_list for IS in patch_start_position_list for RL in patch_start_position_list ]
    
    logger.debug(start_pos_list)

    seg_patches = [extract_roi(seg_img,patch_sz,roi_index,type='seg') for roi_index in start_pos_list]
    img_transform = Compose([LoadImage(image_only=True, ensure_channel_first=True),EnsureType(),
                    Resize(spatial_size=(int(ORIG_RES_IN_MM/patch_res), int(ORIG_RES_IN_MM/patch_res)),size_mode='all', mode='bilinear', align_corners=True)])
    ap_patches = get_xray_ap_patches(xray_ap_path, img_transform, patch_sz, NUM_PATCH)
    lat_patches = get_xray_lat_patches(xray_lat_path, img_transform, patch_sz, NUM_PATCH)

    # repeat AP and LAT patches to correspond to Seg patches
    ap_patches = (ap_patches,)*NUM_PATCH
    ap_patches = [patch for patch_list in ap_patches for patch in patch_list]
    lat_patches = [[patch,]*NUM_PATCH for patch in lat_patches]
    lat_patches = [patch for patch_list in lat_patches for patch in patch_list]

    logger.debug(f'{len(seg_patches)} {len(ap_patches)} {len(lat_patches)}')
    for patch_id, (ap_patch, lat_patch, seg_patch) in enumerate(zip(ap_patches, lat_patches, seg_patches)):

        logger.debug(f'Writing seg patch of size {seg_patch.GetSize()}, ap patch of size {ap_patch.shape},{ap_patch.dtype} {np.min(ap_patch),np.max(ap_patch)} lat patch of size {lat_patch.shape},{lat_patch.dtype},{np.min(lat_patch),np.max(lat_patch)}')
        xray_ap_patch_template = output_fileformats['xray_ap_patch']
        xray_lat_patch_template = output_fileformats['xray_lat_patch']
        seg_roi_patch_template = output_fileformats['seg_roi_patch']

        xray_ap_roi_name = xray_ap_patch_template.format(id=subject_id,patch_id=patch_id)
        xray_ap_roi_path = DIR_TEMPLATE.format(output_type='xray_from_ct_patch')

        xray_lat_roi_name = xray_lat_patch_template.format(id=subject_id,patch_id=patch_id)
        xray_lat_roi_path = DIR_TEMPLATE.format(output_type='xray_from_ct_patch')

        Path(xray_ap_roi_path).mkdir(exist_ok=True,parents=True)
        Path(xray_lat_roi_path).mkdir(exist_ok=True,parents=True)

        seg_roi_patch_name = seg_roi_patch_template.format(id=subject_id, patch_id=patch_id)
        seg_roi_patch_path = DIR_TEMPLATE.format(output_type='seg_roi_patch')

        Path(seg_roi_patch_path).mkdir(exist_ok=True,parents=True)

        xray_ap_patch_fullpath = Path(xray_ap_roi_path)/(xray_ap_roi_name)
        xray_lat_patch_fullpath = Path(xray_lat_roi_path)/(xray_lat_roi_name)
        seg_roi_patch_fullpath = Path(seg_roi_patch_path)/(seg_roi_patch_name)

        logger.debug(f'{xray_ap_patch_fullpath},{xray_lat_patch_fullpath},{seg_roi_patch_fullpath}')

        # write patches
        sitk.WriteImage(seg_patch, str(seg_roi_patch_fullpath))
        write_2d_patch(ap_patch, xray_ap_patch_fullpath)
        write_2d_patch(lat_patch, xray_lat_patch_fullpath)


def get_xray_ap_patches(xray_ap_path, img_transform, patch_sz, num_patch):
    xray_ap = img_transform(xray_ap_path)
    patch_generator = PatchIter(patch_size=(patch_sz,patch_sz))
    xray_ap_reversed = np.swapaxes(xray_ap,1,2) # [u,v] -> [v,u]
    xray_ap_patches = [np.swapaxes(patch,1,2) for patch,coord in  patch_generator(xray_ap_reversed)]
    logger.debug(f'Generating {len(xray_ap_patches)} of size {(patch_sz,patch_sz)} from image of size {xray_ap.shape}')
    # now reverse the image order [[0,1],[2,3]] -> [[1,0],[3,2]]
    reverse_ordering = np.flip(np.array(list(range(len(xray_ap_patches)))).reshape(num_patch,num_patch),axis=1).flatten()
    return np.array(xray_ap_patches)[reverse_ordering]

def get_xray_lat_patches(xray_lat_path, img_transform, patch_sz, num_patch):
    '''num_patch is ignored'''
    xray_lat = img_transform(xray_lat_path)
    patch_generator = PatchIter(patch_size=(patch_sz,patch_sz))
    
    xray_lat_patches =  [patch for patch, coord in patch_generator(xray_lat)]
    logger.debug(f'Generating {len(xray_lat_patches)} of size {(patch_sz,patch_sz)} from image of size {xray_lat.shape}')
    return xray_lat_patches

def write_2d_patch(patch, save_to):
    logger.debug(f'patch dtype {patch.dtype} size {patch.shape}')
    writer = PILWriter(output_dtype=np.uint8, scale=None)
    writer.set_data_array(patch)
    writer.write(save_to)

def extract_roi(img, PATCH_SZ, patch_roi_start_index, type='ct') :
    if type not in ['ct','seg']:
        raise ValueError(f'type should be one of [ct,seg]. got {type}')

    patch_roi = sitk.RegionOfInterest(img, (PATCH_SZ,) * img.GetDimension(), patch_roi_start_index)

    if type == 'ct':
        patch_roi = sitk.Cast(patch_roi,sitk.sitkInt16)
    elif type == 'seg':
        patch_roi = sitk.Cast(patch_roi, sitk.sitkUInt8)
    return patch_roi


if __name__ == '__main__':
    import argparse
    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--patch_sz',type=int)
    parser.add_argument('--patch_res',type=float)

    args = parser.parse_args()
    config = read_config_and_load_components(args.config_file)

    # create logger
    dataset_name = get_stem(args.config_file) + '-' + 'patchify'
    logger = get_logger(dataset_name)

    logger.debug(f'Generating patch dataset {dataset_name} with patch size {args.patch_sz} and patch_res {args.patch_res}')
    logger.debug(f'Configuration {config}')

    subject_list = pd.read_csv(config['subjects']['subject_list'], header=None).to_numpy().flatten()

    logger.debug(f'found {len(subject_list)} subjects')
    logger.debug(subject_list)

    num_workers = (os.cpu_count() or 2) // 2 


    

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures_to_subject_id = {}
        for subject_id in subject_list:
            f = executor.submit(process_total_segmentator_subject_helper,subject_id, config, args.patch_sz, args.patch_res) 
            futures_to_subject_id[f] = subject_id

        for future in tqdm(concurrent.futures.as_completed(futures_to_subject_id),total=len(futures_to_subject_id)):
            subject_id = futures_to_subject_id[future]
            try:
                future.result()
            except Exception as e:
                print(f'{e} {subject_id}')

