from pathlib import Path
import numpy as np
import pandas as pd


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()

base_dir = args.dir
ap_imgs_regex = f'{base_dir}/*ap*.png'
lat_imgs_regex = f'{base_dir}/*lat*.png'
seg_imgs_regex = f'{base_dir}/*seg*.nii.gz'

ap_img_paths = sorted(Path('.').rglob(ap_imgs_regex))
lat_img_paths = sorted(Path('.').rglob(lat_imgs_regex))
seg_img_paths = sorted(Path('.').rglob(seg_imgs_regex))

print(len(ap_img_paths), len(lat_img_paths), len(seg_img_paths))

ap_img_paths = np.array(ap_img_paths)
lat_img_paths = np.array(lat_img_paths)
seg_img_paths = np.array(seg_img_paths)


# write csv
dicts_path = {'ap':ap_img_paths, 'lat': lat_img_paths, 'seg': seg_img_paths}
df = pd.DataFrame(data=dicts_path)    
df.to_csv('configs/test/rib_high_res_test.csv')
