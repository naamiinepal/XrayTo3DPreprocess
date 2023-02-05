#!/usr/bin/bash
python external/XrayTo3DPreprocess/workflow/data/totalsegmentor_ribs/totalsegmentor_combine_ribs.py --base_path 2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset --generate_ribs --generate_stats
python external/XrayTo3DPreprocess/preprocess_totalsegmentor_ribs.py