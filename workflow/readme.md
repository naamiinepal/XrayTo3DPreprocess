# Workflow for Preprocessing TotalSegmentator-Femur Dataset

- Download dataset 
    Zenodo link:https://doi.org/10.5281/zenodo.6802613
- Collect Statistics
  ```python
  python collect_stats.py
  ``` 
  This generates csv of CT Scan sample IDs that are usable (based on voxel threshold)
- Prepare config
    - Setup paths to dataset directory, provide list of subjects to preprocess, output formats
- Start Preprocessing
  ```bash
  python preprocess_total_segmentor_femur.py config.yaml
  ```

# Workflow for Preprocessing TotalSegmentator-Rib Dataset

- Download dataset 
    Zenodo link:https://doi.org/10.5281/zenodo.6802613
    - skip if already done 
- Collect Statistics
  ```shell
  python totalsegmentor_combine_ribs.py --generate_ribs --generate_stats --base_path path/to/dataset
  ``` 
  - combine individual rib segmentations into a single nifti
  ```shell
  python process_metadata.py
  ```
  - This generates csv of CT Scan sample IDs that are usable based on whether the CT Scan contains full set of ribs
- Prepare config
    - Setup paths to dataset directory, provide list of subjects to preprocess, output formats
- Start Preprocessing
  ```python
  python preprocess_total_segmentor_ribs.py config.yaml
  ```

# Workflow for Preprocessing TotalSegmentator-Hip Dataset

- Download dataset 
    Zenodo link:https://doi.org/10.5281/zenodo.6802613
- Collect Statistics
  ```python
  python total_segmentor_hip_stats.py
  ``` 
  The CT Scans were choosen based on whether full bone shape were available (partial hip scans were rejected manually).
- Prepare config
    - Setup paths to dataset directory, provide list of subjects to preprocess, output formats
- Start Preprocessing
  ```bash
  python preprocess_total_segmentor_hip.py config.yaml
  ```