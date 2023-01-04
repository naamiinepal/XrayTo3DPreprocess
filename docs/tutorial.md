# Tutorial

This tutorial walks through basic usage of generating data.

## Get Started

---

1. **Define Configs**
```yaml
# xray image properties
ap:
    # rotation angles 
lat:
    # rotation angles
res: # resolution
size: 
drr_from_ct_mask: # no tissues outside of ROI
drr_from_mask: # no bone density information

# ROI extraction of vertebra around the vertebra body centroid
extraction_ratio:
ct_padding: 
seg_padding: 

# input filename convention
# output vertebra convention
# output directories
```
---

2. **Run data preprocessing pipeline**
```shell
$> python preprocessing_main.py config-file.yaml

---