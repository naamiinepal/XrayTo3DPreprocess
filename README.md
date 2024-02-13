
```
Installation 
To get started create an environment using conda or mamba
1) Mamba: 
- mamba env create --name xrayto3dpreprocess --file requirements.yaml
- pip install .
2) Conda:
- conda create -n env_name python=3.8
- conda activate env_name
- conda env update -f requirements.yaml
- pip install .

Downloading the dataset:
- python3 workflow/verse19/download.py
- python3 workflow/verse19/subjectwise_directory.py

Requirements for Data Preprocessing:
- Detailed explanation on : docs/install_requirements.md

Adding above file you installed to your path:
- After above, you've created a folder named external,get into external/ITK-5.3.0/build/bin 
- Now copy its path,  for example : /home/shirshak/external/ITK-5.3.0/build/bin 
-  In shell type:
    vim .bashrc or nano .bashrc 
    Paste following path at the end:
        export PATH="/home/shirshak/external/ITK-5.3.0/build/bin:$PATH" 
    Hit :wq to save in vim or in  ctrl s followed by cntl x in nano

- python3 preprocessing_vertebra.py configs/full/Verse2019-DRR-full.yaml --dataset verse2019 --parallel




```

Time to download the dataset
|Dataset        | Time              | Size (GB) | #CT Scans         |
|---            | ---               | ---       |---           |
|Verse20        | 2 hr @ 4 MB/s     |36           | 214         |
|Verse19        |                   |12           |160         |
|Totalsegmentor |                   |           |           |
|RibFrac        |                   |56           |         |
|LIDC           |                   |           |           |
|CTPelvic1k     |                   |           |           |
|CTSpine1k      |                   |           |           |
|RSNACervical   |                   |           |           |
