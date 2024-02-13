
## Installation
To get started, create an environment using conda
```sh
conda create -n env_name python=3.8
conda activate env_name
conda env update -f requirements.yaml
pip install .

```

## Downloading the dataset:
```sh
python3 workflow/verse19/download.py
python3 workflow/verse19/subjectwise_directory.py
```

### Requirements for Data Preprocessing:
- Compile DRR Generator and add to path
 
Detailed explanation on : [docs/install_requirements.md](docs/install_requirements.md)


## Sample Command for dataset preprocessing

```sh
python3 preprocessing_vertebra.py configs/full/Verse2019-DRR-full.yaml --dataset verse2019 --parallel
```





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
