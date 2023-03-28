"""utils specific to ctpelvic1k"""
import pandas as pd


def get_series_id(patient_id: str):
    """find series id"""
    csv_path = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/CTColonography_MetaData.csv"
    )
    df = pd.read_csv(csv_path)
    subdf = df[df["Patient Id"].str.contains(patient_id)]
    subdf = subdf[~subdf["Series Description"].str.contains("Topo|topo")][
        "Series UID"
    ]  # avoid topograms
    return subdf.to_numpy()


def get_segmentation_series_number(filename: str):
    """dataset2_1.3.6.1.4.1.9328.50.4.0001_3_325_mask_4label.nii.gz -> 3"""
    return int(filename.split("_")[2])


def get_value_from_tcia_json_metadata(metadata_json, key):
    """return value from metadata"""
    return metadata_json[0][key]
