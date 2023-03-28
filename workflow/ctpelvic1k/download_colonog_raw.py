import argparse
import shutil
import tempfile
from json import JSONDecodeError

import pandas as pd
from tqdm import tqdm
from xrayto3d_preprocess import (
    call_rest_api,
    download_wget,
    get_image_metadata_tcia_restapi_url,
    get_image_tcia_restapi_url,
    zip_dicom_to_nifti,
)


def get_series_id(patient_id: str):
    """find series id"""
    csv_path = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/CTColonography_MetaData.csv"
    )
    df = pd.read_csv(csv_path)
    subdf = df[df["Patient Id"].str.contains(patient_id)]
    subdf = subdf[
        ~(subdf["Series Description"].str.contains("Topo|topo", na=False))
    ]  # avoid topograms
    return subdf["Series UID"].to_numpy()


def get_segmentation_series_number(filename: str):
    """dataset2_1.3.6.1.4.1.9328.50.4.0001_3_325_mask_4label.nii.gz -> 3"""
    return int(filename.split("_")[2])


def parse_args():
    description = "download COLONOG images and save nifti to dir"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-path")
    parser.add_argument("--start-from", type=int, default=0)

    args = parser.parse_args()

    return args


def generate_aux_metadata():
    seg_csv = "external/XrayTo3DPreprocess/workflow/ctpelvic1k/colonog_seg.csv"
    out_csv = "external/XrayTo3DPreprocess/workflow/ctpelvic1k/colonog_seg_metadata.csv"
    metadata_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/CTColonography_MetaData.csv"
    )
    metadata_df = pd.read_csv(metadata_csv)

    row_list = []
    seg_df = pd.read_csv(seg_csv)
    for index, row in tqdm(seg_df.iterrows(), total=len(seg_df)):
        patient_id = str(row["Patient Id"])
        colonog_id = patient_id.split(".")[-1]
        segmentation_filename = str(row["segmentation-filename"])
        for sid in get_series_id(patient_id):
            url = get_image_metadata_tcia_restapi_url(sid)
            try:
                out_json = call_rest_api(url)
            except JSONDecodeError as e:
                print(e)
                continue
            series_number = int(float(out_json[0]["Series Number"]))
            if series_number == get_segmentation_series_number(segmentation_filename):
                # find the metadata for the corresponding series id in the COLONOG Metadata csv
                metadata = metadata_df[metadata_df["Series UID"] == sid]
                row_list.append(
                    {
                        "COLONOG-Id": f"COLONOG-{colonog_id}",
                        "Patient Id": patient_id,
                        "Series UID": sid,
                        "Series Description": metadata["Series Description"].values[0],
                    }
                )
                out_df = pd.DataFrame(
                    row_list,
                    columns=[
                        "COLONOG-Id",
                        "Patient Id",
                        "Series UID",
                        "Series Description",
                    ],
                )
                out_df.to_csv(out_csv, index=False)


def main():
    """entry point"""
    args = parse_args()
    OUTPUT_DIR = args.out_path
    START_FROM = args.start_from
    seg_meta_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/colonog_seg_metadata.csv"
    )
    df = pd.read_csv(seg_meta_csv)
    for current_row, (index, row) in enumerate(
        tqdm(df.iterrows(), total=len(df) - START_FROM)
    ):
        if current_row < START_FROM:
            continue
        patient_id, series_uid = row["Patient Id"], row["Series UID"]
        url = get_image_tcia_restapi_url(series_uid=series_uid)
        with tempfile.TemporaryDirectory() as default_temp_dir:
            dicom_filepath = f"{default_temp_dir}/{patient_id}.zip"
            try:
                download_wget(url, dicom_filepath, ".")
                zip_dicom_to_nifti(dicom_filepath, output_dir=OUTPUT_DIR)
            except RuntimeError as e:
                print(f"{e}")
            shutil.rmtree(default_temp_dir)


if __name__ == "__main__":
    main()
    # generate_aux_metadata()
