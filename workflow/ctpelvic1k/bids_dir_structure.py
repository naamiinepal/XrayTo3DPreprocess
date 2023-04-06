"""bring raw downloaded data into BIDS directory structure"""
from pathlib import Path
from xrayto3d_preprocess import copy_subjects_to_individual_dir

BASE_PATH = "2D-3D-Reconstruction-Datasets"
CTPELVIC1K_PATH = Path(BASE_PATH) / "ctpelvic1k"


def bids_kits():
    """KITS19 dataset"""
    kits_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/kits_subjects.csv"
    )
    kits_bids_basepath = CTPELVIC1K_PATH / "raw" / "KITS19" / "BIDS"
    kits19_raw_img_basepath = CTPELVIC1K_PATH / "raw" / "KITS19" / "img"
    kits19_raw_seg_basepath = (
        CTPELVIC1K_PATH
        / "raw"
        / "KITS19"
        / "seg"
        / "CTPelvic1K_dataset4_mask_mappingback"
    )

    copy_subjects_to_individual_dir(
        kits_bids_csv,
        kits19_raw_img_basepath,
        kits19_raw_seg_basepath,
        kits_bids_basepath,
        dest_img_file_pattern="ct.nii.gz",
        dest_seg_file_pattern="hip.nii.gz",
    )


def bids_msd():
    """MSD-T10 dataset"""
    msd_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/msd_t10-subjects.csv"
    )
    msd_bids_basepath = CTPELVIC1K_PATH / "raw" / "MSD-T10" / "BIDS"
    msdt10_raw_img_basepath = (
        CTPELVIC1K_PATH / "raw" / "MSD-T10" / "img" / "Task10_Colon"
    )
    msdt10_raw_seg_basepath = (
        CTPELVIC1K_PATH
        / "raw"
        / "MSD-T10"
        / "seg"
        / "CTPelvic1K_dataset3_mask_mappingback"
    )

    copy_subjects_to_individual_dir(
        msd_bids_csv,
        msdt10_raw_img_basepath,
        msdt10_raw_seg_basepath,
        msd_bids_basepath,
        dest_img_file_pattern="ct.nii.gz",
        dest_seg_file_pattern="hip.nii.gz",
    )


def bids_cervix():
    """CERVIX dataset"""
    cervix_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/cervix_subjects.csv"
    )
    cervix_bids_path = CTPELVIC1K_PATH / "raw" / "CERVIX" / "BIDS"
    cervix_raw_img_basepath = CTPELVIC1K_PATH / "raw" / "CERVIX" / "img" / "RawData"
    cervix_raw_seg_basepath = CTPELVIC1K_PATH / "raw" / "CERVIX" / "seg"
    copy_subjects_to_individual_dir(
        cervix_bids_csv,
        cervix_raw_img_basepath,
        cervix_raw_seg_basepath,
        cervix_bids_path,
        dest_img_file_pattern="ct.nii.gz",
        dest_seg_file_pattern="hip.nii.gz",
    )


def bids_abdomen():
    """ABDOMEN dataset"""
    abdomen_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/abdomen_subjects.csv"
    )
    abmdomen_bids_path = CTPELVIC1K_PATH / "raw" / "ABDOMEN" / "BIDS"
    abmdomen_raw_img_basepath = CTPELVIC1K_PATH / "raw" / "ABDOMEN" / "img" / "RawData"
    abmdomen_raw_seg_basepath = CTPELVIC1K_PATH / "raw" / "ABDOMEN" / "seg"
    copy_subjects_to_individual_dir(
        abdomen_bids_csv,
        abmdomen_raw_img_basepath,
        abmdomen_raw_seg_basepath,
        abmdomen_bids_path,
        dest_img_file_pattern="ct.nii.gz",
        dest_seg_file_pattern="hip.nii.gz",
    )


def bids_clinic():
    """CLINIC dataset"""
    clinic_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/clinic_subjects.csv"
    )
    clinic_bids_path = CTPELVIC1K_PATH / "raw" / "CLINIC" / "BIDS"
    clinic_raw_img_basepath = CTPELVIC1K_PATH / "raw" / "CLINIC" / "img"
    clinic_raw_seg_basepath = CTPELVIC1K_PATH / "raw" / "CLINIC" / "seg"
    copy_subjects_to_individual_dir(
        clinic_bids_csv,
        clinic_raw_img_basepath,
        clinic_raw_seg_basepath,
        clinic_bids_path,
        dest_img_file_pattern="ct.nii.gz",
        dest_seg_file_pattern="hip.nii.gz",
    )


def bids_clinic_metal():
    """CLINIC-METAL dataset"""
    clinic_metal_bids_csv = "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/clinic_metal_subjects.csv"
    clinic_metal_bids_path = CTPELVIC1K_PATH / "raw" / "CLINIC-METAL" / "BIDS"
    clinic_metal_raw_img_basepath = CTPELVIC1K_PATH / "raw" / "CLINIC-METAL" / "img"
    clinic_metal_raw_seg_basepath = CTPELVIC1K_PATH / "raw" / "CLINIC-METAL" / "seg"
    copy_subjects_to_individual_dir(
        clinic_metal_bids_csv,
        clinic_metal_raw_img_basepath,
        clinic_metal_raw_seg_basepath,
        clinic_metal_bids_path,
        dest_img_file_pattern="ct.nii.gz",
        dest_seg_file_pattern="hip.nii.gz",
    )


def bids_colonog():
    """COLONOG dataset"""
    colonog_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/colonog_subjects.csv"
    )
    colonog_bids_path = CTPELVIC1K_PATH / "raw" / "COLONOG" / "BIDS"
    colonog_raw_img_basepath = CTPELVIC1K_PATH / "zips" / "COLONOG" / "img"
    colonog_raw_seg_basepath = (
        CTPELVIC1K_PATH
        / "zips"
        / "COLONOG"
        / "seg"
        / "CTPelvic1K_dataset2_mask_mappingback"
    )
    copy_subjects_to_individual_dir(
        colonog_bids_csv,
        colonog_raw_img_basepath,
        colonog_raw_seg_basepath,
        colonog_bids_path,
        dest_img_file_pattern="ct.nii.gz",
        dest_seg_file_pattern="hip.nii.gz",
    )


def main():
    """entrypoint"""
    bids_abdomen()
    bids_cervix()
    bids_msd()
    bids_clinic()
    bids_clinic_metal()
    bids_kits()
    # bids_colonog()


if __name__ == "__main__":
    main()
