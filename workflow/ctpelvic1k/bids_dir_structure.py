from pathlib import Path
from xrayto3d_preprocess import copy_subjects_to_individual_dir

BASE_PATH = "2D-3D-Reconstruction-Datasets"
CTPELVIC1K_PATH = Path(BASE_PATH) / "ctpelvic1k"


def bids_kits():
    kits_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/kits_subjects.csv"
    )
    KITS_BIDS_BASEPATH = CTPELVIC1K_PATH / "raw" / "KITS19" / "BIDS"
    KITS19_RAW_IMG_BASEPATH = CTPELVIC1K_PATH / "raw" / "KITS19" / "img"
    KITS19_RAW_SEG_BASEPATH = (
        CTPELVIC1K_PATH
        / "raw"
        / "KITS19"
        / "seg"
        / "CTPelvic1K_dataset4_mask_mappingback"
    )

    copy_subjects_to_individual_dir(
        kits_bids_csv,
        KITS19_RAW_IMG_BASEPATH,
        KITS19_RAW_SEG_BASEPATH,
        KITS_BIDS_BASEPATH,
    )


def bids_msd():
    msd_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/msd_t10-subjects.csv"
    )
    MSD_BIDS_BASEPATH = CTPELVIC1K_PATH / "raw" / "MSD-T10" / "BIDS"
    MSDT10_RAW_IMG_BASEPATH = (
        CTPELVIC1K_PATH / "raw" / "MSD-T10" / "img" / "Task10_Colon"
    )
    MSDT10_RAW_SEG_BASEPATH = (
        CTPELVIC1K_PATH
        / "raw"
        / "MSD-T10"
        / "seg"
        / "CTPelvic1K_dataset3_mask_mappingback"
    )

    copy_subjects_to_individual_dir(
        msd_bids_csv,
        MSDT10_RAW_IMG_BASEPATH,
        MSDT10_RAW_SEG_BASEPATH,
        MSD_BIDS_BASEPATH,
    )


def bids_cervix():
    cervix_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/cervix_subjects.csv"
    )
    CERVIX_BIDS_PATH = CTPELVIC1K_PATH / "raw" / "CERVIX" / "BIDS"
    CERVIX_RAW_IMG_BASEPATH = CTPELVIC1K_PATH / "raw" / "CERVIX" / "img" / "RawData"
    CERVIX_RAW_SEG_BASEPATH = CTPELVIC1K_PATH / "raw" / "CERVIX" / "seg"
    copy_subjects_to_individual_dir(
        cervix_bids_csv,
        CERVIX_RAW_IMG_BASEPATH,
        CERVIX_RAW_SEG_BASEPATH,
        CERVIX_BIDS_PATH,
    )


def bids_abdomen():
    abdomen_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/abdomen_subjects.csv"
    )
    ABMDOMEN_BIDS_PATH = CTPELVIC1K_PATH / "raw" / "ABDOMEN" / "BIDS"
    ABMDOMEN_RAW_IMG_BASEPATH = CTPELVIC1K_PATH / "raw" / "ABDOMEN" / "img" / "RawData"
    ABMDOMEN_RAW_SEG_BASEPATH = CTPELVIC1K_PATH / "raw" / "ABDOMEN" / "seg"
    copy_subjects_to_individual_dir(
        abdomen_bids_csv,
        ABMDOMEN_RAW_IMG_BASEPATH,
        ABMDOMEN_RAW_SEG_BASEPATH,
        ABMDOMEN_BIDS_PATH,
    )


def bids_clinic():
    clinic_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/clinic_subjects.csv"
    )
    CLINIC_BIDS_PATH = CTPELVIC1K_PATH / "raw" / "CLINIC" / "BIDS"
    CLINIC_RAW_IMG_BASEPATH = CTPELVIC1K_PATH / "raw" / "CLINIC" / "img"
    CLINIC_RAW_SEG_BASEPATH = CTPELVIC1K_PATH / "raw" / "CLINIC" / "seg"
    copy_subjects_to_individual_dir(
        clinic_bids_csv,
        CLINIC_RAW_IMG_BASEPATH,
        CLINIC_RAW_SEG_BASEPATH,
        CLINIC_BIDS_PATH,
    )


def bids_clinic_metal():
    clinic_metal_bids_csv = "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/clinic_metal_subjects.csv"
    CLINIC_METAL_BIDS_PATH = CTPELVIC1K_PATH / "raw" / "CLINIC-METAL" / "BIDS"
    CLINIC_METAL_RAW_IMG_BASEPATH = CTPELVIC1K_PATH / "raw" / "CLINIC-METAL" / "img"
    CLINIC_METAL_RAW_SEG_BASEPATH = CTPELVIC1K_PATH / "raw" / "CLINIC-METAL" / "seg"
    copy_subjects_to_individual_dir(
        clinic_metal_bids_csv,
        CLINIC_METAL_RAW_IMG_BASEPATH,
        CLINIC_METAL_RAW_SEG_BASEPATH,
        CLINIC_METAL_BIDS_PATH,
    )


def bids_colonog():
    colonog_bids_csv = (
        "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/colonog_subjects.csv"
    )
    COLONOG_BIDS_PATH = CTPELVIC1K_PATH / "raw" / "COLONOG" / "BIDS"
    COLONOG_RAW_IMG_BASEPATH = CTPELVIC1K_PATH / "zips" / "COLONOG" / "img"
    COLONOG_RAW_SEG_BASEPATH = (
        CTPELVIC1K_PATH
        / "zips"
        / "COLONOG"
        / "seg"
        / "CTPelvic1K_dataset2_mask_mappingback"
    )
    copy_subjects_to_individual_dir(
        colonog_bids_csv,
        COLONOG_RAW_IMG_BASEPATH,
        COLONOG_RAW_SEG_BASEPATH,
        COLONOG_BIDS_PATH,
    )


def main():
    # bids_abdomen()
    # bids_cervix()
    # bids_msd()
    # bids_clinic()
    # bids_clinic_metal()
    # bids_kits()
    bids_colonog()


if __name__ == "__main__":
    main()
