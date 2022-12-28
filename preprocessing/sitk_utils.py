import SimpleITK as sitk

def merge_segmentations(img: sitk.Image, mapping_dict) -> sitk.Image:
    """use SimplITK AggregateLabelMapFilter to merge all segmentation labels to first label. This is used to obtain the bounding box of all the labels """
    fltr = sitk.ChangeLabelImageFilter()
    fltr.SetChangeMap(mapping_dict)
    return fltr.Execute(img)

def keep_only_label(segmentation:sitk.Image, label_id) -> sitk.Image:
    """If the segmentation contains more than one labels, keep only label_id"""
    return sitk.Threshold(segmentation, label_id, label_id, 0)
