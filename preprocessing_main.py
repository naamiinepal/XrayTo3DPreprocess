
VERSE19_PATH =  'VERSE2019/Verse2019-DRR/BIDS/{id}/raw'
verse19_fileformat = {'ct':'{id}_ct.nii.gz','seg':'{id}_seg-vert_msk.nii.gz','ctd':'{id}_seg-vb_ctd.json','vert_ct':'{id}_vert-{vert}_ct.nii.gz','vert_seg':'{id}_vert-{vert}-seg-vert_msk.nii.gz'}
VERSE19_VERTEBRA_PATH = 'VERSE2019/Verse2019-DRR/BIDS/{id}/vertebra/'
VERSE19_VERTEBRA_CT_PATH = 'VERSE2019/Verse2019-DRR/BIDS/{id}/vertebra/ct_roi/{id}_vert-{vert}_ct.nii.gz'
VERSE19_VERTEBRA_SEG_PATH = 'VERSE2019/Verse2019-DRR/BIDS/{id}/vertebra/seg_roi/{id}_vert-{vert}-seg-vert_msk.nii.gz'

subject_list = [{'id':'sub-verse010'}]


from preprocessing import dest_path,write_image

if __name__ == '__main__':
    from pathlib import Path
    from preprocessing import load_centroids, read_image, extract_around_centroid_v2


    subject = subject_list[0]
    id = subject['id']
    ct_path =  Path(VERSE19_PATH.format(id=id))/verse19_fileformat['ct'].format(id=id)
    seg_path = Path(VERSE19_PATH.format(id=id))/verse19_fileformat['seg'].format(id=id)
    json_path = Path(VERSE19_PATH.format(id=id))/verse19_fileformat['ctd'].format(id=id)

    centroid_orientation, centroids = load_centroids(json_path)
    ct = read_image(ct_path)
    seg = read_image(seg_path)

    print(f'Image Spacing {ct.GetSpacing()} Voxel Size {ct.GetSize()}')
    for vb_id, *ctd in centroids:
        print(f'Extract vertebra {vb_id}')
        roi, centroid_heatmap = extract_around_centroid_v2(img=ct, physical_size=(96,96,96),
        centroid_index=ctd,extraction_ratio={'L': 0.5, 'A': 0.7, 'S' :0.5},padding_value=-1024,verbose=False)
        write_image(roi, VERSE19_VERTEBRA_CT_PATH.format(id=id,vert=vb_id))

        roi, centroid_heatmap = extract_around_centroid_v2(img=seg, physical_size=(96,96,96),
        centroid_index=ctd,extraction_ratio={'L': 0.5, 'A': 0.7, 'S' :0.5},padding_value=-1024,verbose=False)
        write_image(roi, VERSE19_VERTEBRA_SEG_PATH.format(id=id,vert=vb_id))