import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    metadata_path = '2D-3D-Reconstruction-Datasets/TotalSegmentor-full/rib_meta.csv'
    df = pd.read_csv(metadata_path)
    print('Median',df['total_voxels'].median(),'20th percentile',f'{df.total_voxels.quantile(0.2):.2f}')

    print('Empty rows',len(df[df['total_voxels'] == 0]))

    print(df[df.transpose().all()])

    full_ribs_df = df[df.transpose().all()].copy()
    full_ribs_df.rename( columns={'Unnamed: 0' :'subject_id'}, inplace=True )
    print(full_ribs_df)
    full_ribs_df.to_csv(Path(metadata_path).with_name('rib_full_meta.csv'),columns=['subject_id'],index=False)