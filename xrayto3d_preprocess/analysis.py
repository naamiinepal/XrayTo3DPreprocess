if __name__ == '__main__':
    from xrayto3d_preprocess import read_image, load_centroids
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    json_path = 'local_disk/dataset-verse19test/sub-verse012/sub-verse012_seg-vb_ctd.json'
    seg_path = 'local_disk/dataset-verse19test/sub-verse012/sub-verse012_seg-vert_msk.nii.gz'

    direction, centroids = load_centroids(json_path)
    seg_sitk = read_image(seg_path)
    seg_numpy = sitk.GetArrayFromImage(seg_sitk)
    for vb_id, *ctd in centroids:
        print(vb_id)
        seg_label_indices = np.where(seg_numpy == vb_id)
        
        centroid_index = seg_sitk.TransformPhysicalPointToContinuousIndex(ctd)

            # flip point from [x,y,z] to [z,y,x]
        centroid_index = list(map(int, centroid_index))
        flipped_coords = np.flip(centroid_index, 0)

        squared_distances = [ (x- flipped_coords[0])**2 + (y - flipped_coords[1])**2 + (z-flipped_coords[2])**2 for x, y, z in zip(*seg_label_indices)]
        hist, edges = np.histogram(squared_distances,bins='scott')
        freq = hist / float(hist.sum())
        
        smooth = gaussian_filter1d(freq, 2)

        # compute second derivative
        smooth_d2 = np.gradient(np.gradient(smooth))

        # find switching points
        infls = np.where(np.diff(np.sign(smooth_d2)))[0]

        t1 , t2 = edges[infls[1]+1],edges[infls[2]+1]

        print(edges[infls])
        # vertebra_colormap = np.zeros_like(squared_distances)
        # vertebra_colormap[np.where(squared_distances > t1)] = 0.5
        # vertebra_colormap[np.where(squared_distances >= t2)] = 1.0

        vertebra_colormap = np.ones_like(squared_distances)
        vertebra_colormap[(np.where((squared_distances > edges[infls][1]) & (squared_distances < edges[infls][2])))] = 0.0
        print(vertebra_colormap)

        plt.figure()
        plt.plot(smooth,label='smoothed')
        plt.plot(np.gradient(smooth),label='1st deriv')
        plt.plot(smooth_d2,label='2nd deriv')
        plt.axhline(0)
        for i, infl in enumerate(infls, 1):
            plt.axvline(x=infl+1, color='k', label=f'Inflection Point {i}')
        plt.legend()

        fig = plt.figure()
        plt.plot(edges[1:],freq,marker='o')
        # plt.plot(edges[1:], smooth_d2 / np.max(smooth_d2), label='Second Derivative (scaled)')
        for i, infl in enumerate(infls, 1):
            plt.axvline(x=edges[infl+1], color='k', label=f'Inflection Point {i}')
        plt.legend(bbox_to_anchor=(1.55, 1.0))


        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(*seg_label_indices,s=2,c=squared_distances, alpha=0.2)


        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(*seg_label_indices,s=2,c=vertebra_colormap,cmap='viridis', alpha=0.2)
        
        plt.show()
        break