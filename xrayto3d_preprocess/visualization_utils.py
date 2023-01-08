from fury import window, actor
import numpy as np
import vtk
from vtk.util import numpy_support
# original code from https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/preview.py
# work in progress

def generate_preview(ct_in, file_out,label_id=1):
    from xvfbwrapper import Xvfb
    with Xvfb() as xvfb:
        plot_subject(ct_in, file_out,label_id=label_id)

def plot_subject(ct_img, output_path,label_id=1):
    window_size = (500, 500)

    scene = window.Scene()
    scene.background((1.0,1.0,1.0))
    window.ShowManager(scene, size=window_size, reset_camera=False).initialize()

    data = ct_img.get_fdata()
    data = data.transpose(1,2,0)
    data = data[::-1, :, :]
    # value_range = (-115, 225)

    roi_data = data == label_id
    affine = ct_img.affine
    affine[:3, 3] = 0
    roi_actor = plot_mask(scene, roi_data,affine,orientation='sagittal')
    scene.add(roi_actor)

    # slice_actor = actor.slicer(data, ct_img.affine)
    # slice_actor.SetPosition(0,0,0)
    # scene.add(slice_actor)

    scene.projection(proj_type='parallel')
    scene.reset_camera_tight(margin_factor=1.0)

    window.record(scene,size=window_size,out_path=output_path,reset_camera=False)
    scene.clear()

def plot_mask(scene, data:np.ndarray,affine,orientation='sagittal'):
    # 3D bundle
    data = data[::-1, :, :]
    if orientation == 'transverse':
        data = data.transpose(1,2,0)
        data = data[:,:, ::-1]

    data_actor = contour_from_roi(data,affine,color=(1.,193./255,149/255),smoothing=10)
    data_actor.SetPosition(0,0,0)
    return data_actor

def contour_from_roi(data, affine, color=(1,0,0),opacity=1,smoothing=0):
    
    vtk_major_version = vtk.vtkVersion.GetVTKMajorVersion()

    vol = np.interp(data,xp=[data.min(),data.max()],fp=[0,255])

    im = vtk.vtkImageData()
    if vtk_major_version <= 5:
        im.SetScalarTypeToUnsignedChar()    
    di, dj, dk = vol.shape[:3]
    im.SetDimensions(di, dj, dk)
    voxsz = (1.,1.,1.)
    im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    if vtk_major_version <= 5:
        im.AllocateScalars()
        im.SetNumberOfScalarComponents(1)
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)    
    # copy data
    vol = np.swapaxes(vol, 0, 2)
    vol = np.ascontiguousarray(vol)
    vol = vol.ravel()

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    if affine is None:
        affine = np.eye(4)
   # Set the transform (identity if none given)
    transform = vtk.vtkTransform()
    transform_matrix = vtk.vtkMatrix4x4()
    transform_matrix.DeepCopy((
        affine[0][0], affine[0][1], affine[0][2], affine[0][3],
        affine[1][0], affine[1][1], affine[1][2], affine[1][3],
        affine[2][0], affine[2][1], affine[2][2], affine[2][3],
        affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
    transform.SetMatrix(transform_matrix)
    transform.Inverse()

    # Set the reslicing
    image_resliced = vtk.vtkImageReslice()
    set_input(image_resliced, im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()

    # Adding this will allow to support anisotropic voxels
    # and also gives the opportunity to slice per voxel coordinates

    rzs = affine[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    image_resliced.SetOutputSpacing(*zooms)

    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    # skin_extractor = vtk.vtkContourFilter()
    skin_extractor = vtk.vtkMarchingCubes()
    if vtk_major_version <= 5:
        skin_extractor.SetInput(image_resliced.GetOutput())
    else:
        skin_extractor.SetInputData(image_resliced.GetOutput())
    skin_extractor.SetValue(0, 100)

    if smoothing > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(skin_extractor.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing)
        smoother.SetRelaxationFactor(0.1)
        smoother.SetFeatureAngle(60)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.SetConvergence(0)
        smoother.Update()

    skin_normals = vtk.vtkPolyDataNormals()
    if smoothing > 0:
        skin_normals.SetInputConnection(smoother.GetOutputPort())
    else:
        skin_normals.SetInputConnection(skin_extractor.GetOutputPort())

    skin_normals.SetFeatureAngle(60.0)
    
    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(skin_normals.GetOutputPort())
    skin_mapper.ScalarVisibilityOff()

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetOpacity(opacity)
    skin_actor.GetProperty().SetColor(color)

    return skin_actor

def set_input(vtk_object, inp):
    """Set Generic input function which takes into account VTK 5 or 6.

    Parameters
    ----------
    vtk_object: vtk object
    inp: vtkPolyData or vtkImageData or vtkAlgorithmOutput

    Returns
    -------
    vtk_object

    Notes
    -------
    This can be used in the following way::
        from fury.utils import set_input
        poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)

    This function is copied from dipy.viz.utils
    """
    if isinstance(inp, (vtk.vtkPolyData, vtk.vtkImageData)):
        vtk_object.SetInputData(inp)
    elif isinstance(inp, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(inp)
    vtk_object.Update()
    return vtk_object

if __name__ == '__main__':
    import nibabel as nib
    from pathlib import Path

    # ct_in_path = '/mnt/driveD/dataset-vis/synthetic_hip/COLONOG-0001_seg_crop_iso_LPI.nii.gz'
    # ct_in_path = '/mnt/driveD/dataset-vis/COLONOG-HIP/COLONOG-0001_seg_crop_iso_LPI.nii.gz'
    ct_in_path = '/mnt/driveD/dataset-vis/synthetic_vertebra/verse005_20_seg_seg.nii.gz'
    ct_in = nib.load(ct_in_path)    
    file_out = Path(ct_in_path).with_suffix('.png')
    generate_preview(ct_in, file_out,label_id=20)
