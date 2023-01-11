from pathlib import Path
import os
# from torchio/utils.py

def get_nifti_stem(path):
    """
    '/home/user/image.nii.gz' -> 'image'
    1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235.nii.gz ->1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235
    """
    def _get_stem(path_string) -> str:
        name_subparts = Path(path_string).name.split('.')
        return '.'.join(name_subparts[:-2]) # get rid of nii.gz
    if isinstance(path, (str, os.PathLike)):
        return _get_stem(path)

def get_stem(path):
    return Path(path).stem
    
def get_file_format_suffix(path):
    """
    '/home/user/image.nii.gz' -> 'nii.gz'
    """
    def _get_stem(path_string) -> str:
        return '.'.join(Path(path_string).name.split('.')[1:])
    if isinstance(path, (str, os.PathLike)):
        return _get_stem(path)    

def dest_path(source_path, dest_dir,dest_format=None):
    """
    '/home/user/image.nii.gz' , /home/dest -> /home/dest/image.nii.gz 
    """
    filename = get_nifti_stem(source_path)
    if dest_format is None:
        dest_format = get_file_format_suffix(source_path)
    return Path(dest_dir)/f'{filename}.{dest_format}'


def get_verse_subject_id(file_path):
    """ verse subject id samples
    sub-verse417_split-verse277_ct.nii.gz -> sub-verse417_split-verse277
    sub-verse149_ct.nii.gz -> sub-verse149
    """
    file_stem = get_nifti_stem(file_path)
    file_components = file_stem.split('_')
    if len(file_components) > 1 and 'split' in file_components[1]:
        return '_'.join(file_components[0:2])
    else:
        return file_components[0]
    
        
if __name__ == '__main__':
    print(get_file_format_suffix('/home/user/image.nii.gz'))
    print(dest_path('/home/user/image.nii.gz','/home/dest','png'))