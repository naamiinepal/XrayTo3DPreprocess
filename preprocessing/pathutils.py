from pathlib import Path
import os
# from torchio/utils.py

def get_stem(path):
    """
    '/home/user/image.nii.gz' -> 'image'
    """
    def _get_stem(path_string) -> str:
        return Path(path_string).name.split('.')[0]
    if isinstance(path, (str, os.PathLike)):
        return _get_stem(path)


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
    filename = get_stem(source_path)
    if dest_format is None:
        dest_format = get_file_format_suffix(source_path)
    return Path(dest_dir)/f'{filename}.{dest_format}'

if __name__ == '__main__':
    print(get_file_format_suffix('/home/user/image.nii.gz'))
    print(dest_path('/home/user/image.nii.gz','/home/dest','png'))