"""misc utils"""
from shutil import which

from .pathutils import get_nifti_stem


def command_exists(command_name):
    """check if a command exists or is in path"""
    return which(command_name) is not None


def range_inclusive(start, end):
    """return a generator for sequential [start,end]"""
    return range(start, end + 1)


def get_drr_command():
    """return full path to the DRR generation tool"""
    possible_commands = [
        "TwoProjectionRegistrationTestDriver GetDRRSiddonJacobsRayTracing",
        "DRRSiddonJacobs",
    ]
    for cmd in possible_commands:
        if command_exists(cmd.split(sep=" ", maxsplit=1)[0]):
            return cmd
    raise ValueError("DRR not found")


def get_drrsiddonjacobs_command_string(
    input_filepath, output_filepath, orientation, config
):
    """command with filled arguments to create DRR"""
    # DRRSiddonJacobs has to be in path
    res = config["res"]  # DRR resolution
    size = config["size"]  # DRR size
    drr_command_executable = get_drr_command()
    rx, ry, rz = (
        config[orientation]["rx"],
        config[orientation]["ry"],
        config[orientation]["rz"],
    )
    # suppress screen output
    command = f"{drr_command_executable} {input_filepath} -o {output_filepath} "
    command += f"-rx {rx} -ry {ry} -rz {rz} -res {res} {res} -size {size} {size} "
    command += "> /dev/null 2>&1"

    return command


def get_verse_subject_id(file_path) -> str:
    """verse subject id samples
    sub-verse417_split-verse277_ct.nii.gz -> sub-verse417_split-verse277
    sub-verse149_ct.nii.gz -> sub-verse149
    """
    file_stem = get_nifti_stem(file_path)
    file_components = file_stem.split("_")
    if len(file_components) > 1 and "split" in file_components[1]:
        return "_".join(file_components[0:2])
    else:
        return file_components[0]
