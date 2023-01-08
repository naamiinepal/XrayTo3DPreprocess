# from https://github.com/shagunsodhani/torch-template
# These utility functions are used to read a custom hierarchical yaml file
# each major module configuration keys are the top of the hierarchy (say datasets, architectures etc.)
# the detailed configuration of these modules may be shared across experiments and may be stored in a separate yaml file.
# These separate yaml are merged into main configuration with a special key '_load'

import os
from os.path import join
from pathlib import Path
from typing import Dict, Union, Any, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


ConfigType = Union[DictConfig, ListConfig]


def read_config_and_load_components(filepath, special_key='_load'):
    config = OmegaConf.load(filepath)
    assert isinstance(config, DictConfig)
    for key in config:
        config[key] = load_components(config[key],Path(filepath).parent, special_key)
    return config


def load_components(config: ConfigType, basepath, special_key) -> ConfigType:
    if config is not None and special_key in config:
        loaded_config = OmegaConf.load(basepath/config.pop(special_key))
        updated_config = OmegaConf.merge(loaded_config, config)
        return updated_config
    else:
        return config

def to_dict(config: ConfigType) -> Dict[str, Any]:
    dict_config = cast(Dict[str,Any],OmegaConf.to_container(config))
    return dict_config



if __name__ == '__main__':
    test_configpath = 'configs/data/test/LIDC-DRR-test.yaml'
    config = read_config_and_load_components(test_configpath)
    print(config)
