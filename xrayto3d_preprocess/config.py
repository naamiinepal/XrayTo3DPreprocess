"""
adapted from https://github.com/shagunsodhani/torch-template
These utility functions are used to read a custom hierarchical yaml file.
each major module configuration keys are the top of the hierarchy (say datasets, architectures etc.)
The detailed configuration of these modules may be shared across experiments
and may be stored in a separate yaml file.
These separate yaml are merged into main configuration with a special key '_load'
"""
from pathlib import Path
from typing import Any, Dict, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

ConfigType = Union[DictConfig, ListConfig]


def read_config_and_load_components(filepath, special_key="_load"):
    """read yaml from filepath and load subcomponents"""
    config_dict = OmegaConf.load(filepath)
    assert isinstance(config_dict, DictConfig)
    for key in config_dict:
        config_dict[key] = load_components(
            config_dict[key], Path(filepath).parent, special_key
        )
    return config_dict


def load_components(config: ConfigType, basepath, special_key) -> ConfigType:
    """
    update dict if the key == special_key
    return a updated dict
    """
    if config is not None and special_key in config:
        loaded_config = OmegaConf.load(basepath / config.pop(special_key))
        updated_config = OmegaConf.merge(loaded_config, config)
        return updated_config
    else:
        return config


def to_dict(config: ConfigType) -> Dict[str, Any]:
    """thin wrapper to OmegaConf.to_container"""
    dict_config = cast(Dict[str, Any], OmegaConf.to_container(config))
    return dict_config


if __name__ == "__main__":
    test_configpath = "configs/test/LIDC-DRR-test.yaml"
    config_dict = read_config_and_load_components(test_configpath)
    print(config_dict)
