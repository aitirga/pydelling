"""
Global variables
"""
import yaml
from box import Box
from pathlib import Path


def get_root_path() -> Path:
    """Returns path to the root of the project"""
    return Path(__file__).parent.parent


def get_config_path() -> Path:
    """Returns path to the root of the project"""
    return Path(__file__).parent.parent / "config"


def test_data_path() -> Path:
    """Returns path to the root of the project"""
    return Path(__file__).parent.parent / "tests/test_data"


def runtime_path():
    return Path.cwd()


def read_config(config_file="./config.yaml"):
    """
    Reads the configuration file
    :param config_file:
    :return:
    """
    with open(config_file) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return Box(context, default_box=True)
    # return munch.DefaultMunch.from_dict(context)


def initialize_config(config_file="./config/config.yaml"):
    global config
    try:
        config = read_config(config_file)
    except FileNotFoundError as fe:
        print(f"ERROR: Config file({config_file}) not found.")
        exit(1)

