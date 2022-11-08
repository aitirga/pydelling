"""
Global variables
"""
from pathlib import Path

import yaml
from box import Box

import logging


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

def create_results_folder(folder_name='./results'):
    output_folder = Path(folder_name)
    output_folder.mkdir(exist_ok=True)
    return output_folder

def create_output_folder(folder_name='output'):
    output_folder = Path(folder_name)
    output_folder.mkdir(exist_ok=True)
    return output_folder

def delete_last_logger():
    loggers = [handler for handler in logging.root.handlers if isinstance(handler, logging.StreamHandler)]
    strange_logger = loggers[-1]
    strange_logger.setLevel(logging.ERROR)