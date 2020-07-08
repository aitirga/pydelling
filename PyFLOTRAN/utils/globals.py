"""
Global variables
"""
# from ..utils.common import read_config
import yaml
from box import Box


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

