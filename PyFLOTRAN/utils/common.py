"""
Contains general purpose utility functions
"""
import munch
import yaml


def read_config(config_file="./config/config.yaml"):
    """
    Reads the configuration file
    :param config_file:
    :return:
    """
    with open(config_file) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return munch.DefaultMunch().from_dict(context)

