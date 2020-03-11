"""
Global variables
"""
# from ..utils.common import read_config
import yaml
import munch

def read_config(config_file="./config.yaml"):
    """
    Reads the configuration file
    :param config_file:
    :return:
    """
    with open(config_file) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return munch.DefaultMunch.fromDict(context)
    # return munch.DefaultMunch.from_dict(context)


def initialize_config(config_file="./config/config.yaml"):
    global config
    config = read_config(config_file)

