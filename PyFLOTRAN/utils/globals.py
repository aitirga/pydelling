"""
Global variables
"""
from ..utils.common import read_config


def initialize_config(config_file="./config/config.yaml"):
    global config
    config = read_config(config_file)

