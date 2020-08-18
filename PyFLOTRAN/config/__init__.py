from box import Box
import yaml
from pathlib import Path
import os
import logging.config
from pathlib import Path

def read_config(config_file: Path="./config.yaml"):
    """
    Reads the configuration file
    :param config_file:
    :return:
    """
    with open(config_file) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return Box(context, default_box=True)



_config_file = list(Path(os.getcwd()).glob("**/*config.yml") and Path(os.getcwd()).glob("**/*config.yaml"))
assert len(_config_file) == 1, "Please provide a configuration file that has a '*config.yaml' name structure"
config = read_config(config_file=_config_file[0])

os.makedirs(config.path.logs if config.path.logs else Path().cwd() / "logs", exist_ok=True)
with open(Path(__file__).parent / "logger_config.yml", "r") as ymlfile:
    log_config = yaml.safe_load(ymlfile)
logging.config.dictConfig(log_config)
