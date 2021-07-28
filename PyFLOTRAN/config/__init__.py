from box import Box
import yaml
from pathlib import Path
import os
import logging.config
from pathlib import Path
from PyFLOTRAN.utils.configuration_utils import get_config_path


def read_config(config_file: Path="./config.yaml"):
    """
    Reads the configuration file
    :param config_file:
    :return:
    """
    with open(config_file) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return Box(context, default_box=True)


_config_file = list(Path(os.getcwd()).glob("**/*config.yml") and Path(os.getcwd()).glob("**/*config.yaml") and Path().cwd().glob("*config*.yml") and Path().cwd().glob("*config*.yaml"))
_config_file = _config_file if _config_file else list(Path(__file__).parent.glob("config.yml"))
assert len(_config_file) == 1, "Please provide a configuration file that has a '*config.yaml' name structure"
config = read_config(config_file=_config_file[0])

# Add global configuration
if not config.globals.is_globals_loaded:
    with open(get_config_path() / "global_config.yml", "r") as yml_file:
        local_yaml_file = yaml.safe_load(yml_file)
        config.globals = local_yaml_file
        config.globals.is_globals_loaded = True

os.makedirs(config.path.logs if config.path.logs else Path().cwd() / "logs", exist_ok=True)
with open(Path(__file__).parent / "logger_config.yml", "r") as ymlfile:
    log_config = yaml.safe_load(ymlfile)
logging.config.dictConfig(log_config)
