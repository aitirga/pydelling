import logging.config
import os
from pathlib import Path
from rich.logging import RichHandler
import pkg_resources

import yaml
from box import Box

from pydelling.utils.configuration_utils import get_config_path


def read_config(config_file: Path="./local_config.yaml"):
    """
    Reads the configuration file
    :param config_file:
    :return:
    """
    with open(config_file) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return Box(context, default_box=True)


_config_file = list(Path.cwd().glob("*config*"))
if not _config_file:
    _config_file = [get_config_path() / 'local_config.yaml']
    logging.warning('Using default configuration file (not the user defined one)')
assert len(_config_file) >= 1, "Please provide a configuration file that has a '*config.yaml' name structure"
config = read_config(config_file=_config_file[0])

# Add global configuration
if not config.globals.is_globals_loaded:
    with open(get_config_path() / "global_config.yaml", "r") as yml_file:
        local_yaml_file = yaml.safe_load(yml_file)
        config.globals = local_yaml_file
        config.globals.is_globals_loaded = True

# Allow to override the global configuration with a local file
for key in config:
    if key in config.globals:
        if isinstance(config[key], dict):
            config.globals[key].update(config[key])
        else:
            config.globals[key] = config[key]

os.makedirs(config.path.logs if config.path.logs else Path().cwd() / "logs", exist_ok=True)
with open(Path(__file__).parent / "logger_config.yaml", "r") as ymlfile:
    log_config = yaml.safe_load(ymlfile)
logging.config.dictConfig(log_config)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, show_path=False)])

# # Fix logging issue caused by streamlit
# loggers = [handler for handler in logging.root.handlers if isinstance(handler, logging.StreamHandler)]
# strange_logger = loggers[-1]
# strange_logger.setLevel(logging.ERROR)

# Capture warnings
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

# Add welcome message, get the version from the setup.py file
try:
    with open(Path(__file__).parent.parent.parent / "pyproject.toml", "r") as setup_file:
        setup_file = setup_file.read()
        version = setup_file.split("version = \"")[1].split("\"")[0]
except:
    version = pkg_resources.get_distribution("pydelling").version
logging.info(f"-----------------------------------")
logging.info(f"[blue bold]Welcome to Pydelling {version}", extra={"markup": True})
logging.info(f"-----------------------------------")



