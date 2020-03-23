# PyFLOTRAN
Set of pre- and post-processing scripts for PFLOTRAN


## Recommendations

Using and IDE like PyCharm is recommended to keep things easy. It may be useful if you need to debug your scripts,
or edit them.

Using PyCharm, you can follow the installation step and then run your scripts in PyFLOTRAN env automatically.
Otherwise, you can follow the steps below to run PyFLOTRAN manually.


## Installation
Configure the conda environment running

`conda env create -f /config/conda_env.yml`


## Templates and Examples

Some examples of usage of PyFLOTRAN may be found inside `templates` folder. Use `config.yaml` to setup data sources and 
other variables.


## Running template scripts

In order to execute PyFLOTRAN, `PyFLOTRAN` conda env needs to be loaded.


### Loading conda environment

`conda activate PyFLOTRAN`


### Running script

Run the script and set the config file as argument. If the config file is not specified, `./config.yaml` will be taken as default.

Example: 

`python templates\interpolate_top_BC\interpolate_BC.py templates\interpolate_top_BC\config.yaml`

