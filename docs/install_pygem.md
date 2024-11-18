(install_pygem_target)=
# Installing PyGEM
The Python Glacier Evolution Model has been packaged using Poetry, such that all dependencies should install seamlessly.  It is recommended that users create a [Anaconda](https://anaconda.org/) environment from which to install the model dependencies and core code. [(see model structure)](model_structure_and_workflow_target) for specifics on the model code structure.

### Setup Conda Environment
Anaconda is a Python dependency management tool. An Anaconda (conda) environment is essentially a directory that contains a specific collection of installed packages. The use of environments reduces issues caused by package dependencies. It is recommended that users first create conda environment from which to install PyGEM and its dependencies (if you do not yet have conda installed, see [conda's documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install) for instructions).  We recommend a conda environment with python >=3.10, <3.13.

A new conda environment can be created from the command line such as:
```
conda create --name pygem python=3.12
```

### PyPI installation
Ensure you've activated your PyGEM environment
```
conda activate pygem
```

Next, install PyGEM via the Python Package Index ([PyPI](https://pypi.org/project/pygem/)):
```
pip install pygem
```

This will install all PyGEM dependencies within your conda environment, and set up PyGEM command line tools to run core model scripts.

Note, [sample files](https://drive.google.com/drive/folders/13kiU00Zz2swN5OzwXiWIQTj_JLEHnDgZ) are produced to support trial runs and model testing.

### Example notebooks
Following PyPI installation, download a series of example demonstration notebooks by running the following command from the command line:
```
download_notebooks
```

After doing so, you may find a set of Jupyter notebooks in `~/PyGEM/example_notebooks/` which can be referenced and utilized for demonstrational purposes on how to run the Python Glacier Evolution Model and perform several analyses.

### Configuration file
***Note, this step is not required, as PyGEM's configuration file will be initialized by default the first time any PyGEM project script is run.***

The first time a PyGEM script is run, a configuration file will be initialized within `~/PyGEM/`. The preceding step will initialze this configuration file. If skipped, users may manually initialize their configuration file in python like so (note, first enter `python` from the command line with your pygem conda environment activated):
```
import pygem.setup.config as config
config.ensure_config()
```

The configuration file can now be found at `~/PyGEM/config.yaml`


### Developing PyGEM
Are you interested in contributing to the development of PyGEM? If so, we recommend forking the [PyGEM's github repository](https://github.com/PyGEM-Community/PyGEM) and then [cloning the github repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) onto your local machine.

Note, if PyGEM was already installed via PyPI, first uninstall:
```
pip uninstall pygem
````

You can then use pip to install your locally cloned fork of PyGEM in 'editable' mode like so:
```
pip install -e /path/to/your/PyGEM/clone
```