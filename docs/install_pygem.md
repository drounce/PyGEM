(install_pygem_target)=
# Installing PyGEM
The Python Glacier Evolution Model has been packaged using Poetry and is hosted on the Python Package Index ([PyPI](https://pypi.org/project/pygem/)), such that all dependencies should install seamlessly. It is recommended that users create a [Anaconda](https://anaconda.org/) environment from which to install the model dependencies and core code.

### Setup Conda Environment
Anaconda is a Python dependency management tool. An Anaconda (conda) environment is essentially a directory that contains a specific collection of installed packages. The use of environments reduces issues caused by package dependencies. It is recommended that users first create conda environment from which to install PyGEM and its dependencies (if you do not yet have conda installed, see [conda's documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install) for instructions).  We recommend a conda environment with python >=3.10, <3.13.

A new conda environment can be created from the command line such as:
```
conda create --name <environment_name> python=3.12
```

### PyPI installation
Ensure you've activated your PyGEM environment
```
conda activate <environment_name>
```

Next, install PyGEM via [PyPI](https://pypi.org/project/pygem/):
```
pip install pygem
```

This will install all PyGEM dependencies within your conda environment, and set up PyGEM command line tools to run core model scripts.

### Setup
Following installation, an initialization script should to be executed.

The initialization script accomplishes two things:
1. Initializes the PyGEM configuration file *~/PyGEM/config.yaml*. If this file already exists, an overwrite prompt will appear.
2. Downloads and unzips a series of sample data files to *~/PyGEM/*, which can also be manually downloaded [here](https://drive.google.com/file/d/1Wu4ZqpOKxnc4EYhcRHQbwGq95FoOxMfZ/view?usp=drive_link).

Run the initialization script by entering the following in the terminal:
```
initialize
```

### Demonstration Notebooks
A series of accompanying Jupyter notebooks have been produces for demonstrating the functionality of PyGEM. These can be acquired and installed from [GitHub](https://github.com/PyGEM-Community/PyGEM-notebooks).