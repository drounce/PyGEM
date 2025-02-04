## Python Glacier Evolution Model (PyGEM)

Overview: Python Glacier Evolution Model (PyGEM) is an open-source glacier evolution model coded in Python that models the transient evolution of glaciers. Each glacier is modeled independently using a monthly timestep. PyGEM has a modular framework that allows different schemes to be used for model calibration or model physics (e.g., climatic mass balance, glacier dynamics).

Manual: Details concerning the model physics, installation, and running the model may be found here: [https://github.com/drounce/PyGEM/wiki](https://pygem.readthedocs.io/en/latest/)

Usage: PyGEM is meant for large-scale glacier evolution modeling.  PyGEM<1.0.0 are no longer being actively being supported. We recommend using the new documentation listed above and contacting the lead developer (David Rounce) if you're interested in using the version that is actively being developed.

***

### Installation
PyGEM can be downloaded from the Python Package Index ([PyPI](https://pypi.org/project/pygem/)).  We recommend creating a dedicated [Anaconda](https://anaconda.org/) environment to house PyGEM.
```
conda create --name <environment_name> python=3.12
conda activate <environment_name>
pip install pygem
```
This will install all PyGEM dependencies within your conda environment, and set up PyGEM command line tools to run core model scripts.

***

### Setup
Following installation, an initialization script should to be executed.

The initialization script accomplishes two things:
1. Initializes the PyGEM configuration file *~/PyGEM/config.yaml*.  If this file already exists, an overwrite prompt will appear.
2. Downloads and unzips a series of sample data files to *~/PyGEM/*, which can also be manually downloaded [here](https://drive.google.com/file/d/1Wu4ZqpOKxnc4EYhcRHQbwGq95FoOxMfZ/view?usp=drive_link).

Run the initialization script by entering the following in the terminal:
```
initialize
```

***

### Development
If you are interested in contributing to further development of PyGEM, we recommend forking [PyGEM](https://github.com/PyGEM-Community/PyGEM) and then [cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) onto your local machine.

Note, if PyGEM was already installed via PyPI, first uninstall:
```
pip uninstall pygem
````

You can then use pip to install your locally cloned fork of PyGEM in 'editable' mode to easily facilitate development like so:
```
pip install -e /path/to/your/cloned/pygem/fork/
```

***

### Model Testing

To support model testing and demonstration, a series of Jupyter notebooks can be found within a separate [PyGEM-notebooks](https://github.com/PyGEM-Community/PyGEM-notebooks) repository. 

***

### About

|  |  |
|---|---|
| **Version** | [![Pypi version](https://img.shields.io/pypi/v/pygem.svg)](https://pypi.python.org/pypi/pygem) &nbsp; [![Supported python versions](https://img.shields.io/pypi/pyversions/pygem.svg)](https://pypi.python.org/pypi/pygem) |
| **Citation** | [![Rounce et al. (2023; Science)](https://img.shields.io/badge/Citation-Rounce%20et%20al.%20(2023;%20Science)%20paper-orange.svg)](https://www.science.org/doi/10.1126/science.abo1324) |
| **License** | [![BSD-3-Clause License](https://img.shields.io/pypi/l/pygem.svg)](https://github.com/PyGEM-Community/PyGEM/blob/master/LICENSE) |
