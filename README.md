## Python Glacier Evolution Model (PyGEM)

Overview: Python Glacier Evolution Model (PyGEM) is an open-source glacier evolution model coded in Python that models the transient evolution of glaciers. Each glacier is modeled independently using a monthly timestep. PyGEM has a modular framework that allows different schemes to be used for model calibration or model physics (e.g., climatic mass balance, glacier dynamics).  In the newest version under development, PyGEM is working to become compatible with the Open Global Glacier Model (OGGM; https://oggm.org/).

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

Credits: If using PyGEM for scientific applications, please cite the following:
Rounce, D.R., Hock, R., Maussion, F., Hugonnet, R., Kochtitzky, W., Huss, M., Berthier, E., Brinkerhoff, D., Compagno, L., Copland, L., Farinotti, D., Menounos, B., and McNabb, R.W. “Global glacier change in the 21st century: Every increase in temperature matters”, Science, 379(6627), pp. 78-83, (2023), doi:10.1126/science.abo1324.

License: PyGEM uses an MIT license.
