(install_pygem_target)=
# Installing PyGEM
The model is stored in two repositories [(see model structure)](model_structure_and_workflow_target) that are installed via PyPI and github as described below.

## Setup Conda Environment
A conda environment is a directory that contains a specific collection of installed packages. The use of environments reduces issues caused by package dependencies. The model is designed to be compatible with OGGM. We therefore get started by following the [installation instructions from OGGM](https://docs.oggm.org/en/stable/installing-oggm.html).

Once your conda environment is setup for OGGM, add the core of PyGEM using pip.
```
pip install pygem
```

This will provide you with a conda environment that has the basic functionality to run simple calibration options (e.g., 'HH2015', 'HH2015mod') and simulations. If you want to use the emulators and Bayesian inference, the advanced environment is required.

### Developing PyGEM
Are you interested in developing PyGEM? If so, we recommend forking the [PyGEM's github repository](https://github.com/PyGEM-Community/PyGEM) and then [cloning the github repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) onto your local machine.

Note, if PyGEM was already installed via PyPI, first uninstall:
```
pip uninstall pygem
````

You can then use pip to install your locally cloned fork of PyGEM in 'editable' mode like so:
```
pip install -e /path/to/your/PyGEM/clone
```

### Advanced environment: GPyTorch (emulator only)
If you want to use the emulators additional packages are required. The simplest way to construct this environment is to add <em>- gpytorch</em> to the oggm_env.yml. Then create the environment using:
```
conda env create -f oggm_env_wemulator.yml
```

The only way to find out if your package dependencies work is to test it by running the model. Make sure to install PyGEM-Scripts and then [test the model](test_model_target).

**If your environment is not set up properly, errors will arise related to missing modules. We recommend that you work through adding the missing modules and use StackOverflow to identify any additional debugging issues related to potential missing modules or module dependencies.** As of July 2023, adding GPyTorch to OGGM's existing environment was quite simple using the .yml file provided by [OGGM](https://docs.oggm.org/en/stable/installing-oggm.html).


### Advanced environment: PyMC2 and GPyTorch
If you want to use the emulators or Bayesian inference associated with PyGEM additional packages are required.

```{warning}
The current dependencies are fairly tricky as PyMC2 is now fairly old and no longer supported. We anticipate developing code that relies on more user-friendly packages in the future, but for the time being have patience and do your best to work through the environment issues.
```
PyMC2 requires Python3.8. Therefore, you may want to re-install your original environment and explicitly specify Python 3.8. Once your environment is setup, activate your environment.

Next, install the modules required for the emulator.
```
pip install torch
pip install gpytorch
```

Next, install the modules required for Bayesian inference.
```
pip install pymc
```

```{warning}
You may try to replace pip install with conda install as conda may help solve dependencies. However, creating this environment can take a long time (> 1 hr), so be patient.
```

The only way to find out if your package dependencies work is to test it by running the model. Make sure to install PyGEM-Scripts and then [test the model](test_model_target).

**If your environment is not set up properly, errors will arise related to missing modules. We recommend that you work through adding the missing modules and use StackOverflow to identify any additional debugging issues related to potential missing modules or module dependencies.** Getting a correct package installed took the lead developer over a day and unfortunately other users have commented that the directions used by the lead developer have not worked for others due to newer computers or different operating systems.


## Install PyGEM-Scripts
The scripts that are used to run PyGEM are located in the [PyGEM-Scripts repository](https://github.com/PyGEM-Community/PyGEM-scripts) on github. To run the model, you can either (i) clone the repository or (ii) fork the repository to develop/add your own scripts. For instructions, follow github’s instructions on [cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) or [forking a repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo). Once the repository is installed on your local machine, you can run the model from this directory.

```{note}
Be sure that your [directory structure](directory_structure_target) is setup properly before you try running the model!
```