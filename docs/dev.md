(dev_pygem_target)=
# Development
Are you interested in contributing to the development of PyGEM? If so, we recommend forking the [PyGEM's GitHub repository](https://github.com/PyGEM-Community/PyGEM) and then [cloning the GitHub repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) onto your local machine.

Note, if PyGEM was already installed via PyPI, first uninstall:
```
pip uninstall pygem
````

You can then use pip to install your locally cloned fork of PyGEM in 'editable' mode like so:
```
pip install -e /path/to/your/PyGEM/clone
```

Installing a package in editable mode (also called development mode) creates a symbolic link to your source code directory (*/path/to/your/PyGEM/clone*), rather than copying the package files into the site-packages directory. This allows you to modify the package code without reinstalling it. Changes to the source code take effect immediately without needing to reinstall the package, thus efficiently facilitating development.<br><br>
Pull requests can  be made to [PyGEM's GitHub repository](https://github.com/PyGEM-Community/PyGEM).
