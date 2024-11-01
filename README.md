# phaethon
Calculate the structure, composition and emission spectra of lava-planet atmospheres! 

## Quick install

If you want to use a GUI to install the code, particularly if you are a Windows or Spyder user, see [here](https://gist.github.com/djbower/c82b4a70a3c3c74ad26dc572edefdd34). Otherwise, follow the instructions below to install the code using the terminal on a Mac or Linux system.

### 1. Obtain the source code

Navigate to a location on your computer and obtain the source code:

    git clone git@github.com:ExPlanetology/phaethon.git
    cd phaethon

### 2. Create a Python environment

The basic procedure is to install *phaethon* into a Python environment. For example, if you are using a Conda distribution to create Python environments (e.g. [Anaconda](https://www.anaconda.com/download)), create a new environment to install *phaethon*. *phaethon* requires Python >= 3.10:

    conda create -n phaethon python
    conda activate phaethon

### 3. Install into the environment

Install *phaethon* into the environment using either (a) [Poetry](https://python-poetry.org) or (b) [pip](https://pip.pypa.io/en/stable/getting-started/). This [Gist](https://gist.github.com/djbower/e9538e7eb5ed3deaf3c4de9dea41ebcd) provides further information.

#### 3a. Option 1: Poetry

This requires that you have you have [Poetry](https://python-poetry.org) installed:

    poetry install

#### 3b. Option 2: pip

Alternatively, use `pip`, where you can include the `-e` option if you want an [editable install ](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

    pip install .

If desired, you will need to manually install the dependencies for testing and documentation (these are automatically installed by Poetry but not when using `pip`). See the additional dependencies to install in `pyproject.toml`. For example:

    pip install pytest

### 4. Optain or create opaciy files

#### Option 1: Precomputed opacities
The opacity files used for Seidler et al. 2024 can be found on zenodo at https://doi.org/10.5281/zenodo.13837367. Please download the repository and extract it. You can find the opacity files in zenodo/input/opacities/. Note that the relevant files are the interpolated k-tables, indicated by their name which should end in '_ip_kdistr.h5'. When running phaethon, you should point it to the location of the opacity files. This is done via the 'opacity_path' keyword:

    pipeline: PhaethonPipeline(
        opacity_path="path/to/opacities/",
        **other_args,  # Additional necessary keyword arguments
    )

Please have a look at 'phaethon_demo.py' for a hands-on example!

#### Option 2: Compile them yourself

If you want to compile your own opacities, please follow the instructions for HELIOS: https://heliosexo.readthedocs.io/en/latest/sections/tutorial.html#include-more-opacities
Similarly to Option 1, pass the location of the opacity files to phaethon.

## Developer install

See this [developer setup guide](https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3) to set up your system to develop *phaethon* using [VS Code](https://code.visualstudio.com) and [Poetry](https://python-poetry.org).

## Tests

You can confirm that all tests pass by running `pytest` in the root directory of *phaethon*. Please add more tests if you add new features. Note that `pip install .` in the *Quick install* instructions will not install `pytest` so you will need to install `pytest` into the environment separately.

## License

This project is licensed under the Affero General Public License (AGPL). See the [LICENSE](LICENSE) file for more details.
