# phaethon
Atmosphere-interior interactions on lava ocean planets: coupling muspell, fastchem and HELIOS.

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

## Developer install

See this [developer setup guide](https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3) to set up your system to develop *phaethon* using [VS Code](https://code.visualstudio.com) and [Poetry](https://python-poetry.org).

## Tests

You can confirm that all tests pass by running `pytest` in the root directory of *phaethon*. Please add more tests if you add new features. Note that `pip install .` in the *Quick install* instructions will not install `pytest` so you will need to install `pytest` into the environment separately.

## License

Do not share this project without the authors permission!
