from setuptools import setup, find_packages

setup(
    name='phaethon',
    version='1.0.0',
    author='Fabian L. Seidler',
    author_email='fabian.seidler@erdw.ethz.ch',
    description='Coupling magma oceans and their atmospheres',
    long_description='Long description of my package',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        # Add other dependencies here
    ],
    package_data={
        "": [
            "phaethon/data",
        ]
    },
)
