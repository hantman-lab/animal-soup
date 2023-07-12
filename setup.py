from setuptools import setup, find_packages
from pathlib import Path


install_requires = [
    "mesmerize-core",
    "fastplotlib",
    "ipydatagrid",
    "pandas"
\
]

with open(Path(__file__).parent.joinpath("README.md")) as f:
    readme = f.read()

with open(Path(__file__).parent.joinpath("animal_soup", "VERSION"), "r") as f:
    ver = f.read().split("\n")[0]

setup(
    name='animal_soup',
    long_description=readme,
    long_description_content_type='text/markdown',
    version=ver,
    install_requires=install_requires,
    python_requires='>=3.8',
    packages=find_packages(),
    include_package_data=True,
    author="Caitlin Lewis",
    author_email='',
    url='https://github.com/hantman-lab/animal-soup',
    license='GPL v3.0',
    description='Hantman Lab behavioral visualization package using fastplotlib'
)