from setuptools import setup, find_packages
from pathlib import Path

install_requires = [
    "numpy",
    "fastplotlib",
    "ipydatagrid",
    "pandas>=1.5.0",
    "decord",
    "ipywidgets==8.0",
    "pytest",
    "glfw",
    "tqdm",
    "requests",
    "tables",
    "jupyter-rfb",
    "jupyterlab",
    "kornia",
    "omegaconf",
    "jupyterlab-widgets==3.0.7",
    "nbmake",
    "vidio",
    "lightning",
    "tensorflow",
    "matplotlib"
]

with open(Path(__file__).parent.joinpath("README.md")) as f:
    readme = f.read()

with open(Path(__file__).parent.joinpath("animal_soup", "VERSION"), "r") as f:
    ver = f.read().split("\n")[0]

setup(
    name='animal-soup',
    version=ver,
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/hantman-lab/animal-soup',
    license='GPL-3.0 license',
    author='clewis7',
    author_email='',
    python_requires='>=3.8',
    install_requires=install_requires,
    include_package_data=True,
    description='Hantman lab automated behavioral classification tool'
)
