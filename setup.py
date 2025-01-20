import os
import sys
from setuptools import find_packages, setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

NAME = "survivalenv"
DESCRIPTION = "An environment"
REPOSITORY = "https://github.com/ljmanso/survivalenv"
EMAIL = "luis.manso@gmail.com"
AUTHOR = "Luis J. Manso"
VERSION = "0.1.0"

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

REQUIRED = [
    "gymnasium==0.29.1",
#    "mujoco==2.3.0",
    "numpy",
    "torch",
    "torchvideo"
]

EXCLUDES=["environment_configs", "tests"]

setup(
    include_package_data=True,
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=REPOSITORY,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=REQUIRED,
    license="GPL-3",
)
