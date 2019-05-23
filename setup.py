from __future__ import absolute_import
import os
from setuptools import setup

CURRENT_DIR = os.path.dirname(__file__)

setup(
    name="shakespeare",
    version=open(os.path.join(CURRENT_DIR, 'VERSION')).read().strip(),
    author="Yage (Jacob) Wang, William Kinsman, Chenyu (Oliver) Ha, Harshal Samant",
    author_email="ywang2@inovalon.com; wkinsman@inovalon.com; cha@inovalon.com; hsamant@inovalon.com",
    maintainer='Yage (Jacob) Wang',
    maintainer_email='ywang2@inovalon.com',
    packages=["shakespeare"],
    install_requires=[
            "six>=1.12.0",
            "pytz>=2018.9",
            "python_dateutil>=2.7.3",
            "kiwisolver>=1.0.1",
            "pyparsing>=2.3.1",
            "cycler>=0.10.0",
            "scipy>=1.2.0",
            "numpy>=1.16.1",
            "pandas>=0.24.1",
            "pyodbc>=4.0.25",
            "matplotlib>=3.0.2",
            "scikit-learn>=0.20.2",
            "xgboost==0.72"],
    include_package_data=True,
    url="inovalon.com",
    description='''A python package designed to detect CARA condition gaps'''
)