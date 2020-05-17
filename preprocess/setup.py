
import setuptools
from setuptools import find_packages

setuptools.setup(
    name='preproc',
    version='V1',
    install_requires=['apache-beam[gcp]==2.16.0',
                        'google-cloud-bigquery',
                        'google-cloud'],
    packages=setuptools.find_packages()
)
