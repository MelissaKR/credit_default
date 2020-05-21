
from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['matplotlib','seaborn','scikit-learn','pandas','gcsfs']

setup(
    name='gradient_boosted_model',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True
)
