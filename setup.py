#!/usr/bin/env python
from setuptools import setup

setup(
      name='DD-GAN-AE',
      version='1.0',
      description='Library for compression methods for DD-GAN',
      author='Zef Wolffs',
      packages=['ddganAE'],
      package_dir={'ddganAE': 'ddganAE'},
      package_data={'ddganAE': ['*.csv', '*.txt']},
      include_package_data=True
      )
