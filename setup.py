#!/usr/bin/env python
from setuptools import setup

setup(name='DD-GAN-AE',
      version='1.0',
      description='Library for compression methods for DD-GAN',
      author='Zef Wolffs',
      packages=['ddgan-ae'],
      package_dir={'ddgan-ae': 'ddgan-ae'},
      package_data={'ddgan-ae': ['*.csv', '*.txt']},
      include_package_data=True
      )
