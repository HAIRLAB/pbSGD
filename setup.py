#!/usr/bin/env python

from distutils.core import setup

setup(name='pbSGD',
      version='1.0',
      description='pbSGD Optimization Algorithm for Deep Learning',
      author='Beitong Zhou',
      author_email='zhoubt@hust.edu.cn',
      url='https://github.com/HAIRLAB/pbSGD',
      packages=['pbSGD'],
      install_requires=[
          'torch>=0.4.0',
      ],
      zip_safe=False,
      python_requires='>=3.6.0')
