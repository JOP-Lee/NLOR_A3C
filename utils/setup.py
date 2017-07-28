#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: setup.py
# Author: Fan Wu <jxwufan@gmail.com>

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
        Extension(
            "bbox",
            ["bbox.pyx"],
            include_dirs=[numpy.get_include()]
            )
        ]

#setup(
#    ext_modules=cythonize("./bbox.pyx"),
#    include_dirs=[numpy.get_include()]
#)

setup(
    ext_modules=cythonize(ext_modules)
)
