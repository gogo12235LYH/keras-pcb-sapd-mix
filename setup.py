import setuptools
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        Extension(
            'utils.compute_overlap',
            ['utils/compute_overlap.pyx']
        )),
    include_dirs=[numpy.get_include()]
)

# python setup.py build_ext --inplace
