from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

setup(
  name = 'BaysianEconometrics Framework',
  ext_modules = cythonize(["KFUC.pyx","KalmanFilter_UniCy.pyx",]),
  include_dirs=[numpy.get_include()]
)



# python setup.py build_ext --inplace