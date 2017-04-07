from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

setup(
  name = 'BayesianEconometrics Framework',
  ext_modules = cythonize([
                          "KF.pyx",
                            "KFMV.pyx",
                           # "KFUC.pyx",
                           # "KalmanFilter_UniCy.pyx",
                           ]),
  include_dirs=[numpy.get_include()]
)



# python setup.py build_ext --inplace
# cython -a KF.pyx