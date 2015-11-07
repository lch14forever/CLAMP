from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'DTW module',
  ext_modules = cythonize("dtw.pyx"),
)
