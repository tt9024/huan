from distutils.core import setup, Extension
import numpy
import os
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()

#replace any -O with -O3
extra_compile_args += ['-DNDEBUG','-O3']

trace_module_np = Extension('trace_module_np', \
                             sources = ['trace_np.c'], \
                             include_dirs = [numpy.get_include()], \
                             extra_compile_args=extra_compile_args)
setup(ext_modules = [trace_module_np])

