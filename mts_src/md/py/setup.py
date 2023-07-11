from distutils.core import setup, Extension
import numpy
import os
import sysconfig

# to compile, run:
#     python3 setup.py build_ext --inplace

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()

#replace any -O with -O3
#extra_compile_args += ['-DNDEBUG','-O3']
#extra_compile_args += ['-DTD_PARSER_TRACE']
extra_compile_args += ['-DNUMPY_COMPILE'] #['TD_PARSER_TRACE']
extra_compile_args += ['-O3','-I/home/mts/dev/src/util','-I/home/mts/dev/src/util/plcc', '-I/home/mts/dev/src/md']#, '-lrt', '-lpthread', '-static-libstdc++','-static-libgcc'] #,'shared', '-fPIC', '-static-libstdc++', '-static-libgcc']

tp_parser_module_np = Extension('td_parser_module_np', \
                             #sources = ['/home/mts/dev/src/util/plcc/ConfigureReader.cpp','/home/mts/dev/src/util/plcc/JsonUtil.cpp','/home/mts/dev/src/util/plcc/PLCC.cpp','/home/mts/dev/src/util/time_util.cpp', '/home/mts/dev/src/util/rate_limiter.cpp','/home/mts/dev/src/util/symbol_map.cpp','td_parser_np.cpp'], \
                             sources = ['td_parser_np.cpp'], \
                             include_dirs = [numpy.get_include()], \
                             extra_compile_args=extra_compile_args)
setup(ext_modules = [tp_parser_module_np])



