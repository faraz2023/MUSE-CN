
from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import os

# Ensure the build directory is inside cy_modules
build_dir = os.path.join('cy_modules', 'build')
if not os.path.exists(build_dir):
    os.makedirs(build_dir)

# Set build_lib to cy_modules directory
class custom_build_ext(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return os.path.join("cy_modules", os.path.basename(filename))

    def get_ext_fullpath(self, ext_name):
        return os.path.join(os.path.dirname(__file__), 
                          self.get_ext_filename(ext_name))


print("Building extensions...")

extensions = [
                    Extension('PrepareBatchGraph', sources = ['cy_modules/PrepareBatchGraph.pyx','cy_modules/src/lib/PrepareBatchGraph.cpp','cy_modules/src/lib/graph.cpp','cy_modules/src/lib/graph_struct.cpp',  'cy_modules/src/lib/disjoint_set.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                   Extension('graph', sources=['cy_modules/graph.pyx', 'cy_modules/src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11'], include_dirs=[np.get_include()]),
                    Extension('mvc_env', sources=['cy_modules/mvc_env.pyx', 'cy_modules/src/lib/mvc_env.cpp', 'cy_modules/src/lib/graph.cpp','cy_modules/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['cy_modules/utils.pyx', 'cy_modules/src/lib/utils.cpp', 'cy_modules/src/lib/graph.cpp', 'cy_modules/src/lib/graph_utils.cpp', 'cy_modules/src/lib/disjoint_set.cpp', 'cy_modules/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['cy_modules/nstep_replay_mem.pyx', 'cy_modules/src/lib/nstep_replay_mem.cpp', 'cy_modules/src/lib/graph.cpp', 'cy_modules/src/lib/mvc_env.cpp', 'cy_modules/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem_prioritized',sources=['cy_modules/nstep_replay_mem_prioritized.pyx', 'cy_modules/src/lib/nstep_replay_mem_prioritized.cpp','cy_modules/src/lib/graph.cpp', 'cy_modules/src/lib/mvc_env.cpp', 'cy_modules/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['cy_modules/graph_struct.pyx', 'cy_modules/src/lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11']),

                   ]
setup(
    name='graph_algorithms',
    packages=['cy_modules'],
    package_dir={'cy_modules': 'cy_modules'},
    cmdclass={'build_ext': custom_build_ext},
    ext_modules=extensions,
)