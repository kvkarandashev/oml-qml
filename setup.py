import sys
from numpy.distutils.core import Extension, setup

from mkldiscover import mkl_exists

__author__ = "Anders S. Christensen"
__copyright__ = "Copyright 2016"
__credits__ = ["Anders S. Christensen et al. (2016) https://github.com/qmlcode/qml"]
__license__ = "MIT"
__version__ = "0.4.0.12"
__maintainer__ = "Anders S. Christensen"
__email__ = "andersbiceps@gmail.com"
__status__ = "Beta"
__description__ = "Quantum Machine Learning"
__url__ = "https://github.com/qmlcode/qml"


FORTRAN = "f90"

# GNU (default)
#COMPILER_FLAGS = ["-O3", "-fopenmp", "-m64", "-march=native", "-fPIC",
#                    "-Wno-maybe-uninitialized", "-Wno-unused-function", "-Wno-cpp"]
COMPILER_FLAGS = ["-g", "-fcheck=all", "-Wall", "-fbacktrace", "-m64", "-march=native", "-fPIC",
                    "-Wno-maybe-uninitialized", "-Wno-unused-function", "-Wno-cpp"]
LINKER_FLAGS = ["-lgomp"]
MATH_LINKER_FLAGS = ["-lblas", "-llapack"]

# UNCOMMENT TO FORCE LINKING TO MKL with GNU compilers:
if mkl_exists(verbose=True):
    LINKER_FLAGS = ["-lgomp", " -lpthread", "-lm", "-ldl"]
    MATH_LINKER_FLAGS = ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]

# For clang without OpenMP: (i.e. most Apple/mac system)
if sys.platform == "darwin" and all(["gnu" not in arg for arg in sys.argv]):
    COMPILER_FLAGS = ["-O3", "-m64", "-march=native", "-fPIC"]
    LINKER_FLAGS = []
    MATH_LINKER_FLAGS = ["-lblas", "-llapack"]


# Intel
if any(["intelem" in arg for arg in sys.argv]):
    COMPILER_FLAGS = ["-xHost", "-O3", "-axAVX", "-qopenmp"]
    LINKER_FLAGS = ["-liomp5", " -lpthread", "-lm", "-ldl"]
    MATH_LINKER_FLAGS = ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]




ext_ffchl_module = Extension(name = 'ffchl_module',
                          sources = [
                                'qml/ffchl_module.f90',
                                'qml/ffchl_scalar_kernels.f90',
                            ],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS ,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_ffchl_scalar_kernels = Extension(name = 'ffchl_scalar_kernels',
                          sources = ['qml/ffchl_scalar_kernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_ffchl_vector_kernels = Extension(name = 'ffchl_vector_kernels',
                          sources = ['qml/ffchl_vector_kernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS + MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_farad_kernels = Extension(name = 'farad_kernels',
                          sources = ['qml/farad_kernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fcho_solve = Extension(name = 'fcho_solve',
                          sources = ['qml/fcho_solve.f90',
                                    ],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = MATH_LINKER_FLAGS + LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fdistance = Extension(name = 'fdistance',
                          sources = ['qml/fdistance.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fkernels = Extension(name = 'fkernels',
                          sources = ['qml/fkernels.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_frepresentations = Extension(name = 'frepresentations',
                          sources = ['qml/frepresentations.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = MATH_LINKER_FLAGS + LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fslatm = Extension(name = 'fslatm',
                          sources = ['qml/fslatm.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_foml_module = Extension(name = 'foml_module',
                          sources = ['qml/foml_module.f90','qml/foml_kernels.f90',],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS+MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_foml_representations = Extension(name = 'foml_representations',
                          sources = ['qml/foml_representations.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS+MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])


ext_foml_kernels = Extension(name = 'foml_kernels',
                          sources = ['qml/foml_kernels.f90', 'qml/foml_module.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS+MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_factive_learning = Extension(name = 'factive_learning',
                          sources = ['qml/factive_learning.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS+MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

ext_fkernels_wders = Extension(name = 'fkernels_wders',
                          sources = ['qml/fkernels_wders.f90', 'qml/fkernels_wders_module.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS+MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])


ext_fkernels_wders_module=Extension(name = 'fkernels_wders_module',
                          sources = ['qml/fkernels_wders_module.f90', 'qml/fkernels_wders.f90'],
                          extra_f90_compile_args = COMPILER_FLAGS,
                          extra_f77_compile_args = COMPILER_FLAGS,
                          extra_compile_args = COMPILER_FLAGS,
                          extra_link_args = LINKER_FLAGS+MATH_LINKER_FLAGS,
                          language = FORTRAN,
                          f2py_options=['--quiet'])

# use README.md as long description
def readme():
    with open('README.md') as f:
        return f.read()

def setup_pepytools():

    setup(

        name="qml",
        packages=['qml'],

        # metadata
        version=__version__,
        author=__author__,
        author_email=__email__,
        platforms = 'Any',
        description = __description__,
        long_description = readme(),
        keywords = ['Machine Learning', 'Quantum Chemistry'],
        classifiers = [],
        url = __url__,

        # set up package contents

        ext_package = 'qml',
        ext_modules = [
              ext_ffchl_module,
              ext_farad_kernels,
              ext_fcho_solve,
              ext_fdistance,
              ext_fkernels,
              ext_fslatm,
              ext_frepresentations,
              ext_foml_module,
              ext_foml_representations,
              ext_foml_kernels,
              ext_factive_learning,
              ext_fkernels_wders,
              ext_fkernels_wders_module
        ],
)

if __name__ == '__main__':

    setup_pepytools()
