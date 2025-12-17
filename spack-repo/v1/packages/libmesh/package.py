from spack.package import *
from spack.pkg.builtin.libmesh import Libmesh


class Libmesh(Libmesh):
    """The libMesh library provides a framework for the numerical simulation of
    partial differential equations using arbitrary unstructured
    discretizations on serial and parallel platforms.

    this inherits from the builtin spack package for Libmesh in order to add version 1.8.4 as an option
    """

    version("1.8.4", commit="82c9bc42ae2296437bc751bb72830cfcf3fb405f", submodules=True)
 
