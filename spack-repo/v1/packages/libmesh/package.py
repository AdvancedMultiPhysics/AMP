from spack.package import *
from spack.pkg.builtin.libmesh import Libmesh


class Libmesh(Libmesh):
    """The libMesh library provides a framework for the numerical simulation of
    partial differential equations using arbitrary unstructured
    discretizations on serial and parallel platforms.

    this inherits from the builtin spack package for Libmesh in order to add version 1.8.4 as an option
    """

    homepage = "https://libmesh.github.io/"
    url = "https://github.com/libMesh/libmesh/releases/download/v1.0.0/libmesh-1.0.0.tar.bz2"
    git = "https://github.com/libMesh/libmesh.git"

    license("LGPL-2.1-or-later")

    version("1.8.4", commit="82c9bc42ae2296437bc751bb72830cfcf3fb405f", submodules=True)
 
