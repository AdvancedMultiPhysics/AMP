# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack.package import *


class Timerutility(CMakePackage):
    """A library for profiling and tracing."""

    homepage = "https://github.com/AdvancedMultiPhysics/timerutility"
    git = "https://github.com/AdvancedMultiPhysics/timerutility.git"

    maintainers("bobby-philip", "gllongo", "rbberger")

    license("UNKNOWN")

    version("master", branch="master")

    variant("mpi", default=True, description="build with mpi")
    variant("shared", default=False, description="Build shared libraries")
    variant("pic", default=False, description="Produce position-independent code")
    variant(
        "cxxstd",
        default="17",
        values=("17", "20", "23"),
        multi=False,
        description="C++ standard",
    )

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("fortran", type="build")

    depends_on("cmake@3.26.0:", type="build")
    depends_on("mpi", when="+mpi")

    def cmake_args(self):
        args = [
            self.define("Timer_INSTALL_DIR", self.prefix),
            self.define_from_variant("USE_MPI", "mpi"),
            self.define_from_variant("ENABLE_SHARED", "shared"),
            self.define("DISABLE_NEW_OVERLOAD", True),
            self.define("CFLAGS", self.compiler.cc_pic_flag),
            self.define("CXXFLAGS", self.compiler.cxx_pic_flag),
            self.define("FFLAGS", self.compiler.fc_pic_flag),
            self.define('CMAKE_C_COMPILER',   spack_cc),
            self.define('CMAKE_CXX_COMPILER', spack_cxx),
            self.define('CMAKE_Fortran_COMPILER', spack_fc),
            self.define_from_variant("CMAKE_CXX_STANDARD", "cxxstd")
        ]

        return args

