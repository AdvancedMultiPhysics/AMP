# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage
from spack.package import *


class Amp(CMakePackage, CudaPackage, ROCmPackage):
    """The Advanced Multi-Physics (AMP) package.

    The Advanced Multi-Physics (AMP) package is an open source parallel
    object-oriented computational framework that is designed with single
    and multi-domain multi-physics applications in mind.
    """

    homepage = "https://github.com/AdvancedMultiPhysics/AMP"
    git = "https://github.com/AdvancedMultiPhysics/AMP.git"

    maintainers("bobby-philip", "gllongo", "rbberger")

    license("UNKNOWN")

    version("master", branch="master")
    version("4.0.0", tag="4.0.0", commit="7ebbcfef5b5c9d36e828a2da2d27e2106499e454")
    version("3.1.0", tag="3.1.0", commit="c8a52e6f3124e43ebce944ee3fae8b9a994c4dbe")

    variant("mpi", default=True, description="Build with MPI support")
    variant("hypre", default=False, description="Build with support for hypre")
    variant("kokkos", default=False, description="Build with support for Kokkos")
    variant("openmp", default=False, description="Build with OpenMP support")
    variant("shared", default=False, description="Build shared libraries")
    variant("libmesh", default=False, description="Build with support for libmesh")
    variant("petsc", default=False, description="Build with support for petsc")
    variant("timerutility", default=False, description="Build with support for TimerUtility")
    variant("trilinos", default=False, description="Build with support for Trilinos")

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("fortran", type="build")

    depends_on("cmake@3.26.0:")
    depends_on("tpl-builder+stacktrace")
    depends_on("tpl-builder+stacktrace+timerutility", when="+timerutility")

    tpl_depends = ["hypre", "kokkos", "mpi", "openmp", "cuda", "rocm", "shared","libmesh", "petsc", "trilinos"]

    for v in tpl_depends:
        depends_on(f"tpl-builder+{v}", when=f"+{v}")
        depends_on(f"tpl-builder~{v}", when=f"~{v}")

    for _flag in CudaPackage.cuda_arch_values:
        depends_on(f"tpl-builder+cuda cuda_arch={_flag}", when=f"+cuda cuda_arch={_flag}")

    for _flag in ROCmPackage.amdgpu_targets:
        depends_on(f"tpl-builder+rocm amdgpu_target={_flag}", when=f"+rocm amdgpu_target={_flag}")

    def flag_handler(self, name, flags):
        wrapper_flags = []
        build_system_flags = []
        if self.spec.satisfies("+mpi+cuda") or self.spec.satisfies("+mpi+rocm"):
            if self.spec.satisfies("^cray-mpich"):
                gtl_lib = self.spec["cray-mpich"].package.gtl_lib
                build_system_flags.extend(gtl_lib.get(name) or [])
            # we need to pass the flags via the build system.
            build_system_flags.extend(flags)
        else:
            wrapper_flags.extend(flags)
        return (wrapper_flags, [], build_system_flags)

    def cmake_args(self):
        spec = self.spec

        options = [
            self.define("TPL_DIRECTORY", spec["tpl-builder"].prefix),
            self.define("AMP_ENABLE_TESTS", self.run_tests),
            self.define("EXCLUDE_TESTS_FROM_ALL", not self.run_tests),
            self.define("AMP_ENABLE_EXAMPLES", False),
            self.define("CXX_STD", "17"),
            # prevent TPL-builder to set something else
            self.define("CMAKE_C_COMPILER", spack_cc),
            self.define("CMAKE_CXX_COMPILER", spack_cxx),
            self.define("CMAKE_Fortran_COMPILER", spack_fc),
        ]

        if "+rocm" in spec:
            options.append(self.define("COMPILE_CXX_AS_HIP", True))
            # since there is no Spack compiler wrapper for HIP compiler, pass extra rpaths directly
            options.append(self.define("CMAKE_EXE_LINKER_FLAGS", " ".join([f"-Wl,-rpath={p}" for p in self.compiler.extra_rpaths])))

        return options
