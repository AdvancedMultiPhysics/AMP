# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
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
    version("4.0.3", tag="4.0.3", commit="079f6b0ecf5c851b3323c875295d1240638db8b9")
    version("4.0.1", tag="4.0.1", commit="808071edd31ea15e3c92b90e63bf7165e83e0588")
    version("4.0.0", tag="4.0.0", commit="7ebbcfef5b5c9d36e828a2da2d27e2106499e454")
    version("3.1.0", tag="3.1.0", commit="c8a52e6f3124e43ebce944ee3fae8b9a994c4dbe")
    
    variant("mpi", default=True, description="Build with MPI support")
    variant("hypre", default=False, description="Build with support for hypre")
    variant("kokkos", default=False, description="Build with support for Kokkos")
    variant("kokkos-kernels", default=False, description="Build with support for KokkosKernels")
    variant("openmp", default=False, description="Build with OpenMP support")
    variant("shared", default=False, description="Build shared libraries")
    variant("libmesh", default=False, description="Build with support for libmesh")
    variant("petsc", default=False, description="Build with support for petsc")
    variant("timerutility", default=False, description="Build with support for TimerUtility")
    variant("trilinos", default=False, description="Build with support for Trilinos")
    variant(
        "cxxstd",
        default="17",
        values=("17", "20", "23"),
        multi=False,
        description="C++ standard",
    )

    conflicts("cxxstd=20", when="@:4.0.0") #c++ 20 is only compatible with amp 4.0.1 and up
    conflicts("cxxstd=23", when="@:4.0.0") #c++ 23 is only compatible with amp 4.0.1 and up

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("fortran", type="build")

    depends_on("cmake@3.26.0:")
    depends_on("tpl-builder+stacktrace")
    depends_on("tpl-builder+stacktrace+timerutility", when="+timerutility")

    tpl_depends = ["hypre", "kokkos", "kokkos-kernels", "mpi", "openmp", "cuda", "rocm", "shared","libmesh", "petsc", "trilinos"]

    for v in tpl_depends:
        depends_on(f"tpl-builder+{v}", when=f"+{v}")
        depends_on(f"tpl-builder~{v}", when=f"~{v}")

    depends_on(f"tpl-builder cxxstd=17", when="cxxstd=17")
    depends_on(f"tpl-builder cxxstd=20", when="cxxstd=20")
    depends_on(f"tpl-builder cxxstd=23", when="cxxstd=23")
 
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
        ]

        if "+rocm" in spec:
            options.append(self.define("COMPILE_CXX_AS_HIP", True))
            # since there is no Spack compiler wrapper for HIP compiler, pass extra rpaths directly
            options.append(self.define("CMAKE_EXE_LINKER_FLAGS", " ".join([f"-Wl,-rpath={p}" for p in self.compiler.extra_rpaths])))

        return options
