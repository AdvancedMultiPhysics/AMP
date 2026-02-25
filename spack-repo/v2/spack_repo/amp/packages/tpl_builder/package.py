# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage
from spack.package import *

class TplBuilder(CMakePackage, CudaPackage, ROCmPackage):
    """A CMake wrapper for dependencies used by SAMRApps and AMP."""

    homepage = "https://github.com/AdvancedMultiPhysics/TPL-builder"
    git = "https://github.com/AdvancedMultiPhysics/TPL-builder.git"

    maintainers("bobby-philip", "gllongo", "rbberger")

    license("UNKNOWN")

    version("master", branch="master")
    version("2.1.3", tag="2.1.3", commit="1c861c271154fa692259dade523d7e8b7728a58c")
    version("2.1.2", tag="2.1.2", commit="1a9c083972e13e1f54eda2a0e70dc297342d84e5")
    version("2.1.0", tag="2.1.0", commit="f2018b32623ea4a2f61fd0e7f7087ecb9b955eb5")

    variant('cxxstd', default='17', values=('17', '20', '23'), description='Build with support for C++17, 20 or 23')
    variant("stacktrace", default=False, description="Build with support for Stacktrace")
    variant("timerutility", default=False, description="Build with support for TimerUtility")
    variant("lapack", default=False, description="Build with support for lapack")
    variant("lapackwrappers", default=False, description="Build with support for lapackwrappers")
    variant("hypre", default=False, description="Build with support for hypre")
    variant("kokkos", default=False, description="Build with support for Kokkos")
    variant("kokkos-kernels", default=False, description="Build with support for KokkosKernels")
    variant("mpi", default=False, description="Build with MPI support")
    variant("openmp", default=False, description="Build with OpenMP support")
    variant("shared", default=False, description="Build shared libraries")
    variant("libmesh", default=False, description="Build with support for libmesh")
    variant("petsc", default=False, description="Build with support for petsc")
    variant("trilinos", default=False, description="Build with support for trilinos")
    variant("test_gpus", default="auto", values=lambda x: x == "auto" or x.isdigit(), multi=False, description="Build with NUMBER_OF_GPUs setting, defaults to use the number of gpus available")
    variant("no_implicit_links", default=False, description="turn off injection of implicit lib links")
    variant(
        "cxxstd",
        default="17",
        values=("17", "20", "23"),
        multi=False,
        description="C++ standard",
    )

    conflicts("cxxstd=20", when="@:2.1.0") #c++ 20 is only compatible with tpl-builder 2.1.2 and up
    conflicts("cxxstd=23", when="@:2.1.0") #c++ 23 is only compatible with tpl-builder 2.1.2 and up

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("fortran", type="build")

    depends_on("git", type="build")

    depends_on("stacktrace~shared", when="~shared+stacktrace")
    depends_on("stacktrace+shared", when="+shared+stacktrace")
    depends_on("stacktrace+mpi", when="+mpi+stacktrace")
    depends_on("stacktrace~mpi", when="~mpi+stacktrace")
    depends_on("stacktrace+timerutility", when="+timerutility+stacktrace")
    depends_on("stacktrace~timerutility", when="~timerutility+stacktrace")
    depends_on(f"stacktrace cxxstd=17", when=f"+stacktrace cxxstd=17")
    depends_on(f"stacktrace cxxstd=20", when=f"+stacktrace cxxstd=20")
    depends_on(f"stacktrace cxxstd=23", when=f"+stacktrace cxxstd=23")

    depends_on("timerutility~shared", when="~shared+timerutility")
    depends_on("timerutility+shared", when="+shared+timerutility")
    depends_on("timerutility~no_implicit_links", when="~no_implicit_links+timerutility")
    depends_on("timerutility+no_implicit_links", when="+no_implicit_links+timerutility")
    depends_on("timerutility+mpi", when="+mpi+timerutility")
    depends_on("timerutility~mpi", when="~mpi+timerutility")
    depends_on(f"timerutility cxxstd=17", when=f"+timerutility cxxstd=17")
    depends_on(f"timerutility cxxstd=20", when=f"+timerutility cxxstd=20")
    depends_on(f"timerutility cxxstd=23", when=f"+timerutility cxxstd=23")

    depends_on("lapackwrappers~shared", when="~shared+lapackwrappers")
    depends_on("lapackwrappers+shared", when="+shared+lapackwrappers")

    depends_on("lapack", when="+lapack")

    depends_on("hypre", when="+hypre")

    depends_on("kokkos", when="+kokkos")
    depends_on("kokkos-kernels", when="+kokkos-kernels")

    depends_on("blas", when="+lapack")

    depends_on("libmesh+exodusii+netcdf+metis", when="+libmesh")

    depends_on("petsc", when="+petsc")
    depends_on("trilinos", when="+trilinos")

    for _flag in list(CudaPackage.cuda_arch_values):
        depends_on(f"hypre cuda_arch={_flag}", when=f"+hypre+cuda cuda_arch={_flag}")
        depends_on(f"kokkos cuda_arch={_flag}", when=f"+kokkos+cuda cuda_arch={_flag}")
        depends_on(f"trilinos cuda_arch={_flag}", when=f"+trilinos+cuda cuda_arch={_flag}")

    for _flag in ROCmPackage.amdgpu_targets:
        depends_on(f"hypre amdgpu_target={_flag}", when=f"+hypre+rocm amdgpu_target={_flag}")
        depends_on(f"kokkos amdgpu_target={_flag}", when=f"+kokkos+rocm amdgpu_target={_flag}")
        depends_on(f"trilinos amdgpu_target={_flag}", when=f"+trilinos+rocm amdgpu_target={_flag}")

    # MPI related dependencies
    depends_on("mpi", when="+mpi")

    phases = ["cmake", "build"]

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
            self.define("INSTALL_DIR", spec.prefix),
            self.define("DISABLE_ALL_TESTS", True),
            self.define_from_variant("CMAKE_CXX_STANDARD", "cxxstd"),
            self.define_from_variant("ENABLE_SHARED", "shared"),
            self.define_from_variant("USE_MPI", "mpi"),
            self.define("MPI_SKIP_SEARCH", False),
            self.define_from_variant("USE_OPENMP", "openmp"),
            self.define("DISABLE_GOLD", True),
            self.define("CFLAGS", self.compiler.cc_pic_flag),
            self.define("CXXFLAGS", self.compiler.cxx_pic_flag),
            self.define("FFLAGS", self.compiler.fc_pic_flag),
            self.define('CMAKE_C_COMPILER',   spack_cc),
            self.define('CMAKE_CXX_COMPILER', spack_cxx),
            self.define('CMAKE_Fortran_COMPILER', spack_fc),
        ]


        if spec.satisfies("+trilinos"):
            if '~kokkos' in spec['trilinos']:
                options.extend( [ self.define("DISABLE_TRILINOS_KOKKOS", True) ] )

        if spec.satisfies("+cuda"):
            cuda_arch = spec.variants["cuda_arch"].value
            cuda_flags = ["-extended-lambda", "--expt-relaxed-constexpr"]
            if cuda_arch[0] != "none":
                options.extend(
                    [
                        self.define("USE_CUDA", True),
                        self.define(
                            "CMAKE_CUDA_COMPILER", join_path(spec["cuda"].prefix.bin, "nvcc")
                        ),
                        self.define("CMAKE_CUDA_ARCHITECTURES", cuda_arch),
                        self.define("CMAKE_CUDA_FLAGS", " ".join(cuda_flags)),
                        self.define_from_variant("CMAKE_CUDA_STANDARD", "cxxstd"),
                    ]
                )

        hip_flags = []

        if spec.satisfies("+shared +rocm"):
            hip_flags.append(self.compiler.cxx_pic_flag)
            
        if spec.satisfies("+mpi +rocm"):
            hip_flags.append(spec['mpi'].headers.include_flags)
            options.extend( [self.define('CMAKE_HIP_HOST_COMPILER', spec['mpi'].mpicxx),])

        if spec.satisfies("+rocm"):
            amdgpu_target = spec.variants["amdgpu_target"].value
            if amdgpu_target[0] != "none":
                hip_flags.append("")
                options.extend(
                    [
                        self.define("USE_HIP", True),
                        self.define(
                            "CMAKE_HIP_COMPILER",
                            join_path(spec["llvm-amdgpu"].prefix.bin, "amdclang++"),
                        ),
                        self.define("CMAKE_HIP_ARCHITECTURES", amdgpu_target),
                        self.define("CMAKE_HIP_FLAGS", " ".join(hip_flags)),
                        self.define_from_variant("CMAKE_HIP_STANDARD", "cxxstd"),
                    ]
                )

        tpl_list = []

        if spec.satisfies("+lapack"):
            tpl_list.append("LAPACK")
            if spec.satisfies("^[virtuals=lapack] intel-mkl"):
                options.append(self.define("LAPACK_INSTALL_DIR", spec["lapack"].prefix.mkl))
            elif spec.satisfies("^[virtuals=lapack] intel-oneapi-mkl"):
                options.append(
                    self.define(
                        "LAPACK_INSTALL_DIR", spec["intel-oneapi-mkl"].package.component_prefix
                    )
                )
            else:
                options.append(self.define("LAPACK_INSTALL_DIR", spec["lapack"].prefix))

            blas, lapack = spec["blas"].libs, spec["lapack"].libs
            options.extend(
                [
                    self.define("BLAS_LIBRARY_NAMES", ";".join(blas.names)),
                    self.define("BLAS_LIBRARY_DIRS", ";".join(blas.directories)),
                    self.define("LAPACK_LIBRARY_NAMES", ";".join(lapack.names)),
                    self.define("LAPACK_LIBRARY_DIRS", ";".join(lapack.directories)),
                ]
            )

        if spec.variants["test_gpus"].value != "auto":
            options.append(self.define("NUMBER_OF_GPUS", spec.variants["test_gpus"].value))

        for vname in ("stacktrace", "hypre", "kokkos", "kokkos-kernels", "libmesh", "petsc", "timerutility", "lapackwrappers", "trilinos"):
            if spec.satisfies(f"+{vname}"):
                tpl_name = "TIMER" if vname == "timerutility" else "LAPACK_WRAPPERS" if vname == "lapackwrappers" else "KOKKOSKERNELS" if vname == "kokkos-kernels" else vname.upper()
                tpl_list.append(tpl_name)
                options.append(self.define(f"{tpl_name}_INSTALL_DIR", spec[vname].prefix))

        options.append(self.define("TPL_LIST", ";".join(tpl_list)))
        return options

    @run_after("build")
    def filter_compilers(self):
        kwargs = {"ignore_absent": True, "backup": False, "string": True}
        filenames = [join_path(self.prefix, "TPLsConfig.cmake")]

        filter_file(spack_cc, self.compiler.cc, *filenames, **kwargs)
        filter_file(spack_cxx, self.compiler.cxx, *filenames, **kwargs)
        filter_file(spack_fc, self.compiler.fc, *filenames, **kwargs)
