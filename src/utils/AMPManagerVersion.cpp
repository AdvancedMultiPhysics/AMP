#include "AMP/AMP_TPLs.h"
#include "AMP/AMP_Version.h"
#include "AMP/utils/AMPManager.h"

#include <sstream>


#ifdef AMP_USE_LAPACK_WRAPPERS
    #include "LapackWrappers.h"
#endif
#ifdef AMP_USE_PETSC
    #include "petscversion.h"
#endif
#ifdef AMP_USE_TIMER
    #include "TimerUtilityVersion.h"
#endif
#ifdef AMP_USE_TRILINOS
    #include "Trilinos_version.h"
#endif
#ifdef AMP_USE_LIBMESH
    #include "libmesh/libmesh_version.h"
    #ifndef LIBMESH_MAJOR_VERSION
int LIBMESH_MAJOR_VERSION = 0;
    #endif
    #ifndef LIBMESH_MINOR_VERSION
int LIBMESH_MINOR_VERSION = 0;
    #endif
    #ifndef LIBMESH_MICRO_VERSION
int LIBMESH_MICRO_VERSION = 0;
    #endif
#endif
#ifdef AMP_USE_HDF5
    #include "H5public.h"
#endif
#ifdef AMP_USE_SUNDIALS
    #include "sundials/sundials_config.h"
#endif
#ifdef AMP_USE_SILO
    #include "silo.h"
#endif
#ifdef AMP_USE_SAMRAI
    #include "SAMRAI/tbox/SAMRAIManager.h"
#endif
#ifdef AMP_USE_HYPRE
    #undef HAVE_UNISTD_H
    #include "HYPRE_config.h"
#endif
#ifdef AMP_USE_KOKKOS
    #include "AMP/utils/KokkosManager.h"
#endif


/****************************************************************************
 *  Functions to return version info                                         *
 ****************************************************************************/
[[maybe_unused]] static std::ostream &operator<<( std::ostream &out,
                                                  const std::array<int, 3> &version )
{
    out << version[0] << "." << version[1] << "." << version[2];
    return out;
}
[[maybe_unused]] static std::ostream &operator<<( std::ostream &out,
                                                  std::initializer_list<int> &version )
{
    if ( version.size() > 0 ) {
        auto it = version.begin();
        out << '.' << *it;
        ++it;
        for ( ; it != version.end(); ++it )
            out << '.' << *it;
    }
    return out;
}
std::array<int, 3> AMP::AMPManager::revision()
{
    return { { AMP::Version::major, AMP::Version::minor, AMP::Version::build } };
}
std::string AMP::AMPManager::info()
{
    std::stringstream out;
    out << "AMP:" << std::endl;
    out << "   Version: " << revision() << std::endl;
    out << "   Hash: " << AMP::Version::short_hash << std::endl;
    out << "   C Compiler: " << AMP::Version::C << std::endl;
    out << "   C++ Compiler: " << AMP::Version::CXX << std::endl;
    out << "   Fortran Compiler: " << AMP::Version::Fortran << std::endl;
    out << "   C Compiler ID: " << AMP::Version::C_ID << std::endl;
    out << "   C++ Compiler ID: " << AMP::Version::CXX_ID << std::endl;
    out << "   Fortran Compiler ID: " << AMP::Version::Fortran_ID << std::endl;
    out << "   C Compiler Version: " << AMP::Version::C_VERSION << std::endl;
    out << "   C++ Compiler Version: " << AMP::Version::CXX_VERSION << std::endl;
    out << "   Fortran Compiler Version: " << AMP::Version::Fortran_VERSION << std::endl;
    out << "   C Flags: " << AMP::Version::C_FLAGS << std::endl;
    out << "   C++ Flags: " << AMP::Version::CXX_FLAGS << std::endl;
    out << "   Fortran Flags: " << AMP::Version::Fortran_FLAGS << std::endl;
#ifdef AMP_USE_TIMER
    auto TimerVersion = { Timer::Version::major, Timer::Version::minor, Timer::Version::build };
    out << "ProfilerApp: " << TimerVersion << std::endl;
#endif
#ifdef AMP_USE_SAMRAI
    auto SAMRAIVersion = { SAMRAI_VERSION_MAJOR, SAMRAI_VERSION_MINOR, SAMRAI_VERSION_PATCHLEVEL };
    out << "SAMRAI: " << SAMRAIVersion << std::endl;
#endif
#ifdef AMP_USE_PETSC
    auto PETSCVersion = { PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR };
    out << "PETSc: " << PETSCVersion << std::endl;
#endif
#ifdef AMP_USE_TRILINOS
    out << "Trilinos: " << TRILINOS_VERSION_STRING << std::endl;
#endif
#ifdef SUNDIALS_PACKAGE_VERSION
    out << "Sundials: " << SUNDIALS_PACKAGE_VERSION << std::endl;
#elif defined( SUNDIALS_VERSION )
    out << "Sundials: " << SUNDIALS_VERSION << std::endl;
#endif
#ifdef HYPRE_RELEASE_VERSION
    out << "Hypre: " << HYPRE_RELEASE_VERSION << std::endl;
#elif defined( HYPRE_PACKAGE_VERSION )
    out << "Hypre: " << HYPRE_PACKAGE_VERSION << std::endl;
#endif
#ifdef AMP_USE_LIBMESH
    auto libmeshVersion = { LIBMESH_MAJOR_VERSION, LIBMESH_MINOR_VERSION, LIBMESH_MICRO_VERSION };
    out << "libMesh: " << libmeshVersion << std::endl;
#endif
#ifdef AMP_USE_HDF5
    auto HDF5Version = { H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE };
    out << "HDF5: " << HDF5Version << std::endl;
#endif
#ifdef AMP_USE_SILO
    auto SILOVersion = { SILO_VERS_MAJ, SILO_VERS_MIN, SILO_VERS_PAT };
    out << "SILO: " << SILOVersion << std::endl;
#endif
#ifdef AMP_USE_MPI
    out << "MPI: " << AMP::AMP_MPI::info();
#endif
#ifdef AMP_USE_LAPACK_WRAPPERS
    out << "Lapack: " << Lapack<double>::info();
#endif
    return out.str();
}
