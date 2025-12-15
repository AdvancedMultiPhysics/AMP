#include "AMP/solvers/petsc/PetscMonitor.h"
#include "AMP/utils/UtilityMacros.h"

#include "petsc/private/petscimpl.h"
#include "petscksp.h"
#include "petscsnes.h"

#include <iomanip>
#include <iostream>

namespace AMP {


/********************************************************************
 *  Constructors/Deconstructors                                      *
 ********************************************************************/
PetscMonitor::PetscMonitor( AMP_MPI comm )
{
    d_comm = comm;
    AMP_ASSERT( !d_comm.isNull() );
}
PetscMonitor::~PetscMonitor() = default;


/********************************************************************
 *  Remove the monitor option from the input string                  *
 ********************************************************************/
std::string PetscMonitor::removeMonitor( std::string options )
{
    size_t i2 = options.find( "monitor" );
    if ( i2 == std::string::npos )
        return options;
    i2 += 6;
    size_t i1 = options.find_last_of( "-", i2 );
    options.erase( i1, i2 - i1 + 1 );
    return options;
}


/********************************************************************
 *  Routines to provide petsc with function pointers for monitoring  *
 ********************************************************************/
PetscErrorCode PetscMonitor::monitorKSP( KSP ksp, PetscInt iteration, PetscReal L2norm, void *ctx )
{
    auto *monitor = reinterpret_cast<PetscMonitor *>( ctx );
    monitor->printKSPStatus( ksp, iteration, L2norm );
    return 0;
}
PetscErrorCode
PetscMonitor::monitorSNES( SNES snes, PetscInt iteration, PetscReal L2norm, void *ctx )
{
    auto *monitor = reinterpret_cast<PetscMonitor *>( ctx );
    monitor->printSNESStatus( snes, iteration, L2norm );
    return 0;
}
void PetscMonitor::printKSPStatus( KSP ksp, PetscInt iteration, PetscReal L2norm )
{
    if ( d_comm.getRank() == 0 ) {
        std::string indent = "  ";
        for ( PetscInt i = 0; i < ( (PetscObject) ksp )->tablevel; i++ )
            indent += "  ";
        AMP::pout << indent << iteration << " KSP Residual norm " << std::scientific
                  << std::setprecision( 12 ) << L2norm << std::endl;
    }
}
void PetscMonitor::printSNESStatus( SNES snes, PetscInt iteration, PetscReal L2norm )
{
    if ( d_comm.getRank() == 0 ) {
        std::string indent = "  ";
        for ( PetscInt i = 0; i < ( (PetscObject) snes )->tablevel; i++ )
            indent += "  ";
        AMP::pout << indent << iteration << " SNES Function norm " << std::scientific
                  << std::setprecision( 12 ) << L2norm << std::endl;
    }
}


} // namespace AMP
