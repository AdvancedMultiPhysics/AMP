#ifndef included_AMP_PetscMonitor
#define included_AMP_PetscMonitor

#include "AMP/utils/AMP_MPI.h"


// Forward declare a few types for PETSc
typedef int PetscErrorCode;
typedef struct _p_SNES *SNES;
typedef struct _p_KSP *KSP;

#include "petscsystypes.h"

namespace AMP {


/**
 * \class PetscMonitor
 *
 * @brief Provide a monitor that petsc can use instead of the default.
 * @details  This class provides routines for convergenece monitoring.
 *   The default convergence monitoring by petsc does not properly cleanup
 *   until PetscFinialize and creates problems when used on sub-communicators.
 */
class PetscMonitor
{
public:
    /**
     *\brief  Default constructor
     *\details  This is the default constructor that should be used
     *\param comm  Communicator to use for the monitor
     */
    explicit PetscMonitor( AMP::AMP_MPI comm );

    //!  Empty deconstructor
    ~PetscMonitor();

    //! Routine to pass to petsc for monitoring KSP
    static PetscErrorCode monitorKSP( KSP, PetscInt, PetscReal, void * );

    //! Routine to pass to petsc for monitoring SNES
    static PetscErrorCode monitorSNES( SNES, PetscInt, PetscReal, void * );

    //! Routine to pass to petsc for monitoring KSP delete
    // static PetscErrorCode (*)(void**)  getKSPMonitorDelete();

    //! Routine to pass to petsc for monitoring SNES delete
    // static PetscErrorCode (*)(void**)  getSNESMonitorDelete();

    // Function to remove the monitor options from the options
    static std::string removeMonitor( std::string options );

private:
    PetscMonitor();

    AMP::AMP_MPI d_comm;

    void printKSPStatus( KSP ksp, PetscInt iteration, PetscReal L2norm );
    void printSNESStatus( SNES snes, PetscInt iteration, PetscReal L2norm );
};
} // namespace AMP

#endif
