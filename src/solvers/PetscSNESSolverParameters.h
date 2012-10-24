#ifndef included_AMP_PetscSNESSolverParameters
#define included_AMP_PetscSNESSolverParameters

#ifndef included_Pointer
#include "boost/shared_ptr.hpp"
#endif


#include "utils/Database.h"
#include "utils/AMP_MPI.h"
#include "SolverStrategyParameters.h"

extern "C"{
#ifdef MPICH_SKIP_MPICXX
#define _FIX_FOR_PETSC_MPI_CXX
#undef MPICH_SKIP_MPICXX
#endif

#ifdef OMPI_SKIP_MPICXX
#define _FIX_FOR_PETSC_OMPI_CXX
#undef OMPI_SKIP_MPICXX
#endif

#include "petsc.h"

#ifdef _FIX_FOR_PETSC_OMPI_CXX
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif
#endif

#ifdef _FIX_FOR_PETSC_MPI_CXX
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#endif
#endif
}

#include "PetscKrylovSolver.h"

namespace AMP {
namespace Solver {

  /**
   * Class PetscSNESSolverParameters provides a uniform mechanism to pass
   * initialization parameters to the PetscSNESSolver solver. It contains
   * shared pointers to a PertscKrylovSolver object and a vector
   * for initial guesses. All member variables are public.
   */
  class PetscSNESSolverParameters: public SolverStrategyParameters{
  public:
    PetscSNESSolverParameters(){}
    PetscSNESSolverParameters(const boost::shared_ptr<AMP::Database> &db);
    virtual ~PetscSNESSolverParameters(){}

    AMP_MPI d_comm;

    boost::shared_ptr<PetscKrylovSolver> d_pKrylovSolver;
    boost::shared_ptr<AMP::LinearAlgebra::Vector> d_pInitialGuess;

  protected:
  private:
    
  };

}
}

#endif
