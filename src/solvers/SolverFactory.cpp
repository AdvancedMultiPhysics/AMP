/*
Copyright 2005, The Regents of the University
of California. This software was produced under
a U.S. Government contract (W-7405-ENG-36)
by Los Alamos National Laboratory, which is
operated by the University of California for the
U.S. Department of Energy. The U.S.
Government is licensed to use, reproduce, and
distribute this software. Permission is granted
to the public to copy and use this software
without charge, provided that this Notice and
any statement of authorship are reproduced on
all copies. Neither the Government nor the
University makes any warranty, express or
implied, or assumes any liability or
responsibility for the use of this software.
*/


#include "AMP/solvers/SolverFactory.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/solvers/BandedSolver.h"
#include "AMP/solvers/BiCGSTABSolver.h"
#include "AMP/solvers/CGSolver.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/DiagonalSolver.h"
#include "AMP/solvers/GMRESRSolver.h"
#include "AMP/solvers/GMRESSolver.h"
#include "AMP/solvers/NonlinearKrylovAccelerator.h"
#include "AMP/solvers/QMRCGSTABSolver.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/TFQMRSolver.h"
#include "AMP/solvers/amg/SASolver.h"
#include "AMP/utils/memory.h"
#include "AMP/solvers/UASolver.h"

#ifdef AMP_USE_PETSC
    #include "AMP/solvers/petsc/PetscKrylovSolver.h"
    #include "AMP/solvers/petsc/PetscSNESSolver.h"
#endif

#ifdef AMP_USE_HYPRE
    #include "AMP/solvers/hypre/BoomerAMGSolver.h"
    #include "AMP/solvers/hypre/HypreBiCGSTABSolver.h"
    #include "AMP/solvers/hypre/HypreGMRESSolver.h"
    #include "AMP/solvers/hypre/HyprePCGSolver.h"

    #include "HYPRE.h"
    #include "HYPRE_IJ_mv.h"
    #include "HYPRE_utilities.h"
#endif

#ifdef AMP_USE_TRILINOS_ML
    #include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#endif

#ifdef AMP_USE_TRILINOS_MUELU
    #include "AMP/solvers/trilinos/muelu/TrilinosMueLuSolver.h"
#endif
#ifdef AMP_USE_TRILINOS_NOX
    #include "AMP/solvers/trilinos/nox/TrilinosNOXSolver.h"
#endif


namespace AMP::Solver {

// Create the operator
std::unique_ptr<SolverStrategy>
SolverFactory::create( std::shared_ptr<SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters != nullptr );
    auto inputDatabase = parameters->d_db;
    AMP_ASSERT( inputDatabase );
    auto objectName = inputDatabase->getString( "name" );

    return FactoryStrategy<SolverStrategy, std::shared_ptr<SolverStrategyParameters>>::create(
	    objectName, parameters );
}


} // namespace AMP::Solver


// register all known solver factories
template<>
void AMP::FactoryStrategy<AMP::Solver::SolverStrategy,
                          std::shared_ptr<AMP::Solver::SolverStrategyParameters>>::registerDefault()
{
    using namespace AMP::Solver;
#ifdef AMP_USE_TRILINOS_MUELU
    d_factories["TrilinosMueLuSolver"] = TrilinosMueLuSolver::createSolver;
#endif

#ifdef AMP_USE_TRILINOS_ML
    d_factories["TrilinosMLSolver"] = TrilinosMLSolver::createSolver;
#endif

#ifdef AMP_USE_TRILINOS_NOX
    d_factories["TrilinosNOXSolver"] = TrilinosNOXSolver::createSolver;
#endif

#ifdef AMP_USE_HYPRE
    d_factories["BoomerAMGSolver"]     = BoomerAMGSolver::createSolver;
    d_factories["HyprePCGSolver"]      = HyprePCGSolver::createSolver;
    d_factories["HypreGMRESSolver"]    = HypreGMRESSolver::createSolver;
    d_factories["HypreBiCGSTABSolver"] = HypreBiCGSTABSolver::createSolver;
#endif

#ifdef AMP_USE_PETSC
    d_factories["SNESSolver"]        = PetscSNESSolver::createSolver;
    d_factories["PetscSNESSolver"]   = PetscSNESSolver::createSolver;
    d_factories["PetscKrylovSolver"] = PetscKrylovSolver::createSolver;
#endif

    d_factories["CGSolver"]        = CGSolver<double>::createSolver;
    d_factories["GMRESSolver"]     = GMRESSolver<double>::createSolver;
    d_factories["GMRESRSolver"]    = GMRESRSolver<double>::createSolver;
    d_factories["BiCGSTABSolver"]  = BiCGSTABSolver<double>::createSolver;
    d_factories["TFQMRSolver"]     = TFQMRSolver<double>::createSolver;
    d_factories["QMRCGSTABSolver"] = QMRCGSTABSolver<double>::createSolver;
    d_factories["UASolver"]        = UASolver::createSolver;


    d_factories["CGSolver<double>"]        = CGSolver<double>::createSolver;
    d_factories["GMRESSolver<double>"]     = GMRESSolver<double>::createSolver;
    d_factories["GMRESRSolver<double>"]    = GMRESRSolver<double>::createSolver;
    d_factories["BiCGSTABSolver<double>"]  = BiCGSTABSolver<double>::createSolver;
    d_factories["TFQMRSolver<double>"]     = TFQMRSolver<double>::createSolver;
    d_factories["QMRCGSTABSolver<double>"] = QMRCGSTABSolver<double>::createSolver;

    d_factories["CGSolver<float>"]        = CGSolver<float>::createSolver;
    d_factories["GMRESSolver<float>"]     = GMRESSolver<float>::createSolver;
    d_factories["GMRESRSolver<float>"]    = GMRESRSolver<float>::createSolver;
    d_factories["BiCGSTABSolver<float>"]  = BiCGSTABSolver<float>::createSolver;
    d_factories["TFQMRSolver<float>"]     = TFQMRSolver<float>::createSolver;
    d_factories["QMRCGSTABSolver<float>"] = QMRCGSTABSolver<float>::createSolver;

    d_factories["NKASolver"]         = NonlinearKrylovAccelerator<double>::createSolver;
    d_factories["NKASolver<double>"] = NonlinearKrylovAccelerator<double>::createSolver;
    d_factories["NKASolver<float>"]  = NonlinearKrylovAccelerator<float>::createSolver;

    d_factories["DiagonalSolver"]         = DiagonalSolver<double>::createSolver;
    d_factories["DiagonalSolver<double>"] = DiagonalSolver<double>::createSolver;
    d_factories["DiagonalSolver<float>"]  = DiagonalSolver<float>::createSolver;

    d_factories["BandedSolver"] = BandedSolver::createSolver;

    d_factories["ColumnSolver"] = ColumnSolver::createSolver;

    d_factories["SASolver"] = AMG::SASolver::createSolver;
    d_factories["HybridGS"] = AMG::HybridGS::createSolver;
    d_factories["JacobiL1"] = AMG::JacobiL1::createSolver;
}
