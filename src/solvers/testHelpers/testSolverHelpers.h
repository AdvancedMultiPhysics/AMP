#ifndef included_testSolverHelpers_H_
#define included_testSolverHelpers_H_

#include "AMP/mesh/Mesh.h"
#include "AMP/utils/UnitTest.h"

#include <functional>

namespace AMP {

class Database;

namespace Discretization {
class DOFManager;
}

namespace LinearAlgebra {
class Vector;
}

namespace Solver {
class SolverStrategy;
}
} // namespace AMP

// Check the solution of the form: T = a + b*z + c*z*z
bool checkAnalyticalSolution( const std::string &exeName,
                              std::function<double( double, double, double )> fun,
                              const AMP::Mesh::MeshIterator &iterator,
                              std::shared_ptr<const AMP::LinearAlgebra::Vector> vec );

std::shared_ptr<AMP::Mesh::Mesh> createMesh( std::shared_ptr<AMP::Database> input_db );

std::pair<std::shared_ptr<AMP::Discretization::DOFManager>,
          std::shared_ptr<AMP::Discretization::DOFManager>>
getDofMaps( std::shared_ptr<const AMP::Mesh::Mesh> mesh );

std::shared_ptr<AMP::LinearAlgebra::Vector>
constructNeutronicsPowerSource( std::shared_ptr<AMP::Database> input_db,
                                std::shared_ptr<AMP::Mesh::Mesh> mesh );

std::tuple<int, double, double, bool>
get_regression_solution( std::shared_ptr<const AMP::Database> input_db );

// Test for validity of solution by referencing map above and following
// the rules laid out there.
// Unrecognized input files just check if convergence reason is ok
void checkConvergence( AMP::Solver::SolverStrategy *solver,
                       std::shared_ptr<const AMP::Database> input_db,
                       const std::string &inputFile,
                       AMP::UnitTest &ut );

#endif
