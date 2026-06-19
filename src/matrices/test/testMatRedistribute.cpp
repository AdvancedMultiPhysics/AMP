#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

namespace {

void testRedistribute( AMP::UnitTest *ut, const std::string &input_file )
{
    auto input_db = AMP::Database::parseInputFile( input_file );
    auto mesh_db  = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    auto dof =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );

    auto in_var  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto out_var = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );
    auto x       = AMP::LinearAlgebra::createVector( dof, in_var );
    auto y       = AMP::LinearAlgebra::createVector( dof, out_var );
    auto A       = AMP::LinearAlgebra::createMatrix( x, y, "CSRMatrix" );
    AMP::LinearAlgebra::fillWithPseudoLaplacian( A );
    A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    auto csr = std::dynamic_pointer_cast<
        AMP::LinearAlgebra::CSRMatrix<AMP::LinearAlgebra::DefaultHostCSRConfig>>( A );
    AMP_INSIST( csr, "testMatRedistribute requires the default host CSR matrix" );

    auto world         = AMP::AMP_MPI( AMP_COMM_WORLD );
    const int nout     = std::max( 1, world.getSize() / 2 );
    auto redistributed = csr->redistribute( nout );

    if ( redistributed == nullptr ) {
        if ( redistributed == nullptr ) {
            ut->passes( "Inactive ranks correctly drop out of redistributed matrix" );
        } else {
            ut->failure( "Inactive ranks should not retain redistributed matrix state" );
        }
        return;
    }

    auto red = std::dynamic_pointer_cast<
        AMP::LinearAlgebra::CSRMatrix<AMP::LinearAlgebra::DefaultHostCSRConfig>>( redistributed );
    AMP_INSIST( red, "Redistributed matrix should still be a host CSR matrix" );

    auto xr = red->createInputVector();
    auto yr = red->createOutputVector();
    xr->setToScalar( 1.0 );
    xr->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    yr->zero();
    red->mult( xr, yr );
    yr->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    const auto norm = yr->L1Norm();
    if ( norm == static_cast<double>( red->numGlobalRows() ) ) {
        ut->passes( "Redistributed CSR matrix preserves pseudo-Laplacian matvec" );
    } else {
        AMP::pout << "Redistributed L1 norm " << norm << ", rows " << red->numGlobalRows()
                  << std::endl;
        ut->failure( "Redistributed CSR matrix failed pseudo-Laplacian matvec test" );
    }
}

} // namespace

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string input = argc > 1 ? argv[1] : "input_testMatVecPerf-1";
    testRedistribute( &ut, input );

    ut.report();
    const int failures = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return failures;
}
