#ifndef included_MeshVectorTests
#define included_MeshVectorTests

#include "ampmesh/testHelpers/meshTests.h"

#include "discretization/DOF_Manager.h"
#include "discretization/DOF_ManagerParameters.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Variable.h"
#include "vectors/Vector.h"
#include "vectors/VectorBuilder.h"

#include "vectors/testHelpers/VectorTests.h"


namespace AMP {
namespace Mesh {



// Simple nodal vector tests
template <int DOF_PER_NODE, bool SPLIT>
void meshTests::simpleNodalVectorTests( AMP::UnitTest *utils,
                             AMP::Mesh::Mesh::shared_ptr mesh,
                             AMP::Discretization::DOFManager::shared_ptr DOFs,
                             int gcw )
{

    // Create a nodal variable
    AMP::LinearAlgebra::Variable::shared_ptr variable(
        new AMP::LinearAlgebra::Variable( "test vector" ) );

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr vectora =
        AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
    AMP::LinearAlgebra::Vector::shared_ptr vectorb =
        AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );

    // Check the size of the vector
    size_t num_dofs = mesh->numGlobalElements( AMP::Mesh::Vertex ) * DOF_PER_NODE;
    if ( vectora->getGlobalSize() == num_dofs )
        utils->passes( "global vector size" );
    else
        utils->failure( "global vector size" );
    num_dofs = mesh->numLocalElements( AMP::Mesh::Vertex ) * DOF_PER_NODE;
    if ( vectora->getLocalSize() == num_dofs )
        utils->passes( "local vector size" );
    else
        utils->failure( "local vector size" );
    num_dofs = mesh->numGhostElements( AMP::Mesh::Vertex, gcw ) * DOF_PER_NODE;
    if ( vectora->getGhostSize() == num_dofs )
        utils->passes( "ghost vector size" );
    else
        utils->failure( "ghost vector size" );

    // Try some trival operations
    vectora->setRandomValues();
    double t1 = vectora->L2Norm();
    vectora->abs( vectora );
    if ( fabs( vectora->L2Norm() - t1 ) < 0.0000001 )
        utils->passes( "non-trivial random vector" );
    else
        utils->failure( "non-trivial random vector" );
    vectorb->setToScalar( 3. );
    vectora->multiply( vectora, vectorb );
    if ( fabs( vectora->L2Norm() - 3. * t1 ) < 0.00000001 )
        utils->passes( "trivial usage" );
    else
        utils->failure( "trivial usage" );
}


// VerifyGetVectorTest
template <int DOF_PER_NODE, bool SPLIT>
void meshTests::VerifyGetVectorTest( AMP::UnitTest *utils, AMP::Mesh::Mesh::shared_ptr mesh )
{

    for ( int gcw = 0; gcw <= 1; gcw++ ) {

        // Create the DOF_Manager
        AMP::Discretization::DOFManagerParameters::shared_ptr DOFparams(
            new AMP::Discretization::DOFManagerParameters( mesh ) );
        AMP::Discretization::DOFManager::shared_ptr DOFs =
            AMP::Discretization::simpleDOFManager::create(
                mesh, AMP::Mesh::Vertex, gcw, DOF_PER_NODE, SPLIT );

        // Run some basic nodal vector tests
        simpleNodalVectorTests<DOF_PER_NODE, SPLIT>( utils, mesh, DOFs, gcw );

        // Run the vector tests
        globalMeshForMeshVectorFactory = mesh;
        AMP::shared_ptr<AMP::LinearAlgebra::VectorFactory> factory( 
            new MeshVectorFactory( mesh, AMP::Mesh::Vertex, gcw, DOF_PER_NODE, SPLIT ) );
        AMP::LinearAlgebra::VectorTests vectorTests( factory );
        vectorTests.testManagedVector( utils );
        vectorTests.testParallelVectors( utils );
        globalMeshForMeshVectorFactory.reset();
    }
}


} // namespace Mesh
} // namespace AMP

#endif
