#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/map/StridedZAxisMap.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <functional>


double fooFunctionOfSpace( AMP::Mesh::Point const &xyz )
{
    AMP_ASSERT( xyz.size() == 3 );
    return 1.0 + xyz[2];
}

double barFunctionOfSpace( AMP::Mesh::Point const &xyz )
{
    AMP_ASSERT( xyz.size() == 3 );
    return 3.0 * std::cos( xyz[2] );
}

static void project( AMP::Mesh::MeshIterator const &meshIterator,
                     AMP::LinearAlgebra::Vector::shared_ptr vector,
                     size_t const dof,
                     size_t const dofsPerNode,
                     std::function<double( AMP::Mesh::Point const & )> functionOfSpace )
{
    AMP_INSIST( dof < dofsPerNode, "WRONG!" );
    auto dofManager = vector->getDOFManager();
    std::vector<size_t> dofIndices;
    for ( auto &elem : meshIterator ) {
        dofManager->getDOFs( elem.globalID(), dofIndices );
        AMP_ASSERT( dofIndices.size() == dofsPerNode );
        auto coord   = elem.coord();
        double value = functionOfSpace( coord );
        vector->setValuesByGlobalID( 1, &dofIndices[dof], &value );
    }
}


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string inputFile = "input_" + exeName;
    std::string logFile   = "output_" + exeName;

    AMP::logOnlyNodeZero( logFile );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // parse input file
    auto inputDatabase = AMP::Database::parseInputFile( inputFile );
    inputDatabase->print( AMP::plog );

    // read the meshe
    auto meshDatabase = inputDatabase->getDatabase( "Mesh" );
    auto meshParams   = std::make_shared<AMP::Mesh::MeshParameters>( meshDatabase );
    meshParams->setComm( globalComm );
    auto mesh    = AMP::Mesh::MeshFactory::create( meshParams );
    auto fooMesh = mesh->Subset( "Foo" );
    auto barMesh = mesh->Subset( "Bar" );

    // Build the dof managers
    auto mapDB         = inputDatabase->getDatabase( "MapOperator" );
    bool split         = true;
    int ghostWidth     = 0;
    int fooDof         = mapDB->getScalar<int>( "InputStride" );
    int barDof         = mapDB->getScalar<int>( "OutputStride" );
    int fooDofsPerNode = mapDB->getScalar<int>( "InputDOFsPerObject" );
    int barDofsPerNode = mapDB->getScalar<int>( "OutputDOFsPerObject" );
    auto fooDofManager = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, ghostWidth, fooDofsPerNode );
    auto barDofManager = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, ghostWidth, barDofsPerNode );

    // Build the vectors
    auto var1 =
        std::make_shared<AMP::LinearAlgebra::Variable>( mapDB->getString( "VariableName1" ) );
    auto var2 =
        std::make_shared<AMP::LinearAlgebra::Variable>( mapDB->getString( "VariableName2" ) );
    auto srcVector = AMP::LinearAlgebra::createVector( fooDofManager, var1, split );
    auto ansVector = AMP::LinearAlgebra::createVector( barDofManager, var2, split );
    auto dstVector = AMP::LinearAlgebra::createVector( barDofManager, var2, split );

    // fill the vectors
    srcVector->zero();
    ansVector->zero();
    dstVector->zero();
    int fooBoundaryID = 2;
    int barBoundaryID = 1;
    auto fooMeshIterator =
        fooMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, fooBoundaryID );
    auto barMeshIterator =
        barMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, barBoundaryID );
    project( fooMeshIterator, srcVector, fooDof, fooDofsPerNode, fooFunctionOfSpace );
    project( barMeshIterator, ansVector, barDof, barDofsPerNode, fooFunctionOfSpace );

    // make map operator
    auto mapOperatorDatabase = inputDatabase->getDatabase( "MapOperator" );
    auto mapOperatorParameters =
        std::make_shared<AMP::Operator::Map3to1to3Parameters>( mapOperatorDatabase );
    mapOperatorParameters->d_Mesh1       = fooMesh;
    mapOperatorParameters->d_Mesh2       = barMesh;
    mapOperatorParameters->d_BoundaryID1 = fooBoundaryID;
    mapOperatorParameters->d_BoundaryID2 = barBoundaryID;
    mapOperatorParameters->d_MapComm     = globalComm;
    auto mapOperator = std::make_shared<AMP::Operator::StridedZAxisMap>( mapOperatorParameters );

    // apply
    AMP::LinearAlgebra::Vector::shared_ptr dummyVector;
    mapOperator->setVector( dstVector );
    mapOperator->apply( srcVector, dummyVector );

    double tolerance = 1.0e-14 * static_cast<double>( ansVector->L2Norm() );
    dstVector->subtract( *ansVector, *dstVector );

    double errNorm = static_cast<double>( dstVector->L2Norm() );
    AMP::pout << "errNorm  " << errNorm << std::endl;

    if ( errNorm < tolerance )
        ut->passes( exeName );
    else
        ut->failure( exeName );
}


int testStridedZAxisMap( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testStridedZAxisMap" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();
    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
