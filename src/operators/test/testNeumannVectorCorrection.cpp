#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    auto var            = std::make_shared<AMP::LinearAlgebra::Variable>( "myVar" );
    auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto nodalVectorDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 2, true );

    auto scalarVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, var, true );
    auto vectorVec = AMP::LinearAlgebra::createVector( nodalVectorDOF, var, true );

    scalarVec->zero();
    vectorVec->zero();

    auto bnd_db = input_db->getDatabase( "NeumannVectorCorrection1" );
    auto vectorCorrectionParameters =
        std::make_shared<AMP::Operator::NeumannVectorCorrectionParameters>( bnd_db );
    vectorCorrectionParameters->d_variable = var;
    vectorCorrectionParameters->d_Mesh     = mesh;
    auto neumannBndOp =
        std::make_shared<AMP::Operator::NeumannVectorCorrection>( vectorCorrectionParameters );

    neumannBndOp->addRHScorrection( scalarVec );

    bnd_db = input_db->getDatabase( "NeumannVectorCorrection2" );
    vectorCorrectionParameters =
        std::make_shared<AMP::Operator::NeumannVectorCorrectionParameters>( bnd_db );
    vectorCorrectionParameters->d_variable = var;
    vectorCorrectionParameters->d_Mesh     = mesh;
    neumannBndOp =
        std::make_shared<AMP::Operator::NeumannVectorCorrection>( vectorCorrectionParameters );

    neumannBndOp->addRHScorrection( vectorVec );
    auto vectorVec2 = vectorVec->select( AMP::LinearAlgebra::VS_Stride( 1, 2 ) );

    ut->passes( "Ran to completion" );
}


int testNeumannVectorCorrection( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testNeumannVectorCorrection";

    myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
