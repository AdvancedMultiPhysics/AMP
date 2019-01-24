//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   operators/test/testNekOperator.cc
 * \brief  Tests Nek pipe problem run through NekMoabOperator
 *
 * This test is intended to test our ability to use a Nek-generated Moab
 * instance.  It duplicates the behavior of testNekPipe, but maps the Nek
 * output onto an actual AMP mesh.
 */
//---------------------------------------------------------------------------//

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "AMP/utils/AMPManager.h"
#include "AMP/utils/InputDatabase.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/utils/Writer.h"
#include "AMP/vectors/VectorBuilder.h"

#include "AMP/operators/moab/MoabMapOperator.h"

// Nek includes
#include "nek/NekMoabOperator.h"
#include "nek/NekMoabOperatorParameters.h"

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

static void nekPipeOperator( AMP::UnitTest *ut )
{
#ifdef USE_EXT_NEK
    // Print Banner
    AMP::Utilities::printBanner();

    // Log all nodes
    AMP::PIO::logAllNodes( "output_testNekOperator" );

    // Build new database
    AMP::pout << "Building Input Database" << std::endl;
    AMP::shared_ptr<AMP::InputDatabase> nekDB( new AMP::InputDatabase( "Nek_DB" ) );
    nekDB->putString( "NekProblemName", "pipe" );

    // Build operator params
    typedef AMP::Operator::NekMoabOperatorParameters NekOpParams;
    typedef AMP::shared_ptr<NekOpParams> SP_NekOpParams;

    AMP::pout << "Building Nek Operator Parameters" << std::endl;
    SP_NekOpParams nekParams( new NekOpParams( nekDB ) );

    // Build operator
    typedef AMP::Operator::NekMoabOperator NekOp;
    typedef AMP::shared_ptr<NekOp> SP_NekOp;

    typedef AMP::Operator::MoabBasedOperator MoabBasedOp;
    typedef AMP::shared_ptr<MoabBasedOp> SP_MoabBasedOp;

    AMP::pout << "Building Nek Operator" << std::endl;
    SP_MoabBasedOp nekOp( new NekOp( nekParams ) );

    // Call apply
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    nekOp->apply( nullVec, nullVec, nullVec, 0.0, 0.0 );

    // Read AMP pellet mesh from file
    AMP::shared_ptr<AMP::Database> meshDB = nekDB->putDatabase( "Mesh" );
    meshDB->putString( "FileName", "pellet_1x.e" );
    meshDB->putString( "MeshName", "fuel" );
    meshDB->putString( "MeshType", "libMesh" );
    meshDB->putInteger( "dim", 3 );
    meshDB->putDouble( "x_offset", 0.0 );
    meshDB->putDouble( "y_offset", 0.0 );
    meshDB->putDouble( "z_offset", 0.0 );
    meshDB->putInteger( "NumberOfElements", 300 );


    // Create Mesh
    AMP::pout << "Creating AMP mesh" << std::endl;
    typedef AMP::Mesh::MeshParameters MeshParams;
    typedef AMP::shared_ptr<MeshParams> SP_MeshParams;

    typedef AMP::Mesh::Mesh AMPMesh;
    typedef AMP::Mesh::Mesh::shared_ptr SP_AMPMesh;

    SP_MeshParams meshParams( new MeshParams( meshDB ) );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    SP_AMPMesh mesh = AMP::Mesh::Mesh::buildMesh( meshParams );


    // Create Parameters for Map Operator
    AMP::pout << "Creating map operator" << std::endl;
    typedef AMP::Operator::MoabMapOperatorParameters MoabMapParams;
    typedef AMP::shared_ptr<MoabMapParams> SP_MoabMapParams;

    typedef AMP::Operator::MoabMapOperator MoabMap;
    typedef AMP::shared_ptr<MoabMap> SP_MoabMap;

    nekDB->putString( "MoabMapVariable", "VPRESS" );
    nekDB->putString( "InterpolateToType", "GaussPoint" );
    SP_MoabMapParams mapParams( new MoabMapParams( nekDB ) );
    mapParams->setMoabOperator( nekOp );
    mapParams->setMesh( mesh );

    AMP::pout << "Creating GP-Based Moab Map Operator" << std::endl;
    SP_MoabMap moabGPMap( new MoabMap( mapParams ) );

    // Create variable to hold pressure data
    typedef AMP::LinearAlgebra::Variable AMPVar;
    typedef AMP::shared_ptr<AMPVar> SP_AMPVar;

    SP_AMPVar allGPPressures( new AMPVar( "AllGaussPointPressures" ) );
    SP_AMPVar allNodePressures( new AMPVar( "AllNodalPressures" ) );

    // Create DOF managers
    size_t DOFsPerElement    = 8;
    size_t DOFsPerNode       = 1;
    int gaussPointGhostWidth = 1;
    int nodalGhostWidth      = 1;
    bool split               = true;
    AMP::Discretization::DOFManager::shared_ptr gaussPointDofMap =
        AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Volume, gaussPointGhostWidth, DOFsPerElement, split );
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap =
        AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // Have mesh manager create vector over all meshes
    AMP::LinearAlgebra::Vector::shared_ptr r_gp =
        AMP::LinearAlgebra::createVector( gaussPointDofMap, allGPPressures );
    AMP::LinearAlgebra::Vector::shared_ptr r_node =
        AMP::LinearAlgebra::createVector( nodalDofMap, allNodePressures );

    AMP::pout << "Gauss Point MultiVector size: " << r_gp->getGlobalSize() << std::endl;
    AMP::pout << "Nodal MultiVector size: " << r_node->getGlobalSize() << std::endl;

    AMP::pout << "Calling apply" << std::endl;
    moabGPMap->apply( nullVec, nullVec, r_gp, 0.0, 0.0 );

    AMP::pout << "Creating Node-Based Moab Map Operator" << std::endl;
    nekDB->putString( "InterpolateToType", "GeomType::Vertex" );
    SP_MoabMap moabNodeMap( new MoabMap( mapParams ) );

    moabNodeMap->apply( nullVec, nullVec, r_node, 0.0, 0.0 );

    // Did we actually get data?
    typedef AMP::LinearAlgebra::Vector AMPVec;
    AMPVec::iterator myIter;
    int ctr      = 0;
    bool nonZero = false;
    for ( myIter = r_gp->begin(); myIter != r_gp->end(); ++myIter ) {
        // AMP::pout << "GP Vector Element " << ctr << " is " << *myIter << std::endl;

        if ( *myIter != 0.0 )
            nonZero = true;

        ctr++;
    }

    if ( nonZero )
        ut->passes( "Gauss point vector is not identically zero" );
    else
        ut->failure( "Gauss point vector is identically zero" );

    ctr     = 0;
    nonZero = false;
    for ( myIter = r_node->begin(); myIter != r_node->end(); ++myIter ) {
        if ( *myIter != 0.0 )
            nonZero = true;

        ctr++;
    }

    if ( nonZero )
        ut->passes( "Nodal vector is not identically zero" );
    else
        ut->failure( "Nodal vector is identically zero" );

        // How about some output?

#ifdef USE_EXT_SILO
    AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter( "Silo" );
    siloWriter->registerMesh( mesh );
    siloWriter->registerVector( r_gp, mesh, AMP::Mesh::GeomType::Volume, "AllGaussPointPressures" );
    siloWriter->registerVector( r_node, mesh, AMP::Mesh::GeomType::Vertex, "AllNodalPressures" );
    siloWriter->writeFile( "Nek_Pressure", 0 );
#endif


    // Finalize Nek Operator
    nekOp->finalize();

#else
    ut->passes( "Nek was not used." );
#endif

    if ( ut->NumPassGlobal() == 0 )
        ut->failure( "if it doesn't pass, it must have failed." );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    nekPipeOperator( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}

//---------------------------------------------------------------------------//
//                 end of testNekOperator.cc
//---------------------------------------------------------------------------//