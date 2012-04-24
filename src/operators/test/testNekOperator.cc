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

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/PIO.h"
#include "utils/InputDatabase.h"

#include "ampmesh/MeshManager.h"
#include "ampmesh/MeshVariable.h"
#include "ampmesh/SiloIO.h"

#include "operators/moab/MoabMapOperator.h"

// Nek includes
#include "nek/NekMoabOperator.h"
#include "nek/NekMoabOperatorParameters.h"

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

void nekPipeOperator(AMP::UnitTest *ut)
{
#ifdef USE_NEK     
    // Log all nodes
    AMP::PIO::logAllNodes( "output_testNekPipe" );

    // Build new database
    AMP::pout << "Building Input Database" << std::endl;
    boost::shared_ptr< AMP::InputDatabase > nekDB( new AMP::InputDatabase("Nek_DB") );
    nekDB->putString("NekProblemName","pipe");

    // Build operator params
    typedef AMP::Operator::NekMoabOperatorParameters NekOpParams;
    typedef boost::shared_ptr< NekOpParams >         SP_NekOpParams;

    AMP::pout << "Building Nek Operator Parameters" << std::endl;
    SP_NekOpParams nekParams( new NekOpParams( nekDB ) );

    // Build operator
    typedef AMP::Operator::NekMoabOperator   NekOp;
    typedef boost::shared_ptr< NekOp >       SP_NekOp;

    typedef AMP::Operator::MoabBasedOperator MoabBasedOp;
    typedef boost::shared_ptr< MoabBasedOp > SP_MoabBasedOp;

    AMP::pout << "Building Nek Operator" << std::endl;
    SP_MoabBasedOp nekOp( new NekOp( nekParams ) );

    // Call apply
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    nekOp->apply( nullVec, nullVec, nullVec, 0.0, 0.0 );

    // Read AMP pellet mesh from file
    nekDB->putInteger("NumberOfMeshes",1);
    boost::shared_ptr<AMP::Database> meshDB = nekDB->putDatabase("Mesh_1");
    meshDB->putString("Filename","pellet_1x.e");
    meshDB->putString("MeshName","fuel");
    meshDB->putDouble("x_offset",0.0);
    meshDB->putDouble("y_offset",0.0);
    meshDB->putDouble("z_offset",-0.03);


    // Create Mesh Manager
    AMP::pout << "Creating mesh manager" << std::endl;
    typedef AMP::Mesh::MeshManagerParameters    MeshMgrParams;
    typedef boost::shared_ptr< MeshMgrParams >  SP_MeshMgrParams;

    typedef AMP::Mesh::MeshManager              MeshMgr;
    typedef boost::shared_ptr< MeshMgr >        SP_MeshMgr;

    SP_MeshMgrParams mgrParams( new MeshMgrParams( nekDB ) );
    SP_MeshMgr       manager(   new MeshMgr( mgrParams ) );
    

    // Create Parameters for Map Operator
    AMP::pout << "Creating map operator" << std::endl;
    typedef AMP::Operator::MoabMapOperatorParameters    MoabMapParams;
    typedef boost::shared_ptr< MoabMapParams >          SP_MoabMapParams;

    typedef AMP::Operator::MoabMapOperator              MoabMap;
    typedef boost::shared_ptr< MoabMap>                 SP_MoabMap;

    nekDB->putString("MoabMapVariable","VPRESS");
    nekDB->putString("InterpolateToType","GaussPoint");
    SP_MoabMapParams mapParams( new MoabMapParams( nekDB ) );
    mapParams->setMoabOperator( nekOp );
    mapParams->setMeshManager( manager );

    AMP::pout << "Creating GP-Based Moab Map Operator" << std::endl;
    SP_MoabMap moabGPMap( new MoabMap( mapParams ) );

    // Create variable to hold pressure data
    typedef AMP::LinearAlgebra::Variable      AMPVar;
    typedef boost::shared_ptr< AMPVar >       SP_AMPVar;

    typedef AMP::LinearAlgebra::MultiVariable AMPMultiVar;
    typedef boost::shared_ptr< AMPMultiVar >  SP_AMPMultiVar;

    SP_AMPMultiVar allGPPressures( new AMPMultiVar( "AllPressures" ) );
    SP_AMPMultiVar allNodePressures( new AMPMultiVar( "AllPressures" ) );

    MeshMgr::MeshIterator currentMesh;
    for( currentMesh  = manager->beginMeshes();
         currentMesh != manager->endMeshes();
         currentMesh++ )
    {
        // Make variable on this mesh
        SP_AMPVar thisGPVar( new AMP::LinearAlgebra::VectorVariable<AMP::Mesh::IntegrationPointVariable, 8>("NekGPPressure", *currentMesh));
        SP_AMPVar thisNodeVar( new AMP::Mesh::NodalScalarVariable("NekNodePressure", *currentMesh));

        // Add variable on this mesh to multivariable
        allGPPressures->add( thisGPVar );
        allNodePressures->add( thisNodeVar );
    }

    // Have mesh manager create vector over all meshes
    AMP::LinearAlgebra::Vector::shared_ptr r_gp = manager->createVector( allGPPressures );
    AMP::LinearAlgebra::Vector::shared_ptr r_node = manager->createVector( allNodePressures );

    AMP::pout << "Gauss Point MultiVector size: " << r_gp->getGlobalSize() << std::endl; 
    AMP::pout << "Nodal MultiVector size: " << r_node->getGlobalSize() << std::endl; 

    AMP::pout << "Calling apply" << std::endl;
    moabGPMap->apply( nullVec, nullVec, r_gp, 0.0, 0.0 );

    AMP::pout << "Creating Node-Based Moab Map Operator" << std::endl;
    nekDB->putString("InterpolateToType","Vertex");
    SP_MoabMap moabNodeMap( new MoabMap( mapParams ) );

    moabNodeMap->apply( nullVec, nullVec, r_node, 0.0, 0.0 );

    // Did we actually get data?
    typedef AMP::LinearAlgebra::Vector AMPVec;
    AMPVec::iterator myIter;
    int ctr=0;
    bool nonZero = false;
    for( myIter  = r_gp->begin();
         myIter != r_gp->end();
         myIter++ )
    {
        AMP::pout << "GP Vector Element " << ctr << " is " << *myIter << std::endl;

        if( *myIter != 0.0 )
            nonZero = true;

        ctr++;
    }

    if( nonZero )
        ut->passes("Gauss point vector is not identically zero");
    else
        ut->failure("Gauss point vector is identically zero");

    ctr=0;
    nonZero = false;
    for( myIter  = r_node->begin();
         myIter != r_node->end();
         myIter++ )
    {
        AMP::pout << "Nodal Vector Element " << ctr << " is " << *myIter << std::endl;

        if( *myIter != 0.0 )
            nonZero = true;

        ctr++;
    }

    if( nonZero )
        ut->passes("Nodal vector is not identically zero");
    else
        ut->failure("Nodal vector is identically zero");

    // How about some output?
    
#ifdef USE_SILO
    manager->registerVectorAsData( r_gp );
    manager->registerVectorAsData( r_node );
    manager->writeFile<AMP::Mesh::SiloIO>( "Nek_Pressure", 0 );
#endif


    // Finalize Nek Operator
    nekOp->finalize();

#else
    ut->passes("Nek was not used.");
#endif

    if (ut->NumPassGlobal() == 0) ut->failure("if it doesn't pass, it must have failed.");
} 


int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;
    
    try {
        nekPipeOperator(&ut);
	      ut.passes("Nek ran pipe to completion.");
    } catch (std::exception &err) {
        std::cout << "ERROR: While testing "<<argv[0] << err.what() << std::endl;
        ut.failure("ERROR: While testing");
    } catch( ... ) {
        std::cout << "ERROR: While testing "<<argv[0] << "An unknown exception was thrown." << std::endl;
        ut.failure("ERROR: While testing");
    }
   
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}

//---------------------------------------------------------------------------//
//                 end of testNekOperator.cc
//---------------------------------------------------------------------------//