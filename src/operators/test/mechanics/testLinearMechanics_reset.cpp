#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/IsotropicElasticModel.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"
#include <memory>

#include <iostream>
#include <string>


static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testLinearMechanics-reset-1" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );


    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    AMP_INSIST( input_db->keyExists( "Isotropic_Model" ), "Key ''Isotropic_Model'' is missing!" );
    auto matModel_db = input_db->getDatabase( "Isotropic_Model" );
    auto matModelParams =
        std::make_shared<AMP::Operator::MechanicsMaterialModelParameters>( matModel_db );
    auto isotropicModel = std::make_shared<AMP::Operator::IsotropicElasticModel>( matModelParams );

    for ( int useReduced = 0; useReduced < 2; useReduced++ ) {

        std::string mechElemDbStr;
        if ( useReduced ) {
            AMP_INSIST( input_db->keyExists( "Mechanics_Linear_Element_Reduced" ),
                        "Key ''Mechanics_Linear_Element_Reduced'' is missing!" );
            mechElemDbStr = "Mechanics_Linear_Element_Reduced";
        } else {
            AMP_INSIST( input_db->keyExists( "Mechanics_Linear_Element_Normal" ),
                        "Key ''Mechanics_Linear_Element_Normal'' is missing!" );
            mechElemDbStr = "Mechanics_Linear_Element_Normal";
        }
        auto elemOp_db = input_db->getDatabase( mechElemDbStr );
        auto elemOpParams =
            std::make_shared<AMP::Operator::ElementOperationParameters>( elemOp_db );
        auto mechLinElem = std::make_shared<AMP::Operator::MechanicsLinearElement>( elemOpParams );

        auto dofMap = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

        AMP_INSIST( input_db->keyExists( "Mechanics_Assembly" ),
                    "Key ''Mechanics_Assembly'' is missing!" );
        auto mechAssembly_db = input_db->getDatabase( "Mechanics_Assembly" );
        auto mechOpParams =
            std::make_shared<AMP::Operator::MechanicsLinearFEOperatorParameters>( mechAssembly_db );
        mechOpParams->d_materialModel = isotropicModel;
        mechOpParams->d_elemOp        = mechLinElem;
        mechOpParams->d_Mesh          = mesh;
        mechOpParams->d_inDofMap      = dofMap;
        mechOpParams->d_outDofMap     = dofMap;
        auto mechOp = std::make_shared<AMP::Operator::MechanicsLinearFEOperator>( mechOpParams );

        auto mechVariable = mechOp->getOutputVariable();

        AMP_INSIST( input_db->keyExists( "Displacement_Boundary" ),
                    "Key ''Displacement_Boundary'' is missing!" );
        auto disp_db = input_db->getDatabase( "Displacement_Boundary" );
        auto dirichletOpParams =
            std::make_shared<AMP::Operator::DirichletMatrixCorrectionParameters>( disp_db );
        dirichletOpParams->d_inputMatrix = mechOp->getMatrix();
        // This is just the variable used to extract the dof_map.
        // This boundary operator itself has an empty input and output variable
        dirichletOpParams->d_variable = mechVariable;
        dirichletOpParams->d_Mesh     = mesh;
        auto dirichletMatOp =
            std::make_shared<AMP::Operator::DirichletMatrixCorrection>( dirichletOpParams );

        auto mechSolVec = AMP::LinearAlgebra::createVector( dofMap, mechVariable, true );
        auto mechRhsVec = mechSolVec->clone();
        auto mechResVec = mechSolVec->clone();

        for ( int i = 0; i < 3; i++ ) {
            mechSolVec->setRandomValues();
            mechRhsVec->setRandomValues();
            mechResVec->setRandomValues();
            mechOp->residual( mechRhsVec, mechSolVec, mechResVec );
        } // end for i

        mechOp->reset( mechOpParams );

        ut->passes( exeName + " : " + mechElemDbStr );

    } // end for useReduced
}

int testLinearMechanics_reset( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
