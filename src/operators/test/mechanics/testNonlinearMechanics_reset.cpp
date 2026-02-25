#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearElement.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/operators/mechanics/VonMisesElastoPlasticModel.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>


static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testNonlinearMechanics-reset-1" );
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

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    // Material model shared by both the linear and nonlinear operators
    AMP_INSIST( input_db->keyExists( "VonMises_Model" ), "Key ''VonMises_Model'' is missing!" );
    auto matModel_db = input_db->getDatabase( "VonMises_Model" );
    auto matModelParams =
        std::make_shared<AMP::Operator::MechanicsMaterialModelParameters>( matModel_db );
    auto matModel = std::make_shared<AMP::Operator::VonMisesElastoPlasticModel>( matModelParams );

    for ( int useReduced = 0; useReduced < 2; useReduced++ ) {

        std::string mechNonlinElemDbStr;
        std::string mechLinElemDbStr;
        if ( useReduced ) {
            AMP_INSIST( input_db->keyExists( "Mechanics_Nonlinear_Element_Reduced" ),
                        "Key ''Mechanics_Nonlinear_Element_Reduced'' is missing!" );
            AMP_INSIST( input_db->keyExists( "Mechanics_Linear_Element_Reduced" ),
                        "Key ''Mechanics_Linear_Element_Reduced'' is missing!" );
            mechNonlinElemDbStr = "Mechanics_Nonlinear_Element_Reduced";
            mechLinElemDbStr    = "Mechanics_Linear_Element_Reduced";
        } else {
            AMP_INSIST( input_db->keyExists( "Mechanics_Nonlinear_Element_Normal" ),
                        "Key ''Mechanics_Nonlinear_Element_Normal'' is missing!" );
            AMP_INSIST( input_db->keyExists( "Mechanics_Linear_Element_Normal" ),
                        "Key ''Mechanics_Linear_Element_Normal'' is missing!" );
            mechNonlinElemDbStr = "Mechanics_Nonlinear_Element_Normal";
            mechLinElemDbStr    = "Mechanics_Linear_Element_Normal";
        }

        auto nonLinElemOp_db = input_db->getDatabase( mechNonlinElemDbStr );
        auto linElemOp_db    = input_db->getDatabase( mechLinElemDbStr );
        auto nonLinElemOpParams =
            std::make_shared<AMP::Operator::ElementOperationParameters>( nonLinElemOp_db );
        auto mechNonlinElem =
            std::make_shared<AMP::Operator::MechanicsNonlinearElement>( nonLinElemOpParams );

        AMP_INSIST( input_db->keyExists( "Mechanics_Nonlinear_Assembly" ),
                    "Key ''Mechanics_Nonlinear_Assembly'' is missing!" );
        auto mechNonlinAssembly_db = input_db->getDatabase( "Mechanics_Nonlinear_Assembly" );
        mechNonlinAssembly_db->putDatabase( "MechanicsLinearElement",
                                            linElemOp_db->cloneDatabase() );
        auto mechNonlinOpParams =
            std::make_shared<AMP::Operator::MechanicsNonlinearFEOperatorParameters>(
                mechNonlinAssembly_db );
        mechNonlinOpParams->d_materialModel                                  = matModel;
        mechNonlinOpParams->d_elemOp                                         = mechNonlinElem;
        mechNonlinOpParams->d_Mesh                                           = mesh;
        mechNonlinOpParams->d_dofMap[AMP::Operator::Mechanics::DISPLACEMENT] = dofMap;
        auto mechNonlinOp =
            std::make_shared<AMP::Operator::MechanicsNonlinearFEOperator>( mechNonlinOpParams );

        auto var = mechNonlinOp->getOutputVariable();

        auto linElemOpParams =
            std::make_shared<AMP::Operator::ElementOperationParameters>( linElemOp_db );
        auto mechLinElem =
            std::make_shared<AMP::Operator::MechanicsLinearElement>( linElemOpParams );

        AMP_INSIST( input_db->keyExists( "Mechanics_Linear_Assembly" ),
                    "Key ''Mechanics_Linear_Assembly'' is missing!" );
        auto mechLinAssembly_db = input_db->getDatabase( "Mechanics_Linear_Assembly" );
        auto mechLinOpParams = std::make_shared<AMP::Operator::MechanicsLinearFEOperatorParameters>(
            mechLinAssembly_db );
        mechLinOpParams->d_materialModel = matModel;
        mechLinOpParams->d_elemOp        = mechLinElem;
        mechLinOpParams->d_Mesh          = mesh;
        mechLinOpParams->d_inDofMap      = dofMap;
        mechLinOpParams->d_outDofMap     = dofMap;
        auto mechLinOp =
            std::make_shared<AMP::Operator::MechanicsLinearFEOperator>( mechLinOpParams );

        AMP_INSIST( input_db->keyExists( "Displacement_Boundary" ),
                    "Key ''Displacement_Boundary'' is missing!" );
        auto disp_db = input_db->getDatabase( "Displacement_Boundary" );
        auto dirichletOpParams =
            std::make_shared<AMP::Operator::DirichletMatrixCorrectionParameters>( disp_db );
        dirichletOpParams->d_inputMatrix = mechLinOp->getMatrix();
        // This is just the variable used to extract the dof_map.
        // This boundary operator itself has an empty input and output variable
        dirichletOpParams->d_variable = var;
        dirichletOpParams->d_Mesh     = mesh;
        auto dirichletMatOp =
            std::make_shared<AMP::Operator::DirichletMatrixCorrection>( dirichletOpParams );

        auto dirichletDispInVecParams =
            std::make_shared<AMP::Operator::DirichletVectorCorrectionParameters>( disp_db );
        // This has an in-place apply. So, it has an empty input variable and
        // the output variable is the same as what it is operating on.
        dirichletDispInVecParams->d_variable = var;
        dirichletDispInVecParams->d_Mesh     = mesh;
        auto dirichletDispInVecOp =
            std::make_shared<AMP::Operator::DirichletVectorCorrection>( dirichletDispInVecParams );

        auto dirichletDispOutVecParams =
            std::make_shared<AMP::Operator::DirichletVectorCorrectionParameters>( disp_db );
        // This has an in-place apply. So, it has an empty input variable and
        // the output variable is the same as what it is operating on.
        dirichletDispOutVecParams->d_variable = var;
        dirichletDispOutVecParams->d_Mesh     = mesh;
        auto dirichletDispOutVecOp =
            std::make_shared<AMP::Operator::DirichletVectorCorrection>( dirichletDispOutVecParams );

        AMP_INSIST( input_db->keyExists( "LinearBVPOperator" ),
                    "Key ''LinearBVPOperator'' is missing!" );
        auto linBvpOp_db = input_db->getDatabase( "LinearBVPOperator" );
        auto linBvpOperatorParams =
            std::make_shared<AMP::Operator::BVPOperatorParameters>( linBvpOp_db );
        linBvpOperatorParams->d_volumeOperator   = mechLinOp;
        linBvpOperatorParams->d_boundaryOperator = dirichletMatOp;
        auto linBvpOperator =
            std::make_shared<AMP::Operator::LinearBVPOperator>( linBvpOperatorParams );

        AMP_INSIST( input_db->keyExists( "NonlinearBVPOperator" ),
                    "Key ''NonlinearBVPOperator'' is missing!" );
        auto nonlinBvpOp_db = input_db->getDatabase( "NonlinearBVPOperator" );
        auto nonlinBvpOperatorParams =
            std::make_shared<AMP::Operator::BVPOperatorParameters>( nonlinBvpOp_db );
        nonlinBvpOperatorParams->d_volumeOperator   = mechNonlinOp;
        nonlinBvpOperatorParams->d_boundaryOperator = dirichletDispOutVecOp;
        auto nonlinBvpOperator =
            std::make_shared<AMP::Operator::NonlinearBVPOperator>( nonlinBvpOperatorParams );

        AMP_INSIST( input_db->keyExists( "Load_Boundary" ), "Key ''Load_Boundary'' is missing!" );
        auto load_db = input_db->getDatabase( "Load_Boundary" );
        auto dirichletLoadVecParams =
            std::make_shared<AMP::Operator::DirichletVectorCorrectionParameters>( load_db );
        // This has an in-place apply. So, it has an empty input variable and
        // the output variable is the same as what it is operating on.
        dirichletLoadVecParams->d_variable = var;
        dirichletLoadVecParams->d_Mesh     = mesh;
        auto dirichletLoadVecOp =
            std::make_shared<AMP::Operator::DirichletVectorCorrection>( dirichletLoadVecParams );

        AMP::LinearAlgebra::Vector::shared_ptr nullVec;

        auto mechNlSolVec = AMP::LinearAlgebra::createVector( dofMap, var, true );
        auto mechNlRhsVec = mechNlSolVec->clone();
        auto mechNlResVec = mechNlSolVec->clone();

        mechNlRhsVec->setToScalar( 0.0 );
        dirichletLoadVecOp->apply( nullVec, mechNlRhsVec );

        for ( int i = 0; i < 3; i++ ) {
            // Initial guess for NL solver must satisfy the displacement boundary
            // conditions
            mechNlSolVec->setRandomValues();
            dirichletDispInVecOp->apply( nullVec, mechNlSolVec );
            mechNlSolVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            nonlinBvpOperator->residual( mechNlRhsVec, mechNlSolVec, mechNlResVec );
        } // end for i

        mechNonlinOp->reset( mechNonlinOpParams );

        ut->passes( exeName + " : " + mechNonlinElemDbStr );

    } // end for useReduced
}

int testNonlinearMechanics_reset( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
