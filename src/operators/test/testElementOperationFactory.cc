#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionNonlinearElement.h"
#include "AMP/operators/libmesh/MassLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsNonlinearElement.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/InputDatabase.h"
#include "AMP/utils/InputManager.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/shared_ptr.h"


static void ElementOperationFactoryTest( AMP::UnitTest *ut )
{
    std::string exeName( "testElementOperationFactory-1" );
    std::string outerInput_file = "input_" + exeName;
    std::string log_file        = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );

    auto outerInput_db = AMP::make_shared<AMP::InputDatabase>( "outerInput_db" );
    AMP::InputManager::getManager()->parseInputFile( outerInput_file, outerInput_db );
    outerInput_db->printClassData( AMP::plog );

    AMP_INSIST( outerInput_db->keyExists( "number_of_tests" ), "key missing!" );
    int numTests = outerInput_db->getInteger( "number_of_tests" );

    for ( int i = 0; i < numTests; i++ ) {
        char key[256];
        sprintf( key, "test_%d", i );

        AMP_INSIST( outerInput_db->keyExists( key ), "key missing!" );
        auto inputFile     = outerInput_db->getString( key );
        auto innerInput_db = AMP::make_shared<AMP::InputDatabase>( "innerInput_db" );
        AMP::InputManager::getManager()->parseInputFile( inputFile, innerInput_db );
        innerInput_db->printClassData( AMP::plog );

        AMP_INSIST( innerInput_db->keyExists( "ElementOperation" ),
                    "Key ''ElementOperation'' is missing!" );
        auto elemOp_db = innerInput_db->getDatabase( "ElementOperation" );
        auto name      = elemOp_db->getString( "name" );
        AMP::shared_ptr<AMP::Operator::ElementOperation> elementOperation =
            AMP::Operator::ElementOperationFactory::createElementOperation( elemOp_db );

        if ( name == "MechanicsLinearElement" ) {
            auto mechOperation = AMP::dynamic_pointer_cast<AMP::Operator::MechanicsLinearElement>(
                elementOperation );
            if ( mechOperation.get() != nullptr ) {
                ut->passes( exeName + " : " + inputFile + " : MechanicsLinearElement" );
            } else {
                ut->failure( exeName + " : " + inputFile + " : MechanicsLinearElement" );
            }
        } else if ( name == "DiffusionLinearElement" ) {
            auto diffusionOperation =
                AMP::dynamic_pointer_cast<AMP::Operator::DiffusionLinearElement>(
                    elementOperation );
            if ( diffusionOperation.get() != nullptr ) {
                ut->passes( exeName + " : " + inputFile + " : DiffusionLinearElement" );
            } else {
                ut->failure( exeName + " : " + inputFile + " : DiffusionLinearElement" );
            }
        } else if ( name == "MechanicsNonlinearElement" ) {
            auto mechOperation =
                AMP::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearElement>(
                    elementOperation );
            if ( mechOperation.get() != nullptr ) {
                ut->passes( exeName + " : " + inputFile + " : MechanicsNonlinearElement" );
            } else {
                ut->failure( exeName + " : " + inputFile + " : MechanicsNonlinearElement" );
            }
        } else if ( name == "DiffusionNonlinearElement" ) {
            auto diffusionOperation =
                AMP::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearElement>(
                    elementOperation );
            if ( diffusionOperation.get() != nullptr ) {
                ut->passes( exeName + " : " + inputFile + " : DiffusionNonlinearElement" );
            } else {
                ut->failure( exeName + " : " + inputFile + " : DiffusionNonlinearElement" );
            }
        } else if ( name == "MassLinearElement" ) {
            auto massOperation =
                AMP::dynamic_pointer_cast<AMP::Operator::MassLinearElement>( elementOperation );
            if ( massOperation.get() != nullptr ) {
                ut->passes( exeName + " : " + inputFile + " : MassLinearElement" );
            } else {
                ut->failure( exeName + " : " + inputFile + " : MassLinearElement" );
            }
        }
    }
}

int testElementOperationFactory( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    ElementOperationFactoryTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}