#include "AMP/operators/OperatorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"


class InstantiateOperatorParameter
{
public:
    static const char *get_test_name() { return "instantiate OperatorParameters"; }

    template<typename UTILS>
    static void run_test( UTILS *utils )
    {
        auto new_db = std::make_shared<AMP::Database>( "Dummy db" );
        AMP::Operator::OperatorParameters params( new_db );
        utils->passes( "instantiate OperatorParameters" );
    }
};


int testOperatorParameterUnitTest( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    InstantiateOperatorParameter::run_test( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
