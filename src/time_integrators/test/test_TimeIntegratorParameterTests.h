#include "../../utils/InputDatabase.h"
#include "../TimeIntegratorParameters.h"


class InstantiateTimeIntegratorParameter
{
public:
    static const char *get_test_name() { return "instantiate TimeIntegratorParameters"; }

    template <typename UTILS>
    static void run_test( UTILS *utils )
    {
        AMP::shared_ptr<AMP::InputDatabase> new_db( new AMP::InputDatabase( "Dummy db" ) );
        AMP::TimeIntegrator::TimeIntegratorParameters params( new_db );
        utils->passes( "instantiate TimeIntegratorParameters" );
    }
};
