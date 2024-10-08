#include "AMP/time_integrators/sundials/IDATimeIntegratorParameters.h"

namespace AMP::TimeIntegrator {

IDATimeIntegratorParameters::IDATimeIntegratorParameters( std::shared_ptr<AMP::Database> db )
    : TimeIntegratorParameters( db )

{
    // Question: does this automatically call the constructor of its parent class?
    /*
    d_db = db;
    d_ic_vector.reset();:
    d_operator.reset();
     */
    d_ic_vector_prime.reset();
}

IDATimeIntegratorParameters::~IDATimeIntegratorParameters() = default;
} // namespace AMP::TimeIntegrator
