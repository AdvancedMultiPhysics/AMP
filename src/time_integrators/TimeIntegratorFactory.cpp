/*
Copyright 2005, The Regents of the University
of California. This software was produced under
a U.S. Government contract (W-7405-ENG-36)
by Los Alamos National Laboratory, which is
operated by the University of California for the
U.S. Department of Energy. The U.S.
Government is licensed to use, reproduce, and
distribute this software. Permission is granted
to the public to copy and use this software
without charge, provided that this Notice and
any statement of authorship are reproduced on
all copies. Neither the Government nor the
University makes any warranty, express or
implied, or assumes any liability or
responsibility for the use of this software.
*/
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/time_integrators/BDFIntegrator.h"
#include "AMP/time_integrators/ExplicitEuler.h"
#include "AMP/time_integrators/RK12TimeIntegrator.h"
#include "AMP/time_integrators/RK23TimeIntegrator.h"
#include "AMP/time_integrators/RK2TimeIntegrator.h"
#include "AMP/time_integrators/RK34TimeIntegrator.h"
#include "AMP/time_integrators/RK45TimeIntegrator.h"
#include "AMP/time_integrators/RK4TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"


namespace AMP::TimeIntegrator {


// Create the operator
std::unique_ptr<TimeIntegrator>
TimeIntegratorFactory::create( std::shared_ptr<TimeIntegratorParameters> parameters )
{
    AMP_ASSERT( parameters );
    auto inputDatabase = parameters->d_db;
    AMP_ASSERT( inputDatabase );
    auto objectName = inputDatabase->getString( "name" );
    return FactoryStrategy<TimeIntegrator, std::shared_ptr<TimeIntegratorParameters>>::create(
        objectName, parameters );
}

std::shared_ptr<TimeIntegrator> TimeIntegratorFactory::create( int64_t fid,
                                                               AMP::IO::RestartManager *manager )
{
    std::string type;
    AMP::IO::readHDF5( fid, "type", type );
    std::shared_ptr<TimeIntegrator> ti;
    if ( type == "ExplicitEuler" )
        ti = std::make_shared<RK12TimeIntegrator>( fid, manager );
    else if ( type == "RK12" )
        ti = std::make_shared<RK12TimeIntegrator>( fid, manager );
    else if ( type == "RK23" )
        ti = std::make_shared<RK23TimeIntegrator>( fid, manager );
    else if ( type == "RK34" )
        ti = std::make_shared<RK34TimeIntegrator>( fid, manager );
    else if ( type == "RK45" )
        ti = std::make_shared<RK45TimeIntegrator>( fid, manager );
    else if ( type == "RK2" )
        ti = std::make_shared<RK2TimeIntegrator>( fid, manager );
    else if ( type == "RK4" )
        ti = std::make_shared<RK4TimeIntegrator>( fid, manager );
    else if ( type == "BDFIntegrator" )
        ti = std::make_shared<BDFIntegrator>( fid, manager );
    else {
        ti = FactoryStrategy<TimeIntegrator, int64_t, AMP::IO::RestartManager *>::create(
            type, fid, manager );
    }
    return ti;
}


} // namespace AMP::TimeIntegrator


// register all known time integrator factories
template<>
void AMP::FactoryStrategy<
    AMP::TimeIntegrator::TimeIntegrator,
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters>>::registerDefault()
{
    using namespace AMP::TimeIntegrator;
    d_factories["ExplicitEuler"]      = AMP::TimeIntegrator::ExplicitEuler::createTimeIntegrator;
    d_factories["ImplicitIntegrator"] = BDFIntegrator::createTimeIntegrator;
    d_factories["Backward Euler"]     = BDFIntegrator::createTimeIntegrator;
    d_factories["BDF1"]               = BDFIntegrator::createTimeIntegrator;
    d_factories["BDF2"]               = BDFIntegrator::createTimeIntegrator;
    d_factories["BDF3"]               = BDFIntegrator::createTimeIntegrator;
    d_factories["BDF4"]               = BDFIntegrator::createTimeIntegrator;
    d_factories["BDF5"]               = BDFIntegrator::createTimeIntegrator;
    d_factories["BDF6"]               = BDFIntegrator::createTimeIntegrator;
    d_factories["CN"]                 = BDFIntegrator::createTimeIntegrator;
    d_factories["RK2"]  = AMP::TimeIntegrator::RK2TimeIntegrator::createTimeIntegrator;
    d_factories["RK4"]  = AMP::TimeIntegrator::RK4TimeIntegrator::createTimeIntegrator;
    d_factories["RK12"] = AMP::TimeIntegrator::RK12TimeIntegrator::createTimeIntegrator;
    d_factories["RK23"] = AMP::TimeIntegrator::RK23TimeIntegrator::createTimeIntegrator;
    d_factories["RK34"] = AMP::TimeIntegrator::RK34TimeIntegrator::createTimeIntegrator;
    d_factories["RK45"] = AMP::TimeIntegrator::RK45TimeIntegrator::createTimeIntegrator;
}
template<>
void AMP::FactoryStrategy<AMP::TimeIntegrator::TimeIntegrator, int64_t, AMP::IO::RestartManager *>::
    registerDefault()
{
}
