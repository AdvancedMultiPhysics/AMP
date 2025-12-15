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

#ifndef included_TimeIntegratorFactory_H_
#define included_TimeIntegratorFactory_H_

#include "AMP/utils/FactoryStrategy.hpp"

#include <memory>


namespace AMP::IO {
class RestartManager;
}

namespace AMP::TimeIntegrator {

class TimeIntegrator;
class TimeIntegratorParameters;


//! Operator factory class
class TimeIntegratorFactory :
    public FactoryStrategy<TimeIntegrator, std::shared_ptr<TimeIntegratorParameters>>
{
public:
    static std::unique_ptr<TimeIntegrator>
    create( std::shared_ptr<TimeIntegratorParameters> parameters );

    //! Create the vector from the restart file
    static std::shared_ptr<TimeIntegrator> create( int64_t fid, AMP::IO::RestartManager *manager );
};


} // namespace  AMP::TimeIntegrator


#endif // included_TimeIntegratorFactory_H_
