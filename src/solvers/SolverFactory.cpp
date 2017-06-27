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
#include "solvers/SolverFactory.h"

#include "solvers/SolverStrategy.h"
#include "solvers/SolverStrategyParameters.h"

namespace AMP {
    
// register all known solver factories
void registerSolverFactories()
{
    auto &solverFactory = SolverFactory::getFactory();

}

} // end namespace AMP
