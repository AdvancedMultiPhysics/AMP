#ifndef FICKSORETNONLINEARFEOPERATORPARAMETERS_H_
#define FICKSORETNONLINEARFEOPERATORPARAMETERS_H_

/*
 * FickSoretNonlinearFEOperatorParameters.h
 *
 *  Created on: Jun 11, 2010
 *      Author: gad
 */

#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/vectors/Variable.h"
#include <string>

namespace AMP::Operator {

class FickSoretNonlinearFEOperator;

class FickSoretNonlinearFEOperatorParameters : public OperatorParameters
{
public:
    explicit FickSoretNonlinearFEOperatorParameters( std::shared_ptr<Database> &db )
        : OperatorParameters( db )
    {
    }

    std::shared_ptr<DiffusionNonlinearFEOperator> d_FickOperator;
    std::shared_ptr<DiffusionNonlinearFEOperator> d_SoretOperator;
    std::shared_ptr<DiffusionNonlinearFEOperatorParameters> d_FickParameters;
    std::shared_ptr<DiffusionNonlinearFEOperatorParameters> d_SoretParameters;
};
} // namespace AMP::Operator

#endif /* FICKSORETNONLINEARFEOPERATORPARAMETERS_H_ */
