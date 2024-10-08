#ifndef included_AMP_NeutronicsRhsExtrasParameters
#define included_AMP_NeutronicsRhsExtrasParameters

#include "AMP/operators/OperatorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/Vector.h"
#include <memory>

namespace AMP::Operator {

/**
  A class for encapsulating the parameters that are required for constructing the
  neutronics source operator.
  @see NeutronicsRhsExtras
  */
class NeutronicsRhsExtrasParameters : public OperatorParameters
{
public:
    explicit NeutronicsRhsExtrasParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db )
    {
        d_numExtras = 0;
    }

    //      std::shared_ptr<AMP::MeshUtils> d_MeshUtils;
    int d_numExtras;
    std::vector<std::string> d_extrasName;
};
} // namespace AMP::Operator

#endif
