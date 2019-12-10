
#ifndef included_AMP_BlockOperatorParameters
#define included_AMP_BlockOperatorParameters

#include "AMP/operators/OperatorParameters.h"

#include "AMP/utils/shared_ptr.h"

namespace AMP {
namespace Operator {

class BlockOperatorParameters : public OperatorParameters
{
public:
    explicit BlockOperatorParameters( AMP::shared_ptr<AMP::Database> db ) : OperatorParameters( db )
    {
    }

    virtual ~BlockOperatorParameters() {}

    std::vector<std::vector<AMP::shared_ptr<OperatorParameters>>> d_blockParams;
};
} // namespace Operator
} // namespace AMP

#endif
