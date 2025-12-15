#include "AMP/operators/OperatorParameters.h"


namespace AMP::Operator {


OperatorParameters::OperatorParameters( std::shared_ptr<AMP::Database> db,
                                        std::shared_ptr<AMP::Mesh::Mesh> mesh )
    : ParameterBase( db ), d_Mesh( mesh )
{
}

} // namespace AMP::Operator
