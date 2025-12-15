#ifndef included_AMP_EpetraMatrixOperatorParameters_h
#define included_AMP_EpetraMatrixOperatorParameters_h

#include "AMP/matrices/trilinos/EpetraMatrixOperations.h"

namespace AMP::Operator {
class EpetraMatrixOperatorParameters : public OperatorParameters
{
public:
    Epetra_CrsMatrix *d_Matrix;

    explicit EpetraMatrixOperatorParameters( std::shared_ptr<Database> db )
        : OperatorParameters( db ), d_Matrix( nullptr )
    {
    }
};
} // namespace AMP::Operator

#endif
