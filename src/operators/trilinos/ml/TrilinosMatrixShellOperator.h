#ifndef included_AMP_TrilinosMatrixShellOperator
#define included_AMP_TrilinosMatrixShellOperator

#include "AMP/operators/LinearOperator.h"

DISABLE_WARNINGS
#include "ml_include.h"
ENABLE_WARNINGS

#include <functional>


namespace AMP::Operator {


class TrilinosMatrixShellOperator : public LinearOperator
{
public:
    explicit TrilinosMatrixShellOperator( std::shared_ptr<const OperatorParameters> params );

    virtual ~TrilinosMatrixShellOperator() {}

    void setOperator( std::shared_ptr<Operator> op );

    void setNodalDofMap( std::shared_ptr<AMP::Discretization::DOFManager> dofMap );

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;
    /**
     * Column specific implementation of the residual: f-L(u)
     * \param f: shared pointer to const vector rhs
     * \param u: shared pointer to const vector u
     * \param r: shared pointer to vector residual
     */
    virtual void residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                           AMP::LinearAlgebra::Vector::const_shared_ptr u,
                           AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

    static int
    matVec( ML_Operator *data, int in_length, double in[], int out_length, double out[] );

    static int getRow( ML_Operator *data,
                       int N_requested_rows,
                       int requested_rows[],
                       int allocated_space,
                       int columns[],
                       double values[],
                       int row_lengths[] );

    void setGetRow(
        std::function<void(
            void *object, int row, std::vector<size_t> &cols, std::vector<double> &values )> );

    void getColumn( int column, std::vector<size_t> &rows, std::vector<double> &values );

    size_t getMatrixSize();

private:
    std::shared_ptr<AMP::Discretization::DOFManager> d_nodalDofMap;

    std::shared_ptr<Operator> d_operator;

    std::function<void( void *, int, std::vector<size_t> &, std::vector<double> & )> d_getRow;
};
} // namespace AMP::Operator

#endif
