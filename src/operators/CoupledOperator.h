#ifndef included_AMP_CoupledOperator
#define included_AMP_CoupledOperator

#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/CoupledOperatorParameters.h"
#include "AMP/vectors/Vector.h"

#include <vector>


namespace AMP::Operator {


/**
   A class for representing a coupled operator combining a NodeToGaussPointOperator,
   a CopyOperator, a MapOperator, and a d_BVPOperator.
  */
class CoupledOperator : public ColumnOperator
{
public:
    explicit CoupledOperator( std::shared_ptr<const OperatorParameters> params );

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    //! Return the name of the operator
    std::string type() const override { return "CoupledOperator"; }

    /**
     * Default base class implementation of the residual: f-L(u)
     * \param f: shared pointer to const vector rhs
     * \param u: shared pointer to const vector u
     * \param r: shared pointer to vector residual
     */
    void residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                   AMP::LinearAlgebra::Vector::const_shared_ptr u,
                   AMP::LinearAlgebra::Vector::shared_ptr r ) override;


    std::shared_ptr<AMP::Operator::Operator> getMapOperator() { return d_operators[2]; }

    void setMapOperator( std::shared_ptr<AMP::Operator::Operator> op ) { d_operators[2] = op; }

    std::shared_ptr<AMP::Operator::Operator> getBVPOperator() { return d_operators[3]; }

    void setBVPOperator( std::shared_ptr<AMP::Operator::Operator> op ) { d_operators[3] = op; }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_operators[3]->getOutputVariable();
    }

    void append( std::shared_ptr<Operator> op ) override
    {
        AMP_ASSERT( d_operators.size() < 4 );
        AMP_ASSERT( op );
        d_operators.push_back( op );
    }

    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> u ) override
    {
        return d_operators[3]->isValidVector( u );
    }

    void setFrozenGaussPointVector( AMP::LinearAlgebra::Vector::shared_ptr u )
    {
        d_frozenGaussPointVector = u;
    }

    std::shared_ptr<OperatorParameters>
    getParameters( const std::string &type,
                   AMP::LinearAlgebra::Vector::const_shared_ptr u,
                   std::shared_ptr<OperatorParameters> params = nullptr ) override
    {
        return d_operators[3]->getParameters( type, u, params );
    }

    virtual ~CoupledOperator() {}

protected:
    AMP::LinearAlgebra::Vector::shared_ptr d_frozenGaussPointVector;
};
} // namespace AMP::Operator

#endif
