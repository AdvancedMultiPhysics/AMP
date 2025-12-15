#ifndef included_AMP_GradientOperator
#define included_AMP_GradientOperator


#include "AMP/discretization/createLibmeshElements.h"
#include "AMP/operators/Operator.h"


namespace AMP::Operator {


/**
 * Class GradientOperator computes the derivative of a nodal vector
 */
class GradientOperator : public Operator
{
public:
    //! Default constructor
    GradientOperator( void );

    //! Constructor
    explicit GradientOperator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~GradientOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "GradientOperator"; }

    /**
     * This function is useful for re-initializing/updating an operator
     * \param params
     *    parameter object containing parameters to change
     */
    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    /**
      This base class can not give a meaningful definition of apply. See the derived classes for
      how they define apply. Each operator is free to define apply in a way that is appropriate
      for that operator.
      \param u: shared pointer to const input vector u
      \param f: shared pointer to output vector storing result of applying this operator
      */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;


    //! Return the output variable
    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outputVar;
    }

    //! Return the input variable
    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_inputVar;
    }

private:
    std::shared_ptr<AMP::Database> d_db;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inputVar, d_outputVar;
    AMP::Discretization::createLibmeshElements d_elem;
    std::vector<AMP::Mesh::MeshElementID> d_nodes;
    std::vector<int> d_elementsPerNode;
};


} // namespace AMP::Operator

#endif
