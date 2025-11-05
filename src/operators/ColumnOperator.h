
#ifndef included_AMP_ColumnOperator
#define included_AMP_ColumnOperator

#include "AMP/operators/ColumnOperatorParameters.h"
#include "AMP/operators/Operator.h"

#include <vector>

namespace AMP::Operator {

/**
  A class for representing a composite operator, F=(F1, F2, F3, .., Fk),
  where each of F1,.., Fk are operators themselves. The user is expected to have
  created and initialized the operators F1,.., Fk
  */
class ColumnOperator : public Operator
{

public:
    //! Empty constructor;
    explicit ColumnOperator();

    //! Default constructor;
    explicit ColumnOperator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~ColumnOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "ColumnOperator"; }

    //! The apply routine for the column operator calls apply on each of the component operators
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    /**
     * Column specific implementation of the residual: f-L(u)
     * \param f: shared pointer to const vector rhs
     * \param u: shared pointer to const vector u
     * \param r: shared pointer to vector residual
     */
    void residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                   AMP::LinearAlgebra::Vector::const_shared_ptr u,
                   AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    /**
      A function for computing the information necessary to construct the jacobian.
     * \param type: std:string specifying type of return operator parameters
     *      being requested. Currently the valid option is Jacobian
     * \param u: const pointer to current solution vector
     * \param params: pointer to additional parameters that might be required
     *      to construct the return parameters
      @return The parameters required to construct the jacobian.
      */
    std::shared_ptr<OperatorParameters>
    getParameters( const std::string &type,
                   AMP::LinearAlgebra::Vector::const_shared_ptr u,
                   std::shared_ptr<OperatorParameters> params = nullptr ) override;

    /**
     * \param op
     *            shared pointer to an operator to append to the existing column of operators
     */
    virtual void append( std::shared_ptr<Operator> op );

    /**
     * returns a MultiVariable object corresponding to the ColumnOperator
     * should be called only after all column operators have been appended.
     * no checks to do this right now.
     */
    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> u ) override;

    size_t getNumberOfOperators() { return d_operators.size(); }

    std::shared_ptr<Operator> getOperator( size_t i ) { return d_operators[i]; }

    inline const auto &getOperators() { return d_operators; }

    //! Return an iterator to the beginning of the operators
    inline auto begin() { return d_operators.begin(); }

    //! Return an iterator to the end of the operators
    inline auto end() { return d_operators.end(); }

    //! Return the operator(s) with the given name/type
    std::vector<std::shared_ptr<Operator>> find( const std::string &name );

protected:
    std::vector<std::shared_ptr<Operator>> d_operators;

private:
};
} // namespace AMP::Operator

#endif
