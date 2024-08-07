#ifndef included_AMP_IDATimeOperator
#define included_AMP_IDATimeOperator


#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/time_integrators/TimeOperator.h"
#include "AMP/time_integrators/TimeOperatorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include <memory>


namespace AMP::TimeIntegrator {

typedef TimeOperatorParameters IDATimeOperatorParameters;

/*!
  @brief operator class associated with IDATimeIntegrator

  Class IDATimeOperator is derived from TimeOperator. It
  is the operator class associated with a IDATimeIntegrator.

  @see IDATimeIntegrator
  @see TimeOperator
*/

class IDATimeOperator : public TimeOperator
{
public:
    /**
     * Main constructor.
     @param [in] params: shared pointer to TimeOperatorParameters object.
     */
    explicit IDATimeOperator( std::shared_ptr<AMP::Operator::OperatorParameters> params );

    /**
     * virtual destructor
     */
    virtual ~IDATimeOperator();

    //! Return the name of the operator
    std::string type() const override { return "IDATimeOperator"; }

    // virtual void reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params);

    /**
      The function that computes the residual.
     * @param u: multivector of the state.
     * @param f: The result of apply ( f = A(u) )
     */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    void residual( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                   std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                   std::shared_ptr<AMP::LinearAlgebra::Vector> r ) override;

    std::shared_ptr<AMP::Operator::OperatorParameters>
    getParameters( const std::string &type,
                   AMP::LinearAlgebra::Vector::const_shared_ptr u,
                   std::shared_ptr<AMP::Operator::OperatorParameters> params = nullptr ) override;
    /**
     * registers the time derivative vector provided by IDA with this operator
     @param [in] vec   shared pointer to time derivative computed by IDA
     */
    void registerIDATimeDerivative( std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
    {
        d_pIDATimeDerivative = vec;
    }

    /**
     * registers a source term if any
     @param [in] vec   shared pointer to vector for source term
     */
    void registerSourceTerm( std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
    {
        d_pSourceTerm = vec;
    }

    /**
     * sets the current time for the operator
     @param [in] currentTime   the current time
     */
    void registerCurrentTime( double currentTime ) { d_current_time = currentTime; }

protected:
    IDATimeOperator();

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pIDATimeDerivative;

    bool d_cloningHappened;

    // JL
    // The test we want to run has a source term which depends on time
    // The time comes from TimeIntegrator
    double d_current_time;

private:
};
} // namespace AMP::TimeIntegrator

#endif
