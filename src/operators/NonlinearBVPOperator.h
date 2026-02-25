
#ifndef included_AMP_NonlinearBVPOperator
#define included_AMP_NonlinearBVPOperator

#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/boundary/BoundaryOperator.h"

namespace AMP::Operator {

/**
 * Class NonlinearBVPOperator is meant to wrap a pointer to a nonlinear volume or interior spatial
 * operator
 * with a pointer to a BoundaryOperator that handles spatial surface boundary conditions. The
 * constructor
 * takes a pointer to a BVPOperatorParameters object
 */
class NonlinearBVPOperator : public Operator
{
public:
    /**
     * Main constructor
     * @param[in] parameters The parameters object contains a database object which
     *     must contain the following fields in addition to the fields expected by
     *     the base Operator class:
     *     1. name: VolumeOperator, type: string, (required), name of the database
     *        associated with the volume operator
     *     2. name: BoundaryOperator, type: string, (required), name of the database
     *        associated with theboundary operator
     *     3. name: name, type: string, (required), must be set to LinearBVPOperator
     */
    explicit NonlinearBVPOperator( std::shared_ptr<const OperatorParameters> parameters );

    /**
     * virtual destructor which does nothing
     */
    virtual ~NonlinearBVPOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "NonlinearBVPOperator"; }

    /**
      The apply function for this operator performs the following operation:
      r = b*f+a*A(u), if f is not NULL and
      r = a*A(u), if f is NULL
      Here, A represents the action of the composite volume and boundary operator
      */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr,
                AMP::LinearAlgebra::Vector::shared_ptr ) override;

    /**
     * This function is useful for re-initializing/updating an operator
     * \param params
     *        parameter object containing parameters to change
     */
    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    std::shared_ptr<Operator> getVolumeOperator() { return d_volumeOperator; }

    std::shared_ptr<BoundaryOperator> getBoundaryOperator() { return d_boundaryOperator; }

    void modifyRHSvector( AMP::LinearAlgebra::Vector::shared_ptr rhs );

    void modifyInitialSolutionVector( AMP::LinearAlgebra::Vector::shared_ptr sol );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_volumeOperator->getInputVariable();
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_volumeOperator->getOutputVariable();
    }

    bool isValidVector( AMP::LinearAlgebra::Vector::const_shared_ptr sol ) override
    {
        return d_volumeOperator->isValidVector( sol );
    }

    std::shared_ptr<OperatorParameters>
    getParameters( const std::string &type,
                   AMP::LinearAlgebra::Vector::const_shared_ptr x,
                   std::shared_ptr<OperatorParameters> params = nullptr ) override;

protected:
    /**
     * shared pointer to a nonlinear volume or interior spatial operator
     */
    std::shared_ptr<Operator> d_volumeOperator;

    /**
     *  shared pointer to a boundary or surface operator that is responsible for apply operations on
     * the boundary of the
     * domain
     */
    std::shared_ptr<BoundaryOperator> d_boundaryOperator;

private:
};
} // namespace AMP::Operator

#endif
