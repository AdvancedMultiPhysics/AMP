#ifndef included_AMP_LinearBVPOperator
#define included_AMP_LinearBVPOperator

#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/LinearOperator.h"

namespace AMP::Operator {

/**
 * Class LinearBVPOperator is meant to wrap a pointer to a linear volume or interior spatial
 * operator
 * with a pointer to a BoundaryOperator that handles spatial surface boundary conditions. The
 * constructor
 * takes a pointer to a BVPOperatorParameters object.
 */
class LinearBVPOperator : public LinearOperator
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
    explicit LinearBVPOperator( std::shared_ptr<const OperatorParameters> parameters );

    /**
     * virtual destructor which does nothing
     */
    virtual ~LinearBVPOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "LinearBVPOperator"; }

    /**
     * This function is useful for re-initializing/updating an operator
     */
    void reset( std::shared_ptr<const OperatorParameters> ) override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_volumeOperator->getInputVariable();
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_volumeOperator->getOutputVariable();
    }

    std::shared_ptr<LinearOperator> getVolumeOperator() { return d_volumeOperator; }

    std::shared_ptr<BoundaryOperator> getBoundaryOperator() { return d_boundaryOperator; }

    void modifyRHSvector( AMP::LinearAlgebra::Vector::shared_ptr rhs );

protected:
    /**
     * shared pointer to a nonlinear volume or interior spatial operator
     */
    std::shared_ptr<LinearOperator> d_volumeOperator;

    /**
     *  shared pointer to a boundary or surface operator that is responsible
     for apply operations on the boundary of the domain
     */
    std::shared_ptr<BoundaryOperator> d_boundaryOperator;

private:
};
} // namespace AMP::Operator

#endif
