#ifndef included_AMP_AsyncMapOperator
#define included_AMP_AsyncMapOperator

#include "AMP/mesh/Mesh.h"
#include "AMP/operators/AsynchronousOperator.h"

namespace AMP {
namespace Operator {

/** \brief  A base class for asynchronous map operations between meshes.
 * A map operation involves two meshes and a communicator spanning those meshes.
 * For some processors one of the meshes may be NULL.
 * The constructor may require syncronous communication, but the apply calls
 * should be implemented asynchronously.
 * Note: Maps may impose a serial thread or even deadlock in parallel if
 * implemented synchronously without great care.
 */
class AsyncMapOperator : public AsynchronousOperator
{
public:
    //! Constructor
    explicit AsyncMapOperator( std::shared_ptr<const OperatorParameters> );

    virtual ~AsyncMapOperator();

    //! Return the name of the operator
    std::string type() const override { return "AsyncMapOperator"; }

    /** \brief  Set a frozen vector for results of the apply operation.
     * \param[in]  p  The vector to set
     */
    virtual void setVector( AMP::LinearAlgebra::Vector::shared_ptr p ) = 0;

    // Overload the apply operator to include makeConsistent
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    // Function to determine if a makeConsistentSet is required
    virtual bool requiresMakeConsistentSet();

    /**
     * Get the meshes this map uses.
     * \param[in] which 1 for d_mesh1 and 2 for d_mesh2
     */
    std::shared_ptr<AMP::Mesh::Mesh> getMesh( int which );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_inpVariable;
    }
    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outVariable;
    }

protected:
    // Communicator for the Map
    AMP_MPI d_MapComm;

    // Variables to store the individual meshes and the DOFManager
    std::shared_ptr<AMP::Mesh::Mesh> d_mesh1;
    std::shared_ptr<AMP::Mesh::Mesh> d_mesh2;
    std::shared_ptr<AMP::Discretization::DOFManager> d_DOFManager;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVariable;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;

    // Frozen vector for the output results
    AMP::LinearAlgebra::Vector::shared_ptr d_OutputVector;
};
} // namespace Operator
} // namespace AMP


#endif
