
#ifndef included_AMP_DTK_DTKMapOperator
#define included_AMP_DTK_DTKMapOperator

#include "operators/Operator.h"

#include "utils/AMP_MPI.h"

#include "ampmesh/Mesh.h"

#include "vectors/Vector.h"

#include "DTKAMPMeshManager.h"

#include <DTK_MapOperator.hpp>

namespace AMP {
namespace Operator {

class DTKMapOperatorParameters : public OperatorParameters
{
public:
    // Constructor.
    explicit DTKMapOperatorParameters( const AMP::shared_ptr<AMP::Database> &db )
        : OperatorParameters( db )
    { /* ... */
    }

    AMP_MPI d_globalComm;
    // Domain mesh. A manager for the mesh that is the data source.
    AMP::shared_ptr<AMP::Mesh::Mesh> d_domain_mesh;

    // Range mesh. A manager for the mesh that is the data target.
    AMP::shared_ptr<AMP::Mesh::Mesh> d_range_mesh;

    // Domain DOF manager. A DOF manager for the source data.
    AMP::shared_ptr<AMP::Discretization::DOFManager> d_domain_dofs;

    // Range DOF manager. A DOF Manager for the target data.
    AMP::shared_ptr<AMP::Discretization::DOFManager> d_range_dofs;
};


/**
 * AMP operator element implementation for DTK Map operator.
 */
class DTKMapOperator : public Operator
{
public:
    /**
     * Constructor.
     */
    explicit DTKMapOperator( const AMP::shared_ptr<OperatorParameters> &params );

    //! Destructor
    ~DTKMapOperator() {}

    //! Apply function.
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;

private:
    AMP_MPI d_comm;
    Teuchos::RCP<const Teuchos::Comm<int>> d_TeuchosComm;
    bool d_mapOnThisProc;
    // DTK map operator.
    AMP::shared_ptr<DataTransferKit::MapOperator> d_dtk_operator;

    // DTK domain mesh.
    AMP::shared_ptr<DTKAMPMeshManager> d_domain_mesh;

    // DTK range mesh.
    AMP::shared_ptr<DTKAMPMeshManager> d_range_mesh;
};
} // namespace Operator
} // namespace AMP

#endif
