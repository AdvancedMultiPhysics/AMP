
#ifndef included_AMP_DTK_AMPMeshEntity
#define included_AMP_DTK_AMPMeshEntity

#include "ampmesh/MeshElement.h"

#include "utils/AMP_MPI.h"

#include <DTK_Entity.hpp>

namespace AMP {
namespace Operator {


/**
  * AMP Mesh element implementation for DTK EntityExtraData interface.
*/
class AMPMeshEntity : public DataTransferKit::Entity
{
public :

    /**
     * Constructor.
     */
    AMPMeshEntity( const AMP::Mesh::MeshElement& element );

    //! Destructor
    ~AMPMeshEntity() { }
};


}
}

#endif

