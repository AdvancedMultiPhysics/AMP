
#ifndef included_AMP_DTK_AMPMeshEntity
#define included_AMP_DTK_AMPMeshEntity

#include "AMP/mesh/MeshElement.h"

#include "AMP/utils/AMP_MPI.h"

#include <DTK_Entity.hpp>

#include <map>
#include <unordered_map>

namespace AMP::Operator {


/**
 * AMP Mesh element implementation for DTK Entity interface.
 */
class AMPMeshEntity : public DataTransferKit::Entity
{
public:
    /**
     * Constructor.
     */
    explicit AMPMeshEntity(
        const AMP::Mesh::MeshElement &element,
        const std::unordered_map<int, int> &rank_map,
        const std::map<AMP::Mesh::MeshElementID, DataTransferKit::EntityId> &id_map );

    //! Destructor
    ~AMPMeshEntity() {}
};
} // namespace AMP::Operator

#endif
