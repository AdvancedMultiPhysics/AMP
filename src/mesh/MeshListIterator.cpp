#include "AMP/mesh/MeshListIterator.hpp"


using ElementPtr = std::unique_ptr<AMP::Mesh::MeshElement>;


template AMP::Mesh::MeshIterator
    AMP::Mesh::createMeshListIterator<ElementPtr>( std::shared_ptr<std::vector<ElementPtr>>,
                                                   size_t );
template class AMP::Mesh::MeshListIterator<ElementPtr>;
