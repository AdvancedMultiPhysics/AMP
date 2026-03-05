#include "AMP/mesh/MeshListIterator.hpp"


using ElementPtr = std::unique_ptr<AMP::Mesh::MeshElement>;


template class AMP::Mesh::MeshListIterator<ElementPtr>;
