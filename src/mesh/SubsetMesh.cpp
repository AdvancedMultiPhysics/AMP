#include "AMP/mesh/SubsetMesh.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/Vector.h"

#include "ProfilerApp.h"

#include <set>
#include <vector>


namespace AMP::Mesh {


/************************************************************************
 * Subroutine to search a sorted list for a mesh element id              *
 ************************************************************************/
static bool binarySearch( const std::vector<std::unique_ptr<MeshElement>> &x, MeshElementID y )
{
    if ( x.empty() )
        return false;
    if ( ( y <= x[0]->globalID() ) || ( y > x.back()->globalID() ) )
        return y == x[0]->globalID();
    size_t lower = 0;
    size_t upper = x.size() - 1;
    size_t index;
    while ( ( upper - lower ) != 1 ) {
        index = ( upper + lower ) / 2;
        if ( x[index]->globalID() >= y )
            upper = index;
        else
            lower = index;
    }
    index = upper;
    return y == x[index]->globalID();
}
static bool binarySearch( const std::vector<ElementID> &x, ElementID y )
{
    if ( x.empty() )
        return false;
    if ( ( y <= x[0] ) || ( y > x.back() ) )
        return y == x[0];
    size_t lower = 0;
    size_t upper = x.size() - 1;
    size_t index;
    while ( ( upper - lower ) != 1 ) {
        index = ( upper + lower ) / 2;
        if ( x[index] >= y )
            upper = index;
        else
            lower = index;
    }
    index = upper;
    return y == x[index];
}


/********************************************************
 * Create the subset mesh                                *
 ********************************************************/
std::shared_ptr<Mesh> SubsetMesh::create( std::shared_ptr<const Mesh> mesh,
                                          const AMP::Mesh::MeshIterator &iterator,
                                          bool isGlobal )
{
    // Check that all elements are contained within the mesh
    auto mesh_ids = mesh->getBaseMeshIDs();
    for ( const auto &elem : iterator ) {
        MeshID id  = elem.globalID().meshID();
        bool found = false;
        for ( auto &mesh_id : mesh_ids ) {
            if ( id == mesh_id )
                found = true;
        }
        if ( !found )
            AMP_ERROR( "Iterator contains elements not found in parent meshes" );
    }
    // Create the subsets
    std::set<MeshID> allMeshIDs;
    std::vector<std::shared_ptr<Mesh>> subsets;
    for ( auto &mesh_id : mesh_ids ) {
        auto mesh2 = mesh->Subset( mesh_id );
        if ( !mesh2->isBaseMesh() )
            continue;
        auto elements = std::make_shared<std::vector<std::unique_ptr<AMP::Mesh::MeshElement>>>();
        elements->reserve( iterator.size() );
        for ( const auto &elem : iterator ) {
            if ( elem.globalID().meshID() == mesh_id )
                elements->push_back( elem.clone() );
        }
        if ( elements->empty() )
            continue;
        allMeshIDs.insert( mesh_id );
        auto it2 = AMP::Mesh::MeshElementVectorIterator( elements, 0 );
        subsets.emplace_back( new SubsetMesh( mesh2, it2, isGlobal ) );
    }
    // Create the multimesh and return the appropriate subset
    AMP::AMP_MPI comm = isGlobal ? mesh->getComm() : AMP_COMM_SELF;
    comm.setGather( allMeshIDs );
    if ( allMeshIDs.size() == 0 )
        return nullptr;
    if ( allMeshIDs.size() == 1 ) {
        if ( subsets.empty() )
            return nullptr;
        else
            return subsets[0];
    }
    return std::make_shared<MultiMesh>( mesh->getName(), comm, subsets );
}


/********************************************************
 * Constructors                                          *
 ********************************************************/
SubsetMesh::SubsetMesh( std::shared_ptr<const Mesh> mesh,
                        const AMP::Mesh::MeshIterator &iterator_in,
                        bool isGlobal )
    : d_parentMesh( mesh ), d_parentMeshID( mesh->meshID() )
{
    AMP_INSIST( d_parentMesh->isBaseMesh(), "SubsetMesh only supports base meshes" );
    if ( isGlobal ) {
        this->d_comm = mesh->getComm();
    } else {
        for ( const auto &elem : iterator_in ) {
            if ( !elem.globalID().is_local() )
                AMP_ERROR(
                    "Subsetting local mesh for iterator with ghost elements is not supported" );
        }
        this->d_comm = AMP_MPI( AMP_COMM_SELF );
    }
    this->setMeshID();
    this->PhysicalDim = mesh->getDim();
    this->d_name      = mesh->getName() + "_subset";
    // Check the iterator
    auto type = GeomType::Nullity;
    for ( const auto &elem : iterator_in ) {
        if ( type == GeomType::Nullity )
            type = elem.elementType();
        if ( type != elem.elementType() )
            AMP_ERROR( "Subset mesh requires all of the elements to be the same type" );
    }
    int type2 = d_comm.minReduce( (int) type );
    if ( type != GeomType::Nullity && type2 != (int) type )
        AMP_ERROR( "Subset mesh requires all of the elements to be the same type" );
    this->GeomDim = static_cast<GeomType>( type2 );
    // Create a list of all elements of the desired type
    d_max_gcw  = d_parentMesh->getMaxGhostWidth();
    d_elements = std::vector<std::vector<ElementListPtr>>( (int) GeomDim + 1 );
    for ( int t = 0; t <= (int) GeomDim; t++ ) {
        d_elements[t] = std::vector<ElementListPtr>( d_max_gcw + 1 );
        for ( int gcw = 0; gcw <= d_max_gcw; gcw++ )
            d_elements[t][gcw] = std::make_shared<ElementList>();
    }
    int gcw = 0;
    while ( true ) {
        int t          = static_cast<int>( GeomDim );
        auto iterator1 = Mesh::getIterator(
            SetOP::Intersection, iterator_in, mesh->getIterator( GeomDim, gcw ) );
        auto iterator2 = iterator1.begin();
        if ( gcw > 0 )
            iterator2 =
                Mesh::getIterator( SetOP::Complement, iterator1, mesh->getIterator( GeomDim, 0 ) );
        d_elements[t][gcw] = std::make_shared<ElementList>( iterator2.size() );
        for ( size_t i = 0; i < iterator2.size(); i++ ) {
            d_elements[t][gcw]->operator[]( i ) = iterator2->clone();
            ++iterator2;
        }
        AMP::Utilities::quicksort( *d_elements[t][gcw] );
        if ( iterator1.size() == iterator_in.size() )
            break;
        gcw++;
    }
    // Create a list of all elements that compose the elements of GeomType
    for ( int t = 0; t < (int) GeomDim; t++ ) {
        std::vector<std::vector<ElementID>> ids( d_max_gcw + 1 );
        for ( gcw = 0; gcw <= d_max_gcw; gcw++ ) {
            std::vector<std::unique_ptr<MeshElement>> list;
            list.reserve( iterator_in.size() );
            ids[gcw].reserve( iterator_in.size() );
            for ( const auto &elem : this->getIterator( GeomDim, gcw ) ) {
                auto elements = elem.getElements( (GeomType) t );
                for ( auto &element : elements ) {
                    auto id = element->globalID().elemID();
                    if ( gcw == 0 ) {
                        if ( id.is_local() ) {
                            list.push_back( element->clone() );
                            ids[gcw].push_back( id );
                        }
                    } else {
                        bool found = false;
                        for ( int j = 0; j < gcw && !found; j++ )
                            found = found || binarySearch( ids[j], id );
                        if ( !found ) {
                            list.push_back( element->clone() );
                            ids[gcw].push_back( id );
                        }
                    }
                }
            }
            std::vector<size_t> I;
            AMP::Utilities::unique( ids[gcw], I );
            auto &elements = *d_elements[t][gcw];
            elements.reserve( ids[gcw].size() );
            for ( size_t i = 0; i < ids[gcw].size(); i++ )
                elements.push_back( std::move( list[I[i]] ) );
        }
    }
    // For each entity type, we need to check that any ghost elements are owned by somebody
    for ( int t = 0; t < (int) GeomDim; t++ ) {
        // First get a global list of all ghost elements
        std::vector<MeshElementID> ghost_local;
        for ( gcw = 0; gcw <= d_max_gcw; gcw++ ) {
            ghost_local.reserve( ghost_local.size() + d_elements[t][gcw]->size() );
            for ( size_t i = 0; i < d_elements[t][gcw]->size(); i++ ) {
                MeshElementID id = ( *d_elements[t][gcw] )[i]->globalID();
                if ( !id.is_local() )
                    ghost_local.push_back( id );
            }
        }
        size_t N_ghost_local  = ghost_local.size();
        size_t N_ghost_global = d_comm.sumReduce( N_ghost_local );
        if ( N_ghost_global == 0 )
            continue;
        std::vector<MeshElementID> ghost_global( N_ghost_global );
        MeshElementID *send_ptr = nullptr;
        if ( N_ghost_local > 0 ) {
            send_ptr = &ghost_local[0];
        }
        MeshElementID *recv_ptr = &ghost_global[0];
        d_comm.allGather( send_ptr, (int) N_ghost_local, recv_ptr );
        AMP::Utilities::unique( ghost_global );
        // For each ghost, check if we own it, and add it to the list if necessary
        MeshID my_mesh_id    = d_parentMesh->meshID();
        unsigned int my_rank = d_parentMesh->getComm().getRank();
        bool changed         = false;
        for ( auto &elem : ghost_global ) {
            AMP_ASSERT( my_mesh_id == elem.meshID() );
            if ( elem.owner_rank() == my_rank ) {
                bool found = false;
                for ( size_t j = 0; j < d_elements[t][0]->size(); j++ ) {
                    if ( ( *d_elements[t][0] )[j]->globalID() == elem ) {
                        found = true;
                        break;
                    }
                }
                if ( !found ) {
                    MeshElementID tmp = elem;
                    tmp.set_is_local( true ); // We do own this element
                    auto element = d_parentMesh->getElement( tmp );
                    AMP_ASSERT( element->globalID() != MeshElementID() );
                    AMP_ASSERT( element->globalID() == elem );
                    d_elements[t][0]->push_back( element->clone() );
                    changed = true;
                }
            }
        }
        if ( changed )
            AMP::Utilities::quicksort( *d_elements[t][0] );
    }
    // Count the number of elements of each type
    N_global = std::vector<size_t>( (int) GeomDim + 1 );
    for ( int i = 0; i <= (int) GeomDim; i++ )
        N_global[i] = d_elements[i][0]->size();
    d_comm.sumReduce( &N_global[0], (int) N_global.size() );
    for ( int i = 0; i <= (int) GeomDim; i++ )
        AMP_ASSERT( N_global[i] > 0 );
    // Create the bounding box
    d_box_local = std::vector<double>( 2 * PhysicalDim );
    for ( int j = 0; j < PhysicalDim; j++ ) {
        d_box_local[2 * j + 0] = 1e100;
        d_box_local[2 * j + 1] = -1e100;
    }
    for ( const auto &elem : getIterator( GeomType::Vertex, 0 ) ) {
        auto coord = elem.coord();
        for ( int j = 0; j < PhysicalDim; j++ ) {
            if ( coord[j] < d_box_local[2 * j + 0] )
                d_box_local[2 * j + 0] = coord[j];
            if ( coord[j] > d_box_local[2 * j + 1] )
                d_box_local[2 * j + 1] = coord[j];
        }
    }
    d_box = Mesh::reduceBox( d_box_local, d_comm );
    // Create the boundary id sets
    auto boundary_ids = d_parentMesh->getBoundaryIDs();
    std::set<int> new_boundary_ids;
    for ( int t = 0; t <= (int) GeomDim; t++ ) {
        for ( auto &boundary_id : boundary_ids ) {
            for ( gcw = 0; gcw <= d_max_gcw; gcw++ ) {
                if ( gcw > 0 )
                    continue; // Iterators over id sets with ghost values is not supported in
                              // libmesh yet
                auto iterator1 = MeshElementVectorIterator( d_elements[t][gcw], 0 );
                auto iterator2 =
                    d_parentMesh->getBoundaryIDIterator( (GeomType) t, boundary_id, gcw );
                auto iterator = Mesh::getIterator( SetOP::Intersection, iterator1, iterator2 );
                ElementListPtr elements;
                if ( iterator.size() == 0 ) {
                    elements = std::make_shared<ElementList>( 0 );
                } else if ( iterator.size() == iterator1.size() ) {
                    elements = d_elements[t][gcw];
                } else {
                    elements = std::make_shared<ElementList>( iterator.size() );
                    for ( size_t j = 0; j < iterator.size(); j++ ) {
                        elements->operator[]( j ) = iterator->clone();
                        ++iterator;
                    }
                }
                if ( gcw == 0 ) {
                    size_t global_size = d_comm.sumReduce( elements->size() );
                    if ( global_size == 0 )
                        break;
                }
                map_id_struct map_id;
                map_id.id   = boundary_id;
                map_id.type = (GeomType) t;
                map_id.gcw  = gcw;
                d_boundarySets.insert( std::make_pair( map_id, elements ) );
                new_boundary_ids.insert( boundary_id );
            }
        }
    }
    d_boundaryIdSets = std::vector<int>( new_boundary_ids.begin(), new_boundary_ids.end() );
    int *send_ptr    = nullptr;
    if ( d_boundaryIdSets.size() > 0 )
        send_ptr = &d_boundaryIdSets[0];
    size_t recv_size = d_comm.sumReduce( d_boundaryIdSets.size() );
    if ( recv_size > 0 ) {
        std::vector<int> recv_list( recv_size, 0 );
        d_comm.allGather( &send_ptr[0], (int) d_boundaryIdSets.size(), &recv_list[0] );
        for ( auto &elem : recv_list )
            new_boundary_ids.insert( elem );
        d_boundaryIdSets = std::vector<int>( new_boundary_ids.begin(), new_boundary_ids.end() );
    }
    // Create the surface sets
    d_surface = std::vector<std::vector<ElementListPtr>>( (int) GeomDim + 1 );
    for ( int t = 0; t <= (int) GeomDim; t++ ) {
        d_surface[t] = std::vector<ElementListPtr>( d_max_gcw + 1 );
        for ( gcw = 0; gcw <= d_max_gcw; gcw++ ) {
            auto iterator1 = MeshElementVectorIterator( d_elements[t][gcw], 0 );
            auto iterator2 = d_parentMesh->getSurfaceIterator( (GeomType) t, gcw );
            auto iterator  = Mesh::getIterator( SetOP::Intersection, iterator1, iterator2 );
            ElementListPtr elements;
            if ( iterator.size() == 0 ) {
                elements = std::make_shared<ElementList>( 0 );
            } else if ( iterator.size() == iterator1.size() ) {
                elements = d_elements[t][gcw];
            } else {
                elements = std::make_shared<ElementList>( iterator.size() );
                for ( size_t j = 0; j < iterator.size(); j++ ) {
                    elements->operator[]( j ) = iterator->clone();
                    ++iterator;
                }
            }
            d_surface[t][gcw] = elements;
        }
    }
    // Create the block id sets
    if ( GeomDim == d_parentMesh->getGeomType() ) {
        // Currently only elements support the block IDs
        std::vector<int> block_ids = d_parentMesh->getBlockIDs();
        std::set<int> new_block_ids;
        for ( const auto &elem : getIterator( GeomDim, 0 ) ) {
            for ( auto &block_id : block_ids ) {
                if ( elem.isInBlock( block_id ) )
                    new_block_ids.insert( block_id );
            }
        }
        d_blockIdSets = std::vector<int>( new_block_ids.begin(), new_block_ids.end() );
        send_ptr      = nullptr;
        if ( d_boundaryIdSets.size() > 0 )
            send_ptr = &d_boundaryIdSets[0];
        recv_size = d_comm.sumReduce( d_blockIdSets.size() );
        if ( recv_size > 0 ) {
            std::vector<int> recv_list( recv_size, 0 );
            d_comm.allGather( &send_ptr[0], (int) d_blockIdSets.size(), &recv_list[0] );
            for ( auto &elem : recv_list )
                new_block_ids.insert( elem );
            d_blockIdSets = std::vector<int>( new_block_ids.begin(), new_block_ids.end() );
        }
    }
    // Perform some final checks
    for ( int t = 0; t < (int) GeomDim; t++ ) {
        for ( int gcw = 0; gcw <= d_max_gcw; gcw++ ) {
            if ( d_elements[t][gcw] ) {
                auto &list = *d_elements[t][gcw];
                for ( size_t i = 1; i < list.size(); i++ )
                    AMP_ASSERT( *list[i] > *list[i - 1] );
            }
        }
    }
}


/********************************************************
 * Destructor                                            *
 ********************************************************/
SubsetMesh::~SubsetMesh() = default;


/********************************************************
 * Return the class name                                 *
 ********************************************************/
std::string SubsetMesh::meshClass() const
{
    return "SubsetMesh<" + d_parentMesh->meshClass() + ">";
}


/********************************************************
 * Copy the mesh                                         *
 ********************************************************/
std::unique_ptr<Mesh> SubsetMesh::clone() const
{
    AMP_ERROR( "clone is not currently supported with SubsetMesh" );
    return std::unique_ptr<Mesh>();
}


/********************************************************
 * Function to return the meshID composing the mesh      *
 ********************************************************/
std::vector<MeshID> SubsetMesh::getAllMeshIDs() const
{
    auto ids = d_parentMesh->getAllMeshIDs();
    ids.push_back( d_meshID );
    AMP::Utilities::quicksort( ids );
    return ids;
}
std::vector<MeshID> SubsetMesh::getBaseMeshIDs() const { return d_parentMesh->getBaseMeshIDs(); }
std::vector<MeshID> SubsetMesh::getLocalMeshIDs() const
{
    auto ids = d_parentMesh->getLocalMeshIDs();
    ids.push_back( d_meshID );
    AMP::Utilities::quicksort( ids );
    return ids;
}
std::vector<MeshID> SubsetMesh::getLocalBaseMeshIDs() const
{
    return d_parentMesh->getLocalBaseMeshIDs();
}


/********************************************************
 * Function to return the mesh with the given ID         *
 ********************************************************/
std::shared_ptr<Mesh> SubsetMesh::Subset( MeshID meshID ) const
{
    if ( d_meshID == meshID || d_parentMesh->meshID() == meshID )
        return std::const_pointer_cast<Mesh>( shared_from_this() );
    else
        return std::shared_ptr<Mesh>();
}


/********************************************************
 * Function to return the mesh with the given name       *
 ********************************************************/
std::shared_ptr<Mesh> SubsetMesh::Subset( std::string name ) const
{
    if ( d_name == name || d_parentMesh->getName() == name )
        return std::const_pointer_cast<Mesh>( shared_from_this() );
    else
        return std::shared_ptr<Mesh>();
}


/********************************************************
 * Mesh iterators                                        *
 ********************************************************/
MeshIterator SubsetMesh::getIterator( const GeomType type, const int gcw ) const
{
    int gcw2   = gcw;
    auto type2 = static_cast<int>( type );
    if ( (int) d_elements.size() <= type2 || gcw < 0 )
        return MeshIterator();
    if ( d_elements[type2].empty() )
        return MeshIterator();
    if ( gcw2 >= (int) d_elements[type2].size() )
        gcw2 = (int) d_elements[type2].size() - 1;
    if ( gcw2 == 0 )
        return MeshElementVectorIterator( d_elements[type2][0], 0 );
    std::vector<MeshIterator> iterators( gcw2 + 1 );
    for ( int i = 0; i <= gcw2; i++ )
        iterators[i] = MeshElementVectorIterator( d_elements[type2][i], 0 );
    return MultiIterator( iterators, 0 );
}
MeshIterator SubsetMesh::getSurfaceIterator( const GeomType type, const int gcw ) const
{
    auto type2 = static_cast<int>( type );
    if ( gcw == 0 )
        return MeshElementVectorIterator( d_surface[type2][0], 0 );
    if ( gcw >= (int) d_surface[type2].size() )
        AMP_ERROR( "Maximum ghost width exceeded" );
    std::vector<MeshIterator> iterators( gcw + 1 );
    for ( int i = 0; i <= gcw; i++ )
        iterators[i] = MeshElementVectorIterator( d_surface[type2][i], 0 );
    return MultiIterator( iterators, 0 );
}
std::vector<int> SubsetMesh::getBoundaryIDs() const { return d_boundaryIdSets; }
MeshIterator
SubsetMesh::getBoundaryIDIterator( const GeomType type, const int id, const int gcw ) const
{
    std::vector<MeshIterator> iterators;
    iterators.reserve( gcw + 1 );
    for ( int i = 0; i <= gcw; i++ ) {
        map_id_struct map_id;
        map_id.id   = id;
        map_id.type = type;
        map_id.gcw  = i;
        auto map_it = d_boundarySets.find( map_id );
        if ( map_it == d_boundarySets.end() )
            continue;
        iterators.push_back( MeshElementVectorIterator( map_it->second, 0 ) );
    }
    if ( iterators.empty() )
        return MeshIterator();
    if ( iterators.size() == 1 )
        return iterators[0];
    return MultiIterator( iterators, 0 );
}
std::vector<int> SubsetMesh::getBlockIDs() const { return d_blockIdSets; }
MeshIterator
SubsetMesh::getBlockIDIterator( const GeomType type, const int id, const int gcw ) const
{
    std::vector<MeshIterator> iterators;
    iterators.reserve( gcw + 1 );
    for ( int i = 0; i <= gcw; i++ ) {
        map_id_struct map_id;
        map_id.id   = id;
        map_id.type = type;
        map_id.gcw  = i;
        auto map_it = d_blockSets.find( map_id );
        if ( map_it == d_blockSets.end() )
            continue;
        iterators.push_back( MeshElementVectorIterator( map_it->second, 0 ) );
    }
    if ( iterators.empty() )
        return MeshIterator();
    if ( iterators.size() == 1 )
        return iterators[0];
    return MultiIterator( iterators, 0 );
}


/********************************************************
 * Check if the element is a member of the mesh          *
 ********************************************************/
bool SubsetMesh::isMember( const MeshElementID &id ) const
{
    if ( id.meshID() != d_parentMeshID )
        return false;
    auto type = static_cast<int>( id.type() );
    if ( type >= static_cast<int>( d_elements.size() ) )
        return false;
    for ( auto &tmp : d_elements[type] ) {
        if ( tmp == nullptr )
            continue;
        if ( binarySearch( *tmp, id ) )
            return true;
    }
    return false;
}
MeshIterator SubsetMesh::isMember( const MeshIterator &iterator ) const
{
    PROFILE( "isMember" );
    auto elements = std::make_shared<ElementList>();
    elements->reserve( iterator.size() );
    int size = static_cast<int>( d_elements.size() );
    for ( const auto &elem : iterator ) {
        auto id = elem.globalID();
        if ( id.meshID() != d_parentMeshID )
            continue;
        int type = static_cast<int>( id.type() );
        if ( type >= size )
            continue;
        for ( auto &tmp : d_elements[type] ) {
            if ( tmp == nullptr )
                continue;
            if ( binarySearch( *tmp, id ) )
                elements->push_back( elem.clone() );
        }
    }
    return AMP::Mesh::MeshElementVectorIterator( elements, 0 );
}


/********************************************************
 * Function to check if the mesh element is contained    *
 ********************************************************/
bool SubsetMesh::containsElement( const MeshElementID &id ) const
{
    int type = static_cast<int>( id.type() );
    for ( auto &tmp : d_elements[type] ) {
        if ( tmp == nullptr )
            continue;
        if ( binarySearch( *tmp, id ) )
            return true;
    }
    return false;
}


/********************************************************
 * Function to return the element given an ID            *
 ********************************************************/
std::unique_ptr<MeshElement> SubsetMesh::getElement( const MeshElementID &elem_id ) const
{
    return d_parentMesh->getElement( elem_id );
}


/********************************************************
 * Function to return parents of an element              *
 ********************************************************/
std::vector<std::unique_ptr<MeshElement>> SubsetMesh::getElementParents( const MeshElement &elem,
                                                                         const GeomType type ) const
{
    return d_parentMesh->getElementParents( elem, type );
}


/********************************************************
 * Other functions                                       *
 ********************************************************/
size_t SubsetMesh::numLocalElements( const GeomType type ) const
{
    return d_elements[static_cast<int>( type )][0]->size();
}
size_t SubsetMesh::numGlobalElements( const GeomType type ) const
{
    return N_global[static_cast<int>( type )];
}
size_t SubsetMesh::numGhostElements( const GeomType type, int gcw ) const
{
    AMP_ASSERT( type <= GeomDim );
    if ( gcw == 0 )
        return 0;
    if ( gcw >= (int) d_elements[(int) type].size() )
        AMP_ERROR( "Maximum ghost width exceeded" );
    return d_elements[(int) type][gcw]->size();
}
uint64_t SubsetMesh::positionHash() const { return d_parentMesh->positionHash(); }
Mesh::Movable SubsetMesh::isMeshMovable() const { return Mesh::Movable::Fixed; }
void SubsetMesh::displaceMesh( const std::vector<double> & )
{
    AMP_ERROR( "displaceMesh by a constant value does not work for subset mesh" );
}
void SubsetMesh::displaceMesh( const AMP::LinearAlgebra::Vector::const_shared_ptr )
{
    AMP_ERROR( "displaceMesh is not implemented for subset mesh" );
}


/****************************************************************
 * Check if two meshes are equal                                 *
 ****************************************************************/
bool SubsetMesh::operator==( const Mesh &rhs ) const
{
    // Check if &rhs == this
    if ( this == &rhs )
        return true;
    // Check if we can cast to a MultiMesh
    auto mesh = dynamic_cast<const SubsetMesh *>( &rhs );
    if ( !mesh )
        return false;
    // Perform comparison on sub-meshes
    if ( d_parentMesh != mesh->d_parentMesh )
        return false;
    AMP_ERROR( "Not finished" );
    return false;
}


/****************************************************************
 * Write restart data                                            *
 ****************************************************************/
void SubsetMesh::writeRestart( int64_t ) const
{
    AMP_ERROR( "writeRestart is not implimented for SubsetMesh" );
}


} // namespace AMP::Mesh
