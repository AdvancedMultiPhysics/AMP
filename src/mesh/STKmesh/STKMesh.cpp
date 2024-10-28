#include "AMP/mesh/STKmesh/STKMesh.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/STKmesh/STKMeshElement.h"
#include "AMP/mesh/STKmesh/STKMeshIterator.h"
#include "AMP/mesh/STKmesh/initializeSTKMesh.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

// STKMesh include
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/FieldData.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/fem/FEMMetaData.hpp"
#include "stk_mesh/fem/SkinMesh.hpp"

#include "Ionit_Initializer.h"
#include "Ioss_SubSystem.h"

namespace AMP::Mesh {

using CartesianField = stk::mesh::Field<double, stk::mesh::Cartesian>;

struct NullDeleter {
    template<typename T>
    void operator()( T * )
    {
    }
};

/********************************************************
 * Constructors                                          *
 ********************************************************/
STKMesh::STKMesh( std::shared_ptr<const MeshParameters> params_in ) : Mesh( params_in )
{
    PROFILE( "constructor" );
    this->d_max_gcw = 1;
    // Check for valid inputs
    AMP_INSIST( params.get(), "Params must not be null" );
    AMP_INSIST( d_comm != AMP_MPI( AMP_COMM_NULL ), "Communicator must be set" );
    // Intialize STKMesh
    STKmeshInit = std::make_shared<initializeSTKMesh>( d_comm );
    // Load the mesh
    if ( d_db.get() ) {
        // Database exists
        AMP_INSIST( d_db->keyExists( "dim" ), "Variable 'dim' must be set in the database" );
        AMP_INSIST( d_db->keyExists( "MeshName" ), "MeshName must exist in input database" );
        PhysicalDim = d_db->getScalar<int>( "dim" );
        d_name      = d_db->getString( "MeshName" );
        AMP_INSIST( PhysicalDim > 0 && PhysicalDim < 10, "Invalid dimension" );
        GeomDim = (GeomType) PhysicalDim;
        if ( d_db->keyExists( "FileName" ) ) {
            // Read an existing mesh
            d_STKIOFixture =
                std::make_shared<stk::io::util::IO_Fixture>( d_comm.getCommunicator() );
            d_STKIOFixture->initialize_meta_data( d_db->getString( "FileName" ) );
            stk::mesh::fem::FEMMetaData &meta_data = d_STKIOFixture->meta_data();
            stk::mesh::Part &skin_part             = meta_data.declare_part( "skin_part" );
            meta_data.commit();
            d_STKIOFixture->initialize_bulk_data();
            d_STKMeshMeta.reset( &d_STKIOFixture->meta_data(), NullDeleter() );
            d_STKMeshBulk.reset( &d_STKIOFixture->bulk_data(), NullDeleter() );
            d_STKIORegion = d_STKIOFixture->input_ioss_region();
        } else if ( d_db->keyExists( "Generator" ) ) {
            std::string generator = d_db->getString( "Generator" );
            d_STKGMeshFixture     = std::make_shared<stk::io::util::Gmesh_STKmesh_Fixture>(
                d_comm.getCommunicator(), generator );
            d_STKMeshMeta.reset( &d_STKGMeshFixture->getFEMMetaData(), NullDeleter() );
            d_STKMeshBulk.reset( &d_STKGMeshFixture->getBulkData(), NullDeleter() );
        } else {
            AMP_ERROR( "Unable to construct mesh with given parameters" );
        }
        // Initialize all of the internal data
        initialize();
        // Displace the mesh
        std::vector<double> displacement( PhysicalDim, 0.0 );
        if ( d_db->keyExists( "x_offset" ) )
            displacement[0] = d_db->getScalar<double>( "x_offset" );
        if ( d_db->keyExists( "y_offset" ) )
            displacement[1] = d_db->getScalar<double>( "y_offset" );
        if ( d_db->keyExists( "z_offset" ) )
            displacement[2] = d_db->getScalar<double>( "z_offset" );
        bool test = false;
        for ( size_t i = 0; i < displacement.size() && !test; i++ )
            test = displacement[i];
        if ( test )
            displaceMesh( displacement );
    } else {
        AMP_ERROR( "Error: params must contain a database object" );
    }
}

STKMesh::STKMesh( std::shared_ptr<stk::mesh::BulkData> mesh, std::string name )
{
    // Set the base properties
    const auto *meta_data = &mesh->mesh_meta_data();
    auto *fem_meta_data   = &stk::mesh::fem::FEMMetaData::get( *meta_data );
    AMP_INSIST( fem_meta_data, "STKMesh::STKMesh not called with a FEM meta data." );
    d_STKMeshMeta.reset( fem_meta_data, NullDeleter() );
    d_STKMeshBulk.reset( mesh.get(), NullDeleter() );
#ifdef MPI_CLASS
    this->d_comm = AMP_MPI( d_STKMeshBulk->parallel() );
    AMP_ASSERT( d_comm != AMP_MPI( AMP_COMM_NULL ) );
#else
    this->d_comm = AMP_MPI( AMP_COMM_SELF );
#endif
    this->setMeshID();
    this->d_name      = name;
    this->d_max_gcw   = 1;
    this->PhysicalDim = d_STKMeshMeta->spatial_dimension();
    this->GeomDim     = (GeomType) PhysicalDim;
    // Initialize all of the internal data
    initialize();
}


/********************************************************
 * Destructor                                            *
 ********************************************************/
STKMesh::~STKMesh()
{
    // First we need to destroy the elements, surface sets, and boundary sets
    d_localElements.clear();
    d_ghostElements.clear();
    d_localSurfaceElements.clear();
    d_ghostSurfaceElements.clear();
    d_boundarySets.clear();
    // We need to clear all STKmesh objects before STKmeshInit
    STKmeshInit.reset();
}


/********************************************************
 * Return the class name                                 *
 ********************************************************/
std::string STKMesh::meshClass() const { return "STKMesh"; }


/********************************************************
 * Function to copy the mesh                             *
 ********************************************************/
Mesh STKMesh::copy() const { return STKMesh( *this ); }


/********************************************************
 * Function to initialize the STKMesh object             *
 ********************************************************/
void STKMesh::initialize()
{
    PROFILE( "initialize" );
    // Verify STKmesh's rank and size agrees with the rank and size of the comm of the mesh
    AMP_INSIST( stk::parallel_machine_rank( d_STKMeshBulk->parallel() ) ==
                    (unsigned) d_comm.getRank(),
                "rank of the mesh does not agree with STKmesh" );
    AMP_INSIST( stk::parallel_machine_size( d_STKMeshBulk->parallel() ) ==
                    (unsigned) d_comm.getSize(),
                "size of the mesh does not agree with STKmesh" );
    // Count the elements
    const unsigned NDim = PhysicalDim + 1;
    n_local             = std::vector<unsigned>( NDim, 0 );
    n_global            = std::vector<unsigned>( NDim, 0 );
    n_ghost             = std::vector<unsigned>( NDim, 0 );
    stk::mesh::count_entities( d_STKMeshMeta->locally_owned_part(), *d_STKMeshBulk, n_local );
    stk::mesh::count_entities( !stk::mesh::Selector(), *d_STKMeshBulk, n_global );
    for ( unsigned i = 0; i < NDim; i++ )
        n_ghost[i] = n_global[i] - n_local[i];
    std::vector<size_t> global;
    stk::mesh::fem::comm_mesh_counts( *d_STKMeshBulk, global );
    n_global.assign( global.begin(), global.end() );
    ;

    AMP_INSIST( n_local[0] > 0, "We currently require at least 1 node on each processor" );
    AMP_INSIST( n_local[NDim - 1] > 0,
                "We currently require at least 1 element on each processor" );

    // Compute the bounding box of the mesh
    d_box_local = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box_local[2 * i + 0] = std::numeric_limits<double>::max();
        d_box_local[2 * i + 1] = -std::numeric_limits<double>::max();
    }

    auto *coordinates =
        d_STKMeshMeta->get_field<stk::mesh::Field<double, stk::mesh::Cartesian>>( "coordinates" );
    AMP_INSIST( coordinates, "Can not find 'coordinates' in STKMesh file." );
    std::vector<stk::mesh::Entity *> nodes;
    stk::mesh::get_entities( *d_STKMeshBulk, d_STKMeshMeta->node_rank(), nodes );

    for ( unsigned n = 0; n < nodes.size(); ++n ) {
        double *x = stk::mesh::field_data( *coordinates, *nodes[n] );
        AMP_INSIST( x, "Null coordinate field found." );
        for ( int i = 0; i < PhysicalDim; i++ ) {
            if ( x[i] < d_box_local[2 * i + 0] ) {
                d_box_local[2 * i + 0] = x[i];
            }
            if ( x[i] > d_box_local[2 * i + 1] ) {
                d_box_local[2 * i + 1] = x[i];
            }
        }
    }
    d_box = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] = d_comm.minReduce( d_box_local[2 * i + 0] );
        d_box[2 * i + 1] = d_comm.maxReduce( d_box_local[2 * i + 1] );
    }
    // Construct the node neighbor information
    const unsigned rank = d_comm.getRank();
    neighborNodeIDs =
        std::vector<unsigned int>( n_local[0], std::numeric_limits<unsigned int>::max() );
    neighborNodes = std::vector<std::vector<stk::mesh::Entity *>>( n_local[0] );
    for ( size_t e = 0, i = 0; e < nodes.size(); ++e ) {
        const stk::mesh::Entity *node = nodes[e];
        if ( node->owner_rank() == rank )
            neighborNodeIDs[i++] = node->identifier();
    }
    AMP::Utilities::quicksort( neighborNodeIDs );
    std::vector<std::set<stk::mesh::EntityKey>> tmpNeighborNodes( n_local[0] );
    std::vector<stk::mesh::Entity *> elements;
    stk::mesh::get_entities( *d_STKMeshBulk, d_STKMeshMeta->element_rank(), elements );
    for ( size_t e = 0; e < elements.size(); ++e ) {
        const auto *elem = elements[e];
        auto elem_nodes  = elem->node_relations();
        for ( auto i = elem_nodes.begin(); i != elem_nodes.end(); ++i ) {
            const stk::mesh::Entity *node = i->entity();
            if ( node->owner_rank() == rank ) {
                const unsigned j =
                    AMP::Utilities::findfirst( neighborNodeIDs, (unsigned) node->identifier() );
                AMP_ASSERT( neighborNodeIDs[j] == node->identifier() );
                for ( stk::mesh::PairIterRelation::iterator k = elem_nodes.begin();
                      k != elem_nodes.end();
                      ++k ) {
                    if ( k != i )
                        tmpNeighborNodes[j].insert( k->entity()->key() );
                }
            }
        }
    }
    for ( unsigned n = 0; n < neighborNodeIDs.size(); n++ ) {
        neighborNodes[n] = std::vector<stk::mesh::Entity *>( tmpNeighborNodes[n].size() );
        int j            = 0;
        for ( std::set<stk::mesh::EntityKey>::iterator it = tmpNeighborNodes[n].begin();
              it != tmpNeighborNodes[n].end();
              ++it ) {
            neighborNodes[n][j++] = d_STKMeshBulk->get_entity( *it );
        }
    }
    // Construct the list of elements of type side or edge
    for ( int i = 0; i <= (int) GeomDim; i++ ) {
        GeomType type = (GeomType) i;
        if ( type == GeomType::Vertex || type == GeomDim )
            continue;
        // Get a unique list of all elements of the desired type
        std::set<MeshElement> element_list;
        MeshIterator it = getIterator( GeomDim, 1 );
        for ( size_t j = 0; j < it.size(); j++ ) {
            const MeshElement &e         = *it;
            std::vector<MeshElement> tmp = e.getElements( type );
            for ( size_t k = 0; k < tmp.size(); k++ )
                element_list.insert( tmp[k] );
            ++it;
        }
        // Split the new elements into the local and ghost lists
        size_t N_local = 0;
        size_t N_ghost = 0;
        for ( auto it2 = element_list.begin(); it2 != element_list.end(); ++it2 ) {
            MeshElementID id = it2->globalID();
            if ( id.is_local() )
                N_local++;
            else
                N_ghost++;
        }
        size_t N_global = d_comm.sumReduce( N_local );
        AMP_ASSERT( N_global >= n_global[i] );
        auto local_elements = std::make_shared<std::vector<MeshElement>>( N_local );
        auto ghost_elements = std::make_shared<std::vector<MeshElement>>( N_ghost );
        N_local             = 0;
        N_ghost             = 0;
        for ( auto it2 = element_list.begin(); it2 != element_list.end(); ++it2 ) {
            MeshElementID id = it2->globalID();
            if ( id.is_local() ) {
                local_elements->operator[]( N_local ) = *it2;
                N_local++;
            } else {
                ghost_elements->operator[]( N_ghost ) = *it2;
                N_ghost++;
            }
        }
        AMP::Utilities::quicksort(
            *local_elements ); // Make sure the elments are sorted for searching
        AMP::Utilities::quicksort(
            *ghost_elements ); // Make sure the elments are sorted for searching
        std::pair<GeomType, std::shared_ptr<std::vector<MeshElement>>> local_pair( type,
                                                                                   local_elements );
        std::pair<GeomType, std::shared_ptr<std::vector<MeshElement>>> ghost_pair( type,
                                                                                   ghost_elements );
        d_localElements.insert( local_pair );
        d_ghostElements.insert( ghost_pair );
        n_local[type]  = local_elements->size();
        n_global[type] = d_comm.sumReduce( n_local[i] );
        n_ghost[type]  = ghost_elements->size();
    }
    // Construct the boundary elements for Node and Elem
    d_localSurfaceElements =
        std::vector<std::shared_ptr<std::vector<MeshElement>>>( (int) GeomDim + 1 );
    d_ghostSurfaceElements =
        std::vector<std::shared_ptr<std::vector<MeshElement>>>( (int) GeomDim + 1 );
    std::set<stk::mesh::Entity *> localBoundaryElements;
    std::set<stk::mesh::Entity *> ghostBoundaryElements;
    std::set<stk::mesh::Entity *> localBoundaryNodes;
    std::set<stk::mesh::Entity *> ghostBoundaryNodes;

    stk::mesh::Part &skin_part =
        *d_STKMeshMeta->get_part( "skin_part", "Skin part should be defined in constructor." );
    stk::mesh::skin_mesh( *d_STKMeshBulk, d_STKMeshMeta->element_rank(), &skin_part );

    //    stk::mesh::Selector select_skin = skin_part;
    auto select_skin = stk::mesh::Selector( d_STKMeshMeta->locally_owned_part() );

    std::vector<stk::mesh::Entity *> sides;
    const auto &input_buckets = d_STKMeshBulk->buckets( d_STKMeshMeta->side_rank() );
    stk::mesh::get_selected_entities( select_skin, input_buckets, sides );
    for ( size_t s = 0; s < sides.size(); ++s ) {
        const auto *side = sides[s];
        auto side_elem   = side->relations( d_STKMeshMeta->element_rank() );
        for ( auto i = side_elem.begin(); i != side_elem.end(); ++i ) {
            stk::mesh::Entity *elem = i->entity();
            if ( elem->owner_rank() == rank )
                localBoundaryElements.insert( elem );
            else
                ghostBoundaryElements.insert( elem );
        }
        auto side_node = side->relations( d_STKMeshMeta->node_rank() );
        for ( auto i = side_node.begin(); i != side_node.end(); ++i ) {
            stk::mesh::Entity *node = i->entity();
            if ( node->owner_rank() == rank )
                localBoundaryNodes.insert( node );
            else
                ghostBoundaryNodes.insert( node );
        }
    }

    d_localSurfaceElements[GeomDim] =
        std::make_shared<std::vector<MeshElement>>( localBoundaryElements.size() );
    auto elem_iterator = localBoundaryElements.begin();
    for ( size_t i = 0; i < localBoundaryElements.size(); i++ ) {
        ( *d_localSurfaceElements[GeomDim] )[i] =
            STKMeshElement( PhysicalDim, *elem_iterator, rank, d_meshID, this );
        ++elem_iterator;
    }
    AMP::Utilities::quicksort( *d_localSurfaceElements[GeomDim] );


    d_ghostSurfaceElements[GeomDim] =
        std::make_shared<std::vector<MeshElement>>( ghostBoundaryElements.size() );
    elem_iterator = ghostBoundaryElements.begin();
    for ( size_t i = 0; i < ghostBoundaryElements.size(); i++ ) {
        ( *d_ghostSurfaceElements[GeomDim] )[i] =
            STKMeshElement( PhysicalDim, *elem_iterator, rank, d_meshID, this );
        ++elem_iterator;
    }
    AMP::Utilities::quicksort( *d_ghostSurfaceElements[GeomDim] );


    d_localSurfaceElements[GeomType::Vertex] =
        std::make_shared<std::vector<MeshElement>>( localBoundaryNodes.size() );
    auto node_iterator = localBoundaryNodes.begin();
    for ( size_t i = 0; i < localBoundaryNodes.size(); i++ ) {
        ( *d_localSurfaceElements[GeomType::Vertex] )[i] =
            STKMeshElement( PhysicalDim, *node_iterator, rank, d_meshID, this );
        ++node_iterator;
    }
    AMP::Utilities::quicksort( *d_localSurfaceElements[GeomType::Vertex] );


    d_ghostSurfaceElements[GeomType::Vertex] =
        std::make_shared<std::vector<MeshElement>>( ghostBoundaryNodes.size() );
    node_iterator = ghostBoundaryNodes.begin();
    for ( size_t i = 0; i < ghostBoundaryNodes.size(); i++ ) {
        ( *d_ghostSurfaceElements[GeomType::Vertex] )[i] =
            STKMeshElement( PhysicalDim, *node_iterator, rank, d_meshID, this );
        ++node_iterator;
    }
    AMP::Utilities::quicksort( *d_ghostSurfaceElements[GeomType::Vertex] );
    // Construct the boundary elements for all other types
    // An face or edge is on the boundary if all of its nodes are on the surface
    //    size_t element_surface_global_size =
    //    d_comm.sumReduce(d_localSurfaceElements[GeomDim]->size());
    for ( int type2 = 1; type2 < (int) GeomDim; type2++ ) {
        GeomType type = (GeomType) type2;
        std::set<MeshElement> local, ghost;
        MeshIterator it = getIterator( type, 0 );
        for ( size_t i = 0; i < it.size(); i++ ) {
            auto vertices = it->getElements( GeomType::Vertex );
            AMP_ASSERT( !vertices.empty() );
            bool on_boundary = true;
            for ( size_t j = 0; j < vertices.size(); j++ ) {
                if ( !vertices[j].isOnSurface() )
                    on_boundary = false;
            }
            if ( on_boundary ) {
                if ( it->globalID().is_local() )
                    local.insert( *it );
                else
                    ghost.insert( *it );
            }
            ++it;
        }
        d_localSurfaceElements[type2] =
            std::make_shared<std::vector<MeshElement>>( local.begin(), local.end() );
        d_ghostSurfaceElements[type2] =
            std::make_shared<std::vector<MeshElement>>( ghost.begin(), ghost.end() );
        AMP::Utilities::quicksort( *d_localSurfaceElements[type2] );
        AMP::Utilities::quicksort( *d_ghostSurfaceElements[type2] );
        // size_t local_size = d_localSurfaceElements[type2]->size();
        // size_t global_size = d_comm.sumReduce(local_size);
        // AMP_ASSERT(global_size>=element_surface_global_size);
    }
    // Construct the boundary lists
    std::vector<int> bids;
    const auto &side_sets = d_STKIORegion->get_sidesets();
    for ( auto it = side_sets.begin(); it != side_sets.end(); ++it ) {
        const Ioss::SideSet *side_set = *it;
        const std::string &name       = side_set->name();
        const int bid                 = Ioss::Utils::hash( name );
        bids.push_back( bid );
    }
    Utilities::quicksort( bids );

    for ( int type2 = 0; type2 <= (int) GeomDim; type2++ ) {
        GeomType type         = (GeomType) type2;
        MeshIterator iterator = getIterator( type, 0 );
        for ( size_t i = 0; i < bids.size(); i++ ) {
            int id = (int) bids[i];
            // Count the number of elements on the given boundary
            MeshIterator curElem = iterator.begin();
            MeshIterator endElem = iterator.end();
            int N                = 0;
            while ( curElem != endElem ) {
                if ( curElem->isOnBoundary( id ) )
                    N++;
                ++curElem;
            }
            // Create the boundary list
            auto list = std::make_shared<std::vector<MeshElement>>( N );
            curElem   = iterator.begin();
            endElem   = iterator.end();
            N         = 0;
            while ( curElem != endElem ) {
                if ( curElem->isOnBoundary( id ) ) {
                    list->operator[]( N ) = *curElem;
                    N++;
                }
                ++curElem;
            }
            // Store the list
            auto mapid = std::pair<int, GeomType>( id, type );
            std::pair<std::pair<int, GeomType>, std::shared_ptr<std::vector<MeshElement>>> entry(
                mapid, list );
            d_boundarySets.insert( entry );
        }
    }

    // Get a list of all block ids
    std::set<int> block_ids;
    const auto &blocks = d_STKIORegion->get_element_blocks();
    for ( auto it = blocks.begin(); it != blocks.end(); ++it ) {
        const Ioss::ElementBlock *block = *it;
        const std::string &name         = block->name();
        const int bid                   = Ioss::Utils::hash( name );
        block_ids.insert( bid );
    }
    std::vector<int> send_list( block_ids.begin(), block_ids.end() );
    size_t recv_size = d_comm.sumReduce( send_list.size() );
    std::vector<int> recv_list( recv_size, 0 );
    d_comm.allGather( &send_list[0], send_list.size(), &recv_list[0] );
    for ( size_t i = 0; i < recv_list.size(); i++ )
        block_ids.insert( recv_list[i] );
    d_block_ids = std::vector<int>( block_ids.begin(), block_ids.end() );
}


/********************************************************
 * Function to estimate the mesh size                    *
 ********************************************************/
size_t STKMesh::estimateMeshSize( std::shared_ptr<const MeshParameters> params )
{
    std::shared_ptr<AMP::Database> database = params->getDatabase();
    AMP_ASSERT( database.get() != NULL );
    size_t NumberOfElements = 0;
    if ( database->keyExists( "NumberOfElements" ) ) {
        // User specified the number of elements, this should override everything
        NumberOfElements = (size_t) database->getScalar<int>( "NumberOfElements" );
    } else if ( database->keyExists( "FileName" ) ) {
        // Read an existing mesh
        std::string fname = database->getString( "FileName" );
        if ( fname.rfind( ".exd" ) < fname.size() || fname.rfind( ".e" ) < fname.size() ) {
            AMP_MPI comm( AMP_COMM_SELF );
            Ioss::Init::Initializer init_db;
            const std::string dbtype( "exodusII" );
            Ioss::DatabaseIO *ioss =
                Ioss::IOFactory::create( dbtype, fname, Ioss::READ_MODEL, comm.getCommunicator() );
            if ( ioss == NULL || !ioss->ok() ) {
                std::cerr << "ERROR: Could not open database '" << fname << "' of type '" << dbtype
                          << "'\n";
                AMP_ERROR( "STKMesh::estimateMeshSize: could not open file" );
            }
            Ioss::Region region( ioss, "input_model" );
            Ioss::Property p = region.get_implicit_property( "element_count" );
            NumberOfElements = p.get_int();
            AMP_ASSERT( NumberOfElements > 0 );
        } else {
            AMP_ERROR( "Unkown mesh type, use key NumberOfElements to specify the mesh size" );
        }
    } else if ( database->keyExists( "Generator" ) ) {
        // Generate a new mesh
        std::string generator = database->getString( "Generator" );
        if ( generator.compare( "cube" ) == 0 ) {
            // Generate a cube mesh
            AMP_INSIST( database->keyExists( "size" ),
                        "Variable 'size' must be set in the database" );
            std::vector<int> size = database->getVector<int>( "size" );
            NumberOfElements      = 1;
            for ( size_t i = 0; i < size.size(); i++ )
                NumberOfElements *= size[i];
        } else {
            AMP_ERROR( std::string( "Unknown STKmesh generator: " ) + generator );
        }
    } else {
        AMP_ERROR( "Unable to construct mesh with given parameters" );
    }
    // Adjust the number of elements by a weight if desired
    if ( database->keyExists( "Weight" ) ) {
        double weight    = database->getScalar<double>( "Weight" );
        NumberOfElements = (size_t) ceil( weight * ( (double) NumberOfElements ) );
    }
    return NumberOfElements;
}


/****************************************************************
 * Estimate the maximum number of processors                     *
 ****************************************************************/
size_t STKMesh::maxProcs( std::shared_ptr<const MeshParameters> params )
{
    return estimateMeshSize( params );
}


/********************************************************
 * Return the number of elements                         *
 ********************************************************/
size_t STKMesh::numLocalElements( const GeomType type ) const
{
    if ( n_local[type] == static_cast<size_t>( -1 ) )
        AMP_ERROR( "numLocalElements is not implemented for this type" );
    return n_local[type];
}
size_t STKMesh::numGlobalElements( const GeomType type ) const
{
    if ( n_global[type] == static_cast<size_t>( -1 ) )
        AMP_ERROR( "numLocalElements is not implemented for this type" );
    return n_global[type];
}
size_t STKMesh::numGhostElements( const GeomType type, int gcw ) const
{
    if ( gcw == 0 )
        return 0;
    if ( gcw > 1 )
        AMP_ERROR( "STKmesh only supports a ghost cell width of 1" );
    if ( n_ghost[type] == static_cast<size_t>( -1 ) )
        AMP_ERROR( "numLocalElements is not implemented for this type" );
    return n_ghost[type];
}


/********************************************************
 * Return an iterator over the given geometric type      *
 ********************************************************/
MeshIterator STKMesh::getIterator( const GeomType type, const int gcw ) const
{
    stk::mesh::EntityRank entity_rank = 0xFF;
    switch ( type ) {
    case GeomType::Cell:
        entity_rank = d_STKMeshMeta->element_rank();
        break;
    case GeomType::Vertex:
        entity_rank = d_STKMeshMeta->node_rank();
        break;
    case GeomType::Edge:
        entity_rank = d_STKMeshMeta->edge_rank();
        break;
    case GeomType::Face:
        entity_rank = d_STKMeshMeta->face_rank();
        break;
    default:
        AMP_ERROR( "Unsupported element type" );
    }
    std::vector<stk::mesh::Entity *> entities;
    if ( gcw != 0 && gcw != 1 )
        AMP_ERROR( "Unsupported ghost cell width" );
    if ( gcw == 0 ) {
        const std::vector<stk::mesh::Bucket *> &input_buckets =
            d_STKMeshBulk->buckets( entity_rank );
        stk::mesh::get_selected_entities(
            d_STKMeshMeta->locally_owned_part(), input_buckets, entities );
    } else
        stk::mesh::get_entities( *d_STKMeshBulk, entity_rank, entities );
    MeshIterator test = STKMeshIterator( this, gcw, entities );
    return test;
}


/********************************************************
 * Return an iterator over the given boundary ids        *
 * Note: we have not programmed this for ghosts yet      *
 ********************************************************/
MeshIterator STKMesh::getSurfaceIterator( const GeomType type, const int gcw ) const
{
    AMP_ASSERT( type >= 0 && type <= GeomDim );
    auto local = d_localSurfaceElements[type];
    auto ghost = d_ghostSurfaceElements[type];
    if ( local.get() == NULL || ghost.get() == NULL )
        AMP_ERROR( "Surface iterator over the given geometry type is not supported" );
    if ( gcw == 0 ) {
        return MultiVectorIterator( local, 0 );
    } else if ( gcw == 1 ) {
        std::vector<MeshIterator::shared_ptr> iterators( 2 );
        iterators[0] = std::make_shared<MultiVectorIterator>( local, 0 );
        iterators[1] = std::make_shared<MultiVectorIterator>( ghost, 0 );
        return MultiIterator( iterators, 0 );
    } else {
        AMP_ERROR( "STKmesh has maximum ghost width of 1" );
    }
    return MeshIterator();
}


/********************************************************
 * Return an iterator over the given boundary ids        *
 * Note: we have not programmed this for ghosts yet      *
 ********************************************************/
std::vector<int> STKMesh::getBoundaryIDs() const
{
    std::vector<int> bids;
    const auto &side_sets = d_STKIORegion->get_sidesets();
    for ( auto it = side_sets.begin(); it != side_sets.end(); ++it ) {
        const Ioss::SideSet *side_set = *it;
        const std::string &name       = side_set->name();
        const int bid                 = Ioss::Utils::hash( name );
        bids.push_back( bid );
    }
    AMP::Utilities::quicksort( bids );
    return bids;
}
MeshIterator
STKMesh::getBoundaryIDIterator( const GeomType type, const int id, const int gcw ) const
{
    AMP_INSIST( gcw == 0, "Iterator over ghost boundary elements is not supported yet" );
    auto mapid = std::pair<int, GeomType>( id, type );
    auto list  = std::make_shared<std::vector<MeshElement>>();
    auto it    = d_boundarySets.find( mapid );
    if ( it != d_boundarySets.end() )
        list = it->second;
    return MultiVectorIterator( list, 0 );
}


/********************************************************
 * Return an iterator over the given block ids           *
 ********************************************************/
std::vector<int> STKMesh::getBlockIDs() const { return d_block_ids; }
MeshIterator STKMesh::getBlockIDIterator( const GeomType type, const int id, const int gcw ) const
{
    AMP_ERROR( "getBlockIDIterator is not implemented yet" );
    return MeshIterator();
}


/********************************************************
 * Return pointers to the neighbor nodes give a node id  *
 ********************************************************/
std::vector<stk::mesh::Entity *> STKMesh::getNeighborNodes( MeshElementID id ) const
{
    AMP_INSIST( id.type() == GeomType::Vertex, "This function is for nodes" );
    AMP_INSIST( id.meshID() == d_meshID, "Unknown mesh" );
    AMP_INSIST( id.is_local(), "Only owned nodes can return their neighbor lists" );
    int i = AMP::Utilities::findfirst( neighborNodeIDs, id.local_id() );
    AMP_ASSERT( neighborNodeIDs[i] == id.local_id() );
    return neighborNodes[i];
}


/********************************************************
 * Function to return the element given an ID            *
 ********************************************************/
MeshElement STKMesh::getElement( const MeshElementID &elem_id ) const
{
    MeshID mesh_id = elem_id.meshID();
    AMP_INSIST( mesh_id == d_meshID, "mesh id must match the mesh id of the element" );
    unsigned int rank = d_comm.getRank();
    if ( elem_id.type() == PhysicalDim ) {
        // This is a STKMesh element
        auto *element =
            d_STKMeshBulk->get_entity( d_STKMeshMeta->element_rank(), elem_id.local_id() );
        return STKMeshElement( (int) PhysicalDim, element, rank, mesh_id, this );
    } else if ( elem_id.type() == GeomType::Vertex ) {
        // This is a STKMesh node
        auto *node = d_STKMeshBulk->get_entity( d_STKMeshMeta->node_rank(), elem_id.local_id() );
        return STKMeshElement( (int) PhysicalDim, node, rank, mesh_id, this );
    }
    // All other types are stored in sorted lists
    std::shared_ptr<std::vector<MeshElement>> list;
    if ( (int) elem_id.owner_rank() == d_comm.getRank() )
        list = ( d_localElements.find( elem_id.type() ) )->second;
    else
        list = ( d_ghostElements.find( elem_id.type() ) )->second;
    size_t n = list->size();
    AMP_ASSERT( n > 0 );
    const MeshElement *x = &( list->operator[]( 0 ) ); // Use the pointer for speed
    if ( x[0] == elem_id )
        return x[0];
    size_t lower = 0;
    size_t upper = n - 1;
    size_t index;
    while ( ( upper - lower ) != 1 ) {
        index = ( upper + lower ) / 2;
        if ( x[index] >= elem_id )
            upper = index;
        else
            lower = index;
    }
    index = upper;
    if ( x[index] == elem_id )
        return x[index];
    if ( elem_id.is_local() )
        AMP_ERROR( "Local element not found" );
    return MeshElement();
}


/********************************************************
 * Displace a mesh                                       *
 ********************************************************/
void STKMesh::displaceMesh( std::vector<double> x_in )
{
    // Check x
    AMP_INSIST( (short int) x_in.size() == PhysicalDim,
                "Displacement vector size should match PhysicalDim" );
    std::vector<double> x = x_in;
    d_comm.minReduce( &x[0], x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        AMP_INSIST( fabs( x[i] - x_in[i] ) < 1e-12, "x does not match on all processors" );
    // Move the mesh
    std::vector<stk::mesh::Entity *> nodes;
    stk::mesh::get_entities( *d_STKMeshBulk, d_STKMeshMeta->node_rank(), nodes );
    auto cur                    = nodes.begin();
    auto end                    = nodes.end();
    CartesianField *coordinates = d_STKMeshMeta->get_field<CartesianField>( "coordinates" );
    while ( cur != end ) {
        double *X = (double *) stk::mesh::field_data( *coordinates, **cur );
        AMP_INSIST( X, "Null coordinate field found." );
        for ( size_t i = 0; i < PhysicalDim; i++ )
            X[i] += x[i];
        ++cur;
    }
    // Update the bounding box
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] += x[i];
        d_box[2 * i + 1] += x[i];
        d_box_local[2 * i + 0] += x[i];
        d_box_local[2 * i + 1] += x[i];
    }
}
void STKMesh::displaceMesh( const AMP::LinearAlgebra::Vector::const_shared_ptr x )
{
    // Create the position vector with the necessary ghost nodes
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        shared_from_this(),
        getIterator( AMP::Mesh::GeomType::Vertex, 1 ),
        getIterator( AMP::Mesh::GeomType::Vertex, 0 ),
        PhysicalDim );
    auto nodalVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "tmp_pos" );
    auto displacement  = AMP::LinearAlgebra::createVector( DOFs, nodalVariable, false );
    std::vector<size_t> dofs1( PhysicalDim );
    std::vector<size_t> dofs2( PhysicalDim );
    auto cur  = getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto end  = cur.end();
    auto DOFx = x->getDOFManager();
    std::vector<double> data( PhysicalDim );
    while ( cur != end ) {
        AMP::Mesh::MeshElementID id = cur->globalID();
        DOFx->getDOFs( id, dofs1 );
        DOFs->getDOFs( id, dofs2 );
        x->getValuesByGlobalID( PhysicalDim, &dofs1[0], &data[0] );
        displacement->setValuesByGlobalID( PhysicalDim, &dofs2[0], &data[0] );
        ++cur;
    }
    displacement->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    // Move all nodes (including the ghost nodes)
    const unsigned rank = d_comm.getRank();
    std::vector<stk::mesh::Entity *> nodes;
    stk::mesh::get_entities( *d_STKMeshBulk, d_STKMeshMeta->node_rank(), nodes );
    CartesianField *coordinates = d_STKMeshMeta->get_field<CartesianField>( "coordinates" );
    for ( size_t e = 0; e < nodes.size(); ++e ) {
        stk::mesh::Entity *node = nodes[e];
        // Create the element id
        const unsigned int owner_rank = node->owner_rank();
        const unsigned int local_id   = node->identifier();
        const bool is_local           = owner_rank == rank;
        AMP::Mesh::MeshElementID id(
            is_local, AMP::Mesh::GeomType::Vertex, local_id, owner_rank, d_meshID );
        // Get the position of the point
        DOFs->getDOFs( id, dofs2 );
        displacement->getValuesByGlobalID( PhysicalDim, &dofs2[0], &data[0] );
        // Move the point
        double *x = (double *) stk::mesh::field_data( *coordinates, *node );
        for ( int i = 0; i < PhysicalDim; i++ )
            x[i] += data[i];
    }
    // Compute the bounding box of the mesh
    d_box_local = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box_local[2 * i + 0] = 1e200;
        d_box_local[2 * i + 1] = -1e200;
    }
    nodes.clear();
    const auto &input_buckets = d_STKMeshBulk->buckets( d_STKMeshMeta->node_rank() );
    stk::mesh::get_selected_entities( d_STKMeshMeta->locally_owned_part(), input_buckets, nodes );
    for ( size_t e = 0; e < nodes.size(); ++e ) {
        stk::mesh::Entity *node = nodes[e];
        const double *x         = (double *) stk::mesh::field_data( *coordinates, *node );
        for ( int i = 0; i < PhysicalDim; i++ ) {
            if ( x[i] < d_box_local[2 * i + 0] ) {
                d_box_local[2 * i + 0] = x[i];
            }
            if ( x[i] > d_box_local[2 * i + 1] ) {
                d_box_local[2 * i + 1] = x[i];
            }
        }
    }
    d_box = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] = d_comm.minReduce( d_box_local[2 * i + 0] );
        d_box[2 * i + 1] = d_comm.maxReduce( d_box_local[2 * i + 1] );
    }
}


} // namespace AMP::Mesh
