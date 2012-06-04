#include "ampmesh/structured/structuredMeshElement.h"
#include "ampmesh/structured/BoxMesh.h"
#include "ampmesh/MeshElementVectorIterator.h"
#include "utils/Utilities.h"
#ifdef USE_AMP_VECTORS
    #include "vectors/Vector.h"
    #include "vectors/Variable.h"
    #include "vectors/VectorBuilder.h"
#endif
#ifdef USE_AMP_DISCRETIZATION
    #include "discretization/DOF_Manager.h"
    #include "discretization/simpleDOF_Manager.h"
#endif

namespace AMP {
namespace Mesh {


/****************************************************************
* Constructor                                                   *
****************************************************************/
BoxMesh::BoxMesh( const MeshParameters::shared_ptr &params_in ):
    Mesh(params_in)
{
    for (int d=0; d<3; d++) {
        d_size[d] = 0;
        d_isPeriodic[d] = false;
    }
    // Check for valid inputs
    AMP_INSIST(params.get(),"Params must not be null");
    AMP_INSIST(d_comm!=AMP_MPI(AMP_COMM_NULL),"Communicator must be set");
    AMP_INSIST(d_db.get(),"Database must exist");
    // Get mandatory fields from the database
    AMP_INSIST(d_db->keyExists("dim"),"Field 'dim' must exist in database'");
    AMP_INSIST(d_db->keyExists("Generator"),"Field 'Generator' must exist in database'");
    AMP_INSIST(d_db->keyExists("Size"),"Field 'Size' must exist in database'");
    AMP_INSIST(d_db->keyExists("Range"),"Field 'Range' must exist in database'");
    PhysicalDim = d_db->getInteger("dim");
    GeomDim = (GeomType) PhysicalDim;
    std::string generator = d_db->getString("Generator");
    std::vector<int> size = d_db->getIntegerArray("Size");
    std::vector<double> range = d_db->getDoubleArray("Range");
    AMP_INSIST(size.size()==PhysicalDim,"Size of field 'Size' must match dim");
    for (int d=0; d<PhysicalDim; d++)
        AMP_INSIST(size[d]>0,"All dimensions must have a size > 0");
    // Create the logical mesh
    AMP_ASSERT(PhysicalDim<=3);
    for (int d=0; d<PhysicalDim; d++)
        d_size[d] = size[d];
    if ( d_comm.getSize()==1 ) {
        // We are dealing with a serial mesh
        d_localSize[0] = std::vector<int>(1,d_size[0]);
        d_localSize[1] = std::vector<int>(1,d_size[1]);
        d_localSize[2] = std::vector<int>(1,d_size[2]);  
    } else {
        // We are dealing with a parallel mesh
        // First, get the prime factors for number of processors and divide the dimensions
        std::vector<int> factors = AMP::Utilities::factor(d_comm.getSize());
        int block_size[3];
        block_size[0] = d_size[0];
        block_size[1] = d_size[1];
        block_size[2] = d_size[2];
        while ( factors.size() > 0 ) {
            int d = -1;
            if ( block_size[0]>=block_size[1] && block_size[0]>=block_size[2] ) {
                d = 0;
            } else if ( block_size[1]>=block_size[0] && block_size[1]>=block_size[2] ) {
                d = 1;
            } else if ( block_size[2]>=block_size[0] && block_size[2]>=block_size[1] ) {
                d = 2;
            } else {
                AMP_ERROR("Internal error");
            }
            block_size[d] /= factors[factors.size()-1];
            factors.resize(factors.size()-1);
        }
        for (int d=0; d<PhysicalDim; d++) {
            int N_blocks = (d_size[d]+block_size[d]-1)/block_size[d];
            d_localSize[d] = std::vector<int>(N_blocks,block_size[d]);
            d_localSize[d][N_blocks-1] = d_size[d] - (N_blocks-1)*block_size[d];
        }
    }
    // Initialize the logical mesh
    d_max_gcw = 3;
    initialize();
    // Create the appropriate mesh coordinates
    if ( generator.compare("cube")==0 ) {
        AMP_INSIST(range.size()==2*PhysicalDim,"Range must be 2*dim for cube generator");
        fillCartesianNodes( PhysicalDim, &d_size[0], &range[0], d_index, d_coord );
    } else { 
        AMP_ERROR("Unknown generator");
    }
    // Fill in the final info for the mesh
    AMP_INSIST(d_db->keyExists("MeshName"),"MeshName must exist in input database");
    d_name = d_db->getString("MeshName");
    d_box_local = std::vector<double>(2*PhysicalDim);
    for (int d=0; d<PhysicalDim; d++) {
        d_box_local[2*d+0] = 1e100;
        d_box_local[2*d+1] = -1e100;
        for (size_t i=0; i<d_coord[d].size(); i++) {
            if ( d_coord[d][i]<d_box_local[2*d+0])
                d_box_local[2*d+0] = d_coord[d][i];
            if ( d_coord[d][i]>d_box_local[2*d+1])
                d_box_local[2*d+1] = d_coord[d][i];
        }
    }
    d_box = std::vector<double>(PhysicalDim*2);
    for (int i=0; i<PhysicalDim; i++) {
        d_box[2*i+0] = d_comm.minReduce( d_box_local[2*i+0] );
        d_box[2*i+1] = d_comm.maxReduce( d_box_local[2*i+1] );
    } 
    // Displace the mesh
    std::vector<double> displacement(PhysicalDim,0.0);
    if ( d_db->keyExists("x_offset") )
        displacement[0] = d_db->getDouble("x_offset");
    if ( d_db->keyExists("y_offset") )
        displacement[1] = d_db->getDouble("y_offset");
    if ( d_db->keyExists("z_offset") )
        displacement[2] = d_db->getDouble("z_offset");
    bool test = false;
    for (size_t i=0; i<displacement.size(); i++) {
        if ( displacement[i] != 0.0 )
           test = true;
    }        
    if ( test )
        displaceMesh(displacement);
}


/****************************************************************
* Initialize the mesh                                           *
****************************************************************/
void BoxMesh::initialize()
{
    // Compute the element indicies for all local and ghost elements
    for (int d=0; d<=PhysicalDim; d++) 
        d_elements[d] = std::vector<ElementIndexList>(d_max_gcw+1);
    std::vector<int> range = getLocalBlock(d_comm.getRank());
    // First get the list of owned elements of each type
    size_t N_localElements = 1;
    for (int d=0; d<PhysicalDim; d++) 
        N_localElements *= range[2*d+1] - range[2*d+0];
    for (int d=0; d<=PhysicalDim; d++) {
        d_elements[d][0].reset( new std::vector<MeshElementIndex>() );
        if ( d==0 || d==PhysicalDim ) {
            d_elements[d][0]->reserve( N_localElements );
        } else if ( d==1 ) {
            d_elements[d][0]->reserve( 6*N_localElements );     
        } else if ( d==2 ) {
            d_elements[d][0]->reserve( 3*N_localElements );
        } else {
            AMP_ERROR("Internal error");
        }
    }
    for (int d=0; d<=PhysicalDim; d++) {    // Loop through the geometric entities
        int numSides = 0;
        if ( d==0 || d==PhysicalDim ) {
            numSides = 1;
        } else if ( PhysicalDim==2 ) {
            numSides = 2;
        } else if ( PhysicalDim==3 ) {
            numSides = 3;
        }
        AMP_ASSERT(numSides>0);
        for (int s=0; s<numSides; s++) {
            // Extend the range for the last box to include the physical boundary
            std::vector<int> range2 = range;
            if ( d==PhysicalDim ) {
                // We are dealing with an element, do nothing
            } else if ( d==0 ) {
                // We are dealing with a node, we may need to expand all dimensions
                for (int d2=0; d2<PhysicalDim; d2++) {
                    if ( range2[2*d2+1]==d_size[d2] && !d_isPeriodic[d2] && d!=PhysicalDim )
                        range2[2*d2+1]++;
                }
            } else {
                if ( d==1 && PhysicalDim==3 ) {
                    // We are dealing with a edge in 3d
                    for (int d2=0; d2<PhysicalDim; d2++) {
                        if ( range2[2*d2+1]==d_size[d2] && !d_isPeriodic[d2] && d!=PhysicalDim && s!=d2 )
                            range2[2*d2+1]++;
                    }
                } else if ( PhysicalDim-d==1 ) { 
                    // We are dealing with a face or edge in 3d or 2d respectivly
                    int d2 = s;
                    if ( range2[2*d2+1]==d_size[d2] && !d_isPeriodic[d2] && d!=PhysicalDim )
                        range2[2*d2+1]++;
                } else {
                    AMP_ERROR("Internal error");
                }
            }
            // Create the elements
            if ( PhysicalDim==3 ) {
                for (int k=range2[4]; k<range2[5]; k++) {
                    for (int j=range2[2]; j<range2[3]; j++) {
                        for (int i=range2[0]; i<range2[1]; i++) {
                            MeshElementIndex index;
                            index.type = (GeomType) d;
                            index.index[0] = i;
                            index.index[1] = j;
                            index.index[2] = k;
                            index.side = s;
                            d_elements[d][0]->push_back( index );
                        }
                    }
                }
            }
        }
    }
    if ( PhysicalDim==1 ) {
        if ( d_isPeriodic[0] )
            AMP_ASSERT( (int)d_elements[0][0]->size() == d_size[0] );
        else
            AMP_ASSERT( (int)d_elements[0][0]->size() == (d_size[0]+1) );
        AMP_ASSERT( (int)d_elements[1][0]->size() == d_size[0] );
    } else if ( PhysicalDim==2 ) {
        size_t N_faces_global = d_size[0]*d_size[1];
        size_t N_edges_global = 2*d_size[0]*d_size[1];
        if ( !d_isPeriodic[0] )
            N_edges_global += d_size[1];
        if ( !d_isPeriodic[1] )
            N_edges_global += d_size[0];
        size_t N_nodes_global = 1;
        for (int i=0; i<2; i++) {
            if ( d_isPeriodic[i] )
                N_nodes_global *= d_size[i];
            else
                N_nodes_global *= d_size[i]+1;
        }
        AMP_ASSERT( d_elements[0][0]->size() == N_nodes_global );
        AMP_ASSERT( d_elements[1][0]->size() == N_edges_global );
        AMP_ASSERT( d_elements[2][0]->size() == N_faces_global );
    } else if ( PhysicalDim==3 ) {
        size_t N_elements_global = d_size[0]*d_size[1]*d_size[2];
        size_t N_faces_global = 3*d_size[0]*d_size[1]*d_size[2];
        size_t N_edges_global = 3*d_size[0]*d_size[1]*d_size[2];
        if ( !d_isPeriodic[0] ) {
            N_faces_global += d_size[1]*d_size[2];
            N_edges_global += 2*d_size[1]*d_size[2];
        }
        if ( !d_isPeriodic[1] ) {
            N_faces_global += d_size[0]*d_size[2];
            N_edges_global += 2*d_size[0]*d_size[2];
        }
        if ( !d_isPeriodic[2] ) {
            N_faces_global += d_size[0]*d_size[1];
            N_edges_global += 2*d_size[0]*d_size[1];
        }
        if ( !d_isPeriodic[0] && !d_isPeriodic[1] )
            N_edges_global += d_size[2];
        if ( !d_isPeriodic[0] && !d_isPeriodic[2] )
            N_edges_global += d_size[1];
        if ( !d_isPeriodic[1] && !d_isPeriodic[2] )
            N_edges_global += d_size[0];
        size_t N_nodes_global = 1;
        for (int i=0; i<3; i++) {
            if ( d_isPeriodic[i] )
                N_nodes_global *= d_size[i];
            else
                N_nodes_global *= d_size[i]+1;
        }
        AMP_ASSERT( d_elements[0][0]->size() == N_nodes_global );
        AMP_ASSERT( d_elements[1][0]->size() == N_edges_global );
        AMP_ASSERT( d_elements[2][0]->size() == N_faces_global );
        AMP_ASSERT( d_elements[3][0]->size() == N_elements_global );
    } else {
        AMP_ERROR("Not programmed for this dimension yet");
    }
    // Create the ghost elements
    for (int gcw=1; gcw<=d_max_gcw; gcw++) {
        for (int d=0; d<=PhysicalDim; d++) {    // Loop through the geometric entities
            d_elements[d][gcw].reset( new std::vector<MeshElementIndex>() );
        }
    }
    // Compute the number of local, global and ghost elements
    for (int d=0; d<=PhysicalDim; d++) 
        N_global[d] = d_comm.sumReduce(d_elements[d][0]->size());
    // Create the nodes
    MeshIterator nodeIterator = getIterator(Vertex,d_max_gcw);
    for (int d=0; d<PhysicalDim; d++)
        d_coord[d] = std::vector<double>(nodeIterator.size(),0.0);
    d_index = std::vector<MeshElementIndex>(nodeIterator.size());
    for (size_t i=0; i<nodeIterator.size(); i++) {
        MeshElement* elem_ptr = nodeIterator->getRawElement();
        structuredMeshElement *element = dynamic_cast<structuredMeshElement *>( elem_ptr );
        AMP_ASSERT(element!=NULL);
        MeshElementIndex index = element->d_index;
        AMP_ASSERT(index.type==0);
        d_index[i] = index;
        ++nodeIterator;
    }
    AMP::Utilities::quicksort(d_index);
    double range2[6] = {0.0,1.0,0.0,1.0,0.0};
    fillCartesianNodes( PhysicalDim, &d_size[0], range2, d_index, d_coord );
    // Create the boundary info
    for (int d=0; d<=PhysicalDim; d++) {
        d_surface_list[d] = std::vector<ElementIndexList>(d_max_gcw+1);
        for (int gcw=0; gcw<=d_max_gcw; gcw++)
            d_surface_list[d][gcw] = boost::shared_ptr<std::vector<MeshElementIndex> >(
                new std::vector<MeshElementIndex>() );
    }
    d_ids = std::vector<int>();
    d_id_list = std::map<std::pair<int,GeomType>,std::vector<ElementIndexList> >();
}


/****************************************************************
* De-constructor                                                *
****************************************************************/
BoxMesh::~BoxMesh()
{
}


/****************************************************************
* Estimate the mesh size                                        *
****************************************************************/
size_t BoxMesh::estimateMeshSize( const MeshParameters::shared_ptr &params )
{
    // Check for valid inputs
    AMP_INSIST(params.get(),"Params must not be null");
    boost::shared_ptr<AMP::Database> db = params->getDatabase( );
    AMP_INSIST(db.get(),"Database must exist");
    // Get mandatory fields from the database
    AMP_INSIST(db->keyExists("dim"),"Field 'Generator' must exist in database'");
    AMP_INSIST(db->keyExists("dim"),"Field 'dim' must exist in database'");
    AMP_INSIST(db->keyExists("Size"),"Field 'Size' must exist in database'");
    int dim = db->getInteger("dim");
    std::string generator = db->getString("Generator");
    std::vector<int> size = db->getIntegerArray("Size");
    AMP_INSIST((int)size.size()==dim,"Size of field 'Size' must match dim");
    for (int d=0; d<dim; d++)
        AMP_INSIST(size[d]>0,"All dimensions must have a size > 0");
    size_t N_elements = 1;
    for (int d=0; d<dim; d++)
        N_elements *= size[d];
    return N_elements;
}


/****************************************************************
* Function to return the element given an ID                    *
****************************************************************/
MeshElement BoxMesh::getElement ( const MeshElementID &elem_id ) const
{
    std::vector<int> range = getLocalBlock( elem_id.owner_rank() );
    AMP_ASSERT(PhysicalDim<=3);
    // Increase the index range for the boxes on the boundary for all elements except the current dimension
    if ( elem_id.type() != PhysicalDim ) {
        for (int d=0; d<PhysicalDim; d++) {
            if ( range[2*d+1]==d_size[d] && !d_isPeriodic[d] )
                range[2*d+1]++;
        }
    }
    // Get the 3-index from the local id
    size_t myBoxSize[3]={1,1,1};
    for (int d=0; d<PhysicalDim; d++)
        myBoxSize[d] = range[2*d+1] - range[2*d+0];
    MeshElementIndex index;
    index.type = elem_id.type();
    size_t local_id = elem_id.local_id();
    index.index[0] = (int) local_id%myBoxSize[0];
    index.index[1] = (int) (local_id/myBoxSize[0])%myBoxSize[1];
    index.index[2] = (int) (local_id/(myBoxSize[0]*myBoxSize[1]))%myBoxSize[2];
    index.side = (unsigned char) (local_id/(myBoxSize[0]*myBoxSize[1]*myBoxSize[2]));
    return structuredMeshElement( index, this );
}


/****************************************************************
* Functions to return the number of elements                    *
****************************************************************/
size_t BoxMesh::numLocalElements( const GeomType type ) const
{
    return d_elements[(int)type][0]->size();
}
size_t BoxMesh::numGlobalElements( const GeomType type ) const
{
    return N_global[(int)type];
}
size_t BoxMesh::numGhostElements( const GeomType type, int gcw ) const
{
    size_t N_ghost = 0;
    for (int i=1; i<=gcw; i++)
        N_ghost += d_elements[(int)type][gcw]->size();
    return N_ghost;
}


/****************************************************************
* Function to get an iterator                                   *
****************************************************************/
MeshIterator BoxMesh::getIterator( const GeomType type, const int gcw ) const
{
    // Construct a list of elements for the local patch of the given type
    AMP_ASSERT(type<=3);
    AMP_ASSERT(gcw<(int)d_elements[type].size());
    boost::shared_ptr<std::vector<MeshElement> >  elements( new std::vector<MeshElement>() );
    elements->reserve( numLocalElements(type)+numGhostElements(type,gcw) );
    for (int j=0; j<=gcw; j++) {
        for (size_t k=0; k<d_elements[type][j]->size(); k++) {
            BoxMesh::MeshElementIndex index = d_elements[type][j]->operator[](k);
            MeshElement elem = structuredMeshElement( index, this );
            elements->push_back( elem );
        }
    }
    return MultiVectorIterator( elements, 0 );
}


/****************************************************************
* Function to get an iterator over the surface                  *
****************************************************************/
MeshIterator BoxMesh::getSurfaceIterator( const GeomType type, const int gcw ) const
{
    size_t N_elements = 1;
    for (int i=0; i<=gcw; i++)
        N_elements += d_surface_list[type][gcw]->size();
    boost::shared_ptr<std::vector<MeshElement> >  elements( new std::vector<MeshElement>() );
    elements->reserve( N_elements );
    for (int i=0; i<=gcw; i++) {
        for (int j=0; j<d_surface_list[type][gcw]->size(); j++) {
            BoxMesh::MeshElementIndex index = d_surface_list[type][gcw]->operator[](j);
            MeshElement elem = structuredMeshElement( index, this );
            elements->push_back( elem );
        }
    }
    return MultiVectorIterator( elements, 0 );
}


/****************************************************************
* Functions that aren't implimented yet                         *
****************************************************************/
std::vector<int> BoxMesh::getBoundaryIDs ( ) const
{
    return d_ids;
}
MeshIterator BoxMesh::getBoundaryIDIterator ( const GeomType type, const int id, const int gcw) const
{
    std::map<std::pair<int,GeomType>,std::vector<ElementIndexList> >::const_iterator it = d_id_list.find( std::pair<int,GeomType>(id,type) );
    AMP_INSIST(it!=d_id_list.end(),"Boundary elements of the given type and id were not found");
    boost::shared_ptr<std::vector<MeshElement> >  elements( new std::vector<MeshElement>() );
    size_t N_elements = 0;
    for (int i=0; i<=gcw; i++)
        N_elements += it->second[i]->size();
    elements->reserve( N_elements );
    for (int j=0; j<=gcw; j++) {
        for (size_t k=0; k<d_elements[type][j]->size(); k++) {
            BoxMesh::MeshElementIndex index = it->second[j]->operator[](k);
            MeshElement elem = structuredMeshElement( index, this );
            elements->push_back( elem );
        }
    }
    return MultiVectorIterator( elements, 0 );
}
std::vector<int> BoxMesh::getBlockIDs ( ) const
{
    return std::vector<int>(1,0);
}
MeshIterator BoxMesh::getBlockIDIterator ( const GeomType type, const int id, const int gcw ) const
{
    if ( id==0 ) 
        return getIterator( type, gcw );
    return MeshIterator();
}
void BoxMesh::displaceMesh( std::vector<double> x )
{
    AMP_ASSERT(x.size()==PhysicalDim);
    for (int i=0; i<PhysicalDim; i++) {
        for (size_t j=0; j<d_coord[i].size(); j++)
            d_coord[i][j] += x[i];
        d_box[2*i+0] += x[i];
        d_box[2*i+1] += x[i];
        d_box_local[2*i+0] += x[i];
        d_box_local[2*i+1] += x[i];
    }
}
#ifdef USE_AMP_VECTORS
void BoxMesh::displaceMesh( const AMP::LinearAlgebra::Vector::const_shared_ptr x )
{
    std::vector<box_info> d_localBoxList;
}
#endif


/****************************************************************
* Helper function to return the indices of the local block      *
* owned by the given processor                                  *
****************************************************************/
std::vector<int> BoxMesh::getLocalBlock(unsigned int rank) const
{
    size_t num_blocks = 1;
    for (int d=0; d<PhysicalDim; d++)
        num_blocks *= d_localSize[d].size();
    AMP_ASSERT((int)rank<num_blocks);
    std::vector<int> range(2*PhysicalDim);
    size_t tmp = 1;
    for (int d=0; d<PhysicalDim; d++) {
        int i = (int) ((rank/tmp)%d_localSize[d].size());
        tmp *= d_localSize[d].size();
        size_t i0 = 0;
        for (int j=0; j<i; j++)
            i0 += d_localSize[d][j];
        range[2*d+0] = (int) i0;
        range[2*d+1] = (int) (i0+d_localSize[d][i]);
    }
    return range;
}


/****************************************************************
* Helper function to return the indices and rank of the owning  *
* block for a given MeshElementIndex                            *
****************************************************************/
std::vector<int> BoxMesh::getOwnerBlock(const MeshElementIndex index, unsigned int &rank) const
{
    std::vector<int> range(2*PhysicalDim);
    int myBoxIndex[3]={1,1,1};
    for (int d=0; d<PhysicalDim; d++) {
        // Check if the element lies on the physical bounadry
        if ( index.index[d]==d_size[d] ) {
            AMP_ASSERT(index.type<PhysicalDim);
            myBoxIndex[d] = (int) d_localSize[d].size()-1;
            range[2*d+0] = d_size[d]-d_localSize[d][myBoxIndex[d]];
            range[2*d+1] = d_size[d];
            continue;
        }
        // Find the owning box
        range[2*d+0] = 0;
        range[2*d+1] = 0;
        int i=0;
        while ( index.index[d] >= range[2*d+1] ) {
            range[2*d+0] += range[2*d+1];
            range[2*d+1] += d_localSize[d][i];
            myBoxIndex[d] = i;
            i++;
        }
    }
    // Increase the index range for the boxes on the boundary for all elements except the current dimension
    if ( index.type != PhysicalDim ) {
        for (int d=0; d<PhysicalDim; d++) {
            if ( range[2*d+1]==d_size[d] && !d_isPeriodic[d] )
                range[2*d+1]++;
        }
    }
    rank = (unsigned int) ( myBoxIndex[0] + myBoxIndex[1]*d_localSize[0].size() + 
        myBoxIndex[2]*d_localSize[0].size()*d_localSize[1].size() );
    return range;
}


/****************************************************************
* Helper function to fill the cartesian coordinates             *
****************************************************************/
void BoxMesh::fillCartesianNodes(int dim, const int* globalSize, const double *range, 
    const std::vector<MeshElementIndex> &index, std::vector<double> *coord)
{
    AMP_ASSERT(index.size()==coord[0].size());
    for (size_t i=0; i<index.size(); i++) {
        AMP_ASSERT(index[i].type==0);
        for (int d=0; d<dim; d++)
            coord[d][i] = range[2*d+0] + (range[2*d+1]-range[2*d+0])*((double)index[i].index[d])/((double)globalSize[d]);
    }
}


} // Mesh namespace
} // AMP namespace

