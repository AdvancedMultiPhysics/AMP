#include <vector>
#include <set>

#include "ampmesh/SubsetMesh.h"
#include "ampmesh/MeshElementVectorIterator.h"
#include "ampmesh/MultiIterator.h"
#include "vectors/Vector.h"

namespace AMP {
namespace Mesh {


/********************************************************
* Constructors                                          *
********************************************************/
SubsetMesh::SubsetMesh( boost::shared_ptr<const Mesh> mesh, const AMP::Mesh::MeshIterator iterator_in )
{
    this->d_parent_mesh = mesh;
    this->d_comm = mesh->getComm();
    this->setMeshID();
    this->PhysicalDim = mesh->getDim();
    this->d_name = mesh->getName() + "_subset";
    // Check the iterator
    GeomType type = null;
    AMP_ASSERT(iterator_in.size()>0);
    MeshIterator iterator = iterator_in.begin();
    for (size_t i=0; i<iterator.size(); i++) {
        if ( type==null )
            type = iterator->elementType();
        if ( type!= iterator->elementType() )
            AMP_ERROR("Subset mesh requires all of the elements to be the same type");
        ++iterator;
    }
    int type2 = d_comm.maxReduce((int) type);
    if ( type!=null && type2!=(int)type )
        AMP_ERROR("Subset mesh requires all of the elements to be the same type");
    this->GeomDim = (GeomType) type2;
    std::vector<MeshID> ids = mesh->getBaseMeshIDs();
    iterator = iterator_in.begin();
    for (size_t i=0; i<iterator.size(); i++) {
        MeshID id = iterator->globalID().meshID();
        bool found = false;
        for (size_t j=0; j<ids.size(); j++) {
            if ( id == ids[j] )
                found = true;
        }
        if ( !found )
            AMP_ERROR("Iterator contains elements not found in parent meshes");
        ++iterator;
    }
    // Create a list of all elements of the desired type
    d_elements = std::vector<std::vector<boost::shared_ptr<std::vector<MeshElement> > > >((int)GeomDim+1);
    int gcw = 0;
    while ( 1 ) {
        MeshIterator iterator1 = Mesh::getIterator( Intersection, iterator_in, mesh->getIterator(GeomDim,gcw) );
        MeshIterator iterator2 = iterator1.begin();
        if ( gcw>0 ) 
            iterator2 = Mesh::getIterator( Complement, iterator1, mesh->getIterator(GeomDim,0) );
        d_elements[GeomDim].push_back( boost::shared_ptr<std::vector<MeshElement> >( new std::vector<MeshElement>(iterator2.size()) ) );
        for (size_t i=0; i<iterator2.size(); i++) {
            d_elements[GeomDim][gcw]->operator[](i) = *iterator2;
            ++iterator2;
        }
        AMP::Utilities::quicksort( *(d_elements[GeomDim][gcw]) );
        if ( iterator1.size() == iterator_in.size() )
            break;
        gcw++;
    }
    if ( d_elements[GeomDim].size() == 1 )
        d_elements[GeomDim].push_back( boost::shared_ptr<std::vector<MeshElement> >( new std::vector<MeshElement> ) );
    d_max_gcw = d_elements[GeomDim].size()-1;
    // Create a list of all elements that compose the elements of GeomType
    for (int t=0; t<(int)GeomDim; t++) {
        d_elements[t] = std::vector<boost::shared_ptr<std::vector<MeshElement> > >(d_max_gcw+1);
        for (int gcw=0; gcw<=d_max_gcw; gcw++) {
            std::set<MeshElement> list;
            iterator = this->getIterator(GeomDim,gcw);
            for (size_t it=0; it<iterator.size(); it++) {
                std::vector<MeshElement> elements = iterator->getElements((GeomType)t);
                for (size_t i=0; i<elements.size(); i++) {
                    if ( gcw==0 ) {
                        if ( elements[i].globalID().is_local() )
                            list.insert(elements[i]);
                    } else {
                        bool found = false;
                        for (int j=0; j<gcw; j++) {
                            size_t index = AMP::Utilities::findfirst( *(d_elements[t][j]), elements[i] );
                            if ( index==d_elements[t][j]->size() ) { index--; }
                            if ( d_elements[t][j]->operator[](index) == elements[i] )
                                found = true;
                        }
                        if ( !found )
                            list.insert(elements[i]);
                    }
                }
                ++iterator;
            }
            d_elements[t][gcw] = boost::shared_ptr<std::vector<MeshElement> >( 
                new std::vector<MeshElement>(list.begin(),list.end()) );
        }
    }
    N_global = std::vector<size_t>((int)GeomDim+1);
    for (int i=0; i<=(int)GeomDim; i++)
        N_global[i] = d_elements[i][0]->size();
    d_comm.sumReduce( &N_global[0], N_global.size() );
    for (int i=0; i<=(int)GeomDim; i++)
        AMP_ASSERT(N_global[i]>0);
    // Create the bounding box
    d_box = std::vector<double>(2*PhysicalDim);
    for (int j=0; j<PhysicalDim; j++) {
        d_box[2*j+0] = 1e100;
        d_box[2*j+1] = -1e100;
    }
    iterator = getIterator(Vertex,0);
    for (size_t i=0; i<iterator.size(); i++) {
        std::vector<double> coord = iterator->coord();
        for (int j=0; j<PhysicalDim; j++) {
            if ( coord[j] < d_box[2*j+0] )
                d_box[2*j+0] = coord[j];
            if ( coord[j] > d_box[2*j+1] )
                d_box[2*j+1] = coord[j];
        }
        ++iterator;
    }
    for (int j=0; j<PhysicalDim; j++) {
        d_box[2*j+0] = d_comm.minReduce(d_box[2*j+0]);
        d_box[2*j+1] = d_comm.maxReduce(d_box[2*j+1]);
    }
}


/********************************************************
* De-constructor                                        *
********************************************************/
SubsetMesh::~SubsetMesh()
{
}


/********************************************************
* Function to return the meshID composing the mesh      *
********************************************************/
std::vector<MeshID> SubsetMesh::getAllMeshIDs() const
{
    return std::vector<MeshID>(1,d_meshID);
}
std::vector<MeshID> SubsetMesh::getBaseMeshIDs() const
{
    return std::vector<MeshID>(1,d_meshID);
}


/********************************************************
* Function to return the mesh with the given ID         *
********************************************************/
boost::shared_ptr<Mesh>  SubsetMesh::Subset( MeshID meshID ) const
{
    if ( d_meshID==meshID ) 
        return boost::const_pointer_cast<Mesh>( shared_from_this() );
    else
        return boost::shared_ptr<Mesh>();
}


/********************************************************
* Function to return the mesh with the given name       *
********************************************************/
boost::shared_ptr<Mesh>  SubsetMesh::Subset( std::string name ) const {
    if ( d_name==name ) 
        return boost::const_pointer_cast<Mesh>( shared_from_this() );
    else
        return boost::shared_ptr<Mesh>();
}


/********************************************************
* Subset mesh                                           *
********************************************************/
boost::shared_ptr<Mesh> SubsetMesh::Subset( MeshIterator::shared_ptr & ) const
{
    AMP_ERROR("Subset is not implimented for the base class");
    return boost::shared_ptr<Mesh>();
}
boost::shared_ptr<Mesh> SubsetMesh::Subset( Mesh & ) const
{
    AMP_ERROR("Subset is not implimented for the base class");
    return boost::shared_ptr<Mesh>();
}


/********************************************************
* Mesh iterators                                        *
********************************************************/
MeshIterator SubsetMesh::getIterator( const GeomType type, const int gcw ) const
{
    if ( gcw == 0 )
        return MultiVectorIterator( d_elements[type][0], 0 );
    if ( gcw >= (int) d_elements[type].size() ) 
        AMP_ERROR("Maximum ghost width exceeded");
    std::vector<boost::shared_ptr<MeshIterator> > iterators(gcw+1);
    for (int i=0; i<=gcw; i++)
        iterators[i] = boost::shared_ptr<MeshIterator>( new MultiVectorIterator( d_elements[type][i], 0 ) );
    return MultiIterator( iterators, 0 );
}
MeshIterator SubsetMesh::getSurfaceIterator( const GeomType, const int gcw ) const
{
    AMP_ERROR("getSurfaceIterator is not implimented for the subset mesh yet");
    return MeshIterator();
}
std::vector<int> SubsetMesh::getIDSets ( ) const
{
    AMP_ERROR("getIDSets is not implimented for subset mesh yet");
    return std::vector<int>();
}
MeshIterator SubsetMesh::getIDsetIterator ( const GeomType, const int id, const int gcw ) const
{
    AMP_ERROR("getIDsetIterator is not implimented for subset mesh yet");
    return MeshIterator();
}


/********************************************************
* Other functions                                       *
********************************************************/
size_t SubsetMesh::numLocalElements( const GeomType type ) const
{
    return d_elements[type][0]->size();
}
size_t SubsetMesh::numGlobalElements( const GeomType type ) const
{
    return N_global[type];
}
size_t SubsetMesh::numGhostElements( const GeomType type, int gcw ) const
{
    AMP_ASSERT(type<=GeomDim);
    if ( gcw == 0 )
        return 0;
    if ( gcw >= (int) d_elements[type].size() )
        AMP_ERROR("Maximum ghost width exceeded");
    return d_elements[type][gcw]->size();
}
void SubsetMesh::displaceMesh( std::vector<double> x )
{
    AMP_ERROR("displaceMesh by a constant value does not work for subset mesh");
}
#ifdef USE_AMP_VECTORS
void SubsetMesh::displaceMesh( const AMP::LinearAlgebra::Vector::const_shared_ptr x )
{
    AMP_ERROR("displaceMesh is not implimented for subset mesh");
}
#endif


} // Mesh namespace
} // AMP namespace

