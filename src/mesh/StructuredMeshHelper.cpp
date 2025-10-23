#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/structuredMeshIterator.h"

#include <tuple>
#include <numeric>

namespace AMP::Mesh {


/************************************************************
 * Function to return the coordinates of a cube mesh         *
 ************************************************************/
void StructuredMeshHelper::getXYZCoordinates( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                              std::vector<double> &x_out,
                                              std::vector<double> &y_out,
                                              std::vector<double> &z_out,
                                              bool check )
{
    AMP_ASSERT( mesh );
    std::set<double> x, y, z;
    auto it = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    for ( size_t i = 0; i < it.size(); i++ ) {
        auto coord = it->coord();
        AMP_ASSERT( coord.size() == 3 );
        x.insert( coord[0] );
        y.insert( coord[1] );
        z.insert( coord[2] );
        ++it;
    }
    mesh->getComm().setGather( x );
    mesh->getComm().setGather( y );
    mesh->getComm().setGather( z );
    x_out.resize( 0 );
    y_out.resize( 0 );
    z_out.resize( 0 );
    x_out.reserve( x.size() );
    y_out.reserve( y.size() );
    z_out.reserve( z.size() );
    x_out.push_back( *( x.begin() ) );
    y_out.push_back( *( y.begin() ) );
    z_out.push_back( *( z.begin() ) );
    for ( const auto &elem : x ) {
        if ( !Utilities::approx_equal( x_out.back(), elem, 1e-12 ) )
            x_out.push_back( elem );
    }
    for ( const auto &elem : y ) {
        if ( !Utilities::approx_equal( y_out.back(), elem, 1e-12 ) )
            y_out.push_back( elem );
    }
    for ( const auto &elem : z ) {
        if ( !Utilities::approx_equal( z_out.back(), elem, 1e-12 ) )
            z_out.push_back( elem );
    }
    size_t Nx = x.size() - 1;
    size_t Ny = y.size() - 1;
    size_t Nz = z.size() - 1;
    if ( check )
        AMP_ASSERT( Nx * Ny * Nz == mesh->numGlobalElements( AMP::Mesh::GeomType::Cell ) );
}


/************************************************************
 * Functions to iterators over particular sets of faces      *
 ************************************************************/
AMP::Mesh::MeshIterator
StructuredMeshHelper::getXYFaceIterator( std::shared_ptr<AMP::Mesh::Mesh> mesh, int gcw )
{
    return getFaceIterator( mesh, gcw, 2 );
}
AMP::Mesh::MeshIterator
StructuredMeshHelper::getXZFaceIterator( std::shared_ptr<AMP::Mesh::Mesh> mesh, int gcw )
{
    return getFaceIterator( mesh, gcw, 1 );
}
AMP::Mesh::MeshIterator
StructuredMeshHelper::getYZFaceIterator( std::shared_ptr<AMP::Mesh::Mesh> mesh, int gcw )
{
    return getFaceIterator( mesh, gcw, 0 );
}
AMP::Mesh::MeshIterator StructuredMeshHelper::getFaceIterator(
    std::shared_ptr<AMP::Mesh::Mesh> mesh, int gcw, int direction )
{
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    auto boxmesh   = std::dynamic_pointer_cast<AMP::Mesh::BoxMesh>( mesh );
    if ( multimesh != nullptr ) {
        // Optimization for multi-meshes
        auto meshlist = multimesh->getMeshes();
        if ( meshlist.size() == 1 ) {
            return getFaceIterator( meshlist[0], gcw, direction );
        } else {
            std::vector<AMP::Mesh::MeshIterator> iterators( meshlist.size() );
            for ( size_t i = 0; i < meshlist.size(); i++ )
                iterators[i] = getFaceIterator( meshlist[i], gcw, direction );
            return AMP::Mesh::MultiIterator( iterators );
        }
    } else if ( boxmesh != nullptr ) {
        // Optimization for AMP structured meshes
        auto box      = boxmesh->getLocalBox( gcw );
        auto periodic = boxmesh->periodic();
        int Nx        = box.last[0] - box.first[0] + 1;
        int Ny        = box.last[1] - box.first[1] + 1;
        int Nz        = box.last[2] - box.first[2] + 1;
        int last[3]   = { 1, 1, 1 };
        if ( gcw == 0 ) {
            auto global_box = boxmesh->getGlobalBox( 0 );
            for ( int d = 0; d < 3; d++ ) {
                if ( periodic[0] || box.last[d] != global_box.last[d] )
                    last[d] = 0;
            }
        }
        auto face_list = std::make_shared<std::vector<BoxMesh::MeshElementIndex>>();
        face_list->reserve( 4 * Nx * Ny * Nz );
        if ( direction == 0 ) {
            face_list->reserve( ( Nx + 1 ) * Ny * Nz );
            for ( int k = box.first[2]; k <= box.last[2]; k++ ) {
                for ( int j = box.first[1]; j <= box.last[1]; j++ ) {
                    for ( int i = box.first[0]; i <= box.last[0] + last[0]; i++ )
                        face_list->push_back( AMP::Mesh::BoxMesh::MeshElementIndex(
                            AMP::Mesh::GeomType::Face, 0, i, j, k ) );
                }
            }
        } else if ( direction == 1 ) {
            face_list->reserve( Nx * ( Ny + 1 ) * Nz );
            for ( int k = box.first[2]; k <= box.last[2]; k++ ) {
                for ( int i = box.first[0]; i <= box.last[0]; i++ ) {
                    for ( int j = box.first[1]; j <= box.last[1] + last[1]; j++ )
                        face_list->push_back( AMP::Mesh::BoxMesh::MeshElementIndex(
                            AMP::Mesh::GeomType::Face, 1, i, j, k ) );
                }
            }
        } else if ( direction == 2 ) {
            face_list->reserve( Nx * Ny * ( Nz + 1 ) );
            for ( int j = box.first[1]; j <= box.last[1]; j++ ) {
                for ( int i = box.first[0]; i <= box.last[0]; i++ ) {
                    for ( int k = box.first[2]; k <= box.last[2] + last[2]; k++ )
                        face_list->push_back( AMP::Mesh::BoxMesh::MeshElementIndex(
                            AMP::Mesh::GeomType::Face, 2, i, j, k ) );
                }
            }
        } else {
            AMP_ERROR( "Unfinished" );
        }
        return structuredMeshIterator( face_list, boxmesh.get(), 0 );
    } else {
        // General case
        auto iterator = mesh->getIterator( AMP::Mesh::GeomType::Face, gcw );
        std::vector<std::unique_ptr<AMP::Mesh::MeshElement>> face_list;
        std::vector<double> face_index;
        face_list.reserve( iterator.size() );
        face_index.reserve( iterator.size() );
        std::vector<double> x, y, z;
        getXYZCoordinates( mesh, x, y, z, false );
        std::vector<std::tuple<int, int, int>> index;
        index.reserve( iterator.size() );
        for ( size_t i = 0; i < iterator.size(); ++i ) {
            auto nodes    = iterator->getElements( AMP::Mesh::GeomType::Vertex );
            auto center   = iterator->centroid();
            bool is_valid = true;
            for ( auto &node : nodes ) {
                auto coord = node->coord();
                if ( !AMP::Utilities::approx_equal( coord[direction], center[direction], 1e-12 ) )
                    is_valid = false;
            }
            if ( is_valid ) {
                int t1 = 0, t2 = 0, t3 = 0;
                if ( direction == 0 && center.size() == 3 ) {
                    t1 = Utilities::findfirst( z, center[2] - 1e-12 );
                    t2 = Utilities::findfirst( y, center[1] - 1e-12 );
                    t3 = Utilities::findfirst( x, center[0] - 1e-12 );
                } else if ( direction == 1 && center.size() == 3 ) {
                    t1 = Utilities::findfirst( z, center[2] - 1e-12 );
                    t2 = Utilities::findfirst( x, center[0] - 1e-12 );
                    t3 = Utilities::findfirst( y, center[1] - 1e-12 );
                } else if ( direction == 2 && center.size() == 3 ) {
                    t1 = Utilities::findfirst( y, center[1] - 1e-12 );
                    t2 = Utilities::findfirst( x, center[0] - 1e-12 );
                    t3 = Utilities::findfirst( z, center[2] - 1e-12 );
                } else {
                    AMP_ERROR( "Not finished" );
                }
                face_list.push_back( iterator->clone() );
                index.emplace_back( t1, t2, t3 );
            }
            ++iterator;
        }
        // Sort the points in the direction first, then the coordinates
        std::vector<int> I( index.size() );
        std::iota( I.begin(), I.end(), 0 );
        Utilities::quicksort( index, I );
        auto elements =
            std::make_shared<std::vector<std::unique_ptr<AMP::Mesh::MeshElement>>>( index.size() );
        for ( size_t i = 0; i < index.size(); i++ )
            elements->operator[]( i ) = std::move( face_list[I[i]] );
        return AMP::Mesh::MeshElementVectorIterator( elements );
    }
}


AMP::Mesh::MeshIterator StructuredMeshHelper::getGapFaceIterator( std::shared_ptr<AMP::Mesh::Mesh>,
                                                                  int )
{
    AMP_ERROR( "Not finished" );
    return AMP::Mesh::MeshIterator();
}


} // namespace AMP::Mesh


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
#include "AMP/utils/Utilities.hpp"
using intTuple3 = std::tuple<int, int, int>;
template void AMP::Utilities::quicksort<intTuple3, int>( size_t N, intTuple3 *x, int *y );
template void AMP::Utilities::quicksort<intTuple3, int>( std::vector<intTuple3> &x,
                                                         std::vector<int> &y );
