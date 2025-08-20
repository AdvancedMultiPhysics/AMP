// Some basic utility functions

#ifndef UTILS_HPP_ // Checks if UTILS_HPP_ is NOT defined
#define UTILS_HPP_ // Defines UTILS_HPP_ if it's not already defined

#include "AMP/IO/PIO.h"

#include "AMP/utils/AMPManager.h"

#include "AMP/vectors/Vector.h"
#include "AMP/vectors/MultiVector.h"

#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/structured/BoxMesh.h"


#include <iostream>
#include <iomanip>


// Object to hold column indices and associated data
struct colsDataPair {
    std::vector<size_t> cols;
    std::vector<double> data;
};

/* --------------------------------------
    Implementation of utility functions 
----------------------------------------- */

/* Convert the grid index i to the corresponding DOF index */
inline size_t grid_inds_to_DOF(int i,
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh, 
    std::shared_ptr<AMP::Discretization::DOFManager> DOFMan) {
    AMP::Mesh::BoxMesh::MeshElementIndex ind(
                    AMP::Mesh::GeomType::Vertex, 0, i );
    AMP::Mesh::MeshElementID id = mesh->convert( ind );
    std::vector<size_t> dof;
    DOFMan->getDOFs(id, dof);
    return dof[0];
};

// As above, but just gets a localNodeBox from the mesh
inline AMP::Mesh::BoxMesh::Box getLocalNodeBox( std::shared_ptr<AMP::Mesh::BoxMesh> mesh ) {
    auto local  = mesh->getLocalBox();
    auto global = mesh->getGlobalBox();
    for ( int d = 0; d < 3; d++ ) {
        if ( local.last[d] == global.last[d] )
            local.last[d]++;
    }
    return local;
};

// As above, but just gets a GlobalNodeBox from the mesh
inline AMP::Mesh::BoxMesh::Box getGlobalNodeBox( std::shared_ptr<AMP::Mesh::BoxMesh> mesh ) {
    auto global = mesh->getGlobalBox();
    for ( int d = 0; d < 3; d++ ) {
        global.last[d]++;
    }
    return global;
};

/* Fill CSR matrix with data from CSRData */
inline void fillMatWithLocalCSRData( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix,
                          std::shared_ptr<AMP::Discretization::DOFManager> DOFMan,
                          std::map<size_t, colsDataPair> CSRData ) {

    size_t nrows = 1;
    // Iterate through local rows in matrix
    for ( size_t dof = DOFMan->beginDOF(); dof != DOFMan->endDOF(); dof++ ) {
        size_t ncols = CSRData[dof].cols.size();
        matrix->setValuesByGlobalID<double>( nrows, ncols, &dof, CSRData[dof].cols.data(), CSRData[dof].data.data() );
    }
}


/* In 1D: We want the computational vertices at the mid points of each cell. 
The n cells look like:
    [0, 1]*h, [1, 2]*h, ..., [n-1, n]*h.
Since there are n cells in [0,1], the width of each must be 1/n
The computational vertices are the mid points of each cell, and we have n active vertices:
        0.5*h, h + 0.5*h, ..., 1 - (h + 0.5*), 1 - 0.5*h. 

    */
inline static std::shared_ptr<AMP::Mesh::BoxMesh> createBoxMesh( AMP::AMP_MPI comm, std::unique_ptr<AMP::Database> &mesh_db )
{
    auto n   = mesh_db->getScalar<int>( "n" );
    auto dim = mesh_db->getScalar<int>( "dim" );
    double h = 1.0 / n;
    mesh_db->putScalar( "h",   h );
    
    auto mesh_db_internal = std::make_shared<AMP::Database>( "Mesh" );
    mesh_db_internal->putScalar<std::string>( "MeshName", "AMP::cube" );
    mesh_db_internal->putScalar<std::string>( "Generator", "cube" );
    if ( dim == 1 ) {
        mesh_db_internal->putVector<int>( "Size", { n-1 } ); // mesh has n points
        mesh_db_internal->putVector<double>( "Range", { 0.5*h, 1.0 - 0.5*h } );
    } else if ( dim == 2 ) {
        mesh_db_internal->putVector<int>( "Size", { n-1, n-1 } ); // mesh has n x n points
        mesh_db_internal->putVector<double>( "Range", { 0.5*h, 1.0 - 0.5*h, 0.5*h, 1.0 - 0.5*h } );
    }

    // Create MeshParameters
    auto mesh_params = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db_internal );
    mesh_params->setComm( comm );
    
    // Create Mesh
    static std::shared_ptr<AMP::Mesh::BoxMesh> boxMesh = AMP::Mesh::BoxMesh::generate( mesh_params );

    return boxMesh;
}

/* Compute discrete norms of vector u */
inline std::vector<double> getDiscreteNorms(double h,  
                    std::shared_ptr<const AMP::LinearAlgebra::Vector> u) {
    // Compute norms
    double uL1Norm  = static_cast<double>( u->L1Norm()  ) * h*h;
    double uL2Norm  = static_cast<double>( u->L2Norm()  ) * h;
    double uMaxNorm = static_cast<double>( u->maxNorm() );

    std::vector<double> unorms = { uL1Norm, uL2Norm, uMaxNorm }; 
    return unorms;
}



#endif // UTILS_HPP_