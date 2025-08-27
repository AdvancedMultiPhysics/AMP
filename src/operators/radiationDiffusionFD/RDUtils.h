#ifndef RD_UTILS_ 
#define RD_UTILS_ 

#include "AMP/IO/PIO.h"

#include "AMP/utils/AMPManager.h"

#include "AMP/vectors/Vector.h"
#include "AMP/vectors/MultiVector.h"

#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/structured/BoxMesh.h"


#include <iostream>
#include <iomanip>


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
    } else if ( dim == 3 ) {
        mesh_db_internal->putVector<int>( "Size", { n-1, n-1, n-1 } ); // mesh has n x n x n points
        mesh_db_internal->putVector<double>( "Range", { 0.5*h, 1.0 - 0.5*h, 0.5*h, 1.0 - 0.5*h, 0.5*h, 1.0 - 0.5*h } );
    }

    // Create MeshParameters
    auto mesh_params = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db_internal );
    mesh_params->setComm( comm );
    
    // Create Mesh
    static std::shared_ptr<AMP::Mesh::BoxMesh> boxMesh = AMP::Mesh::BoxMesh::generate( mesh_params );

    return boxMesh;
}


#endif // RD_UTILS_