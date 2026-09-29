#include "AMP/AMP_TPLs.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/Writer.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/geometry/LogicalGeometry.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <sstream>
#include <string>


// Calculate the volume of each element
AMP::LinearAlgebra::Vector::shared_ptr calcVolume( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    auto DOF =
        AMP::Discretization::simpleDOFManager::create( mesh, mesh->getGeomType(), 0, 1, false );
    auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "volume" );
    auto vec = AMP::LinearAlgebra::createVector( DOF, var, true );
    vec->zero();
    std::vector<size_t> dofs;
    for ( const auto &elem : mesh->getIterator( mesh->getGeomType(), 0 ) ) {
        double volume = elem.volume();
        DOF->getDOFs( elem.globalID(), dofs );
        AMP_ASSERT( dofs.size() == 1 );
        vec->addValuesByGlobalID( 1, &dofs[0], &volume );
    }
    return vec;
}


// Write the mesh data
void writeMesh( std::shared_ptr<AMP::Mesh::Mesh> mesh, const std::string &fname )
{

    // Create the writer and get it's properties
    AMP::IO::Writer::WriterParameters params;
    params.decomposition = AMP::IO::Writer::DecompositionType::SINGLE;
    auto writer          = AMP::IO::Writer::buildWriter( "HDF5", params );

    // Create a simple DOFManager
    auto pointType  = AMP::Mesh::GeomType::Vertex;
    auto volumeType = mesh->getGeomType();
    auto DOFparams  = std::make_shared<AMP::Discretization::DOFManagerParameters>( mesh );
    auto DOF_scalar = AMP::Discretization::simpleDOFManager::create( mesh, pointType, 1, 1, true );
    auto DOF_volume = AMP::Discretization::simpleDOFManager::create( mesh, volumeType, 1, 1, true );
    auto DOF_vector = AMP::Discretization::simpleDOFManager::create( mesh, pointType, 1, 3, true );

    // Create the vectors
    auto rank_var     = std::make_shared<AMP::LinearAlgebra::Variable>( "rank" );
    auto position_var = std::make_shared<AMP::LinearAlgebra::Variable>( "position" );
    auto gp_var       = std::make_shared<AMP::LinearAlgebra::Variable>( "gp_var" );
    auto id_var       = std::make_shared<AMP::LinearAlgebra::Variable>( "ids" );
    auto norm_var     = std::make_shared<AMP::LinearAlgebra::Variable>( "normal" );
    auto meshID_var   = std::make_shared<AMP::LinearAlgebra::Variable>( "MeshID" );
    auto block_var    = std::make_shared<AMP::LinearAlgebra::Variable>( "block" );
    auto rank_vec     = AMP::LinearAlgebra::createVector( DOF_scalar, rank_var, true );
    auto position     = AMP::LinearAlgebra::createVector( DOF_vector, position_var, true );
    auto meshID_vec   = AMP::LinearAlgebra::createVector( DOF_scalar, meshID_var, true );
    auto block_vec    = AMP::LinearAlgebra::createVector( DOF_volume, block_var, true );

    // Register the data
    using VectorType = AMP::IO::Writer::VectorType;
    int level        = 1; // How much detail do we want to register
    writer->registerMesh( mesh, level );
    writer->registerVector( meshID_vec, mesh, pointType, "MeshID", VectorType::INT );
    writer->registerVector( block_vec, mesh, volumeType, "BlockID", VectorType::INT, true );
    writer->registerVector( rank_vec, mesh, pointType, "rank", VectorType::INT, true );
    writer->registerVector( position, mesh, pointType, "position", VectorType::DOUBLE );

    // For each submesh: store the mesh id and volume
    auto meshIDs = mesh->getBaseMeshIDs();
    for ( size_t i = 0; i < meshIDs.size(); i++ ) {
        auto mesh2 = mesh->Subset( meshIDs[i] );
        if ( mesh2 ) {
            auto volume = calcVolume( mesh2 );
            AMP::LinearAlgebra::VS_Mesh meshSelector( mesh2 );
            auto meshID_vec2 = meshID_vec->select( meshSelector );
            meshID_vec2->setToScalar( i + 1 );
            writer->registerMesh( mesh2, level );
            writer->registerVector( volume, mesh2, mesh2->getGeomType(), "volume" );
        }
    }

    // Initialize the data
    rank_vec->setToScalar( mesh->getComm().getRank() );
    rank_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    std::vector<size_t> dofs;
    for ( auto &elem : DOF_vector->getIterator() ) {
        DOF_vector->getDOFs( elem.globalID(), dofs );
        auto pos = elem.coord();
        position->setValuesByGlobalID( dofs.size(), dofs.data(), pos.data() );
    }
    position->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    block_vec->setToScalar( -1 );
    for ( auto &id : mesh->getBlockIDs() ) {
        double val = double( id );
        try {
            for ( auto &elem : mesh->getBlockIDIterator( volumeType, id, 0 ) ) {
                DOF_volume->getDOFs( elem.globalID(), dofs );
                block_vec->setValuesByGlobalID( 1, &dofs[0], &val );
            }
        } catch ( ... ) {
        }
    }
    block_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    writer->writeFile( fname, 0 );
}


std::shared_ptr<AMP::Mesh::Mesh> loadSTL( const std::string &fname )
{
    auto db     = AMP::Database::create( "MeshName", "stl", "FileName", fname, "MeshType", "AMP" );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( std::move( db ) );
    return AMP::Mesh::MeshFactory::create( params );
}


int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );

    if ( argc != 2 ) {
        std::cerr << "viewSTL file.stl\n";
        return -1;
    }
    if ( AMP::AMP_MPI( AMP_COMM_WORLD ).getSize() != 1 ) {
        std::cerr << "viewSTL must be run in serial\n";
        return -1;
    }

    auto mesh = loadSTL( argv[1] );

    auto fname = AMP::Utilities::strrep( argv[1], ".stl", "" );
    writeMesh( mesh, fname );


    AMP::AMPManager::shutdown();
    return 0;
}
