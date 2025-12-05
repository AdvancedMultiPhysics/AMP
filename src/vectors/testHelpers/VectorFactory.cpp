#include "AMP/vectors/testHelpers/VectorFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/vectors/data/VectorDataNull.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * CloneFactory                                                  *
 ****************************************************************/
CloneFactory::CloneFactory( std::shared_ptr<const VectorFactory> factory ) : d_factory( factory ) {}
AMP::LinearAlgebra::Vector::shared_ptr CloneFactory::getVector() const
{
    auto vec = d_factory->getVector();
    return vec->clone();
}
std::string CloneFactory::name() const { return "CloneFactory<" + d_factory->name() + ">"; }


/****************************************************************
 * NullVectorFactory                                             *
 ****************************************************************/
AMP::LinearAlgebra::Vector::shared_ptr NullVectorFactory::getVector() const
{
    return std::make_shared<AMP::LinearAlgebra::Vector>( "null" );
}
std::string NullVectorFactory::name() const { return "NullVectorFactory"; }
AMP::LinearAlgebra::Vector::shared_ptr NullVectorDataFactory::getVector() const
{
    auto ops  = std::make_shared<VectorOperationsDefault<double>>();
    auto data = std::make_shared<VectorDataNull<double>>();
    auto var  = std::make_shared<Variable>();
    auto DOFs = std::make_shared<AMP::Discretization::DOFManager>( 0, AMP_COMM_SELF );
    return std::make_shared<Vector>( data, ops, var, DOFs );
}
std::string NullVectorDataFactory::name() const { return "NullVectorDataFactory"; }


/****************************************************************
 * CubeMeshVectorFactory                                         *
 ****************************************************************/
CubeMeshVectorFactory::CubeMeshVectorFactory( int N )
    : AMP::Mesh::meshTests::MeshVectorFactory(
          generateMesh( N ), AMP::Mesh::GeomType::Vertex, 1, 1, true )
{
}
std::shared_ptr<AMP::Mesh::Mesh> CubeMeshVectorFactory::generateMesh( int N )
{
    auto database = std::make_shared<AMP::Database>( "Mesh" );
    database->putScalar<int>( "dim", 3 );
    database->putScalar<std::string>( "MeshName", "AMP::cube" );
    database->putScalar<std::string>( "Generator", "cube" );
    database->putVector<int>( "Size", { N, N, N } );
    database->putVector<double>( "Range", { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 } );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP_COMM_WORLD );
    return AMP::Mesh::BoxMesh::generate( params );
}


/****************************************************************
 * MultiVectorFactory                                            *
 ****************************************************************/
MultiVectorFactory::MultiVectorFactory( std::shared_ptr<const VectorFactory> factory1,
                                        int N1,
                                        std::shared_ptr<const VectorFactory> factory2,
                                        int N2 )
    : NUM1( N1 ), NUM2( N2 ), FACTORY1( factory1 ), FACTORY2( factory2 )
{
}
AMP::LinearAlgebra::Vector::shared_ptr MultiVectorFactory::getVector() const
{
    auto var = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "var1" );

    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs;
    for ( int i = 0; i != NUM1; i++ )
        vecs.push_back( FACTORY1->getVector() );
    for ( int i = 0; i != NUM2; i++ )
        vecs.push_back( FACTORY2->getVector() );
    return AMP::LinearAlgebra::MultiVector::create( var, AMP_COMM_WORLD, vecs );
}
std::string MultiVectorFactory::name() const
{
    return "MultiVectorFactory<" + FACTORY1->name() + "," + std::to_string( NUM1 ) + "," +
           FACTORY2->name() + "," + std::to_string( NUM2 ) + ">";
}


} // namespace AMP::LinearAlgebra
