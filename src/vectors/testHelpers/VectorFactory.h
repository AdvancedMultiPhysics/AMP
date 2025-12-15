#ifndef included_AMP_test_VectorFactory
#define included_AMP_test_VectorFactory

#include "AMP/mesh/testHelpers/meshTests.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.hpp"
#include "AMP/vectors/testHelpers/VectorTests.h"


namespace AMP::LinearAlgebra {


// Null vector factory
class NullVectorFactory : public VectorFactory
{
public:
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override;
    std::string name() const override;
};


// Null vector data factory
class NullVectorDataFactory : public VectorFactory
{
public:
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override;
    std::string name() const override;
};


// CloneFactory factory
class CloneFactory : public VectorFactory
{
public:
    explicit CloneFactory( std::shared_ptr<const VectorFactory> factory );
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override;
    std::string name() const override;

private:
    CloneFactory();
    std::shared_ptr<const VectorFactory> d_factory;
};


// SimpleVector factory
template<class TYPE       = double,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataDefault<TYPE>>
class SimpleVectorFactory : public VectorFactory
{
public:
    SimpleVectorFactory( int i, bool global, std::string name = "SimpleVectorFactory" )
        : I( i ), GLOBAL( global ), NAME( std::move( name ) )
    {
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "simple" );
        if ( GLOBAL )
            return AMP::LinearAlgebra::createSimpleVector<TYPE, VecOps, VecData>(
                I, var, AMP_COMM_WORLD );
        else
            return AMP::LinearAlgebra::createSimpleVector<TYPE, VecOps, VecData>( I, var );
    }
    std::string name() const override { return NAME; }

private:
    SimpleVectorFactory() = delete;
    int I;
    bool GLOBAL;
    std::string NAME;
};


// ArrayVector factory
template<class TYPE = double>
class ArrayVectorFactory : public VectorFactory
{
public:
    ArrayVectorFactory( size_t d, size_t i, bool global ) : D( d ), I( i ), GLOBAL( global ) {}
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "array" );
        if ( GLOBAL ) {
            AMP_MPI comm( AMP_COMM_WORLD );
            ArraySize index( comm.getRank(), 0, 0 );
            return AMP::LinearAlgebra::createArrayVector<TYPE>( { D, I }, index, comm, var );
        } else {
            return AMP::LinearAlgebra::createArrayVector<TYPE>( { D, I }, var );
        }
    }
    std::string name() const override { return "ArrayVectorFactory"; }

private:
    ArrayVectorFactory() = delete;
    size_t D, I;
    bool GLOBAL;
};


// MeshVector factory
class CubeMeshVectorFactory : public AMP::Mesh::meshTests::MeshVectorFactory
{
public:
    CubeMeshVectorFactory( int N );
    std::string name() const override { return "CubeMeshVectorFactory"; }
    static std::shared_ptr<AMP::Mesh::Mesh> generateMesh( int N );

private:
    CubeMeshVectorFactory() = delete;
};


// View factory
template<typename TYPE>
class ViewFactory : public VectorFactory
{
public:
    explicit ViewFactory( std::shared_ptr<const VectorFactory> factory ) : d_factory( factory ) {}
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        auto vec = TYPE::view( d_factory->getVector() );
        AMP_INSIST( vec != nullptr, "Failed to cast view to type" );
        [[maybe_unused]] auto native = vec->getNativeVec();
        return vec->getManagedVec();
    }
    std::string name() const override { return "ViewFactory<" + d_factory->name() + ">"; }

private:
    ViewFactory() = delete;
    std::shared_ptr<const VectorFactory> d_factory;
};


// MultiVector factory
class MultiVectorFactory : public VectorFactory
{
public:
    MultiVectorFactory( std::shared_ptr<const VectorFactory> factory1,
                        int N1,
                        std::shared_ptr<const VectorFactory> factory2,
                        int N2 );
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override;
    std::string name() const override;

private:
    MultiVectorFactory() = delete;
    int NUM1, NUM2;
    std::shared_ptr<const VectorFactory> FACTORY1;
    std::shared_ptr<const VectorFactory> FACTORY2;
};


} // namespace AMP::LinearAlgebra


#endif
