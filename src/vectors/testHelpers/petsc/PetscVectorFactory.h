#ifndef included_AMP_test_PetscVectorFactory
#define included_AMP_test_PetscVectorFactory

#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/petsc/NativePetscVectorData.h"
#include "AMP/vectors/petsc/PetscHelpers.h"
#include "AMP/vectors/testHelpers/VectorTests.h"
#include "AMP/vectors/testHelpers/generateVectorFactories.h"

#include "petscvec.h"


namespace AMP::LinearAlgebra {


//! Helper deleter class
class PetscVecDeleter
{
public:
    PetscVecDeleter( std::shared_ptr<AMP::LinearAlgebra::PetscVector> vec ) : d_vec( vec ) {}
    void operator()( Vec *ptr ) const
    {
        delete ptr;
        d_vec.reset();
    }

private:
    mutable std::shared_ptr<AMP::LinearAlgebra::PetscVector> d_vec;
};


/**
 * \class PetscVectorFactory
 * \brief A helper class to generate vectors
 */
class PetscVectorFactory : public VectorFactory
{
public:
    virtual ~PetscVectorFactory() {}
    virtual std::shared_ptr<Vec> getVec( AMP::LinearAlgebra::Vector::shared_ptr ) const = 0;

protected:
    PetscVectorFactory() {}
    PetscVectorFactory( const PetscVectorFactory & );
};


class PetscCloneFactory : public PetscVectorFactory
{
public:
    PetscCloneFactory() = delete;
    explicit PetscCloneFactory( std::shared_ptr<const PetscVectorFactory> factory )
        : d_factory( factory )
    {
    }
    std::shared_ptr<Vec> getVec( AMP::LinearAlgebra::Vector::shared_ptr vec ) const override
    {
        return d_factory->getVec( vec );
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        return d_factory->getVector()->clone();
    }
    std::string name() const override { return "PetscCloneFactory<" + d_factory->name() + ">"; }

private:
    std::shared_ptr<const PetscVectorFactory> d_factory;
};


class PetscViewFactory : public PetscVectorFactory
{
public:
    explicit PetscViewFactory( std::shared_ptr<const VectorFactory> factory ) : d_factory( factory )
    {
    }
    std::shared_ptr<Vec> getVec( AMP::LinearAlgebra::Vector::shared_ptr vec ) const override
    {
        auto view = AMP::LinearAlgebra::PetscVector::view( vec );
        std::shared_ptr<Vec> ptr( new Vec( view->getVec() ), PetscVecDeleter( view ) );
        return ptr;
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        return d_factory->getVector();
    }
    std::string name() const override { return "PetscViewFactory<" + d_factory->name() + ">"; }

private:
    std::shared_ptr<const VectorFactory> d_factory;
};


class NativePetscVectorFactory : public PetscVectorFactory
{
public:
    std::shared_ptr<Vec> getVec( AMP::LinearAlgebra::Vector::shared_ptr vec ) const override
    {
        auto data = std::dynamic_pointer_cast<NativePetscVectorData>( vec->getVectorData() );
        std::shared_ptr<Vec> ptr( new Vec( data->getVec() ) );
        return ptr;
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        Vec v;
        AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
        VecCreate( globalComm.getCommunicator(), &v );
        PetscInt local_size = 15;
        VecSetSizes( v, local_size, PETSC_DECIDE );
        VecSetType( v, VECMPI ); // this line will have to be modified for the no mpi and cuda cases
        auto newVec = createVector( v, true );
        VecSetFromOptions( v );
        newVec->getVectorData()->assemble();
        newVec->setVariable(
            std::make_shared<AMP::LinearAlgebra::Variable>( "Test NativePetscVector" ) );
        return newVec;
    }
    std::string name() const override { return "NativePetscVectorFactory"; }
};


} // namespace AMP::LinearAlgebra

#endif
