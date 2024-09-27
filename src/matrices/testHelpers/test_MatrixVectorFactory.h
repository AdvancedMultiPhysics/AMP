#ifndef included_AMP_test_MatrixVectorFactory
#define included_AMP_test_MatrixVectorFactory

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/testHelpers/VectorFactory.h"

#if defined( AMP_USE_PETSC )
    #include "AMP/matrices/petsc/PetscMatrix.h"
    #include "AMP/vectors/petsc/PetscHelpers.h"
    #include "AMP/vectors/petsc/PetscVector.h"
    #include "AMP/vectors/testHelpers/petsc/PetscVectorFactory.h"
    #include "petscmat.h"
#endif

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {


// Classes to serve as the vector factories
class AmpInterfaceLeftVectorFactory : public VectorFactory
{
public:
    explicit AmpInterfaceLeftVectorFactory( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
        : d_matrix( matrix )
    {
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE( "AmpInterfaceLeftVectorFactory::getVector" );
        auto vector = d_matrix->getLeftVector();
        vector->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "left" ) );
        return vector;
    }
    std::string name() const override
    {
        return "AmpInterfaceLeftVectorFactory<" + getVector()->type() + ">";
    }

private:
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_matrix;
};


class AmpInterfaceRightVectorFactory : public VectorFactory
{
public:
    explicit AmpInterfaceRightVectorFactory( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
        : d_matrix( matrix )
    {
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE( "AmpInterfaceRightVectorFactory::getVector" );
        auto vector = d_matrix->getRightVector();
        vector->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "right" ) );
        return vector;
    }
    std::string name() const override
    {
        return "AmpInterfaceRightVectorFactory<" + getVector()->type() + ">";
    }

private:
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_matrix;
};


#if defined( AMP_USE_PETSC )

class PETScInterfaceLeftVectorFactory : public PetscVectorFactory
{
public:
    explicit PETScInterfaceLeftVectorFactory( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
        : d_matrix( matrix )
    {
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE( "PETScInterfaceLeftVectorFactory::getVector" );
        auto view = AMP::LinearAlgebra::PetscMatrix::view( d_matrix );
        ::Mat m   = view->getMat();
        ::Vec v;
        MatCreateVecs( m, &v, nullptr );
        auto vector = createVector( v, true );
        vector->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "petsc_left" ) );
        return vector;
    }
    std::shared_ptr<Vec> getVec( AMP::LinearAlgebra::Vector::shared_ptr vec ) const override
    {
        auto data = std::dynamic_pointer_cast<NativePetscVectorData>( vec->getVectorData() );
        auto ptr  = std::make_shared<Vec>( data->getVec() );
        return ptr;
    }
    std::string name() const override { return "PETScInterfaceLeftVectorFactory"; };

private:
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_matrix;
};


class PETScInterfaceRightVectorFactory : public PetscVectorFactory
{
public:
    explicit PETScInterfaceRightVectorFactory( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
        : d_matrix( matrix )
    {
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE( "PETScInterfaceRightVectorFactory::getVector" );
        auto view = AMP::LinearAlgebra::PetscMatrix::view( d_matrix );
        ::Mat m   = view->getMat();
        ::Vec v;
        MatCreateVecs( m, &v, nullptr );
        auto vector = createVector( v, true );
        vector->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "petsc_right" ) );
        return vector;
    }
    std::shared_ptr<Vec> getVec( AMP::LinearAlgebra::Vector::shared_ptr vec ) const override
    {
        auto data = std::dynamic_pointer_cast<NativePetscVectorData>( vec->getVectorData() );
        auto ptr  = std::make_shared<Vec>( data->getVec() );
        return ptr;
    }
    std::string name() const override { return "PETScInterfaceRightVectorFactory"; }

private:
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_matrix;
};

#endif

} // namespace AMP::LinearAlgebra

#endif
