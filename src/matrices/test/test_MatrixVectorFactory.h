#ifndef included_AMP_test_MatrixVectorFactory
#define included_AMP_test_MatrixVectorFactory

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/testHelpers/VectorFactory.h"

#if defined( USE_EXT_PETSC ) && defined( USE_EXT_TRILINOS )
#include "AMP/matrices/petsc/PetscMatrix.h"
#include "AMP/vectors/petsc/PetscHelpers.h"
#include "AMP/vectors/petsc/PetscVector.h"
#include "AMP/vectors/testHelpers/petsc/PetscVectorFactory.h"
#endif


namespace AMP {
namespace LinearAlgebra {


AMP::LinearAlgebra::Matrix::shared_ptr global_cached_matrix =
    AMP::LinearAlgebra::Matrix::shared_ptr();


// Classes to serve as the vector factories
class AmpInterfaceLeftVectorFactory : public VectorFactory
{
public:
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE_START( "AmpInterfaceLeftVectorFactory::getVector" );
        AMP_ASSERT( global_cached_matrix != nullptr );
        auto matrix = global_cached_matrix;
        auto vector = matrix->getLeftVector();
        vector->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "left" ) );
        PROFILE_STOP( "AmpInterfaceLeftVectorFactory::getVector" );
        return vector;
    }
    std::string name() const override
    {
        return "AmpInterfaceLeftVectorFactory<" + getVector()->type() + ">";
    }
};


class AmpInterfaceRightVectorFactory : public VectorFactory
{
public:
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE_START( "AmpInterfaceRightVectorFactory::getVector" );
        AMP_ASSERT( global_cached_matrix != nullptr );
        auto matrix = global_cached_matrix;
        auto vector = matrix->getRightVector();
        vector->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "right" ) );
        PROFILE_STOP( "AmpInterfaceRightVectorFactory::getVector" );
        return vector;
    }
    std::string name() const override
    {
        return "AmpInterfaceRightVectorFactory<" + getVector()->type() + ">";
    }
};


#if defined( USE_EXT_PETSC ) && defined( USE_EXT_TRILINOS )

class PETScInterfaceLeftVectorFactory : public PetscVectorFactory
{
public:
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE_START( "PETScInterfaceLeftVectorFactory::getVector" );
        AMP_ASSERT( global_cached_matrix != nullptr );
        auto matrix = std::dynamic_pointer_cast<AMP::LinearAlgebra::PetscMatrix>(
            AMP::LinearAlgebra::PetscMatrix::createView( global_cached_matrix ) );
        ::Mat m = matrix->getMat();
        ::Vec v;
        DISABLE_WARNINGS
        MatGetVecs( m, &v, nullptr );
        ENABLE_WARNINGS
        auto vector = createVector( v, true );
        vector->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "petsc_left" ) );
        PROFILE_STOP( "PETScInterfaceLeftVectorFactory::getVector" );
        return vector;
    }
    std::shared_ptr<Vec> getVec( AMP::LinearAlgebra::Vector::shared_ptr vec ) const override
    {
        auto data = std::dynamic_pointer_cast<NativePetscVectorData>( vec->getVectorData() );
        auto ptr  = std::make_shared<Vec>( data->getVec() );
        return ptr;
    }
    std::string name() const override { return "PETScInterfaceLeftVectorFactory"; };
};


class PETScInterfaceRightVectorFactory : public PetscVectorFactory
{
public:
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        PROFILE_START( "PETScInterfaceRightVectorFactory::getVector" );
        AMP_ASSERT( global_cached_matrix != nullptr );
        auto matrix = std::dynamic_pointer_cast<AMP::LinearAlgebra::PetscMatrix>(
            AMP::LinearAlgebra::PetscMatrix::createView( global_cached_matrix ) );
        ::Mat m = matrix->getMat();
        ::Vec v;
        DISABLE_WARNINGS
        MatGetVecs( m, &v, nullptr );
        ENABLE_WARNINGS
        auto vector = createVector( v, true );
        vector->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "petsc_right" ) );
        PROFILE_STOP( "PETScInterfaceRightVectorFactory::getVector" );
        return vector;
    }
    std::shared_ptr<Vec> getVec( AMP::LinearAlgebra::Vector::shared_ptr vec ) const override
    {
        auto data = std::dynamic_pointer_cast<NativePetscVectorData>( vec->getVectorData() );
        auto ptr  = std::make_shared<Vec>( data->getVec() );
        return ptr;
    }
    std::string name() const override { return "PETScInterfaceRightVectorFactory"; }
};

#endif

} // namespace LinearAlgebra
} // namespace AMP

#endif
