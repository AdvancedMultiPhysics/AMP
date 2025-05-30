#include "AMP/vectors/testHelpers/VectorTests.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#ifdef AMP_USE_SUNDIALS
    #include "AMP/vectors/sundials/ManagedSundialsVector.h"
    #include "AMP/vectors/sundials/SundialsVector.h"
#endif
#ifdef AMP_USE_PETSC
    #include "AMP/vectors/petsc/PetscVector.h"
#endif

#include "ProfilerApp.h"

#include <algorithm>
#include <numeric>


#define PASS_FAIL( test, MSG )                                                    \
    do {                                                                          \
        if ( test )                                                               \
            ut->passes( d_factory->name() + " - " + __FUNCTION__ + ": " + MSG );  \
        else                                                                      \
            ut->failure( d_factory->name() + " - " + __FUNCTION__ + ": " + MSG ); \
    } while ( 0 )


namespace AMP::LinearAlgebra {


static double getTol( const Vector &x )
{
    double tol = 0;
    for ( size_t i = 0; i < x.numberOfDataBlocks(); i++ ) {
        if ( x.isType<double>( i ) )
            tol = std::max<double>( tol, std::numeric_limits<double>::epsilon() );
        if ( x.isType<float>( i ) )
            tol = std::max<float>( tol, std::numeric_limits<float>::epsilon() );
    }
    return tol;
}


void VectorTests::InstantiateVector( AMP::UnitTest *ut )
{
    PROFILE( "InstantiateVector" );
    auto vector = d_factory->getVector();
    PASS_FAIL( vector, "created " );
}


void VectorTests::CopyVectorConsistency( AMP::UnitTest *ut )
{
    PROFILE( "CopyVectorConsistency" );
    auto vec1        = d_factory->getVector();
    auto vec2        = vec1->clone();
    auto vec3        = vec1->clone();
    auto commList    = vec1->getCommunicationList();
    double *t1       = nullptr;
    double *t2       = nullptr;
    size_t *ndx      = nullptr;
    size_t numGhosts = commList->getGhostIDList().size();
    vec1->setRandomValues();
    vec2->copyVector( vec1 );
    if ( numGhosts ) {
        t1  = new double[numGhosts];
        t2  = new double[numGhosts];
        ndx = new size_t[numGhosts];
        std::copy( commList->getGhostIDList().begin(), commList->getGhostIDList().end(), ndx );
        vec1->getValuesByGlobalID( numGhosts, ndx, t1 );
        vec2->getValuesByGlobalID( numGhosts, ndx, t2 );
        PASS_FAIL( std::equal( t1, t1 + numGhosts, t2 ), "Ghosts are the same (1)" );
    }

    vec1->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    vec3->copyVector( vec1 );
    if ( numGhosts ) {
        vec1->getValuesByGlobalID( numGhosts, ndx, t1 );
        vec3->getValuesByGlobalID( numGhosts, ndx, t2 );
        PASS_FAIL( std::equal( t1, t1 + numGhosts, t2 ), "Ghosts are the same (2)" );
        delete[] t1;
        delete[] t2;
        delete[] ndx;
    }
}


void VectorTests::Bug_728( AMP::UnitTest *ut )
{
    PROFILE( "Bug_728" );
    auto vector = d_factory->getVector();
    auto var1   = vector->getVariable();
    if ( !var1 )
        return;
    auto var2 = var1->clone( var1->getName() );
    PASS_FAIL( vector->subsetVectorForVariable( var1 ), "Found vector for same variable" );
    PASS_FAIL( vector->subsetVectorForVariable( var2 ), "Found vector for cloned variable" );
}

template<typename T>
bool vectorValuesNotEqual( AMP::LinearAlgebra::Vector::shared_ptr vector, T val )
{
    PROFILE( "vectorValuesNotEqual" );
    bool fail   = false;
    auto curVec = vector->begin<T>();
    auto endVec = vector->end<T>();
    while ( curVec != endVec ) {
        if ( *curVec != val ) {
            fail = true;
            break;
        }
        ++curVec;
    }
    return fail;
}

void VectorTests::SetToScalarVector( AMP::UnitTest *ut )
{
    PROFILE( "SetToScalarVector" );
    auto vector = d_factory->getVector();
    vector->setToScalar( 0. );
    vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    ut->passes( "setToScalar ran to completion " + d_factory->name() );
    bool fail = false;
    if ( vector->getVectorData()->isType<double>() ) {
        fail = vectorValuesNotEqual<double>( vector, 0.0 );
    } else if ( vector->getVectorData()->isType<float>() ) {
        fail = vectorValuesNotEqual<float>( vector, 0.0 );
    } else if ( vector->getVectorData()->isType<std::complex<double>>() ) {
        fail = vectorValuesNotEqual<std::complex<double>>( vector, 0.0 );
    } else if ( vector->getVectorData()->isType<std::complex<float>>() ) {
        fail = vectorValuesNotEqual<std::complex<float>>( vector, 0.0 );
    } else {
        AMP_ERROR( "Vector data not of correct scalar type" );
    }

    PASS_FAIL( !fail, "Set data to 0" );
    fail = false;
    vector->setToScalar( 5. );
    vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    if ( vector->getVectorData()->isType<double>() ) {
        fail = vectorValuesNotEqual<double>( vector, 5.0 );
    } else if ( vector->getVectorData()->isType<float>() ) {
        fail = vectorValuesNotEqual<float>( vector, 5.0 );
    } else if ( vector->getVectorData()->isType<std::complex<double>>() ) {
        fail = vectorValuesNotEqual<std::complex<double>>( vector, 5.0 );
    } else if ( vector->getVectorData()->isType<std::complex<float>>() ) {
        fail = vectorValuesNotEqual<std::complex<float>>( vector, 5.0 );
    } else {
        AMP_ERROR( "Vector data not of correct scalar type" );
    }
    PASS_FAIL( !fail, "Set data to 5" );
    auto remoteDofs = vector->getDOFManager()->getRemoteDOFs();
    fail            = false;
    if ( vector->getVectorData()->isType<double>() ) {
        for ( auto &remoteDof : remoteDofs ) {
            if ( vector->getValueByGlobalID( remoteDof ) != 5. )
                fail = true;
        }
    } else if ( vector->getVectorData()->isType<float>() ) {
        for ( auto &remoteDof : remoteDofs ) {
            if ( vector->getValueByGlobalID<float>( remoteDof ) != 5. )
                fail = true;
        }
    } else if ( vector->getVectorData()->isType<std::complex<double>>() ) {
        const std::complex<double> five( 5.0, 0 );
        for ( auto &remoteDof : remoteDofs ) {
            if ( vector->getValueByGlobalID<std::complex<double>>( remoteDof ) != five )
                fail = true;
        }
    } else if ( vector->getVectorData()->isType<std::complex<float>>() ) {
        const std::complex<float> five( 5.0, 0 );
        for ( auto &remoteDof : remoteDofs ) {
            if ( vector->getValueByGlobalID<std::complex<float>>( remoteDof ) != five )
                fail = true;
        }
    } else {
        AMP_ERROR( "Vector data not of correct scalar type" );
    }
    PASS_FAIL( !fail, "Set ghost data to 5" );
}

void VectorTests::CloneVector( AMP::UnitTest *ut )
{
    PROFILE( "CloneVector" );
    auto vector = d_factory->getVector();
    auto clone  = vector->clone( "cloned vector" );
    vector->setToScalar( 3.0 );
    clone->setToScalar( 0.0 );
    ut->passes( "Clone created " + d_factory->name() );
    bool pass = true;
    for ( size_t i = 0; i != vector->numberOfDataBlocks(); i++ ) {
        //        auto *clone_ptr  = clone->getRawDataBlock<double>( i );
        //        auto *vector_ptr = vector->getRawDataBlock<double>( i );
        auto *clone_ptr  = clone->getRawDataBlockAsVoid( i );
        auto *vector_ptr = vector->getRawDataBlockAsVoid( i );
        if ( clone_ptr == vector_ptr )
            pass = false;
    }
    PASS_FAIL( pass, "CloneVector: allocated" );
    clone->setToScalar( 1. );
    const auto t = static_cast<double>( clone->L1Norm() );
    PASS_FAIL( clone->getGlobalSize() == vector->getGlobalSize(),
               "CloneVector: global size equality" );
    PASS_FAIL( clone->getLocalSize() == vector->getLocalSize(),
               "CloneVector: local size equality" );
    PASS_FAIL( fabs( t - (double) clone->getGlobalSize() ) < 0.0000001,
               "CloneVector: trivial set data" );
}


void VectorTests::DotProductVector( AMP::UnitTest *ut )
{
    PROFILE( "DotProductVector" );
    auto vector1( d_factory->getVector() );
    auto vector2( d_factory->getVector() );
    vector1->setToScalar( 1. );
    vector2->setToScalar( 2. );
    auto d11 = vector1->dot( *vector1 );
    auto d12 = vector1->dot( *vector2 );
    auto d21 = vector2->dot( *vector1 );
    auto d22 = vector2->dot( *vector2 );
    PASS_FAIL( 2 * d11 == d12, "dot product 1" );
    PASS_FAIL( 2 * d11 == d21, "dot product 2" );
    PASS_FAIL( 4 * d11 == d22, "dot product 3" );
    PASS_FAIL( d11 == vector1->getGlobalSize(), "dot product 4" );
    PASS_FAIL( d21 == d12, "dot product 5" );
}


void VectorTests::L2NormVector( AMP::UnitTest *ut )
{
    PROFILE( "L2NormVector" );
    auto vector = d_factory->getVector();
    vector->setToScalar( 1. );
    auto norm  = static_cast<double>( vector->L2Norm() );
    auto norm2 = static_cast<double>( vector->dot( *vector ) );
    double tol = 0.000001;
    if ( vector->getVectorData()->isType<float>() )
        tol = 0.00001;
    PASS_FAIL( fabs( static_cast<double>( norm * norm - norm2 ) ) < tol, "L2 norm 1" );
    vector->setRandomValues();
    norm  = static_cast<double>( vector->L2Norm() );
    norm2 = static_cast<double>( vector->dot( *vector ) );
    PASS_FAIL( fabs( static_cast<double>( norm * norm - norm2 ) ) < tol, "L2 norm 2" );
}


void VectorTests::AbsVector( AMP::UnitTest *ut )
{
    PROFILE( "AbsVector" );
    auto vec1 = d_factory->getVector();
    auto vec2 = vec1->clone();
    vec1->setRandomValues();
    vec2->copyVector( vec1 );
    vec2->scale( -1.0 );
    vec2->abs( *vec2 );
    PASS_FAIL( vec1->equals( *vec2 ), "Abs passes" );
}


void VectorTests::L1NormVector( AMP::UnitTest *ut )
{
    PROFILE( "L1NormVector" );
    auto vector = d_factory->getVector();
    auto vector_1( d_factory->getVector() );
    vector->setRandomValues();
    vector_1->setToScalar( 1. );
    auto norm = static_cast<double>( vector->L1Norm() );
    vector->abs( *vector );
    auto norm2 = static_cast<double>( vector->dot( *vector_1 ) );
    double tol = 50 * norm * getTol( *vector );
    if ( fabs( norm - norm2 ) )
        ut->passes( "L1 norm" );
    else
        ut->failure( AMP::Utilities::stringf( "L1 norm (%e) (%e)", fabs( norm - norm2 ), tol ) );
}


void VectorTests::MaxNormVector( AMP::UnitTest *ut )
{
    PROFILE( "MaxNormVector" );
    auto vector = d_factory->getVector();
    vector->setRandomValues();
    auto infNorm = vector->maxNorm();
    vector->abs( *vector );
    if ( vector->getVectorData()->isType<double>() ) {
        auto curData   = vector->begin();
        auto endData   = vector->end();
        auto local_ans = *curData;
        while ( curData != endData ) {
            local_ans = std::max( local_ans, *curData );
            ++curData;
        }
        auto global_ans = vector->getComm().maxReduce( local_ans );
        PASS_FAIL( global_ans == infNorm, "Inf norm" );
    } else if ( vector->getVectorData()->isType<float>() ) {
        auto curData   = vector->begin<float>();
        auto endData   = vector->end<float>();
        auto local_ans = *curData;
        while ( curData != endData ) {
            local_ans = std::max( local_ans, *curData );
            ++curData;
        }
        auto global_ans = vector->getComm().maxReduce( local_ans );
        PASS_FAIL( global_ans == infNorm, "Inf norm" );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
}

template<typename TYPE>
bool scaleTest( AMP::LinearAlgebra::Vector::shared_ptr vector1,
                AMP::LinearAlgebra::Vector::shared_ptr vector2,
                TYPE beta )
{
    PROFILE( "scaleTest" );
    bool pass     = true;
    auto curData1 = vector1->begin<TYPE>();
    auto endData1 = vector1->end<TYPE>();
    auto curData2 = vector2->begin<TYPE>();
    while ( curData1 != endData1 ) {
        if ( *curData1 != beta * *curData2 )
            pass = false;
        ++curData1;
        ++curData2;
    }
    return pass;
}

void VectorTests::ScaleVector( AMP::UnitTest *ut )
{
    PROFILE( "ScaleVector" );
    auto vector1( d_factory->getVector() );
    auto vector2( d_factory->getVector() );
    double beta = 1.2345;
    vector2->setRandomValues();
    vector1->scale( beta, *vector2 );
    bool pass;
    if ( vector1->getVectorData()->isType<double>() &&
         vector2->getVectorData()->isType<double>() ) {
        pass = scaleTest<double>( vector1, vector2, beta );
    } else if ( vector1->getVectorData()->isType<float>() &&
                vector2->getVectorData()->isType<float>() ) {
        pass = scaleTest<float>( vector1, vector2, beta );
    } else if ( vector1->getVectorData()->isType<std::complex<double>>() &&
                vector2->getVectorData()->isType<std::complex<double>>() ) {
        pass = scaleTest<std::complex<double>>( vector1, vector2, beta );
    } else if ( vector1->getVectorData()->isType<std::complex<float>>() &&
                vector2->getVectorData()->isType<std::complex<float>>() ) {
        pass = scaleTest<std::complex<float>>( vector1, vector2, beta );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
    PASS_FAIL( pass, "scale vector 1" );
    vector2->scale( beta );
    vector1->subtract( *vector2, *vector1 );
    double tol = 10 * getTol( *vector1 );
    PASS_FAIL( vector1->maxNorm() < tol, "scale vector 2" );
}


void VectorTests::Bug_491( [[maybe_unused]] AMP::UnitTest *ut )
{
#ifdef AMP_USE_PETSC
    PROFILE( "Bug_491" );
    auto vector1( d_factory->getVector() );
    if ( vector1->getVectorData()->isType<PetscReal>() ) {
        vector1->setRandomValues();
        auto managed_petsc = AMP::LinearAlgebra::PetscVector::view( vector1 );
        auto petsc_vec =
            std::dynamic_pointer_cast<AMP::LinearAlgebra::PetscVector>( managed_petsc );
        Vec managed_vec = petsc_vec->getVec();


        // This sets the petsc cache
        PetscReal n1, n2, ninf;
        VecNormBegin( managed_vec, NORM_1, &n1 );
        VecNormBegin( managed_vec, NORM_2, &n2 );
        VecNormBegin( managed_vec, NORM_INFINITY, &ninf );
        VecNormEnd( managed_vec, NORM_1, &n1 );
        VecNormEnd( managed_vec, NORM_2, &n2 );
        VecNormEnd( managed_vec, NORM_INFINITY, &ninf );
        VecNorm( managed_vec, NORM_1, &n1 );
        VecNorm( managed_vec, NORM_2, &n2 );
        VecNorm( managed_vec, NORM_INFINITY, &ninf );

        // Now, we perform some math on vector1
        vector1->scale( 100000 );
        auto sp_n1  = static_cast<PetscReal>( vector1->L1Norm() );
        auto sp_n2  = static_cast<PetscReal>( vector1->L2Norm() );
        auto sp_inf = static_cast<PetscReal>( vector1->maxNorm() );

        // Check to see if petsc cache has been invalidated
        VecNormBegin( managed_vec, NORM_1, &n1 );
        VecNormBegin( managed_vec, NORM_2, &n2 );
        VecNormBegin( managed_vec, NORM_INFINITY, &ninf );
        VecNormEnd( managed_vec, NORM_1, &n1 );
        VecNormEnd( managed_vec, NORM_2, &n2 );
        VecNormEnd( managed_vec, NORM_INFINITY, &ninf );

        PetscReal tol = 0.00000001 * n1;
        PASS_FAIL( fabs( n1 - sp_n1 ) < tol, "L1 norm -- Petsc interface begin/end" );
        PASS_FAIL( fabs( n2 - sp_n2 ) < tol, "L2 norm -- Petsc interface begin/end" );
        PASS_FAIL( fabs( ninf - sp_inf ) < tol, "Linf norm -- Petsc interface begin/end" );

        VecNorm( managed_vec, NORM_1, &n1 );
        VecNorm( managed_vec, NORM_2, &n2 );
        VecNorm( managed_vec, NORM_INFINITY, &ninf );

        PetscReal L1Norm( vector1->L1Norm() );
        PetscReal L2Norm( vector1->L2Norm() );
        PetscReal maxNorm( vector1->maxNorm() );
        PASS_FAIL( fabs( n1 - L1Norm ) < tol, "L1 norm -- Petsc interface begin/end " );
        PASS_FAIL( fabs( n2 - L2Norm ) < tol, "L2 norm -- Petsc interface begin/end " );
        PASS_FAIL( fabs( ninf - maxNorm ) < tol, "inf norm -- Petsc interface begin/end " );
    }
#endif
}


template<typename T, typename BinaryOperator>
bool binaryOpTest( AMP::LinearAlgebra::Vector::shared_ptr vector1,
                   AMP::LinearAlgebra::Vector::shared_ptr vector2,
                   AMP::LinearAlgebra::Vector::shared_ptr vector3,
                   BinaryOperator op )
{
    PROFILE( "binaryOpTest" );
    bool pass     = true;
    auto curData1 = vector1->begin<T>();
    auto endData1 = vector1->end<T>();
    auto curData2 = vector2->begin<T>();
    auto curData3 = vector3->begin<T>();
    while ( curData1 != endData1 ) {
        if ( *curData3 != op( *curData1, *curData2 ) )
            pass = false;
        ++curData1;
        ++curData2;
        ++curData3;
    }
    return pass;
}

void VectorTests::AddVector( AMP::UnitTest *ut )
{
    PROFILE( "AddVector" );
    auto vector1( d_factory->getVector() );
    auto vector2( d_factory->getVector() );
    auto vector3( d_factory->getVector() );
    vector1->setRandomValues();
    vector2->setRandomValues();
    vector3->add( *vector1, *vector2 );
    bool pass;
    if ( vector1->getVectorData()->isType<double>() && vector2->getVectorData()->isType<double>() &&
         vector3->getVectorData()->isType<double>() ) {
        pass = binaryOpTest<double>( vector1, vector2, vector3, std::plus<double>() );
    } else if ( vector1->getVectorData()->isType<float>() &&
                vector2->getVectorData()->isType<float>() &&
                vector3->getVectorData()->isType<float>() ) {
        pass = binaryOpTest<float>( vector1, vector2, vector3, std::plus<float>() );
    } else if ( vector1->getVectorData()->isType<std::complex<double>>() &&
                vector2->getVectorData()->isType<std::complex<double>>() &&
                vector3->getVectorData()->isType<std::complex<double>>() ) {
        pass = binaryOpTest<std::complex<double>>(
            vector1, vector2, vector3, std::plus<std::complex<double>>() );
    } else if ( vector1->getVectorData()->isType<std::complex<float>>() &&
                vector2->getVectorData()->isType<std::complex<float>>() &&
                vector3->getVectorData()->isType<std::complex<float>>() ) {
        pass = binaryOpTest<std::complex<float>>(
            vector1, vector2, vector3, std::plus<std::complex<float>>() );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
    PASS_FAIL( pass, "add vector" );
}


void VectorTests::SubtractVector( AMP::UnitTest *ut )
{
    PROFILE( "SubtractVector" );
    auto vector1( d_factory->getVector() );
    auto vector2( d_factory->getVector() );
    auto vector3( d_factory->getVector() );
    auto vector4( d_factory->getVector() );
    vector1->setRandomValues();
    vector2->setRandomValues();
    vector3->subtract( *vector1, *vector2 );
    bool pass;
    if ( vector1->getVectorData()->isType<double>() && vector2->getVectorData()->isType<double>() &&
         vector3->getVectorData()->isType<double>() ) {
        pass = binaryOpTest<double>( vector1, vector2, vector3, std::minus<double>() );
    } else if ( vector1->getVectorData()->isType<float>() &&
                vector2->getVectorData()->isType<float>() &&
                vector3->getVectorData()->isType<float>() ) {
        pass = binaryOpTest<float>( vector1, vector2, vector3, std::minus<float>() );
    } else if ( vector1->getVectorData()->isType<std::complex<double>>() &&
                vector2->getVectorData()->isType<std::complex<double>>() &&
                vector3->getVectorData()->isType<std::complex<double>>() ) {
        pass = binaryOpTest<std::complex<double>>(
            vector1, vector2, vector3, std::minus<std::complex<double>>() );
    } else if ( vector1->getVectorData()->isType<std::complex<float>>() &&
                vector2->getVectorData()->isType<std::complex<float>>() &&
                vector3->getVectorData()->isType<std::complex<float>>() ) {
        pass = binaryOpTest<std::complex<float>>(
            vector1, vector2, vector3, std::minus<std::complex<float>>() );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
    PASS_FAIL( pass, "vector subtract 1" );
    vector2->scale( -1. );
    vector4->add( *vector1, *vector2 );
    vector4->subtract( *vector3, *vector4 );
    double tol = 10 * getTol( *vector1 );
    PASS_FAIL( vector4->maxNorm() < tol, "vector subtract 2" );
}


void VectorTests::MultiplyVector( AMP::UnitTest *ut )
{
    PROFILE( "MultiplyVector" );
    auto vector1( d_factory->getVector() );
    auto vector2( d_factory->getVector() );
    auto vector3( d_factory->getVector() );
    vector1->setRandomValues();
    vector2->setToScalar( 3. );
    vector3->multiply( *vector1, *vector2 );
    bool pass;
    if ( vector1->getVectorData()->isType<double>() && vector2->getVectorData()->isType<double>() &&
         vector3->getVectorData()->isType<double>() ) {
        pass = binaryOpTest<double>( vector1, vector2, vector3, std::multiplies<double>() );
    } else if ( vector1->getVectorData()->isType<float>() &&
                vector2->getVectorData()->isType<float>() &&
                vector3->getVectorData()->isType<float>() ) {
        pass = binaryOpTest<float>( vector1, vector2, vector3, std::multiplies<float>() );
    } else if ( vector1->getVectorData()->isType<std::complex<double>>() &&
                vector2->getVectorData()->isType<std::complex<double>>() &&
                vector3->getVectorData()->isType<std::complex<double>>() ) {
        pass = binaryOpTest<std::complex<double>>(
            vector1, vector2, vector3, std::multiplies<std::complex<double>>() );
    } else if ( vector1->getVectorData()->isType<std::complex<float>>() &&
                vector2->getVectorData()->isType<std::complex<float>>() &&
                vector3->getVectorData()->isType<std::complex<float>>() ) {
        pass = binaryOpTest<std::complex<float>>(
            vector1, vector2, vector3, std::multiplies<std::complex<float>>() );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
    PASS_FAIL( pass, "vector::multiply" );
}


void VectorTests::DivideVector( AMP::UnitTest *ut )
{
    PROFILE( "DivideVector" );
    auto vector1( d_factory->getVector() );
    auto vector2( d_factory->getVector() );
    auto vector3( d_factory->getVector() );
    vector1->setRandomValues();
    vector2->setRandomValues();
    vector3->divide( *vector1, *vector2 );
    bool pass;
    if ( vector1->getVectorData()->isType<double>() && vector2->getVectorData()->isType<double>() &&
         vector3->getVectorData()->isType<double>() ) {
        pass = binaryOpTest<double>( vector1, vector2, vector3, std::divides<double>() );
    } else if ( vector1->getVectorData()->isType<float>() &&
                vector2->getVectorData()->isType<float>() &&
                vector3->getVectorData()->isType<float>() ) {
        pass = binaryOpTest<float>( vector1, vector2, vector3, std::divides<float>() );
    } else if ( vector1->getVectorData()->isType<std::complex<double>>() &&
                vector2->getVectorData()->isType<std::complex<double>>() &&
                vector3->getVectorData()->isType<std::complex<double>>() ) {
        pass = binaryOpTest<std::complex<double>>(
            vector1, vector2, vector3, std::divides<std::complex<double>>() );
    } else if ( vector1->getVectorData()->isType<std::complex<float>>() &&
                vector2->getVectorData()->isType<std::complex<float>>() &&
                vector3->getVectorData()->isType<std::complex<float>>() ) {
        pass = binaryOpTest<std::complex<float>>(
            vector1, vector2, vector3, std::divides<std::complex<float>>() );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
    PASS_FAIL( pass, "vector::divide" );
}


void VectorTests::VectorIteratorLengthTest( AMP::UnitTest *ut )
{
    PROFILE( "VectorIteratorLengthTest" );
    auto vector1( d_factory->getVector() );
    size_t i = 0;
    if ( vector1->getVectorData()->isType<double>() ) {
        for ( auto it = vector1->begin(); it != vector1->end(); ++it )
            i++;
    } else if ( vector1->getVectorData()->isType<float>() ) {
        for ( auto it = vector1->begin<float>(); it != vector1->end<float>(); ++it )
            i++;
    } else if ( vector1->getVectorData()->isType<std::complex<double>>() ) {
        for ( auto it = vector1->begin(); it != vector1->end(); ++it )
            i++;
    } else if ( vector1->getVectorData()->isType<std::complex<float>>() ) {
        for ( auto it = vector1->begin<std::complex<float>>();
              it != vector1->end<std::complex<float>>();
              ++it )
            i++;
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
    size_t k = vector1->getLocalSize();
    PASS_FAIL( i == k, "Iterated over the correct number of entries" );
}


void VectorTests::VectorIteratorTests( AMP::UnitTest *ut )
{
    PROFILE( "VectorIteratorTests" );
    auto vector1 = d_factory->getVector();
    if ( vector1->getVectorData()->isType<double>() ) {
        both_VectorIteratorTests<double, AMP::LinearAlgebra::VectorDataIterator<double>>( vector1,
                                                                                          ut );
        both_VectorIteratorTests<const double,
                                 AMP::LinearAlgebra::VectorDataIterator<const double>>( vector1,
                                                                                        ut );
    } else if ( vector1->getVectorData()->isType<float>() ) {
        both_VectorIteratorTests<float, AMP::LinearAlgebra::VectorDataIterator<float>>( vector1,
                                                                                        ut );
        both_VectorIteratorTests<const float, AMP::LinearAlgebra::VectorDataIterator<const float>>(
            vector1, ut );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
}


void VectorTests::VerifyVectorMin( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorMin" );
    auto vec = d_factory->getVector();
    vec->setRandomValues();
    vec->scale( -1.0 ); // make negative
    double min( vec->min() );
    double norm( vec->maxNorm() );
    PASS_FAIL( fabs( min + norm ) < 1.e-10, "minimum of negative vector == ||.||_infty" );
}


void VectorTests::VerifyVectorMax( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorMax" );
    auto vec = d_factory->getVector();
    vec->setRandomValues();
    double max( vec->max() );
    double norm( vec->maxNorm() );
    PASS_FAIL( fabs( max - norm ) < 1.e-10, "maximum of positive vector == ||.||_infty" );
}


void VectorTests::VerifyVectorMaxMin( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorMaxMin" );
    auto vec    = d_factory->getVector();
    bool passes = true;
    for ( size_t i = 0; i != 10; i++ ) {
        vec->setRandomValues();
        vec->addScalar( *vec, -0.5 );
        vec->scale( 2.0 ); // vec i.i.d [-1,1);
        double max( vec->max() );
        double min( vec->min() );
        auto ans = std::max( fabs( max ), fabs( min ) );
        double norm( vec->maxNorm() );
        if ( fabs( ans - norm ) >= 1.e-20 ) {
            passes = false;
        }
    }
    PASS_FAIL( passes, "Max and min correctly predict maxNorm()" );
}


void VectorTests::SetRandomValuesVector( AMP::UnitTest *ut )
{
    PROFILE( "SetRandomValuesVector" );
    auto vector  = d_factory->getVector();
    auto l2norm1 = -1;
    for ( size_t i = 0; i < 5; i++ ) {
        vector->setRandomValues();
        auto l2norm2 = static_cast<double>( vector->L2Norm() );
        PASS_FAIL( fabs( l2norm1 - l2norm2 ) > 0.000001, "Distinct vector created" );
        l2norm1 = l2norm2;
        PASS_FAIL( vector->min() >= 0, "Min value >= 0" );
        PASS_FAIL( vector->max() < 1, "Max value < 1" );
        PASS_FAIL( vector->L2Norm() > 0, "Non-zero vector created" );
    }
}


void VectorTests::ReciprocalVector( AMP::UnitTest *ut )
{
    PROFILE( "ReciprocalVector" );
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    auto vector1 = d_factory->getVector();
    vectora->setRandomValues();
    vectorb->reciprocal( *vectora );
    vector1->setToScalar( 1. );
    vectorc->divide( *vector1, *vectora );
    vectord->subtract( *vectorb, *vectorc );
    double tol = 10 * getTol( *vectora );
    PASS_FAIL( vectord->maxNorm() < tol, "vector::reciprocal" );
}


static void LinearSumVectorRun( std::shared_ptr<const VectorFactory> d_factory,
                                AMP::UnitTest *ut,
                                double alpha,
                                double beta,
                                const char *msg )
{
    PROFILE( "LinearSumVectorRun" );
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    vectora->setRandomValues();
    vectorb->setRandomValues();
    vectorc->linearSum( alpha, *vectora, beta, *vectorb );
    vectora->scale( alpha );
    vectorb->scale( beta );
    vectord->add( *vectora, *vectorb );
    vectord->subtract( *vectorc, *vectord );
    double tol = 10 * getTol( *vectora );
    PASS_FAIL( vectord->maxNorm() < tol, msg );
}
void VectorTests::LinearSumVector( AMP::UnitTest *ut )
{
    PROFILE( "LinearSumVector" );
    LinearSumVectorRun( d_factory, ut, 1.2345, 9.8765, "linear sum 1" );
    LinearSumVectorRun( d_factory, ut, -1.2345, 9.8765, "linear sum 2" );
    LinearSumVectorRun( d_factory, ut, 1.2345, -9.8765, "linear sum 3" );
    LinearSumVectorRun( d_factory, ut, -1.2345, -9.8765, "linear sum 4" );
}


static void AxpyVectorRun( std::shared_ptr<const VectorFactory> d_factory,
                           AMP::UnitTest *ut,
                           double alpha,
                           const char *msg )
{
    PROFILE( "AxpyVectorRun" );
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    vectora->setRandomValues();
    vectorb->setRandomValues();
    vectorc->linearSum( alpha, *vectora, 1., *vectorb );
    vectord->axpy( alpha, *vectora, *vectorb );
    vectorc->subtract( *vectorc, *vectord );
    double err = static_cast<double>( vectorc->maxNorm() );
    double tol = 10 * getTol( *vectora );
    PASS_FAIL( err < tol, msg );
}
void VectorTests::AxpyVector( AMP::UnitTest *ut )
{
    PROFILE( "AxpyVector" );
    AxpyVectorRun( d_factory, ut, 6.38295, "axpy 1" );
    AxpyVectorRun( d_factory, ut, -6.38295, "axpy 2" );
}

template<typename T>
static void AxpbyVectorRun( T alpha,
                            T beta,
                            AMP::LinearAlgebra::Vector::shared_ptr vectora,
                            AMP::LinearAlgebra::Vector::shared_ptr vectorb,
                            AMP::LinearAlgebra::Vector::shared_ptr vectorb1,
                            AMP::LinearAlgebra::Vector::shared_ptr vectorc,
                            AMP::LinearAlgebra::Vector::shared_ptr vectord )
{
    PROFILE( "AxpbyVectorRun" );
    vectorb1->linearSum( alpha, *vectora, beta, *vectorb );
    vectorb->linearSum( alpha, *vectora, beta, *vectorb );
    vectorc->axpby( alpha, beta, *vectora );
    vectord->subtract( *vectorc, *vectorb1 );
}

static void AxpbyVectorRun( std::shared_ptr<const VectorFactory> d_factory,
                            AMP::UnitTest *ut,
                            double alpha,
                            double beta,
                            const char *msg )
{
    PROFILE( "AxpbyVectorRun" );
    auto vectora  = d_factory->getVector();
    auto vectorb  = d_factory->getVector();
    auto vectorb1 = d_factory->getVector();
    auto vectorc  = d_factory->getVector();
    auto vectord  = d_factory->getVector();
    vectora->setRandomValues();
    vectorb->setRandomValues();
    vectorc->copyVector( vectorb );
    if ( vectora->getVectorData()->isType<double>() && vectorb->getVectorData()->isType<double>() &&
         vectorb1->getVectorData()->isType<double>() &&
         vectorc->getVectorData()->isType<double>() &&
         vectord->getVectorData()->isType<double>() ) {
        AxpbyVectorRun<double>( alpha, beta, vectora, vectorb, vectorb1, vectorc, vectord );
    } else if ( vectora->getVectorData()->isType<float>() &&
                vectorb->getVectorData()->isType<float>() &&
                vectorb1->getVectorData()->isType<float>() &&
                vectorc->getVectorData()->isType<float>() &&
                vectord->getVectorData()->isType<float>() ) {
        AxpbyVectorRun<float>( alpha, beta, vectora, vectorb, vectorb1, vectorc, vectord );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }

    auto maxNorm = vectord->maxNorm();
    double tol   = 10 * getTol( *vectora );
    PASS_FAIL( maxNorm < tol, msg );
    vectord->subtract( *vectorc, *vectorb );
    maxNorm = vectord->maxNorm();
    PASS_FAIL( maxNorm < tol, msg );
}
void VectorTests::AxpbyVector( AMP::UnitTest *ut )
{
    PROFILE( "AxpbyVector" );
    AxpbyVectorRun( d_factory, ut, 6.38295, 99.273624, "axpby 1" );
    AxpbyVectorRun( d_factory, ut, 6.38295, -99.273624, "axpby 2" );
    AxpbyVectorRun( d_factory, ut, -6.38295, 99.273624, "axpby 3" );
    AxpbyVectorRun( d_factory, ut, -6.38295, -99.273624, "axpby 4" );
}


void VectorTests::CopyVector( AMP::UnitTest *ut )
{
    PROFILE( "CopyVector" );
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();

    vectora->setRandomValues();
    vectorb->copyVector( vectora );
    vectorc->subtract( *vectora, *vectorb );
    PASS_FAIL( vectorc->maxNorm() == 0, "copy vector 1" );

    vectora->scale( 100. );
    vectorc->subtract( *vectora, *vectorb );
    if ( vectorb->getVectorData()->isType<double>() &&
         vectorc->getVectorData()->isType<double>() ) {
        auto c_maxNorm = vectorc->maxNorm().get<double>();
        auto b_maxNorm = vectorb->maxNorm().get<double>();
        PASS_FAIL( fabs( c_maxNorm - 99 * b_maxNorm ) < 1e-12 * b_maxNorm, "copy vector 2" );
    } else if ( vectorb->getVectorData()->isType<float>() &&
                vectorc->getVectorData()->isType<float>() ) {
        auto c_maxNorm = vectorc->maxNorm().get<float>();
        auto b_maxNorm = vectorb->maxNorm().get<float>();
        //        PASS_FAIL( fabs( c_maxNorm - 99 * b_maxNorm ) < 1e-5 * b_maxNorm, "copy vector 2"
        //        );
        PASS_FAIL( AMP::Utilities::approx_equal( c_maxNorm, 99 * b_maxNorm ), "copy vector 2" );
    } else {
        AMP_ERROR( "CopyVector tests not implemented for provided scalar TYPE" );
    }

    auto simple1 = AMP::LinearAlgebra::createSimpleVector<double>(
        vectora->getLocalSize(), vectora->getVariable(), vectora->getComm() );
    auto simple2 = AMP::LinearAlgebra::createSimpleVector<float>(
        vectora->getLocalSize(), vectora->getVariable(), vectora->getComm() );
    simple1->copyVector( vectora );
    simple2->copyVector( vectora );
    double aNorm( vectora->L2Norm() );
    double norm1( simple1->L2Norm() );
    double norm2( simple2->L2Norm() );
    PASS_FAIL( fabs( aNorm - norm1 ) < 1e-5 * aNorm, "copy vector double" );
    PASS_FAIL( fabs( aNorm - norm2 ) < 1e-5 * aNorm, "copy vector single" );
    // vectorb->copyVector( simple1 );
    // vectorc->copyVector( simple2 );
}


void VectorTests::CopyRawDataBlockVector( AMP::UnitTest *ut )
{
    PROFILE( "CopyRawDataBlockVector" );
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    vectora->setRandomValues();
    vectorb->zero();
    auto buf = new double[vectora->getLocalSize()];
    vectora->getRawData( buf );
    vectorb->putRawData( buf );
    vectorb->getVectorData()->assemble(); // required for petsc
    delete[] buf;
    vectorc->subtract( *vectora, *vectorb );
    PASS_FAIL( vectorc->maxNorm() == 0, "copy raw data block" );
}


void VectorTests::VerifyVectorGhostCreate( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorGhostCreate" );
    AMP_MPI globalComm( AMP_COMM_WORLD );
    auto vector    = d_factory->getVector();
    int num_ghosts = vector->getGhostSize();
    num_ghosts     = globalComm.sumReduce( num_ghosts );
    if ( globalComm.getSize() == 1 ) {
        ut->expected_failure( "No ghost cells for single processor " + d_factory->name() );
    } else {
        PASS_FAIL( num_ghosts > 0, "verify ghosts created " );
    }
}

void VectorTests::VerifyVectorSetZeroGhosts( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorSetZeroGhosts" );
    AMP_MPI globalComm( AMP_COMM_WORLD );
    auto vector = d_factory->getVector();
    vector->setNoGhosts();
    int num_ghosts = vector->getGhostSize();
    bool no_ghosts = !vector->getVectorData()->hasGhosts();
    num_ghosts     = globalComm.sumReduce( num_ghosts );
    PASS_FAIL( no_ghosts && num_ghosts == 0, "verify setNoGhosts " );

    // Test vectors sharing communication lists
    auto v1            = d_factory->getVector();
    auto v2            = v1->clone();
    auto v2_has_ghosts = v2->getVectorData()->hasGhosts();
    v1->setNoGhosts();
    auto nghosts_v1         = v1->getGhostSize();
    nghosts_v1              = globalComm.sumReduce( nghosts_v1 );
    const bool no_ghosts_v1 = !v1->getVectorData()->hasGhosts();
    // if v2 originally had ghosts check again to ensure it still has ghosts, if not set to true for
    // vectors with no ghosts
    v2_has_ghosts = v2_has_ghosts ? v2->getVectorData()->hasGhosts() : true;
    PASS_FAIL( no_ghosts_v1 && nghosts_v1 == 0 && v2_has_ghosts, "verify setNoGhosts " );
}


void VectorTests::VerifyVectorMakeConsistentAdd( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorMakeConsistentAdd" );
    using UpdateState = AMP::LinearAlgebra::UpdateState;
    AMP_MPI globalComm( AMP_COMM_WORLD );
    auto vector = d_factory->getVector();
    auto dofmap = vector->getDOFManager();
    if ( !vector )
        ut->failure( "verify makeConsistent () for add " + d_factory->name() );

    // Zero the vector
    vector->zero();
    if ( vector->getUpdateStatus() != UpdateState::UNCHANGED )
        ut->failure( "zero leaves vector in UpdateState::UNCHANGED state " + d_factory->name() );

    // Set and add local values by global id (this should not interfere with the add)
    const double val = 0.0;
    for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
        vector->setLocalValuesByGlobalID( 1, &i, &val );
        vector->addLocalValuesByGlobalID( 1, &i, &val );
    }
    if ( vector->getUpdateStatus() != UpdateState::LOCAL_CHANGED )
        ut->failure( "local set/add leaves vector in UpdateState::LOCAL_CHANGED state " +
                     d_factory->name() );

    // Add values by global id
    const double zero = 0.0;
    for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ )
        vector->addValuesByGlobalID( 1, &i, &zero );
    if ( vector->getUpdateStatus() != UpdateState::LOCAL_CHANGED )
        ut->failure(
            "addValueByGlobalID(local) leaves vector in UpdateState::LOCAL_CHANGED state " +
            d_factory->name() );
    auto remote = dofmap->getRemoteDOFs();
    if ( !remote.empty() ) {
        for ( size_t i : remote )
            vector->addValuesByGlobalID( 1, &i, &zero );
        if ( vector->getUpdateStatus() != UpdateState::ADDING )
            ut->failure( "addValueByGlobalID(remote) leaves vector in UpdateState::ADDING state " +
                         d_factory->name() );
    }

    // Perform a makeConsistent ADD and check the result
    vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    if ( vector->getUpdateStatus() != UpdateState::UNCHANGED )
        ut->failure( "makeConsistent leaves vector in UpdateState::UNCHANGED state " +
                     d_factory->name() );
}


void VectorTests::VerifyVectorMakeConsistentSet( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorMakeConsistentSet" );
    auto vector = d_factory->getVector();
    auto dofmap = vector->getDOFManager();

    // Zero the vector
    vector->zero();
    if ( vector->getUpdateStatus() != AMP::LinearAlgebra::UpdateState::UNCHANGED )
        ut->failure( "zero leaves vector in UpdateState::UNCHANGED state " + d_factory->name() );

    // Set and add local values by global id (this should not interfere with the add)
    const double val = 0.0;
    for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
        vector->setLocalValuesByGlobalID( 1, &i, &val );
        vector->addLocalValuesByGlobalID( 1, &i, &val );
    }
    if ( vector->getUpdateStatus() != AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED )
        ut->failure( "local set/add leaves vector in UpdateState::LOCAL_CHANGED state " +
                     d_factory->name() + " - " + vector->type() );

    // Set values by global id
    for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
        const auto val = double( i );
        vector->setValuesByGlobalID( 1, &i, &val );
    }
    if ( vector->getUpdateStatus() != AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED &&
         vector->getUpdateStatus() != AMP::LinearAlgebra::UpdateState::SETTING )
        ut->failure( "setValueByGlobalID leaves vector in UpdateState::SETTING or "
                     "UpdateState::LOCAL_CHANGED state " +
                     d_factory->name() );

    // Perform a makeConsistent SET and check the result
    vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    if ( vector->getUpdateStatus() != AMP::LinearAlgebra::UpdateState::UNCHANGED )
        ut->failure( "makeConsistent leaves vector in UpdateState::UNCHANGED state " +
                     d_factory->name() );
    if ( vector->getGhostSize() > 0 ) {
        auto comm_list = vector->getCommunicationList();
        std::vector<double> ghostList( vector->getGhostSize() );
        auto ghostIDList = comm_list->getGhostIDList();
        vector->getValuesByGlobalID(
            vector->getGhostSize(), (size_t *) &( ghostIDList[0] ), &( ghostList[0] ) );
        bool testPassed = true;
        for ( size_t i = 0; i != vector->getGhostSize(); i++ ) {
            if ( fabs( ghostList[i] - (double) ( ghostIDList[i] ) ) > 0.0000001 )
                testPassed = false;
        }
        PASS_FAIL( testPassed, "ghost set correctly in vector" );
    }
    if ( vector->getGhostSize() > 0 ) {
        auto comm_list   = vector->getCommunicationList();
        auto ghostIDList = comm_list->getGhostIDList();
        bool testPassed  = true;
        for ( size_t i = 0; i != vector->getGhostSize(); i++ ) {
            size_t ghostNdx = ghostIDList[i];
            double ghostVal = vector->getValueByGlobalID( ghostNdx );
            if ( fabs( ghostVal - (double) ghostNdx ) > 0.0000001 )
                testPassed = false;
        }
        PASS_FAIL( testPassed, "ghost set correctly in alias " );
    }
}


// Test creating a multivector with multiple copies of the data
// This should always return one copy of the superset of the data
void VectorTests::TestMultivectorDuplicate( AMP::UnitTest *ut )
{
    PROFILE( "TestMultivectorDuplicate" );
    auto vec0 = d_factory->getVector();
    // Create a multivector
    auto var      = std::make_shared<AMP::LinearAlgebra::Variable>( "multivec" );
    auto multiVec = AMP::LinearAlgebra::MultiVector::create( var, vec0->getComm() );
    // Add different views of vec0
    multiVec->addVector( vec0 );
    multiVec->addVector( vec0 );
    multiVec->addVector( multiVec->getVector( 0 ) );
    auto var2 = std::make_shared<AMP::LinearAlgebra::Variable>( "vec2" );
    multiVec->addVector( AMP::LinearAlgebra::MultiVector::create( var2, vec0->getComm() ) );
    // Verify the size of the multivector
    auto dof1 = vec0->getDOFManager();
    auto dof2 = multiVec->getDOFManager();
    bool pass = dof1->numLocalDOF() == dof2->numLocalDOF() &&
                dof1->numGlobalDOF() == dof2->numGlobalDOF() &&
                dof1->beginDOF() == dof2->beginDOF();
    PASS_FAIL( pass, "multivector resolves multiple copies of a vector" );
}


// Test containsGlobalElement()
void VectorTests::TestContainsGlobalElement( AMP::UnitTest *ut )
{
    PROFILE( "TestContainsGlobalElement" );
    auto vec = d_factory->getVector();
    auto DOF = vec->getDOFManager();
    std::set<size_t> dofs;
    for ( auto dof : DOF->getRemoteDOFs() )
        dofs.insert( dof );
    for ( size_t i = DOF->beginDOF(); i < DOF->endDOF(); i++ )
        dofs.insert( i );
    bool pass = true;
    for ( size_t i = 0; i < vec->getGlobalSize(); i++ ) {
        bool available = dofs.find( i ) != dofs.end();
        pass           = pass && ( vec->containsGlobalElement( i ) == available );
    }
    PASS_FAIL( pass, "containsGlobalElement" );
}


} // namespace AMP::LinearAlgebra
