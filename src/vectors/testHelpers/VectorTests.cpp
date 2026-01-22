#include "AMP/vectors/testHelpers/VectorTests.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorHelpers.h"
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


template<typename T>
static void both_VectorIteratorTests( Vector::shared_ptr p, AMP::UnitTest *utils )
{
    int kk = p->getLocalSize();
    if ( ( p->end<T>() - p->begin<T>() ) == (int) p->getLocalSize() )
        utils->passes( "Subtracting begin from end " );
    else
        utils->failure( "Subtracting begin from end " );

    if ( (int) ( p->begin<T>() - p->end<T>() ) == -(int) p->getLocalSize() )
        utils->passes( "Subtracting end from beginning " );
    else
        utils->failure( "Subtracting end from beginning " );

    auto cur1 = p->begin<T>();
    auto cur2 = p->begin<T>();
    auto end  = p->end<T>();
    ++cur1;
    ++cur2;
    int i = 0;
    while ( cur2 != end ) {
        if ( i == 10 )
            break;
        ++cur2;
        i++;
    }
    int tt = ( cur2 - cur1 );
    if ( i == tt )
        utils->passes( "Subtracting arbitrary iterators " );
    else
        utils->failure( "Subtracting arbitrary iterators " );

    p->setToScalar( 5.0 );
    i = 0;
    for ( cur1 = p->begin<T>(); cur1 != end; ++cur1 ) {
        if ( ( *cur1 ) != 5.0 )
            break;
        i++;
    }
    if ( i == (int) p->getLocalSize() )
        utils->passes( "Iterating data access " );
    else
        utils->failure( "Iterating data access" );

    cur1 = end;
    i    = 0;
    do {
        --cur1;
        if ( ( *cur1 ) != 5.0 )
            break;
        i++;
    } while ( cur1 != p->begin<T>() );

    if ( i == kk )
        utils->passes( "Iterating backward data access" );
    else
        utils->failure( "Iterating backward data access" );

    if ( p->getLocalSize() > 7 ) {
        cur1 = p->begin<T>();
        cur2 = cur1 + 5;
        if ( ( cur2 - cur1 ) == 5 )
            utils->passes( "Adding and subtracting" );
        else
            utils->failure( "Adding and subtracting" );
        i = 0;
        while ( cur2 != end ) {
            i++;
            ++cur2;
        }
        if ( i == ( (int) p->getLocalSize() - 5 ) )
            utils->passes( "Adding and iterating" );
        else
            utils->failure( "Adding and iterating" );

        cur1 += 5;
        i = 0;
        while ( cur1 != end ) {
            i++;
            ++cur1;
        }
        if ( i == ( (int) p->getLocalSize() - 5 ) )
            utils->passes( "Add-equal and iterating" );
        else
            utils->failure( "Add-equal and iterating" );
    }
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

    vec1->makeConsistent( ScatterType::CONSISTENT_SET );
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
bool vectorValuesNotEqual( Vector::shared_ptr vector, T val )
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
    vector->makeConsistent( ScatterType::CONSISTENT_SET );
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
    vector->makeConsistent( ScatterType::CONSISTENT_SET );
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
    auto vec = d_factory->getVector();
    vec->setToScalar( 1. );
    auto norm  = static_cast<double>( vec->L2Norm() );
    auto norm2 = static_cast<double>( vec->dot( *vec ) );
    double tol = 0.000001;
    if ( vec->getVectorData()->isType<float>() )
        tol = 0.00001;
    PASS_FAIL( fabs( static_cast<double>( norm * norm - norm2 ) ) < tol, "L2 norm 1" );
    vec->setRandomValues();
    norm  = static_cast<double>( vec->L2Norm() );
    norm2 = static_cast<double>( vec->dot( *vec ) );
    PASS_FAIL( fabs( static_cast<double>( norm * norm - norm2 ) ) < tol, "L2 norm 2" );
    norm      = static_cast<double>( vec->L2Norm() );
    norm2     = static_cast<double>( VectorHelpers::L2Norm( vec, { vec->getName() } )[0] );
    bool pass = fabs( norm - norm2 ) < tol;
    if ( !std::dynamic_pointer_cast<MultiVector>( vec ) ) {
        auto vec2 = vec->clone();
        vec2->setRandomValues();
        vec->setName( "a" );
        vec2->setName( "b" );
        norm2         = static_cast<double>( vec2->L2Norm() );
        auto multivec = MultiVector::create(
            std::make_shared<Variable>( "multivec" ), vec->getComm(), { vec, vec2 } );
        auto norms  = VectorHelpers::L2Norm( multivec, { "a", "b", "c" } );
        double n[3] = { static_cast<double>( norms[0] ),
                        static_cast<double>( norms[1] ),
                        static_cast<double>( norms[2] ) };
        pass        = pass && fabs( n[0] - norm ) < tol;
        pass        = pass && fabs( n[1] - norm2 ) < tol;
        pass        = pass && n[2] == 0;
    }
    PASS_FAIL( pass, "VectorHelpers" );
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
    vec2->makeConsistent( ScatterType::CONSISTENT_SET );
    PASS_FAIL( vec1->equals( *vec2 ), "Abs" );
}


void VectorTests::L1NormVector( AMP::UnitTest *ut )
{
    PROFILE( "L1NormVector" );
    auto vec = d_factory->getVector();
    auto vec1( d_factory->getVector() );
    vec->setRandomValues();
    vec1->setToScalar( 1. );
    auto norm = static_cast<double>( vec->L1Norm() );
    vec->abs( *vec );
    auto norm2 = static_cast<double>( vec->dot( *vec1 ) );
    double tol = 50 * norm * getTol( *vec );
    if ( fabs( norm - norm2 ) < tol )
        ut->passes( "L1 norm" );
    else
        ut->failure( AMP::Utilities::stringf( "L1 norm (%e) (%e)", fabs( norm - norm2 ), tol ) );
    norm      = static_cast<double>( vec->L1Norm() );
    norm2     = static_cast<double>( VectorHelpers::L1Norm( vec, { vec->getName() } )[0] );
    bool pass = fabs( norm - norm2 ) < tol;
    if ( !std::dynamic_pointer_cast<MultiVector>( vec ) ) {
        auto vec2 = vec->clone();
        vec2->setRandomValues();
        vec->setName( "a" );
        vec2->setName( "b" );
        norm2         = static_cast<double>( vec2->L1Norm() );
        auto multivec = MultiVector::create(
            std::make_shared<Variable>( "multivec" ), vec->getComm(), { vec, vec2 } );
        auto norms  = VectorHelpers::L1Norm( multivec, { "a", "b", "c" } );
        double n[3] = { static_cast<double>( norms[0] ),
                        static_cast<double>( norms[1] ),
                        static_cast<double>( norms[2] ) };
        pass        = pass && fabs( n[0] - norm ) < tol;
        pass        = pass && fabs( n[1] - norm2 ) < tol;
        pass        = pass && n[2] == 0;
    }
    PASS_FAIL( pass, "VectorHelpers" );
}


void VectorTests::MaxNormVector( AMP::UnitTest *ut )
{
    PROFILE( "MaxNormVector" );
    auto vec = d_factory->getVector();
    vec->setRandomValues();
    auto infNorm = vec->maxNorm();
    vec->abs( *vec );
    if ( vec->getVectorData()->isType<double>() ) {
        auto curData   = vec->begin();
        auto endData   = vec->end();
        auto local_ans = *curData;
        while ( curData != endData ) {
            local_ans = std::max( local_ans, *curData );
            ++curData;
        }
        auto global_ans = vec->getComm().maxReduce( local_ans );
        PASS_FAIL( global_ans == infNorm, "Inf norm" );
    } else if ( vec->getVectorData()->isType<float>() ) {
        auto curData   = vec->begin<float>();
        auto endData   = vec->end<float>();
        auto local_ans = *curData;
        while ( curData != endData ) {
            local_ans = std::max( local_ans, *curData );
            ++curData;
        }
        auto global_ans = vec->getComm().maxReduce( local_ans );
        PASS_FAIL( global_ans == infNorm, "Inf norm" );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
    auto norm  = static_cast<double>( vec->maxNorm() );
    auto norm2 = static_cast<double>( VectorHelpers::maxNorm( vec, { vec->getName() } )[0] );
    double tol = 2 * getTol( *vec );
    bool pass  = fabs( norm - norm2 ) < tol;
    if ( !std::dynamic_pointer_cast<MultiVector>( vec ) ) {
        auto vec2 = vec->clone();
        vec2->setRandomValues();
        vec->setName( "a" );
        vec2->setName( "b" );
        norm2         = static_cast<double>( vec2->maxNorm() );
        auto multivec = MultiVector::create(
            std::make_shared<Variable>( "multivec" ), vec->getComm(), { vec, vec2 } );
        auto norms  = VectorHelpers::maxNorm( multivec, { "a", "b", "c" } );
        double n[3] = { static_cast<double>( norms[0] ),
                        static_cast<double>( norms[1] ),
                        static_cast<double>( norms[2] ) };
        pass        = pass && fabs( n[0] - norm ) < tol;
        pass        = pass && fabs( n[1] - norm2 ) < tol;
        pass        = pass && n[2] == 0;
    }
    PASS_FAIL( pass, "VectorHelpers" );
}

template<typename TYPE>
bool scaleTest( Vector::shared_ptr vector1, Vector::shared_ptr vector2, TYPE beta )
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
    vector1->makeConsistent( ScatterType::CONSISTENT_SET );
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
    vector1->makeConsistent( ScatterType::CONSISTENT_SET );
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
        auto managed_petsc = PetscVector::view( vector1 );
        auto petsc_vec     = std::dynamic_pointer_cast<PetscVector>( managed_petsc );
        Vec managed_vec    = petsc_vec->getVec();


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
bool binaryOpTest( Vector::shared_ptr vector1,
                   Vector::shared_ptr vector2,
                   Vector::shared_ptr vector3,
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
    vector3->makeConsistent( ScatterType::CONSISTENT_SET );
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
    vector3->makeConsistent( ScatterType::CONSISTENT_SET );
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
    vector4->makeConsistent( ScatterType::CONSISTENT_SET );
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
    vector3->makeConsistent( ScatterType::CONSISTENT_SET );
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
    vector3->makeConsistent( ScatterType::CONSISTENT_SET );
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
        both_VectorIteratorTests<double>( vector1, ut );
        both_VectorIteratorTests<const double>( vector1, ut );
    } else if ( vector1->getVectorData()->isType<float>() ) {
        both_VectorIteratorTests<float>( vector1, ut );
        both_VectorIteratorTests<const float>( vector1, ut );
    } else {
        AMP_ERROR( "VectorIteratorTests not implemented for provided scalar TYPE" );
    }
}


void VectorTests::VerifyVectorSum( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorSum" );
    auto vec = d_factory->getVector();
    vec->setRandomValues();
    double sum( vec->sum() );
    double norm( vec->L1Norm() );
    PASS_FAIL( fabs( sum - norm ) < 1.e-10 * norm, "sum matches L1 norm" );
}


void VectorTests::VerifyVectorMin( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorMin" );
    auto vec = d_factory->getVector();
    vec->setRandomValues();
    vec->scale( -1.0 ); // make negative
    double min( vec->min() );
    double norm( vec->maxNorm() );
    PASS_FAIL( fabs( min + norm ) < 1.e-10 * norm, "minimum of negative vector == ||.||_infty" );
    vec->setMin( -0.5 );
    min = static_cast<double>( vec->min() );
    PASS_FAIL( min >= -0.5, "setMin" );
}


void VectorTests::VerifyVectorMax( AMP::UnitTest *ut )
{
    PROFILE( "VerifyVectorMax" );
    auto vec = d_factory->getVector();
    vec->setRandomValues();
    double max( vec->max() );
    double norm( vec->maxNorm() );
    PASS_FAIL( fabs( max - norm ) < 1.e-10 * norm, "maximum of positive vector == ||.||_infty" );
    vec->setMax( 0.5 );
    max = static_cast<double>( vec->max() );
    PASS_FAIL( max <= 0.5, "setMax" );
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
    vectord->makeConsistent( ScatterType::CONSISTENT_SET );
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
    vectord->makeConsistent( ScatterType::CONSISTENT_SET );
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
    vectorc->makeConsistent( ScatterType::CONSISTENT_SET );
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
                            Vector::shared_ptr vectora,
                            Vector::shared_ptr vectorb,
                            Vector::shared_ptr vectorb1,
                            Vector::shared_ptr vectorc,
                            Vector::shared_ptr vectord )
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

    auto simple1 = createSimpleVector<double>(
        vectora->getLocalSize(), vectora->getVariable(), vectora->getComm() );
    auto simple2 = createSimpleVector<float>(
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
    using UpdateState = UpdateState;
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
        vector->setValuesByGlobalID( 1, &i, &val );
        vector->addValuesByGlobalID( 1, &i, &val );
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
    vector->makeConsistent( ScatterType::CONSISTENT_ADD );
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
    if ( vector->getUpdateStatus() != UpdateState::UNCHANGED )
        ut->failure( "zero leaves vector in UpdateState::UNCHANGED state " + d_factory->name() );

    // Set and add local values by global id (this should not interfere with the add)
    const double val = 0.0;
    for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
        vector->setValuesByGlobalID( 1, &i, &val );
        vector->addValuesByGlobalID( 1, &i, &val );
    }
    if ( vector->getUpdateStatus() != UpdateState::LOCAL_CHANGED )
        ut->failure( "local set/add leaves vector in UpdateState::LOCAL_CHANGED state " +
                     d_factory->name() + " - " + vector->type() );

    // Set values by global id
    for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
        const auto val = double( i );
        vector->setValuesByGlobalID( 1, &i, &val );
    }
    if ( vector->getUpdateStatus() != UpdateState::LOCAL_CHANGED &&
         vector->getUpdateStatus() != UpdateState::SETTING )
        ut->failure( "setValueByGlobalID leaves vector in UpdateState::SETTING or "
                     "UpdateState::LOCAL_CHANGED state " +
                     d_factory->name() );

    // Perform a makeConsistent SET and check the result
    vector->makeConsistent( ScatterType::CONSISTENT_SET );
    if ( vector->getUpdateStatus() != UpdateState::UNCHANGED )
        ut->failure( "makeConsistent leaves vector in UpdateState::UNCHANGED state " +
                     d_factory->name() );
    if ( vector->getGhostSize() > 0 ) {
        auto comm_list = vector->getCommunicationList();
        std::vector<double> ghostList( vector->getGhostSize() );
        auto ghostIDList = comm_list->getGhostIDList();
        vector->getValuesByGlobalID(
            vector->getGhostSize(), (size_t *) &( ghostIDList[0] ), &( ghostList[0] ) );
        bool pass = true;
        for ( size_t i = 0; i != vector->getGhostSize(); i++ ) {
            if ( fabs( ghostList[i] - (double) ( ghostIDList[i] ) ) > 0.0000001 )
                pass = false;
        }
        PASS_FAIL( pass, "ghost set correctly in vector" );
    }
    if ( vector->getGhostSize() > 0 ) {
        auto comm_list   = vector->getCommunicationList();
        auto ghostIDList = comm_list->getGhostIDList();
        bool pass        = true;
        for ( size_t i = 0; i != vector->getGhostSize(); i++ ) {
            size_t ghostNdx = ghostIDList[i];
            double ghostVal = vector->getValueByGlobalID( ghostNdx );
            if ( fabs( ghostVal - (double) ghostNdx ) > 0.0000001 )
                pass = false;
        }
        PASS_FAIL( pass, "ghost set correctly in alias" );
    }
    if ( vector->getGhostSize() > 0 ) {
        auto comm_list   = vector->getCommunicationList();
        auto ghostIDList = comm_list->getGhostIDList();
        size_t N         = vector->getGhostSize();
        std::vector<double> ghost1( N, -1.0 );
        std::vector<double> ghost2( N, -1.0 );
        vector->getGhostValuesByGlobalID( N, ghostIDList.data(), ghost1.data() );
        size_t N2 = vector->getVectorData()->getAllGhostValues( ghost2.data() );
        bool pass = N == N2;
        for ( size_t i = 0; i != N; i++ )
            pass = pass && ghost1[i] == ghost2[i];
        PASS_FAIL( pass, "VectorData::getAllGhostValues" );
    }
}


// Test creating a multivector with multiple copies of the data
// This should always return one copy of the superset of the data
void VectorTests::TestMultivectorDuplicate( AMP::UnitTest *ut )
{
    PROFILE( "TestMultivectorDuplicate" );
    auto vec0 = d_factory->getVector();
    // Create a multivector
    auto var      = std::make_shared<Variable>( "multivec" );
    auto multiVec = MultiVector::create( var, vec0->getComm() );
    // Add different views of vec0
    multiVec->addVector( vec0 );
    multiVec->addVector( vec0 );
    multiVec->addVector( multiVec->getVector( 0 ) );
    auto var2 = std::make_shared<Variable>( "vec2" );
    multiVec->addVector( MultiVector::create( var2, vec0->getComm() ) );
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
