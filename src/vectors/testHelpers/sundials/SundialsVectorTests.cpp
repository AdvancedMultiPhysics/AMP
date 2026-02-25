#include "AMP/vectors/testHelpers/sundials/SundialsVectorTests.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/sundials/ManagedSundialsVector.h"
#include "AMP/vectors/sundials/SundialsVector.h"


#define PASS_FAIL( test, MSG )                                                    \
    do {                                                                          \
        if ( test )                                                               \
            ut->passes( d_factory->name() + " - " + __FUNCTION__ + ": " + MSG );  \
        else                                                                      \
            ut->failure( d_factory->name() + " - " + __FUNCTION__ + ": " + MSG ); \
    } while ( 0 )


namespace AMP::LinearAlgebra {


static inline N_Vector getVec( AMP::LinearAlgebra::Vector::shared_ptr vector )
{
    auto sundials = std::dynamic_pointer_cast<AMP::LinearAlgebra::SundialsVector>( vector );
    AMP_ASSERT( sundials );
    return sundials->getNVector();
}


void SundialsVectorTests::testSundialsVector( AMP::UnitTest *ut )
{
    CloneSundialsVector( ut );
    LinearSumSundialsVector( ut );
    ConstSundialsVector( ut );
    ProdSundialsVector( ut );
    DivSundialsVector( ut );
    ScaleSundialsVector( ut );
    AbsSundialsVector( ut );
    InvSundialsVector( ut );
    AddConstSundialsVector( ut );
    DotProdSundialsVector( ut );
    MaxNormSundialsVector( ut );
    WRMSNormSundialsVector( ut );
    L1NormSundialsVector( ut );
}


void SundialsVectorTests::CloneSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = N_VClone( vec_a );
        auto vectorb   = getVector( vec_b );
        bool pass      = true;
        for ( size_t i = 0; i != vectorb->numberOfDataBlocks(); i++ ) {
            if ( vectorb->getRawDataBlock<double>( i ) == vectora->getRawDataBlock<double>( i ) )
                pass = false;
        }
        PASS_FAIL( pass, "Clone created" );
        N_VDestroy( vec_b );
        ut->passes( "N_VDestroy returned" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}


void SundialsVectorTests::LinearSumSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();
        auto vectorc = d_factory->getVector();
        auto vectord = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );
        N_Vector vec_c = getVec( vectorc );

        vectora->setRandomValues();
        vectorb->setRandomValues();
        N_VLinearSum( .2, vec_a, .5, vec_b, vec_c );
        vectord->linearSum( .2, *vectora, .5, *vectorb );
        vectord->subtract( *vectord, *vectorc );
        PASS_FAIL( vectord->maxNorm() < 0.000001, "random linear sum" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}


void SundialsVectorTests::ConstSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        N_Vector vec_a = getVec( vectora );
        N_VConst( 0., vec_a );
        double maxNorm = static_cast<double>( vectora->maxNorm() );
        PASS_FAIL( maxNorm == 0, "Set vector to 0" );

        N_VConst( 1., vec_a );
        maxNorm       = static_cast<double>( vectora->maxNorm() );
        double L1Norm = static_cast<double>( vectora->L1Norm() );
        PASS_FAIL( ( maxNorm == 1. ) && ( L1Norm == (double) vectora->getGlobalSize() ),
                   "Set vector to 1" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}

void SundialsVectorTests::ProdSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();
        auto vectorc = d_factory->getVector();
        auto vectord = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );
        N_Vector vec_c = getVec( vectorc );

        vectora->setRandomValues();
        vectorb->setRandomValues();
        N_VProd( vec_a, vec_b, vec_c );
        vectord->multiply( *vectora, *vectorb );
        vectord->subtract( *vectorc, *vectord );
        double norm = static_cast<double>( vectord->maxNorm() );
        PASS_FAIL( norm < 0.000001, "Products match" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}


void SundialsVectorTests::DivSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();
        auto vectorc = d_factory->getVector();
        auto vectord = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );
        N_Vector vec_c = getVec( vectorc );

        vectora->setRandomValues();
        vectorb->setRandomValues();
        N_VDiv( vec_a, vec_b, vec_c );
        vectord->divide( *vectora, *vectorb );
        vectord->subtract( *vectorc, *vectord );
        PASS_FAIL( vectord->maxNorm() < 0.000001, "Quotients match" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}

void SundialsVectorTests::ScaleSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();
        auto vectorc = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );

        vectora->setRandomValues();
        N_VScale( 2.0, vec_a, vec_b );
        vectorc->scale( 2.0, *vectora );
        vectorc->subtract( *vectorc, *vectorb );
        PASS_FAIL( vectorc->maxNorm() < 0.000001, "Scalings match" );
    }
}

void SundialsVectorTests::AbsSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();
        auto vectorc = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );

        vectora->setRandomValues();
        vectorc->setToScalar( 0.5 );
        vectora->subtract( *vectora, *vectorc );
        N_VAbs( vec_a, vec_b );
        vectorc->abs( *vectora );
        vectorc->subtract( *vectorc, *vectorb );
        PASS_FAIL( vectorc->maxNorm() < 0.000001, "Values match" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}


void SundialsVectorTests::InvSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();
        auto vectorc = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );

        vectora->setRandomValues();
        N_VInv( vec_a, vec_b );
        vectorc->reciprocal( *vectora );
        vectorc->subtract( *vectorc, *vectorb );
        PASS_FAIL( vectorc->maxNorm() < 0.000001, "Scalings match" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}


void SundialsVectorTests::AddConstSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();
        auto vectorc = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );

        vectora->setRandomValues();
        N_VAddConst( vec_a, .3, vec_b );
        vectorc->addScalar( *vectora, .3 );
        vectorc->subtract( *vectorb, *vectorc );
        double norm = static_cast<double>( vectorc->maxNorm() );
        PASS_FAIL( norm < 0.00000001, "N_VAddConst" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}

void SundialsVectorTests::DotProdSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );

        vectora->setRandomValues();
        vectorb->setRandomValues();
        double d1 = N_VDotProd( vec_a, vec_b );
        double d2 = static_cast<double>( vectora->dot( *vectorb ) );
        PASS_FAIL( fabs( d1 - d2 ) < 0.00000001, "N_VDotProd" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}

void SundialsVectorTests::MaxNormSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {

        N_Vector vec_a = getVec( vectora );

        vectora->setRandomValues();

        double d1 = N_VMaxNorm( vec_a );
        double d2 = static_cast<double>( vectora->maxNorm() );
        PASS_FAIL( fabs( d1 - d2 ) < 0.00000001, "N_VMaxNorm" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}

void SundialsVectorTests::WRMSNormSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();
        auto vectorc = d_factory->getVector();
        if ( !vectorc )
            ut->failure( "N_VWrmsNorm" );

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );

        vectora->setRandomValues();
        vectorb->setRandomValues();

        double d1 = N_VWrmsNorm( vec_a, vec_b );
        double d2 = static_cast<double>( vectorb->wrmsNorm( *vectora, *vectorb ) );
        PASS_FAIL( fabs( d1 - d2 ) < 0.00000001, "N_VWrmsNorm" );
    }
}

void SundialsVectorTests::MinSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {

        N_Vector vec_a = getVec( vectora );

        vectora->setRandomValues();

        double d1 = N_VMin( vec_a );
        double d2 = static_cast<double>( vectora->min() );
        PASS_FAIL( fabs( d1 - d2 ) < 0.00000001, "N_VMin" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}

void SundialsVectorTests::L1NormSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {

        N_Vector vec_a = getVec( vectora );

        vectora->setRandomValues();

        double d1 = N_VL1Norm( vec_a );
        double d2 = static_cast<double>( vectora->L1Norm() );
        PASS_FAIL( fabs( d1 - d2 ) < 0.00000001, "N_VL1Norm" );
    }
}

void SundialsVectorTests::MinQuotientSundialsVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    if ( vectora->getVectorData()->isType<double>() ) {
        auto vectorb = d_factory->getVector();

        N_Vector vec_a = getVec( vectora );
        N_Vector vec_b = getVec( vectorb );

        vectora->setRandomValues();
        vectorb->setRandomValues();

        double d1 = N_VMinQuotient( vec_a, vec_b );
        double d2 = static_cast<double>( vectorb->minQuotient( *vectora ) );
        PASS_FAIL( fabs( d1 - d2 ) < 0.00000001, "N_VMinQuotient" );
    } else {
        ut->expected_failure( "Sundials tests currently only work for double" );
    }
}

} // namespace AMP::LinearAlgebra
