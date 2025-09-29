#include "AMP/vectors/testHelpers/trilinos/tpetra/TpetraVectorTests.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/testHelpers/trilinos/tpetra/TpetraVectorFactory.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVector.h"

#include <algorithm>
#include <string>


namespace AMP::LinearAlgebra {


#define PASS_FAIL( test, MSG )                                                    \
    do {                                                                          \
        if ( test )                                                               \
            ut->passes( d_factory->name() + " - " + __FUNCTION__ + ": " + MSG );  \
        else                                                                      \
            ut->failure( d_factory->name() + " - " + __FUNCTION__ + ": " + MSG ); \
    } while ( 0 )


void TpetraVectorTests::testTpetraVector( AMP::UnitTest *ut ) { VerifyNorms( ut ); }


void TpetraVectorTests::VerifyNorms( AMP::UnitTest *ut )
{
    auto vec  = d_factory->getVector();
    auto view = AMP::LinearAlgebra::TpetraVector::view( vec );
    auto &Vec = view->getTpetra_Vector();
    int NVec  = Vec.getNumVectors();
    AMP_ASSERT( NVec == 1 );

    double ans1, ans2;
    if ( vec->isType<double>( 0 ) ) {
        vec->setRandomValues();

        ans1 = static_cast<double>( vec->L1Norm() );
        ans2 = Vec.norm1();
        PASS_FAIL( fabs( ans1 - ans2 ) < 1e-12 * fabs( ans1 ), "Tpetra L1 norms match" );

        ans1 = static_cast<double>( vec->L2Norm() );
        ans2 = Vec.norm2();
        PASS_FAIL( fabs( ans1 - ans2 ) < 1e-12 * fabs( ans1 ), "Tpetra L2 norms match" );

        ans1 = static_cast<double>( vec->maxNorm() );
        ans2 = Vec.normInf();
        PASS_FAIL( fabs( ans1 - ans2 ) < 1e-12 * fabs( ans1 ), "Tpetra Inf norms match" );
    } else {
        ut->expected_failure( "Tpetra tests currently only work for double" );
    }
}


} // namespace AMP::LinearAlgebra
