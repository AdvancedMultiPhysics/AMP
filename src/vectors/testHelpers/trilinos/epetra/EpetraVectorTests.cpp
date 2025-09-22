#include "AMP/vectors/testHelpers/trilinos/epetra/EpetraVectorTests.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/testHelpers/trilinos/epetra/EpetraVectorFactory.h"
#include "AMP/vectors/trilinos/epetra/EpetraVector.h"

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


void EpetraVectorTests::testEpetraVector( AMP::UnitTest *ut ) { VerifyNorms( ut ); }


void EpetraVectorTests::VerifyNorms( AMP::UnitTest *ut )
{
    auto vec  = d_factory->getVector();
    auto view = AMP::LinearAlgebra::EpetraVector::view( vec );
    auto &Vec = view->getEpetra_Vector();
    int NVec  = Vec.NumVectors();
    AMP_ASSERT( NVec == 1 );

    double ans1, ans2;
    if ( vec->isType<double>( 0 ) ) {
        vec->setRandomValues();

        ans1 = static_cast<double>( vec->L1Norm() );
        Vec.Norm1( &ans2 );
        PASS_FAIL( fabs( ans1 - ans2 ) < 1e-12 * fabs( ans1 ), "Epetra L1 norms match" );

        ans1 = static_cast<double>( vec->L2Norm() );
        Vec.Norm2( &ans2 );
        PASS_FAIL( fabs( ans1 - ans2 ) < 1e-12 * fabs( ans1 ), "Epetra L2 norms match" );

        ans1 = static_cast<double>( vec->maxNorm() );
        Vec.NormInf( &ans2 );
        PASS_FAIL( fabs( ans1 - ans2 ) < 1e-12 * fabs( ans1 ), "Epetra Inf norms match" );
    } else {
        ut->expected_failure( "Epetra tests currently only work for double" );
    }
}


} // namespace AMP::LinearAlgebra


