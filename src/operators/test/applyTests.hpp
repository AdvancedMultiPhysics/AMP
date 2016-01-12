#include "applyTests.h"
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/Utilities.h"
#include "utils/shared_ptr.h"
#include <exception>
#include <utils/Castable.h>

void adjust( AMP::LinearAlgebra::Vector::shared_ptr vec,
             const double *shift,
             const double *scale,
             const size_t nshift )
{
    if ( not vec->isA<AMP::LinearAlgebra::MultiVector>() ) {
        vec->scale( scale[0] );
        vec->addScalar( vec, shift[0] );
    } else {
        AMP::shared_ptr<AMP::LinearAlgebra::MultiVector> mvec =
            AMP::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( vec );
        size_t nvecs = mvec->getNumberOfSubvectors();
        AMP_INSIST( nshift <= nvecs, "not enough subvectors" );
        for ( size_t i = 0; i < nshift; i++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr subvec = mvec->getVector( i );
            subvec->scale( scale[i] );
            subvec->addScalar( subvec, shift[i] );
        }
    }
    vec->makeConsistent( AMP::LinearAlgebra::Vector::CONSISTENT_SET );
}

void adjust( AMP::LinearAlgebra::Vector::shared_ptr vec, const double shift, const double scale )
{
    adjust( vec, &shift, &scale, 1 );
}

void applyTests( AMP::UnitTest *ut,
                 std::string msgPrefix,
                 AMP::shared_ptr<AMP::Operator::Operator>
                     testOperator,
                 AMP::LinearAlgebra::Vector::shared_ptr rhsVec,
                 AMP::LinearAlgebra::Vector::shared_ptr solVec,
                 AMP::LinearAlgebra::Vector::shared_ptr resVec,
                 const double shift,
                 const double scale )
{
    applyTests( ut, msgPrefix, testOperator, rhsVec, solVec, resVec, &shift, &scale, 1 );
}

void applyTests( AMP::UnitTest *ut,
                 std::string msgPrefix,
                 AMP::shared_ptr<AMP::Operator::Operator>
                     testOperator,
                 AMP::LinearAlgebra::Vector::shared_ptr rhsVec,
                 AMP::LinearAlgebra::Vector::shared_ptr solVec,
                 AMP::LinearAlgebra::Vector::shared_ptr resVec,
                 const double *shift,
                 const double *scale,
                 const size_t nshift )
{
    AMP::LinearAlgebra::Variable::shared_ptr testOperatorVariable =
        testOperator->getOutputVariable();
    AMP_ASSERT( testOperatorVariable.get() != nullptr );
    // first test for apply - random values in all three input vectors
    AMP::pout << "ApplyTest #1" << std::endl;
    bool passed = true;
    try {
        for ( int j = 0; j < 3; j++ ) {
            solVec->setRandomValues();
            rhsVec->setRandomValues();
            resVec->setRandomValues();
            adjust( solVec, shift, scale, nshift );
            testOperator->residual( rhsVec, solVec, resVec );
        } // end for j
    } catch ( std::exception ) {
        passed = false;
    }
    if ( passed ) {
        ut->passes( msgPrefix + " : apply with random f, u, r, a=1, b=-1.0" );
    } else {
        ut->failure( msgPrefix + " : apply with random f, u, r, a=1, b=-1.0" );
    }

    // second test for apply - f NULL, u, r, random values
    AMP::pout << "ApplyTest #2" << std::endl;
    passed = true;
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr fVec;
            solVec->setRandomValues();
            resVec->setRandomValues();
            adjust( solVec, shift, scale, nshift );
            testOperator->residual( fVec, solVec, resVec );
        } // end for j
    } catch ( std::exception ) {
        passed = false;
    }
    if ( passed ) {
        ut->passes( msgPrefix + " : apply with f NULL, random u, r, a=1, b=-1.0" );
    } else {
        ut->failure( msgPrefix + " : apply with f NULL, random u, r, a=1, b=-1.0" );
    }

    // R.S.: u is allowed to be NULL for some operators. For example, operators
    // with an in-place apply. However, this test is not meant to be used with those operators.
    // third test for apply - u NULL, f, r, random values
    AMP::pout << "ApplyTest #3" << std::endl;
    passed = false;
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr uVec;
            rhsVec->setRandomValues();
            resVec->setRandomValues();
            testOperator->residual( rhsVec, uVec, resVec );
        } // end for j
    } catch ( std::exception ) {
        passed = true;
    }
    if ( passed ) {
        ut->passes( msgPrefix +
                    " : apply with u NULL, random values in the vectors f,r, a=1, b=-1.0" );
    } else {
        ut->failure( msgPrefix +
                     " : apply with u NULL, random values in the vectors f,r, a=1, b=-1.0" );
    }

    // fourth test for apply - r NULL, f, u, random values
    AMP::pout << "ApplyTest #4" << std::endl;
    passed = false;
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr rVec;
            solVec->setRandomValues();
            rhsVec->setRandomValues();
            adjust( solVec, shift, scale, nshift );
            testOperator->residual( rhsVec, solVec, rVec );
        } // end for j
    } catch ( std::exception ) {
        passed = true;
    }
    if ( passed ) {
        ut->passes( msgPrefix +
                    " : apply with r NULL, random values in the vectors f,u, a=1, b=-1.0" );
    } else {
        ut->failure( msgPrefix +
                     " : apply with r NULL, random values in the vectors f,u, a=1, b=-1.0" );
    }

    // fifth test for apply - f NULL, u NULL, r, random values
    AMP::pout << "ApplyTest #5" << std::endl;
    passed = false;
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr fVec;
            AMP::LinearAlgebra::Vector::shared_ptr uVec;
            resVec->setRandomValues();
            testOperator->residual( fVec, uVec, resVec );
        } // end for j
    } catch ( std::exception ) {
        passed = true;
    }
    if ( passed ) {
        ut->passes( msgPrefix +
                    " : apply with f NULL, u NULL random values in the vector r, a=1, b=-1.0" );
    } else {
        ut->failure( msgPrefix +
                     " : apply with f NULL, u NULL random values in the vector r, a=1, b=-1.0" );
    }

    // sixth test for apply - u NULL, r NULL, f, random values
    AMP::pout << "ApplyTest #6" << std::endl;
    passed = false;
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr uVec;
            AMP::LinearAlgebra::Vector::shared_ptr rVec;
            rhsVec->setRandomValues();
            testOperator->residual( rhsVec, uVec, rVec );
        } // end for j
    } catch ( std::exception ) {
        passed = true;
    }
    if ( passed ) {
        ut->passes( msgPrefix +
                    " : apply with u NULL, r NULL, random values in the vector f, a=1, b=-1.0" );
    } else {
        ut->failure( msgPrefix +
                     " : apply with u NULL, r NULL, random values in the vector f, a=1, b=-1.0" );
    }

    // seventh test for apply - r NULL, f NULL, u random values
    AMP::pout << "ApplyTest #7" << std::endl;
    passed = false;
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr rVec;
            AMP::LinearAlgebra::Vector::shared_ptr fVec;
            solVec->setRandomValues();
            adjust( solVec, shift, scale, nshift );
            testOperator->residual( fVec, solVec, rVec );
        } // end for j
    } catch ( std::exception ) {
        passed = true;
    }
    if ( passed ) {
        ut->passes( msgPrefix +
                    " : apply with f, r NULL, random values in the vector u, a=1, b=-1.0" );
    } else {
        ut->failure( msgPrefix +
                     " : apply with f, r NULL, random values in the vector u, a=1, b=-1.0" );
    }

    // eighth test for apply - r NULL, f NULL, u NULL
    AMP::pout << "ApplyTest #8" << std::endl;
    passed = false;
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr rVec;
            AMP::LinearAlgebra::Vector::shared_ptr fVec;
            AMP::LinearAlgebra::Vector::shared_ptr uVec;
            testOperator->residual( fVec, uVec, rVec );
        } // end for j
    } catch ( std::exception ) {
        passed = true;
    }
    if ( passed ) {
        ut->passes( msgPrefix + " : apply with f, u, r NULL, a=1, b=-1.0" );
    } else {
        ut->failure( msgPrefix + " : apply with f, u, r NULL, a=1, b=-1.0" );
    }

#if 0
  // ninth test for apply - random values in all three input vectors, a=0, b=1
  AMP::pout<<"ApplyTest #9"<<std::endl; 
  rhsVec->setRandomValues();
  resVec->setRandomValues();
  solVec->setRandomValues();
  adjust(solVec, shift, scale, nshift);
  testOperator->apply(rhsVec, solVec, resVec, 0.0, 1.0);
  rhsVec->subtract(rhsVec, resVec);
  double norm = rhsVec->subsetVectorForVariable(testOperatorVariable)->L2Norm();
  if (AMP::Utilities::approx_equal(norm, 0.0)) {
    ut->passes(msgPrefix + " : apply with random values in the vectors f,u,r, a=0.0, b=1.0");
  } else {
    ut->failure(msgPrefix + " : apply with random values in the vectors f,u,r, a=0.0, b=1.0");
  }

  // tenth test for apply - random values in all three input vectors, a=0, b=-1, to test scaling
  AMP::pout<<"ApplyTest #10"<<std::endl; 
  rhsVec->setRandomValues();
  resVec->setRandomValues();
  solVec->setRandomValues();
  adjust(solVec, shift, scale, nshift);
  testOperator->apply(rhsVec, solVec, resVec, 0.0, -1.0);
  rhsVec->add(rhsVec, resVec);
  norm = rhsVec->subsetVectorForVariable(testOperatorVariable)->L2Norm();
  if (AMP::Utilities::approx_equal(norm, 0.0)) {
    ut->passes(msgPrefix + " : apply with random values in the vectors f,u,r, a=0.0, b=-1.0 (test scaling of f)");
  } else {
    ut->failure(msgPrefix + " : apply with random values in the vectors f,u,r, a=0.0, b=-1.0 (test scaling of f)");
  }
#endif

    // eleventh test for apply - f, u, r, random values, u random -negative values
    // RS:  It is not clear why this test is valid. For example, Displacements
    //     could either be negative or positive.
    //
    // GAD: This test is deferred until a reasonably automated valid range facility is established
    //     for operators. Such ranges will frequently depend on underlying material property valid
    //     ranges. So range reporting for material properties will be needed.
    //     Facilities will have to be devised to intersect ranges for compound operators.
    //     Alternatively, one could specify test ranges in the unit test input file and pass into
    //     this
    //     routine, but all the input files would have to be modified.
    //     Alternatively, one could specify test ranges in the calling program, but this is less
    //     desirable
    //     as many unit test authors do not know ahead of time what material or operator is
    //     requested in the
    //     input file.
    AMP::pout << "ApplyTest #11" << std::endl;
    if ( false ) {
        passed = false;
        try {
            for ( int j = 0; j < 3; j++ ) {
                solVec->setRandomValues();
                // introduce negative values
                solVec->scale( -1.0 );
                rhsVec->setRandomValues();
                resVec->setRandomValues();
                testOperator->residual( rhsVec, solVec, resVec );
            } // end for j
        } catch ( ... ) {
            ut->expected_failure(
                msgPrefix +
                " : apply with random negative values in u, random positive f,r, a=1, b=-1.0" );
            passed = true;
        }
        if ( not passed ) {
            ut->failure(
                msgPrefix +
                " : apply with random negative values in u, random positive f,r, a=1, b=-1.0" );
        }
    }
}
