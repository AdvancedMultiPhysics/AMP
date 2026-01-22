#include "AMP/IO/PIO.h"
#include "AMP/operators/mechanics/UpdatedLagrangianUtils.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"


using namespace AMP::Operator::UpdatedLagrangianUtils;


static void myTest( AMP::UnitTest *ut )
{

    double A[3][3] = { { -2, 2, -2 }, { -1, 1, 3 }, { 2, 0, -1 } };

    double R[3][3];
    double U[3][3];
    double F[3][3];
    double R2[3][3], U2[3][3], F2[3][3];
    double initStif[6][6], finalStif[6][6];

    double K = 100.0, G = 70.0;
    for ( int i = 0; i < 6; i++ ) {
        for ( int j = 0; j < 6; j++ ) {
            initStif[i][j]  = 0.0;
            finalStif[i][j] = 0.0;
        }
    }
    for ( int i = 0; i < 3; i++ ) {
        initStif[i][i] += ( 2.0 * G );
        initStif[i + 3][i + 3] += ( 2.0 * G );
    }
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            initStif[i][j] += ( K - ( 2.0 * G / 3.0 ) );
        }
    }

    polarDecomposeRU( A, R, U );
    matMatMultiply( R, U, F );
    polarDecompositionFeqRU_Simo( A, R2, U2 );
    matMatMultiply( R2, U2, F2 );

    pushforwardCorotationalStiffness( R2, initStif, finalStif );
    for ( int i = 0; i < 6; i++ ) {
        for ( int j = 0; j < 6; j++ ) {
            std::cout << "finalStif[" << i << "][" << j << "] = " << finalStif[i][j] << std::endl;
        }
    }

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            std::cout << "U[" << i << "][" << j << "]=" << U[i][j] << " U2[" << i << "][" << j
                      << "]=" << U2[i][j] << std::endl;
            std::cout << "R[" << i << "][" << j << "]=" << R[i][j] << " R2[" << i << "][" << j
                      << "]=" << R2[i][j] << std::endl;
        }
    }

    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
            std::cout << "F[" << i << "][" << j << "]=" << F[i][j] << " F2[" << i << "][" << j
                      << "]=" << F2[i][j] << std::endl;

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            if ( !softEquals( U[i][j], U[j][i] ) ) {
                ut->failure( "U is not symmetric." );
            }
        } // end for j
    } // end for i

    double eVal[3];
    eigenValues( U, eVal );
    for ( auto &elem : eVal ) {
        if ( elem <= 0 ) {
            ut->failure( "U is not positive definite." );
        }
    }

    double eValSum = eVal[0] + eVal[1] + eVal[2];
    double Utrace  = U[0][0] + U[1][1] + U[2][2];
    if ( !softEquals( eValSum, Utrace ) ) {
        ut->failure( "Trace(U) != sum of EigenValues." );
    }

    double eValProd = eVal[0] * eVal[1] * eVal[2];
    double detU     = matDeterminant( U );
    if ( !softEquals( eValProd, detU ) ) {
        ut->failure( "det(U) != product of EigenValues." );
    }

    double Rtran[3][3];
    matTranspose( R, Rtran );

    double RtranR[3][3];
    matMatMultiply( Rtran, R, RtranR );
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            if ( i == j ) {
                if ( !softEquals( RtranR[i][j], 1.0 ) ) {
                    ut->failure( "R is not orthogonal." );
                }
            } else {
                if ( !softEquals( RtranR[i][j], 0.0 ) ) {
                    ut->failure( "R is not orthogonal." );
                }
            }
        } // end for j
    } // end for i

    double RU[3][3];
    matMatMultiply( R, U, RU );
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            if ( !softEquals( RU[i][j], A[i][j] ) ) {
                ut->failure( "RU != A." );
            }
        } // end for j
    } // end for i

    ut->passes( "testStiffnessTransformation" );
}

int testStiffnessTransformation( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
