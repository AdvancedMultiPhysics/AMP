#include "AMP/operators/mechanics/UpdatedLagrangianUtils.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <cmath>
#include <iostream>


namespace AMP::Operator {

void polarDecomposeRU( double A[3][3], double R[3][3], double U[3][3] )
{
    double Atran[3][3];
    matTranspose( A, Atran );

    double Usquare[3][3];
    matMatMultiply( Atran, A, Usquare );

    double eVal[3];
    eigenValues( Usquare, eVal );

    double eVec[3][3];
    eigenVectors( Usquare, eVal, eVec );

    // Transpose(eVec) = Inverse(eVec)
    double eVecInv[3][3];
    matTranspose( eVec, eVecInv );

    double eValSqrt[3];
    vecSqrt( eVal, eValSqrt );

    matDiagMatMultiply( eVec, eValSqrt, eVecInv, U );

    double eValSqrtInv[3];
    vecInv( eValSqrt, eValSqrtInv );

    double Uinv[3][3];
    matDiagMatMultiply( eVec, eValSqrtInv, eVecInv, Uinv );

    matMatMultiply( A, Uinv, R );
}

void matDiagMatMultiply( double P1[3][3], double D[3], double P2[3][3], double A[3][3] )
{
    double Tmp[3][3];

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            Tmp[i][j] = D[i] * P2[i][j];
        }
    }

    matMatMultiply( P1, Tmp, A );
}

void eigenValues( double A[3][3], double val[3] )
{
    double a = 1.0;
    double b = -matTrace( A );
    double d = -matDeterminant( A );
    double Asquare[3][3];
    matMatMultiply( A, A, Asquare );
    double c = 0.5 * ( ( b * b ) - matTrace( Asquare ) );

    cubicRoots( a, b, c, d, val[0], val[1], val[2] );

    std::sort( val, val + 3 );
}

void eigenVectors( double A[3][3], double val[3], double vec[3][3] )
{
    std::vector<double> uniqVals;
    uniqVals.push_back( val[0] );
    for ( int i = 1; i < 3; i++ ) {
        if ( !softEquals( val[i], uniqVals[uniqVals.size() - 1] ) ) {
            uniqVals.push_back( val[i] );
        }
    } // end for i

    if ( uniqVals.size() == 1 ) {
        for ( int i = 0; i < 3; i++ ) {
            vec[i][i] = 1.0;
            for ( int j = i + 1; j < 3; j++ ) {
                vec[i][j] = 0.0;
                vec[j][i] = 0.0;
            } // end for j
        }     // end for i
    } else {
        int vecColCnt = 0;
        for ( auto &uniqVal : uniqVals ) {
            double B[3][3];
            matCopy( A, B );

            for ( int j = 0; j < 3; j++ ) {
                B[j][j] = B[j][j] - uniqVal;
            } // end for j

            std::vector<std::vector<double>> sols;
            solveEquation( B, sols );

            for ( auto &sol : sols ) {
                AMP_ASSERT( vecColCnt < 3 );
                for ( int j = 0; j < 3; j++ ) {
                    vec[j][vecColCnt] = sol[j];
                } // end for j
                vecColCnt++;
            } // end for k
        }     // end for i
        AMP_ASSERT( vecColCnt == 3 );
    }
}

void solveEquation( double A[3][3], std::vector<std::vector<double>> &sols )
{

    if ( softEquals( A[0][1], 0 ) ) {
        if ( softEquals( A[0][2], 0 ) ) {
            if ( softEquals( A[1][1], 0 ) ) {
                if ( softEquals( A[1][2], 0 ) ) {
                    if ( softEquals( A[2][1], 0 ) ) {
                        if ( softEquals( A[2][2], 0 ) ) {
                            std::vector<double> tmp( 3 );
                            tmp[0] = 0;
                            tmp[1] = 1;
                            tmp[2] = 0;
                            sols.push_back( tmp );
                            tmp[1] = 0;
                            tmp[2] = 1;
                            sols.push_back( tmp );
                        } else {
                            std::vector<double> tmp( 3 );
                            tmp[0] = 0;
                            tmp[1] = 1;
                            tmp[2] = 0;
                            sols.push_back( tmp );
                        } // A22
                    } else {
                        std::vector<double> tmp( 3 );
                        tmp[0] = 0;
                        tmp[1] = -A[2][2] / A[2][1];
                        tmp[2] = 1;
                        sols.push_back( tmp );
                    } // A21
                } else {
                    if ( softEquals( A[2][1], 0 ) ) {
                        std::vector<double> tmp( 3 );
                        tmp[0] = 0;
                        tmp[1] = 1;
                        tmp[2] = 0;
                        sols.push_back( tmp );
                    } // A21
                }     // A12
            } else {
                if ( softEquals( ( A[1][1] * A[2][2] ), ( A[1][2] * A[2][1] ) ) ) {
                    std::vector<double> tmp( 3 );
                    tmp[0] = 0;
                    tmp[1] = -A[1][2] / A[1][1];
                    tmp[2] = 1;
                    sols.push_back( tmp );
                }
            } // A11
        } else {
            if ( softEquals( A[1][1], 0 ) ) {
                if ( softEquals( A[2][1], 0 ) ) {
                    std::vector<double> tmp( 3 );
                    tmp[0] = 0;
                    tmp[1] = 1;
                    tmp[2] = 0;
                    sols.push_back( tmp );
                } // A21
            }     // A11
        }         // A02
    } else {
        if ( softEquals( ( A[1][2] * A[0][1] ), ( A[1][1] * A[0][2] ) ) ) {
            if ( softEquals( ( A[2][2] * A[0][1] ), ( A[2][1] * A[0][2] ) ) ) {
                std::vector<double> tmp( 3 );
                tmp[0] = 0;
                tmp[1] = -A[0][2] / A[0][1];
                tmp[2] = 1;
                sols.push_back( tmp );
            }
        }
    } // A01

    if ( softEquals( A[0][1], 0 ) ) {
        if ( softEquals( A[0][2], 0 ) ) {
            if ( softEquals( A[0][0], 0 ) ) {
                if ( softEquals( A[1][1], 0 ) ) {
                    if ( softEquals( A[1][2], 0 ) ) {
                        if ( softEquals( A[1][0], 0 ) ) {
                            if ( softEquals( A[2][1], 0 ) ) {
                                if ( softEquals( A[2][2], 0 ) ) {
                                    if ( softEquals( A[2][0], 0 ) ) {
                                        std::vector<double> tmp( 3 );
                                        tmp[0] = 1;
                                        tmp[1] = 0;
                                        tmp[2] = 0;
                                        sols.push_back( tmp );
                                        tmp[1] = 0;
                                        tmp[2] = 1;
                                        sols.push_back( tmp );
                                        tmp[1] = 1;
                                        tmp[2] = 0;
                                        sols.push_back( tmp );
                                    } // A20
                                } else {
                                    std::vector<double> tmp( 3 );
                                    tmp[0] = 1;
                                    tmp[1] = 0;
                                    tmp[2] = -A[2][0] / A[2][2];
                                    sols.push_back( tmp );
                                    tmp[1] = 1;
                                    sols.push_back( tmp );
                                } // A22
                            } else {
                                std::vector<double> tmp( 3 );
                                tmp[0] = 1;
                                tmp[2] = 0;
                                tmp[1] = -( A[2][0] + ( A[2][2] * tmp[2] ) ) / A[2][1];
                                sols.push_back( tmp );
                                tmp[2] = 1;
                                tmp[1] = -( A[2][0] + ( A[2][2] * tmp[2] ) ) / A[2][1];
                                sols.push_back( tmp );
                            } // A21
                        }     // A10
                    } else {
                        if ( softEquals( A[2][1], 0 ) ) {
                            if ( softEquals( ( A[2][2] * A[1][0] ), ( A[2][0] * A[1][2] ) ) ) {
                                std::vector<double> tmp( 3 );
                                tmp[0] = 1;
                                tmp[1] = 0;
                                tmp[2] = -A[1][0] / A[1][2];
                                sols.push_back( tmp );
                                tmp[1] = 1;
                                sols.push_back( tmp );
                            }
                        } else {
                            std::vector<double> tmp( 3 );
                            tmp[0] = 1;
                            tmp[1] = ( ( A[2][2] * A[1][0] ) - ( A[2][0] * A[1][2] ) ) /
                                     ( A[1][2] * A[2][1] );
                            tmp[2] = -A[1][0] / A[1][2];
                            sols.push_back( tmp );
                        } // A21
                    }     // A12
                } else {
                    if ( softEquals( ( A[1][1] * A[2][2] ), ( A[2][1] * A[1][2] ) ) ) {
                        if ( softEquals( ( A[1][1] * A[2][0] ), ( A[2][1] * A[1][0] ) ) ) {
                            std::vector<double> tmp( 3 );
                            tmp[0] = 1;
                            tmp[2] = 0;
                            tmp[1] = -( A[1][0] + ( A[1][2] * tmp[2] ) ) / A[1][1];
                            sols.push_back( tmp );
                            tmp[2] = 1;
                            tmp[1] = -( A[1][0] + ( A[1][2] * tmp[2] ) ) / A[1][1];
                            sols.push_back( tmp );
                        }
                    } else {
                        std::vector<double> tmp( 3 );
                        tmp[0] = 1;
                        tmp[2] = ( ( A[2][1] * A[1][0] ) - ( A[2][0] * A[1][1] ) ) /
                                 ( ( A[2][2] * A[1][1] ) - ( A[2][1] * A[1][2] ) );
                        tmp[1] = -( A[1][0] + ( A[1][2] * tmp[2] ) ) / A[1][1];
                        sols.push_back( tmp );
                    }
                } // A11
            }     // A00
        } else {
            if ( softEquals( A[1][1], 0 ) ) {
                std::vector<double> tmp( 3 );
                tmp[0] = 1;
                tmp[2] = -A[0][0] / A[0][2];
                if ( softEquals( ( A[1][2] * tmp[2] ), -A[1][0] ) ) {
                    if ( softEquals( A[2][1], 0 ) ) {
                        if ( softEquals( ( A[2][2] * tmp[2] ), -A[2][0] ) ) {
                            tmp[1] = 0;
                            sols.push_back( tmp );
                            tmp[1] = 1;
                            sols.push_back( tmp );
                        }
                    } else {
                        tmp[1] = -( A[2][0] + ( A[2][2] * tmp[2] ) ) / A[2][1];
                        sols.push_back( tmp );
                    } // A21
                }
            } else {
                std::vector<double> tmp( 3 );
                tmp[0] = 1;
                tmp[2] = -A[0][0] / A[0][2];
                tmp[1] = -( A[1][0] + ( A[1][2] * tmp[2] ) ) / A[1][1];
                if ( softEquals( ( ( A[2][1] * tmp[1] ) + ( A[2][2] * tmp[2] ) ), -A[2][0] ) ) {
                    sols.push_back( tmp );
                }
            } // A11
        }     // A02
    } else {
        std::vector<double> tmp( 3 );
        tmp[0]   = 1;
        double b = A[1][2] - ( A[1][1] * A[0][2] / A[0][1] );
        double c = ( A[0][0] * A[1][1] / A[0][1] ) - A[1][0];
        double d = A[2][2] - ( A[2][1] * A[0][2] / A[0][1] );
        double e = ( A[0][0] * A[2][1] / A[0][1] ) - A[2][0];
        if ( softEquals( b, 0 ) ) {
            if ( softEquals( c, 0 ) ) {
                if ( softEquals( d, 0 ) ) {
                    if ( softEquals( e, 0 ) ) {
                        tmp[2] = 0;
                        tmp[1] = -( ( A[0][2] * tmp[2] ) + A[0][0] ) / A[0][1];
                        sols.push_back( tmp );
                        tmp[2] = 1;
                        tmp[1] = -( ( A[0][2] * tmp[2] ) + A[0][0] ) / A[0][1];
                        sols.push_back( tmp );
                    }
                } else {
                    tmp[2] = e / d;
                    tmp[1] = -( ( A[0][2] * tmp[2] ) + A[0][0] ) / A[0][1];
                    sols.push_back( tmp );
                }
            }
        } else {
            tmp[2] = c / b;
            tmp[1] = -( ( A[0][2] * tmp[2] ) + A[0][0] ) / A[0][1];
            if ( softEquals( ( d * tmp[2] ), e ) ) {
                sols.push_back( tmp );
            }
        }
    } // A01

    orthonormalize( sols );
}

void orthonormalize( std::vector<std::vector<double>> &vecs )
{
    std::vector<std::vector<double>> res;
    for ( auto &vec : vecs ) {
        double tmp[] = { 0, 0, 0 };
        for ( auto &re : res ) {
            double dot1 = vecDot( vec, re );
            for ( int k = 0; k < 3; k++ ) {
                tmp[k] += ( dot1 * re[k] );
            } // end for k
        }     // end for j

        std::vector<double> newVec( 3 );
        for ( int k = 0; k < 3; k++ ) {
            newVec[k] = vec[k] - tmp[k];
        } // end for k

        double dot2 = vecDot( newVec, newVec );

        if ( !softEquals( dot2, 0 ) ) {
            vecScale( ( 1.0 / std::sqrt( dot2 ) ), newVec );
            res.push_back( newVec );
        }
    } // end for i

    vecs = res;
}

void cubicRoots( double a, double b, double c, double d, double &r1, double &r2, double &r3 )
{
    // next two lines seem to be unused, leads to compiler warning
    //    double delta = (18.0*a*b*c*d) - (4.0*b*b*b*d) + (b*b*c*c) - (4.0*a*c*c*c) -
    //    (27.0*a*a*d*d);
    //    AMP_ASSERT(delta >= 0.0);

    firstCubicRoot( a, b, c, d, r1 );

    double A = a;
    double B = ( b + ( a * r1 ) );
    double C = ( c + ( b * r1 ) + ( a * r1 * r1 ) );

    quadraticRoots( A, B, C, r2, r3 );

    AMP_ASSERT( softEquals( ( ( a * r1 * r1 * r1 ) + ( b * r1 * r1 ) + ( c * r1 ) + d ), 0 ) );
    AMP_ASSERT( softEquals( ( ( a * r2 * r2 * r2 ) + ( b * r2 * r2 ) + ( c * r2 ) + d ), 0 ) );
    AMP_ASSERT( softEquals( ( ( a * r3 * r3 * r3 ) + ( b * r3 * r3 ) + ( c * r3 ) + d ), 0 ) );
}

void quadraticRoots( double a, double b, double c, double &r1, double &r2 )
{
    double delta = ( ( b * b ) - ( 4.0 * a * c ) );
    AMP_ASSERT( delta >= 0.0 );
    AMP_ASSERT( !softEquals( a, 0 ) );

    r1 = ( -b + std::sqrt( delta ) ) / ( 2.0 * a );
    r2 = ( -b - std::sqrt( delta ) ) / ( 2.0 * a );
}

void firstCubicRoot( double a, double b, double c, double d, double &r1 )
{
    AMP_ASSERT( !softEquals( a, 0 ) );
    double p = ( ( 3.0 * a * c ) - ( b * b ) ) / ( 3.0 * a * a );
    double q =
        ( ( 2.0 * b * b * b ) - ( 9.0 * a * b * c ) + ( 27.0 * a * a * d ) ) / ( 27.0 * a * a * a );

    double t;

    if ( softEquals( q, 0 ) ) {
        t = 0;
    } else if ( softEquals( p, 0 ) ) {
        if ( q > 0 ) {
            t = -std::pow( q, ( 1.0 / 3.0 ) );
        } else {
            t = std::pow( -q, ( 1.0 / 3.0 ) );
        }
    } else if ( softEquals( ( 4.0 * p * p * p ), ( -27.0 * q * q ) ) ) {
        t = 3.0 * q / p;
    } else if ( ( ( 4.0 * p * p * p ) + ( 27.0 * q * q ) ) > 0 ) {
        double uCube = -( q / 2.0 ) + std::sqrt( ( q * q / 4.0 ) + ( p * p * p / 27.0 ) );
        double vCube = -( q / 2.0 ) - std::sqrt( ( q * q / 4.0 ) + ( p * p * p / 27.0 ) );

        double u;
        double v;

        if ( softEquals( uCube, 0 ) ) {
            u = 0;
        } else if ( uCube > 0 ) {
            u = std::pow( uCube, ( 1.0 / 3.0 ) );
        } else {
            u = -std::pow( -uCube, ( 1.0 / 3.0 ) );
        }

        if ( softEquals( vCube, 0 ) ) {
            v = 0;
        } else if ( vCube > 0 ) {
            v = std::pow( vCube, ( 1.0 / 3.0 ) );
        } else {
            v = -std::pow( -vCube, ( 1.0 / 3.0 ) );
        }

        t = u + v;
    } else {
        t = 2.0 * std::sqrt( -p / 3.0 ) *
            cos( 1.0 / 3.0 * acos( ( 3.0 * q / ( 2.0 * p ) ) * std::sqrt( -3.0 / p ) ) );
    }

    r1 = ( t - ( b / ( 3.0 * a ) ) );
}

void matTranspose( double A[3][3], double B[3][3] )
{
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            B[i][j] = A[j][i];
        } // end for j
    }     // end for i
}

void matInverse( double A[3][3], double B[3][3] )
{
    double det = matDeterminant( A );

    AMP_ASSERT( !softEquals( det, 0 ) );

    B[0][0] = ( ( A[2][2] * A[1][1] ) - ( A[2][1] * A[1][2] ) );
    B[0][1] = -( ( A[2][2] * A[0][1] ) - ( A[2][1] * A[0][2] ) );
    B[0][2] = ( ( A[1][2] * A[0][1] ) - ( A[1][1] * A[0][2] ) );

    B[1][0] = -( ( A[2][2] * A[1][0] ) - ( A[2][0] * A[1][2] ) );
    B[1][1] = ( ( A[2][2] * A[0][0] ) - ( A[2][0] * A[0][2] ) );
    B[1][2] = -( ( A[1][2] * A[0][0] ) - ( A[1][0] * A[0][2] ) );

    B[2][0] = ( ( A[2][1] * A[1][0] ) - ( A[2][0] * A[1][1] ) );
    B[2][1] = -( ( A[2][1] * A[0][0] ) - ( A[2][0] * A[0][1] ) );
    B[2][2] = ( ( A[1][1] * A[0][0] ) - ( A[1][0] * A[0][1] ) );

    matScale( B, ( 1.0 / det ) );
}

double matTrace( double A[3][3] )
{
    double tr = 0;

    for ( int i = 0; i < 3; i++ ) {
        tr += A[i][i];
    } // end for i

    return tr;
}

void matScale( double A[3][3], double c )
{
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            A[i][j] *= c;
        } // end for j
    }     // end for i
}

double matDeterminant( double A[3][3] )
{
    double det = 0;

    det = ( A[0][0] * ( ( A[2][2] * A[1][1] ) - ( A[2][1] * A[1][2] ) ) ) -
          ( A[1][0] * ( ( A[2][2] * A[0][1] ) - ( A[2][1] * A[0][2] ) ) ) +
          ( A[2][0] * ( ( A[1][2] * A[0][1] ) - ( A[1][1] * A[0][2] ) ) );

    return det;
}

void matMatMultiply( double A[3][3], double B[3][3], double C[3][3] )
{
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            C[i][j] = 0.0;
            for ( int k = 0; k < 3; k++ ) {
                C[i][j] += ( A[i][k] * B[k][j] );
            } // end for k
        }     // end for j
    }         // end for i
}

void matVecMultiply( double A[3][3], double b[3], double c[3] )
{
    // Here c = A * b is implemented.
    c[0] = c[1] = c[2] = 0.0;
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            c[i] += ( A[i][j] * b[j] );
        }
    }
}

void vecVecAddition( double a[3], double b[3], double c[3] )
{
    for ( int i = 0; i < 3; i++ ) {
        c[i] = a[i] + b[i];
    }
}

void matCopy( double A[3][3], double B[3][3] )
{
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            B[i][j] = A[i][j];
        } // end for j
    }     // end for i
}

void vecSqrt( double val[3], double valSqrt[3] )
{
    for ( int i = 0; i < 3; i++ ) {
        valSqrt[i] = std::sqrt( val[i] );
    }
}

void vecInv( double val[3], double valInv[3] )
{
    for ( int i = 0; i < 3; i++ ) {
        valInv[i] = 1.0 / val[i];
    }
}

void vecScale( double c, double vec[3] )
{
    for ( int i = 0; i < 3; i++ ) {
        vec[i] *= c;
    }
}

void vecScale( double c, std::vector<double> &vec )
{
    for ( int i = 0; i < 3; i++ ) {
        vec[i] *= c;
    }
}

double vecDot( double a[3], double b[3] )
{
    double dot = 0;
    for ( int i = 0; i < 3; i++ ) {
        dot += ( a[i] * b[i] );
    }
    return dot;
}

double vecDot( const std::vector<double> &a, const std::vector<double> &b )
{
    double dot = 0;
    for ( int i = 0; i < 3; i++ ) {
        dot += ( a[i] * b[i] );
    }
    return dot;
}

bool softEquals( double a, double b )
{
    bool res = false;
    if ( fabs( a - b ) < 1.0e-12 ) {
        res = true;
    }
    return res;
}


/* Utils for pushback and pullforwards */

/* QtAQ = Q^t A Q */
void pullbackCorotational( double Q[3][3], double A[3][3], double QtAQ[3][3] )
{
    double AQ[3][3];
    double Qt[3][3];

    matMatMultiply( A, Q, AQ );
    matTranspose( Q, Qt );
    matMatMultiply( Qt, AQ, QtAQ );
}

/* QAQt = Q A Qt */
void pushforwardCorotational( double Q[3][3], double A[3][3], double QAQt[3][3] )
{
    double AQt[3][3];
    double Qt[3][3];

    matTranspose( Q, Qt );
    matMatMultiply( A, Qt, AQt );
    matMatMultiply( Q, AQt, QAQt );
}

void matrixToVectorConversion( double Q[3][3], double T[6][6] )
{
    T[0][0] = Q[0][0] * Q[0][0];
    T[0][1] = Q[0][1] * Q[0][1];
    T[0][2] = Q[0][2] * Q[0][2];
    T[0][3] = 2.0 * Q[0][1] * Q[0][2];
    T[0][4] = 2.0 * Q[0][0] * Q[0][2];
    T[0][5] = 2.0 * Q[0][0] * Q[0][1];

    T[1][0] = Q[1][0] * Q[1][0];
    T[1][1] = Q[1][1] * Q[1][1];
    T[1][2] = Q[1][2] * Q[1][2];
    T[1][3] = 2.0 * Q[1][1] * Q[1][2];
    T[1][4] = 2.0 * Q[1][0] * Q[1][2];
    T[1][5] = 2.0 * Q[1][0] * Q[1][1];

    T[2][0] = Q[2][0] * Q[2][0];
    T[2][1] = Q[2][1] * Q[2][1];
    T[2][2] = Q[2][2] * Q[2][2];
    T[2][3] = 2.0 * Q[2][1] * Q[2][2];
    T[2][4] = 2.0 * Q[2][0] * Q[2][2];
    T[2][5] = 2.0 * Q[2][0] * Q[2][1];

    T[3][0] = Q[1][0] * Q[2][0];
    T[3][1] = Q[1][1] * Q[2][1];
    T[3][2] = Q[1][2] * Q[2][2];
    T[3][3] = 1.0 * ( ( Q[1][1] * Q[2][2] ) + ( Q[1][2] * Q[2][1] ) );
    T[3][4] = 1.0 * ( ( Q[1][0] * Q[2][2] ) + ( Q[1][2] * Q[2][0] ) );
    T[3][5] = 1.0 * ( ( Q[1][0] * Q[2][1] ) + ( Q[1][1] * Q[2][0] ) );

    T[4][0] = Q[0][0] * Q[2][0];
    T[4][1] = Q[0][1] * Q[2][1];
    T[4][2] = Q[0][2] * Q[2][2];
    T[4][3] = 1.0 * ( ( Q[0][1] * Q[2][2] ) + ( Q[0][2] * Q[2][1] ) );
    T[4][4] = 1.0 * ( ( Q[0][0] * Q[2][2] ) + ( Q[0][2] * Q[2][0] ) );
    T[4][5] = 1.0 * ( ( Q[0][0] * Q[2][1] ) + ( Q[0][1] * Q[2][0] ) );

    T[5][0] = Q[0][0] * Q[1][0];
    T[5][1] = Q[0][1] * Q[1][1];
    T[5][2] = Q[0][2] * Q[1][2];
    T[5][3] = 1.0 * ( ( Q[0][1] * Q[1][2] ) + ( Q[0][2] * Q[1][1] ) );
    T[5][4] = 1.0 * ( ( Q[0][0] * Q[1][2] ) + ( Q[0][2] * Q[1][0] ) );
    T[5][5] = 1.0 * ( ( Q[0][0] * Q[1][1] ) + ( Q[0][1] * Q[1][0] ) );
}

/* QQKQtQt = (QxQ) K (QxQ) */
void pushforwardCorotationalStiffness( double Q[3][3], double K[6][6], double QQKQtQt[6][6] )
{
    double tmp1[6][6], T[6][6];

    //  INITIALIZATION

    for ( int j = 0; j < 6; j++ ) {
        for ( int k = 0; k < 6; k++ ) {
            tmp1[j][k]    = 0.0;
            QQKQtQt[j][k] = 0.0;
            T[j][k]       = 0.0;
        }
    }

    // matrixOuterProduct(Q, Q, T);
    matrixToVectorConversion( Q, T );

    //  TMP1 = T * K

    for ( int j = 0; j < 6; j++ ) {
        for ( int k = 0; k < 6; k++ ) {
            for ( int l = 0; l < 6; l++ ) {
                tmp1[j][k] += T[j][l] * K[l][k];
            }
        }
    }

    // QQKQtQt = TMP1 * T^t

    for ( int j = 0; j < 6; j++ ) {
        for ( int k = 0; k < 6; k++ ) {
            for ( int l = 0; l < 6; l++ ) {
                QQKQtQt[j][k] += tmp1[j][l] * T[k][l];
            }
        }
    }
}

/* Tijab = Qia Qjb */
void matrixOuterProduct( double Qia[3][3], double Qjb[3][3], double Tijab[6][6] )
{
    int j, k, j3, k3;
    int m1j, m2j, m1k, m2k;
    const double map1[] = { 1, 2, 3, 2, 3, 1 };
    const double map2[] = { 1, 2, 3, 3, 1, 2 };

    for ( j = 0; j < 3; j++ ) {
        for ( k = 0; k < 3; k++ ) {
            j3 = j + 3;
            k3 = k + 3;

            m1j = map1[j3] - 1;
            m2j = map2[j3] - 1;
            m1k = map1[k3] - 1;
            m2k = map2[k3] - 1;

            Tijab[j][k] = Qia[j][k] * Qjb[j][k];

            Tijab[j][k3] = Qia[j][m1k] * Qjb[j][m2k] + Qia[j][m2k] * Qjb[j][m1k];

            Tijab[j3][k] = ( Qia[m1j][k] * Qjb[m2j][k] + Qia[m2j][k] * Qjb[m1j][k] ) * 0.50;

            Tijab[j3][k3] = ( Qia[m1j][m1k] * Qjb[m2j][m2k] + Qia[m2j][m1k] * Qjb[m1j][m2k] +
                              Qia[m1j][m2k] * Qjb[m2j][m1k] + Qia[m2j][m2k] * Qjb[m1j][m1k] ) *
                            0.50;
        }
    }
}

void polarDecompositionFeqRU_Simo(
    double F[3][3],
    double R[3][3],
    double U[3][3] ) // Decomposes the deformation gradient F into rotation R and stretch U (F = RU)
{
    // F -> The input deformation gradient.
    // R -> The rotation matrix (output).
    // U -> The strecth tensor (output).
    double I1, I2, I3, i1, i2, i3, b, c, x[3], lambda[3], D, Uinv[3][3], C[3][3], Ft[3][3],
        C2[3][3], I[3][3];
    double tol = 1.0e-8;

    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
            Ft[j][i] = F[i][j];

    for ( auto &elem : I )
        for ( double &j : elem )
            j = 0.0;

    I[0][0] = I[1][1] = I[2][2] = 1.0;

    matMatMultiply( Ft, F, C );
    matMatMultiply( C, C, C2 );

    I1 = matTrace( C );
    I2 = ( 1.0 / 2.0 ) * ( ( I1 * I1 ) - matTrace( C2 ) );
    I3 = matDeterminant( C );

    b = I2 - ( ( I1 * I1 ) / 3.0 );
    c = -( ( 2.0 / 27.0 ) * I1 * I1 * I1 ) + ( ( I1 * I2 ) / 3.0 ) - I3;

    if ( fabs( b ) < tol ) {
        if ( fabs( c ) < tol )
            c = fabs( c );
        AMP_INSIST( ( c >= 0.0 ),
                    "Error in the polar decomposition (Simo), value of c is less than zero." );
        for ( auto &elem : x )
            elem = -std::pow( c, ( 1.0 / 3.0 ) );
    } else {
        AMP_INSIST( ( b <= 0.0 ),
                    "Error in the polar decomposition (Simo), value of b is greater than zero." );
        double m     = 2.0 * std::sqrt( -b / 3.0 );
        double n     = ( 3.0 * c ) / ( m * b );
        double term1 = 1.0 - ( n * n );
        term1        = std::max( tol, term1 );
        AMP_INSIST( ( term1 >= 0.0 ),
                    "(1.0 - n*n) is not greater than or equal to zero. Error in "
                    "the polar decomposition (Simo)" );
        double t    = ( atan2( std::sqrt( term1 ), n ) ) / 3.0;
        double pi_d = 3.14159265358979;
        for ( int i = 0; i < 3; i++ )
            x[i] = m * cos( t + ( ( 2.0 * ( (double) ( i ) ) * pi_d ) / 3.0 ) );
    }

    for ( int i = 0; i < 3; i++ ) {
        double term2 = x[i] + ( I1 / 3.0 );
        if ( fabs( term2 ) < tol )
            term2 = fabs( term2 );
        AMP_INSIST( ( term2 >= 0.0 ),
                    "Error in the polar decomposition (Simo), while calculating "
                    "lambda, lambda^2 is negetive." );
        lambda[i] = std::sqrt( term2 );
    }

    i1 = lambda[0] + lambda[1] + lambda[2];
    i2 = ( lambda[0] * lambda[1] ) + ( lambda[0] * lambda[2] ) + ( lambda[1] * lambda[2] );
    i3 = lambda[0] * lambda[1] * lambda[2];

    D = ( i1 * i2 ) - i3;
    AMP_INSIST( ( D > tol ),
                "Error in the polar decomposition (Simo), D should be greater than zero." );

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            U[i][j] = ( 1.0 / D ) *
                      ( -C2[i][j] + ( ( ( i1 * i1 ) - i2 ) * C[i][j] ) + ( i1 * i3 * I[i][j] ) );
            AMP_INSIST(
                ( i3 > tol ),
                "Error in the polar decomposition (Simo), i3 should be greater than zero." );
            Uinv[i][j] = ( 1.0 / i3 ) * ( C[i][j] - ( i1 * U[i][j] ) + ( i2 * I[i][j] ) );
        }
    }

    matMatMultiply( F, Uinv, R );
}

void computeShapeFunctions( double N[8], double xi, double eta, double zeta )
{
    // Initialize the shape functions with zero.
    for ( int i = 0; i < 8; i++ ) {
        N[i] = 0.0;
    }

    // Assign the values of the shape functions.
    double one8 = 1.0 / 8.0;
    N[0]        = one8 * ( 1.0 - xi ) * ( 1.0 - eta ) * ( 1.0 - zeta );
    N[1]        = one8 * ( 1.0 + xi ) * ( 1.0 - eta ) * ( 1.0 - zeta );
    N[2]        = one8 * ( 1.0 + xi ) * ( 1.0 + eta ) * ( 1.0 - zeta );
    N[3]        = one8 * ( 1.0 - xi ) * ( 1.0 + eta ) * ( 1.0 - zeta );
    N[4]        = one8 * ( 1.0 - xi ) * ( 1.0 - eta ) * ( 1.0 + zeta );
    N[5]        = one8 * ( 1.0 + xi ) * ( 1.0 - eta ) * ( 1.0 + zeta );
    N[6]        = one8 * ( 1.0 + xi ) * ( 1.0 + eta ) * ( 1.0 + zeta );
    N[7]        = one8 * ( 1.0 - xi ) * ( 1.0 + eta ) * ( 1.0 + zeta );
}

void computeLocalDerivatives(
    double dNdxi[8], double dNdeta[8], double dNdzeta[8], double xi, double eta, double zeta )
{
    for ( int i = 0; i < 8; i++ ) {
        dNdxi[i]   = 0.0;
        dNdeta[i]  = 0.0;
        dNdzeta[i] = 0.0;
    }

    // Assign the values for the derivatives of the shape functions.
    double one8 = 1.0 / 8.0;
    dNdxi[0]    = -one8 * ( 1.0 - eta ) * ( 1.0 - zeta );
    dNdxi[1]    = one8 * ( 1.0 - eta ) * ( 1.0 - zeta );
    dNdxi[2]    = one8 * ( 1.0 + eta ) * ( 1.0 - zeta );
    dNdxi[3]    = -one8 * ( 1.0 + eta ) * ( 1.0 - zeta );
    dNdxi[4]    = -one8 * ( 1.0 - eta ) * ( 1.0 + zeta );
    dNdxi[5]    = one8 * ( 1.0 - eta ) * ( 1.0 + zeta );
    dNdxi[6]    = one8 * ( 1.0 + eta ) * ( 1.0 + zeta );
    dNdxi[7]    = -one8 * ( 1.0 + eta ) * ( 1.0 + zeta );

    dNdeta[0] = -one8 * ( 1.0 - xi ) * ( 1.0 - zeta );
    dNdeta[1] = -one8 * ( 1.0 + xi ) * ( 1.0 - zeta );
    dNdeta[2] = one8 * ( 1.0 + xi ) * ( 1.0 - zeta );
    dNdeta[3] = one8 * ( 1.0 - xi ) * ( 1.0 - zeta );
    dNdeta[4] = -one8 * ( 1.0 - xi ) * ( 1.0 + zeta );
    dNdeta[5] = -one8 * ( 1.0 + xi ) * ( 1.0 + zeta );
    dNdeta[6] = one8 * ( 1.0 + xi ) * ( 1.0 + zeta );
    dNdeta[7] = one8 * ( 1.0 - xi ) * ( 1.0 + zeta );

    dNdzeta[0] = -one8 * ( 1.0 - xi ) * ( 1.0 - eta );
    dNdzeta[1] = -one8 * ( 1.0 + xi ) * ( 1.0 - eta );
    dNdzeta[2] = -one8 * ( 1.0 + xi ) * ( 1.0 + eta );
    dNdzeta[3] = -one8 * ( 1.0 - xi ) * ( 1.0 + eta );
    dNdzeta[4] = one8 * ( 1.0 - xi ) * ( 1.0 - eta );
    dNdzeta[5] = one8 * ( 1.0 + xi ) * ( 1.0 - eta );
    dNdzeta[6] = one8 * ( 1.0 + xi ) * ( 1.0 + eta );
    dNdzeta[7] = one8 * ( 1.0 - xi ) * ( 1.0 + eta );
}

void computeJInverse( double G[3][3],
                      double dNdxi[8],
                      double dNdeta[8],
                      double dNdzeta[8],
                      double x[8],
                      double y[8],
                      double z[8],
                      double detJ[1] )
{
    double J[3][3];
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            G[i][j] = 0.0;
            J[i][j] = 0.0;
        }
    }

    for ( int i = 0; i < 8; i++ ) {
        J[0][0] += ( dNdxi[i] * x[i] );
        J[0][1] += ( dNdxi[i] * y[i] );
        J[0][2] += ( dNdxi[i] * z[i] );

        J[1][0] += ( dNdeta[i] * x[i] );
        J[1][1] += ( dNdeta[i] * y[i] );
        J[1][2] += ( dNdeta[i] * z[i] );

        J[2][0] += ( dNdzeta[i] * x[i] );
        J[2][1] += ( dNdzeta[i] * y[i] );
        J[2][2] += ( dNdzeta[i] * z[i] );
    }

    detJ[0] = matDeterminant( J );
    AMP_INSIST( ( detJ[0] > 0.0 ),
                "\n#########\nGeomType::Cell of the element is "
                "negetive.\nExiting the code.\n##########\n" );

    matInverse( J, G );
}

void constructShapeFunctionDerivatives( double dNdx[8],
                                        double dNdy[8],
                                        double dNdz[8],
                                        double x[8],
                                        double y[8],
                                        double z[8],
                                        double xi,
                                        double eta,
                                        double zeta,
                                        double detJ[1] )
{
    for ( int i = 0; i < 8; i++ ) {
        dNdx[i] = dNdy[i] = dNdz[i] = 0.0;
    }

    double G[3][3], dNdxi[8], dNdeta[8], dNdzeta[8];
    computeLocalDerivatives( dNdxi, dNdeta, dNdzeta, xi, eta, zeta );
    computeJInverse( G, dNdxi, dNdeta, dNdzeta, x, y, z, detJ );

    for ( int i = 0; i < 8; i++ ) {
        dNdx[i] = ( G[0][0] * dNdxi[i] ) + ( G[0][1] * dNdeta[i] ) + ( G[0][2] * dNdzeta[i] );
        dNdy[i] = ( G[1][0] * dNdxi[i] ) + ( G[1][1] * dNdeta[i] ) + ( G[1][2] * dNdzeta[i] );
        dNdz[i] = ( G[2][0] * dNdxi[i] ) + ( G[2][1] * dNdeta[i] ) + ( G[2][2] * dNdzeta[i] );
    }
}

void computeGradient( double dN_dxnp1o2[8],
                      double dN_dynp1o2[8],
                      double dN_dznp1o2[8],
                      double delta_u[8],
                      double delta_v[8],
                      double delta_w[8],
                      unsigned int num_nodes,
                      double d_np1o2[3][3] )
{
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            d_np1o2[i][j] = 0.0;
        }
    }

    for ( unsigned int i = 0; i < num_nodes; i++ ) {
        d_np1o2[0][0] += ( dN_dxnp1o2[i] * delta_u[i] );
        d_np1o2[0][1] += ( dN_dynp1o2[i] * delta_u[i] );
        d_np1o2[0][2] += ( dN_dznp1o2[i] * delta_u[i] );

        d_np1o2[1][0] += ( dN_dxnp1o2[i] * delta_v[i] );
        d_np1o2[1][1] += ( dN_dynp1o2[i] * delta_v[i] );
        d_np1o2[1][2] += ( dN_dznp1o2[i] * delta_v[i] );

        d_np1o2[2][0] += ( dN_dxnp1o2[i] * delta_w[i] );
        d_np1o2[2][1] += ( dN_dynp1o2[i] * delta_w[i] );
        d_np1o2[2][2] += ( dN_dznp1o2[i] * delta_w[i] );
    }
}

void jaumannToCauchy( double Om[3][3], double Sg[3][3] )
{
    double OmSg[3][3], SgOm[3][3];
    matMatMultiply( Om, Sg, OmSg );
    matMatMultiply( Sg, Om, SgOm );
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            Sg[i][j] = -OmSg[i][j] + SgOm[i][j];
        }
    }
}
} // namespace AMP::Operator
