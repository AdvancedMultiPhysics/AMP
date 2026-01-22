
#ifndef included_AMP_UpdatedLagrangianUtils
#define included_AMP_UpdatedLagrangianUtils

#include <array>
#include <vector>

namespace AMP::Operator::UpdatedLagrangianUtils {

bool softEquals( double a, double b );

void polarDecomposeRU( double A[3][3], double R[3][3], double U[3][3] );

void matMatMultiply( double A[3][3], double B[3][3], double C[3][3] );

void matVecMultiply( double A[3][3], double b[3], double c[3] );

void vecVecAddition( double a[3], double b[3], double c[3] );

void matTranspose( double A[3][3], double B[3][3] );

void matInverse( double A[3][3], double B[3][3] );

void matDiagMatMultiply( double P1[3][3], double D[3], double P2[3][3], double A[3][3] );

void matCopy( double A[3][3], double B[3][3] );

void matScale( double A[3][3], double c );

double matTrace( double A[3][3] );

double matDeterminant( double A[3][3] );

void eigenValues( double A[3][3], double val[3] );

void eigenVectors( double A[3][3], double val[3], double vec[3][3] );

void solveEquation( double A[3][3], std::vector<std::array<double, 3>> &sols );

void orthonormalize( std::vector<std::array<double, 3>> &vecs );

void quadraticRoots( double a, double b, double c, double &r1, double &r2 );

void cubicRoots( double a, double b, double c, double d, double &r1, double &r2, double &r3 );

void firstCubicRoot( double a, double b, double c, double d, double &r1 );

/* Utils for pushback and pullforwards */
void pullbackCorotational( double Q[3][3], double A[3][3], double QtAQ[3][3] );

void pushforwardCorotational( double Q[3][3], double A[3][3], double QAQt[3][3] );

void pushforwardCorotationalStiffness( double Q[3][3], double K[6][6], double QQKQtQt[6][6] );

void matrixOuterProduct( double Qia[3][3], double Qjb[3][3], double Tijab[6][6] ); // T = (QxQ)

void polarDecompositionFeqRU_Simo( double F[3][3],
                                   double R[3][3],
                                   double U[3][3] ); // Decomposes the deformation gradient F into
                                                     // rotation R and stretch U (F = RU)

// This function calculates the derivatives of the shape functions with respect to the global x, y
// and z directions.
// dNdx, dNdy and dNdz are the output arguments which contains the derivatives of the shape
// functions.
// x, y and z contains the nodal coordinates for the current element. These are inputs.
// xi, eta and zeta are the coordinates of the current gauss point. These are given as input.
// detJ contains the determinant of the jacobian matrix. It is used in the numerical integration.
// This is an output.
void constructShapeFunctionDerivatives( double dNdx[8],
                                        double dNdy[8],
                                        double dNdz[8],
                                        double x[8],
                                        double y[8],
                                        double z[8],
                                        double xi,
                                        double eta,
                                        double zeta,
                                        double detJ[1] );

// This function calculates values of all the 8 shape functions at a particular gauss/quadrature
// point.
// xi, eta and zeta are the coordinates of the current gauss point. These are given as input.
// N holds the 8 shape functions. It is the output.
void computeShapeFunctions( double N[8], double xi, double eta, double zeta );

// This function calculates the derivatives of the shape function with respect to the isoparametric
// coordinates xi, eta
// and zeta.
// This is evaluated at each gauss/quadrature point.
void computeLocalDerivatives(
    double dNdxi[8], double dNdeta[8], double dNdzeta[8], double xi, double eta, double zeta );

// This function calculates the matrix J^inverse.
// G is the output matrix. G = J^inverse.
// dNdxi, dNdeta and dNdzeta are the local derivatives of the shape functions. These are given as
// inputs.
// x, y and z contains the nodal coordinates for the current element. These are inputs.
// detJ is an output. It contains the determinant of the J matrix. It also signifies the volume of
// the current element.
void computeJInverse( double G[3][3],
                      double dNdxi[8],
                      double dNdeta[8],
                      double dNdzeta[8],
                      double x[8],
                      double y[8],
                      double z[8],
                      double detJ[1] );

// This function calculates the symmetric part of the deformation gradient also known as the rate of
// deformation tensor
// (d_np1o2).
// dN_dxnp1o2, dN_dynp1o2 and dN_dznp1o2 are the derivatives of the shape functions with respect to
// the np1o2
// configuration.
// delta_u, delta_v and delta_w are the incremental displacements between n-th and (n+1)-th
// configuration.
// num_nodes are the number of nodes in the element.
void computeGradient( double dN_dxnp1o2[8],
                      double dN_dynp1o2[8],
                      double dN_dznp1o2[8],
                      double delta_u[8],
                      double delta_v[8],
                      double delta_w[8],
                      unsigned int num_nodes,
                      double d_np1o2[3][3] );

// This function is called for converting the jaumann rate to cauchy rate.
void jaumannToCauchy( double Om[3][3], double Sg[3][3] );


} // namespace AMP::Operator::UpdatedLagrangianUtils

#endif
