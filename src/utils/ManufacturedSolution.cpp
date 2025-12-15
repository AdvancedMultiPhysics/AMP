#include "AMP/utils/ManufacturedSolution.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include <memory>

#include <cmath>
#include <limits>
#include <string>
#include <vector>


namespace AMP {


static constexpr double pi  = 3.1415926535898;
static constexpr double pi2 = 2 * 3.1415926535898;


/************************************************************************
 *  Unit cube, quadratic, all Neumann BC's.
 *	The normal derivative \f$\frac{\partial u}{\partial n}\f$
 *	has the values \f$c_i, i=0,\dots,5\f$ on the planes \f$x=0, x=1, y=0, y=1, z=0, z=1\f$,
 *	respectively.
 *  Note that \f$\frac{\partial u}{\partial n} = - \frac{\partial u}{\partial x}\f$
 *	at \f$x=0\f$ and similarly for \f$y=0\f$ and \f$z=0\f$. The derivatives up to order
 *	two are returned in the order
 *	\f[ u,
 *	\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}, \frac{\partial u}{\partial z},
 *	\frac{\partial^2 u}{\partial x^2}, \frac{\partial^2 u}{\partial x \partial y},
 *  \frac{\partial^2 u}{\partial x \partial z},
 *	\frac{\partial^2 u}{\partial y^2},  \frac{\partial^2 u}{\partial y \partial z},
 *  \frac{\partial^2 u}{\partial z^2}, \f]
 *
 *	or more succinctly,
 *
 *	[0, x, y, z, xx, xy, xz, yy, yz, zz]
 *	 0  1  2  3   4   5   6   7   8   9
 *
 *	which is also true for cylindrical coordinates [x,y,z]->[r,th,z]
 ************************************************************************/
static std::array<double, 10>
quad_neumann( double x, double y, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 6, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 1, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    // clang-format off
    result[0] = a[0] + x * (-c[0]+x*(c[0]/2+c[1]/2))
                     + y * (-c[2]+y*(c[2]/2+c[3]/2))
                     + z * (-c[4]+z*(c[4]/2+c[5]/2));
    result[1] = -c[0] + x * ( c[0] + c[1] );
    result[2] = -c[2] + y * ( c[2] + c[3] );
    result[3] = -c[4] + z * ( c[4] + c[5] );
    result[4] = c[0] + c[1];
    result[5] = 0;
    result[6] = 0;
    result[7] = c[2] + c[3];
    result[8] = 0;
    result[9] = c[4] + c[5];
    // clang-format on
    return result;
}


static std::array<double, 10>
quad_dirichlet1( double x, double, double, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 2, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 1, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    result[0] = c[0] + x * ( a[0] + x * ( -a[0] / 2 + c[1] / 2 ) );
    result[1] = a[0] + x * ( -a[0] + c[1] );
    result[2] = 0;
    result[3] = 0;
    result[4] = -a[0] + c[1];
    result[5] = 0;
    result[6] = 0;
    result[7] = 0;
    result[8] = 0;
    result[9] = 0;
    return result;
}


static std::array<double, 10>
quad_dirichlet2( double x, double, double, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 2, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 1, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    result[0] = c[0] + x * ( a[0] + x * ( -a[0] - c[0] + c[1] ) );
    result[1] = a[0] + x * ( -2 * a[0] - 2 * c[0] + 2 * c[1] );
    result[2] = 0;
    result[3] = 0;
    result[4] = -2 * a[0] - 2 * c[0] + 2 * c[1];
    result[5] = 0;
    result[6] = 0;
    result[7] = 0;
    result[8] = 0;
    result[9] = 0;
    return result;
}


static std::array<double, 10>
quad_none( double x, double y, double z, const ManufacturedSolution *mfs )
{
    const auto &a = mfs->geta();
    AMP_INSIST( a.size() >= 10, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    // clang-format off
    result[0] = a[0] + z * ( a[1] + z * a[2] )
                     + y * ( a[3] + z * a[4] + y * a[5] )
                     + x * ( a[6] + z * a[7] + y * a[8] + x * a[9] );
    result[1] = a[6] + z * a[7] + y * a[8] + 2 * x * a[9];
    result[2] = a[3] + z * a[4] + 2 * y * a[5] + x * a[8];
    result[3] = a[1] + 2 * z * a[2] + y * a[4] + x * a[7];
    result[4] = 2 * a[9];
    result[5] = a[8];
    result[6] = a[7];
    result[7] = 2 * a[5];
    result[8] = a[4];
    result[9] = 2 * a[2];
    // clang-format o
    return result;
}


static std::array<double, 10>
cubic_neumann( double x, double y, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 6, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 8, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    result[0] =
        a[0] +
        x * ( -c[0] +
              x * ( a[4] + z * z * ( a[5] - ( 2 * z * a[5] ) / 3 ) +
                    y * y *
                        ( a[6] + z * z * ( a[7] - ( 2 * z * a[7] ) / 3 ) +
                          y * ( ( -2 * a[6] ) / 3 +
                                z * z * ( ( -2 * a[7] ) / 3 + ( 4 * z * a[7] ) / 9 ) ) ) +
                    x * ( ( -2 * a[4] ) / 3 + z * z * ( ( -2 * a[5] ) / 3 + ( 4 * z * a[5] ) / 9 ) +
                          y * y *
                              ( ( -2 * a[6] ) / 3 +
                                z * z * ( ( -2 * a[7] ) / 3 + ( 4 * z * a[7] ) / 9 ) +
                                y * ( ( 4 * a[6] ) / 9 +
                                      z * z * ( ( 4 * a[7] ) / 9 - ( 8 * z * a[7] ) / 27 ) ) ) +
                          c[0] / 3 + c[1] / 3 ) ) ) +
        y * ( -c[2] +
              y * ( a[2] + z * z * ( a[3] - ( 2 * z * a[3] ) / 3 ) +
                    y * ( ( -2 * a[2] ) / 3 + z * z * ( ( -2 * a[3] ) / 3 + ( 4 * z * a[3] ) / 9 ) +
                          c[2] / 3 + c[3] / 3 ) ) ) +
        z * ( -c[4] + z * ( a[1] + z * ( ( -2 * a[1] ) / 3 + c[4] / 3 + c[5] / 3 ) ) );
    result[1] =
        -c[0] + x * ( 2 * a[4] + z * z * ( 2 * a[5] - ( 4 * z * a[5] ) / 3 ) +
                      y * y *
                          ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
                            y * ( ( -4 * a[6] ) / 3 +
                                  z * z * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 9 ) ) ) +
                      x * ( -2 * a[4] + z * z * ( -2 * a[5] + ( 4 * z * a[5] ) / 3 ) +
                            y * y *
                                ( -2 * a[6] + z * z * ( -2 * a[7] + ( 4 * z * a[7] ) / 3 ) +
                                  y * ( ( 4 * a[6] ) / 3 +
                                        z * z * ( ( 4 * a[7] ) / 3 - ( 8 * z * a[7] ) / 9 ) ) ) +
                            c[0] + c[1] ) );
    result[2] =
        x * x *
            ( x * y *
                  ( ( -4 * a[6] ) / 3 + z * z * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 9 ) +
                    y * ( ( 4 * a[6] ) / 3 +
                          z * z * ( ( 4 * a[7] ) / 3 - ( 8 * z * a[7] ) / 9 ) ) ) +
              y * ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
                    y * ( -2 * a[6] + z * z * ( -2 * a[7] + ( 4 * z * a[7] ) / 3 ) ) ) ) -
        c[2] +
        y * ( 2 * a[2] + z * z * ( 2 * a[3] - ( 4 * z * a[3] ) / 3 ) +
              y * ( -2 * a[2] + z * z * ( -2 * a[3] + ( 4 * z * a[3] ) / 3 ) + c[2] + c[3] ) );
    result[3] = y * y *
                    ( z * ( 2 * a[3] - 2 * z * a[3] ) +
                      y * z * ( ( -4 * a[3] ) / 3 + ( 4 * z * a[3] ) / 3 ) ) +
                x * x *
                    ( z * ( 2 * a[5] - 2 * z * a[5] ) +
                      y * y *
                          ( z * ( 2 * a[7] - 2 * z * a[7] ) +
                            y * z * ( ( -4 * a[7] ) / 3 + ( 4 * z * a[7] ) / 3 ) ) +
                      x * ( z * ( ( -4 * a[5] ) / 3 + ( 4 * z * a[5] ) / 3 ) +
                            y * y *
                                ( y * z * ( ( 8 * a[7] ) / 9 - ( 8 * z * a[7] ) / 9 ) +
                                  z * ( ( -4 * a[7] ) / 3 + ( 4 * z * a[7] ) / 3 ) ) ) ) -
                c[4] + z * ( 2 * a[1] + z * ( -2 * a[1] + c[4] + c[5] ) );
    result[4] =
        2 * a[4] + z * z * ( 2 * a[5] - ( 4 * z * a[5] ) / 3 ) +
        y * y *
            ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
              y * ( ( -4 * a[6] ) / 3 + z * z * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 9 ) ) ) +
        x * ( -4 * a[4] + z * z * ( -4 * a[5] + ( 8 * z * a[5] ) / 3 ) +
              y * y *
                  ( -4 * a[6] + z * z * ( -4 * a[7] + ( 8 * z * a[7] ) / 3 ) +
                    y * ( ( 8 * a[6] ) / 3 +
                          z * z * ( ( 8 * a[7] ) / 3 - ( 16 * z * a[7] ) / 9 ) ) ) +
              2 * c[0] + 2 * c[1] );
    result[5] = x * ( x * y *
                          ( -4 * a[6] + z * z * ( -4 * a[7] + ( 8 * z * a[7] ) / 3 ) +
                            y * ( 4 * a[6] + z * z * ( 4 * a[7] - ( 8 * z * a[7] ) / 3 ) ) ) +
                      y * ( 4 * a[6] + z * z * ( 4 * a[7] - ( 8 * z * a[7] ) / 3 ) +
                            y * ( -4 * a[6] + z * z * ( -4 * a[7] + ( 8 * z * a[7] ) / 3 ) ) ) );
    result[6] = x * ( z * ( 4 * a[5] - 4 * z * a[5] ) +
                      y * y *
                          ( z * ( 4 * a[7] - 4 * z * a[7] ) +
                            y * z * ( ( -8 * a[7] ) / 3 + ( 8 * z * a[7] ) / 3 ) ) +
                      x * ( z * ( -4 * a[5] + 4 * z * a[5] ) +
                            y * y *
                                ( y * z * ( ( 8 * a[7] ) / 3 - ( 8 * z * a[7] ) / 3 ) +
                                  z * ( -4 * a[7] + 4 * z * a[7] ) ) ) );
    result[7] =
        2 * a[2] + z * z * ( 2 * a[3] - ( 4 * z * a[3] ) / 3 ) +
        x * x *
            ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
              y * ( -4 * a[6] + z * z * ( -4 * a[7] + ( 8 * z * a[7] ) / 3 ) ) +
              x * ( ( -4 * a[6] ) / 3 + z * z * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 9 ) +
                    y * ( ( 8 * a[6] ) / 3 +
                          z * z * ( ( 8 * a[7] ) / 3 - ( 16 * z * a[7] ) / 9 ) ) ) ) +
        y * ( -4 * a[2] + z * z * ( -4 * a[3] + ( 8 * z * a[3] ) / 3 ) + 2 * c[2] + 2 * c[3] );
    result[8] =
        y * ( z * ( 4 * a[3] - 4 * z * a[3] ) + y * z * ( -4 * a[3] + 4 * z * a[3] ) ) +
        x * x *
            ( x * y *
                  ( y * z * ( ( 8 * a[7] ) / 3 - ( 8 * z * a[7] ) / 3 ) +
                    z * ( ( -8 * a[7] ) / 3 + ( 8 * z * a[7] ) / 3 ) ) +
              y * ( z * ( 4 * a[7] - 4 * z * a[7] ) + y * z * ( -4 * a[7] + 4 * z * a[7] ) ) );
    result[9] =
        2 * a[1] +
        y * y * ( 2 * a[3] - 4 * z * a[3] + y * ( ( -4 * a[3] ) / 3 + ( 8 * z * a[3] ) / 3 ) ) +
        x * x *
            ( 2 * a[5] - 4 * z * a[5] +
              y * y *
                  ( 2 * a[7] - 4 * z * a[7] + y * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 3 ) ) +
              x * ( ( -4 * a[5] ) / 3 + ( 8 * z * a[5] ) / 3 +
                    y * y *
                        ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 3 +
                          y * ( ( 8 * a[7] ) / 9 - ( 16 * z * a[7] ) / 9 ) ) ) ) +
        z * ( -4 * a[1] + 2 * c[4] + 2 * c[5] );
    return result;
}


static std::array<double, 10>
cubic_dirichlet1( double x, double y, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 2, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 8, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    result[0] =
        c[0] +
        x * ( a[0] + z * z * ( a[1] - ( 2 * z * a[1] ) / 3 ) +
              y * y *
                  ( a[2] + z * z * ( a[3] - ( 2 * z * a[3] ) / 3 ) +
                    y * ( ( -2 * a[2] ) / 3 +
                          z * z * ( ( -2 * a[3] ) / 3 + ( 4 * z * a[3] ) / 9 ) ) ) +
              x * ( a[4] + z * z * ( a[5] - ( 2 * z * a[5] ) / 3 ) +
                    y * y *
                        ( a[6] + z * z * ( a[7] - ( 2 * z * a[7] ) / 3 ) +
                          y * ( ( -2 * a[6] ) / 3 +
                                z * z * ( ( -2 * a[7] ) / 3 + ( 4 * z * a[7] ) / 9 ) ) ) +
                    x * ( -a[0] / 3 - ( 2 * a[4] ) / 3 +
                          z * z *
                              ( -a[1] / 3 + z * ( ( 2 * a[1] ) / 9 + ( 4 * a[5] ) / 9 ) -
                                ( 2 * a[5] ) / 3 ) +
                          y * y *
                              ( -a[2] / 3 - ( 2 * a[6] ) / 3 +
                                y * ( ( 2 * a[2] ) / 9 + ( 4 * a[6] ) / 9 +
                                      z * z *
                                          ( ( 2 * a[3] ) / 9 +
                                            z * ( ( -4 * a[3] ) / 27 - ( 8 * a[7] ) / 27 ) +
                                            ( 4 * a[7] ) / 9 ) ) +
                                z * z *
                                    ( -a[3] / 3 + z * ( ( 2 * a[3] ) / 9 + ( 4 * a[7] ) / 9 ) -
                                      ( 2 * a[7] ) / 3 ) ) +
                          c[1] / 3 ) ) );
    result[1] =
        a[0] + z * z * ( a[1] - ( 2 * z * a[1] ) / 3 ) +
        y * y *
            ( a[2] + z * z * ( a[3] - ( 2 * z * a[3] ) / 3 ) +
              y * ( ( -2 * a[2] ) / 3 + z * z * ( ( -2 * a[3] ) / 3 + ( 4 * z * a[3] ) / 9 ) ) ) +
        x * ( 2 * a[4] + z * z * ( 2 * a[5] - ( 4 * z * a[5] ) / 3 ) +
              y * y *
                  ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
                    y * ( ( -4 * a[6] ) / 3 +
                          z * z * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 9 ) ) ) +
              x * ( -a[0] - 2 * a[4] +
                    z * z * ( -a[1] - 2 * a[5] + z * ( ( 2 * a[1] ) / 3 + ( 4 * a[5] ) / 3 ) ) +
                    y * y *
                        ( -a[2] - 2 * a[6] +
                          z * z *
                              ( -a[3] - 2 * a[7] + z * ( ( 2 * a[3] ) / 3 + ( 4 * a[7] ) / 3 ) ) +
                          y * ( ( 2 * a[2] ) / 3 + ( 4 * a[6] ) / 3 +
                                z * z *
                                    ( ( 2 * a[3] ) / 3 +
                                      z * ( ( -4 * a[3] ) / 9 - ( 8 * a[7] ) / 9 ) +
                                      ( 4 * a[7] ) / 3 ) ) ) +
                    c[1] ) );
    result[2] =
        x *
        ( y * ( 2 * a[2] + z * z * ( 2 * a[3] - ( 4 * z * a[3] ) / 3 ) +
                y * ( -2 * a[2] + z * z * ( -2 * a[3] + ( 4 * z * a[3] ) / 3 ) ) ) +
          x * ( x * y *
                    ( ( -2 * a[2] ) / 3 - ( 4 * a[6] ) / 3 +
                      z * z *
                          ( ( -2 * a[3] ) / 3 + z * ( ( 4 * a[3] ) / 9 + ( 8 * a[7] ) / 9 ) -
                            ( 4 * a[7] ) / 3 ) +
                      y * ( ( 2 * a[2] ) / 3 + ( 4 * a[6] ) / 3 +
                            z * z *
                                ( ( 2 * a[3] ) / 3 + z * ( ( -4 * a[3] ) / 9 - ( 8 * a[7] ) / 9 ) +
                                  ( 4 * a[7] ) / 3 ) ) ) +
                y * ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
                      y * ( -2 * a[6] + z * z * ( -2 * a[7] + ( 4 * z * a[7] ) / 3 ) ) ) ) );
    result[3] =
        x *
        ( z * ( 2 * a[1] - 2 * z * a[1] ) +
          y * y *
              ( z * ( 2 * a[3] - 2 * z * a[3] ) +
                y * z * ( ( -4 * a[3] ) / 3 + ( 4 * z * a[3] ) / 3 ) ) +
          x * ( z * ( 2 * a[5] - 2 * z * a[5] ) +
                y * y *
                    ( z * ( 2 * a[7] - 2 * z * a[7] ) +
                      y * z * ( ( -4 * a[7] ) / 3 + ( 4 * z * a[7] ) / 3 ) ) +
                x * ( z * ( ( -2 * a[1] ) / 3 - ( 4 * a[5] ) / 3 +
                            z * ( ( 2 * a[1] ) / 3 + ( 4 * a[5] ) / 3 ) ) +
                      y * y *
                          ( y * z *
                                ( ( 4 * a[3] ) / 9 + z * ( ( -4 * a[3] ) / 9 - ( 8 * a[7] ) / 9 ) +
                                  ( 8 * a[7] ) / 9 ) +
                            z * ( ( -2 * a[3] ) / 3 - ( 4 * a[7] ) / 3 +
                                  z * ( ( 2 * a[3] ) / 3 + ( 4 * a[7] ) / 3 ) ) ) ) ) );
    result[4] =
        2 * a[4] + z * z * ( 2 * a[5] - ( 4 * z * a[5] ) / 3 ) +
        y * y *
            ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
              y * ( ( -4 * a[6] ) / 3 + z * z * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 9 ) ) ) +
        x * ( -2 * a[0] - 4 * a[4] +
              z * z * ( -2 * a[1] - 4 * a[5] + z * ( ( 4 * a[1] ) / 3 + ( 8 * a[5] ) / 3 ) ) +
              y * y *
                  ( -2 * a[2] - 4 * a[6] +
                    z * z * ( -2 * a[3] - 4 * a[7] + z * ( ( 4 * a[3] ) / 3 + ( 8 * a[7] ) / 3 ) ) +
                    y * ( ( 4 * a[2] ) / 3 + ( 8 * a[6] ) / 3 +
                          z * z *
                              ( ( 4 * a[3] ) / 3 + z * ( ( -8 * a[3] ) / 9 - ( 16 * a[7] ) / 9 ) +
                                ( 8 * a[7] ) / 3 ) ) ) +
              2 * c[1] );
    result[5] =
        y * ( 2 * a[2] + z * z * ( 2 * a[3] - ( 4 * z * a[3] ) / 3 ) +
              y * ( -2 * a[2] + z * z * ( -2 * a[3] + ( 4 * z * a[3] ) / 3 ) ) ) +
        x * ( x * y *
                  ( -2 * a[2] - 4 * a[6] +
                    z * z * ( -2 * a[3] - 4 * a[7] + z * ( ( 4 * a[3] ) / 3 + ( 8 * a[7] ) / 3 ) ) +
                    y * ( 2 * a[2] + 4 * a[6] +
                          z * z *
                              ( 2 * a[3] + z * ( ( -4 * a[3] ) / 3 - ( 8 * a[7] ) / 3 ) +
                                4 * a[7] ) ) ) +
              y * ( 4 * a[6] + z * z * ( 4 * a[7] - ( 8 * z * a[7] ) / 3 ) +
                    y * ( -4 * a[6] + z * z * ( -4 * a[7] + ( 8 * z * a[7] ) / 3 ) ) ) );
    result[6] =
        z * ( 2 * a[1] - 2 * z * a[1] ) +
        y * y *
            ( z * ( 2 * a[3] - 2 * z * a[3] ) +
              y * z * ( ( -4 * a[3] ) / 3 + ( 4 * z * a[3] ) / 3 ) ) +
        x * ( z * ( 4 * a[5] - 4 * z * a[5] ) +
              y * y *
                  ( z * ( 4 * a[7] - 4 * z * a[7] ) +
                    y * z * ( ( -8 * a[7] ) / 3 + ( 8 * z * a[7] ) / 3 ) ) +
              x * ( z * ( -2 * a[1] - 4 * a[5] + z * ( 2 * a[1] + 4 * a[5] ) ) +
                    y * y *
                        ( y * z *
                              ( ( 4 * a[3] ) / 3 + z * ( ( -4 * a[3] ) / 3 - ( 8 * a[7] ) / 3 ) +
                                ( 8 * a[7] ) / 3 ) +
                          z * ( -2 * a[3] - 4 * a[7] + z * ( 2 * a[3] + 4 * a[7] ) ) ) ) );
    result[7] =
        x *
        ( 2 * a[2] + z * z * ( 2 * a[3] - ( 4 * z * a[3] ) / 3 ) +
          y * ( -4 * a[2] + z * z * ( -4 * a[3] + ( 8 * z * a[3] ) / 3 ) ) +
          x * ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
                y * ( -4 * a[6] + z * z * ( -4 * a[7] + ( 8 * z * a[7] ) / 3 ) ) +
                x * ( ( -2 * a[2] ) / 3 - ( 4 * a[6] ) / 3 +
                      z * z *
                          ( ( -2 * a[3] ) / 3 + z * ( ( 4 * a[3] ) / 9 + ( 8 * a[7] ) / 9 ) -
                            ( 4 * a[7] ) / 3 ) +
                      y * ( ( 4 * a[2] ) / 3 + ( 8 * a[6] ) / 3 +
                            z * z *
                                ( ( 4 * a[3] ) / 3 + z * ( ( -8 * a[3] ) / 9 - ( 16 * a[7] ) / 9 ) +
                                  ( 8 * a[7] ) / 3 ) ) ) ) );
    result[8] =
        x * ( y * ( z * ( 4 * a[3] - 4 * z * a[3] ) + y * z * ( -4 * a[3] + 4 * z * a[3] ) ) +
              x * ( y * ( z * ( 4 * a[7] - 4 * z * a[7] ) + y * z * ( -4 * a[7] + 4 * z * a[7] ) ) +
                    x * y *
                        ( y * z *
                              ( ( 4 * a[3] ) / 3 + z * ( ( -4 * a[3] ) / 3 - ( 8 * a[7] ) / 3 ) +
                                ( 8 * a[7] ) / 3 ) +
                          z * ( ( -4 * a[3] ) / 3 - ( 8 * a[7] ) / 3 +
                                z * ( ( 4 * a[3] ) / 3 + ( 8 * a[7] ) / 3 ) ) ) ) );
    result[9] =
        x *
        ( 2 * a[1] - 4 * z * a[1] +
          y * y * ( 2 * a[3] - 4 * z * a[3] + y * ( ( -4 * a[3] ) / 3 + ( 8 * z * a[3] ) / 3 ) ) +
          x * ( 2 * a[5] - 4 * z * a[5] +
                y * y *
                    ( 2 * a[7] - 4 * z * a[7] + y * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 3 ) ) +
                x * ( ( -2 * a[1] ) / 3 - ( 4 * a[5] ) / 3 +
                      z * ( ( 4 * a[1] ) / 3 + ( 8 * a[5] ) / 3 ) +
                      y * y *
                          ( ( -2 * a[3] ) / 3 +
                            y * ( ( 4 * a[3] ) / 9 + z * ( ( -8 * a[3] ) / 9 - ( 16 * a[7] ) / 9 ) +
                                  ( 8 * a[7] ) / 9 ) -
                            ( 4 * a[7] ) / 3 + z * ( ( 4 * a[3] ) / 3 + ( 8 * a[7] ) / 3 ) ) ) ) );
    return result;
}


static std::array<double, 10>
cubic_dirichlet2( double x, double y, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 2, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 8, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    result[0] =
        c[0] +
        x * ( a[0] + z * z * ( a[1] - ( 2 * z * a[1] ) / 3 ) +
              y * y *
                  ( a[2] + z * z * ( a[3] - ( 2 * z * a[3] ) / 3 ) +
                    y * ( ( -2 * a[2] ) / 3 +
                          z * z * ( ( -2 * a[3] ) / 3 + ( 4 * z * a[3] ) / 9 ) ) ) +
              x * ( a[4] + z * z * ( a[5] - ( 2 * z * a[5] ) / 3 ) +
                    y * y *
                        ( a[6] + z * z * ( a[7] - ( 2 * z * a[7] ) / 3 ) +
                          y * ( ( -2 * a[6] ) / 3 +
                                z * z * ( ( -2 * a[7] ) / 3 + ( 4 * z * a[7] ) / 9 ) ) ) +
                    x * ( -a[0] - a[4] +
                          z * z * ( -a[1] + z * ( ( 2 * a[1] ) / 3 + ( 2 * a[5] ) / 3 ) - a[5] ) +
                          y * y *
                              ( -a[2] - a[6] +
                                y * ( ( 2 * a[2] ) / 3 + ( 2 * a[6] ) / 3 +
                                      z * z *
                                          ( ( 2 * a[3] ) / 3 +
                                            z * ( ( -4 * a[3] ) / 9 - ( 4 * a[7] ) / 9 ) +
                                            ( 2 * a[7] ) / 3 ) ) +
                                z * z *
                                    ( -a[3] + z * ( ( 2 * a[3] ) / 3 + ( 2 * a[7] ) / 3 ) -
                                      a[7] ) ) -
                          c[0] + c[1] ) ) );
    result[1] =
        a[0] + z * z * ( a[1] - ( 2 * z * a[1] ) / 3 ) +
        y * y *
            ( a[2] + z * z * ( a[3] - ( 2 * z * a[3] ) / 3 ) +
              y * ( ( -2 * a[2] ) / 3 + z * z * ( ( -2 * a[3] ) / 3 + ( 4 * z * a[3] ) / 9 ) ) ) +
        x * ( 2 * a[4] + z * z * ( 2 * a[5] - ( 4 * z * a[5] ) / 3 ) +
              y * y *
                  ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
                    y * ( ( -4 * a[6] ) / 3 +
                          z * z * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 9 ) ) ) +
              x * ( -3 * a[0] - 3 * a[4] +
                    z * z * ( -3 * a[1] - 3 * a[5] + z * ( 2 * a[1] + 2 * a[5] ) ) +
                    y * y *
                        ( -3 * a[2] - 3 * a[6] +
                          z * z * ( -3 * a[3] - 3 * a[7] + z * ( 2 * a[3] + 2 * a[7] ) ) +
                          y * ( 2 * a[2] + 2 * a[6] +
                                z * z *
                                    ( 2 * a[3] + z * ( ( -4 * a[3] ) / 3 - ( 4 * a[7] ) / 3 ) +
                                      2 * a[7] ) ) ) -
                    3 * c[0] + 3 * c[1] ) );
    result[2] =
        x *
        ( y * ( 2 * a[2] + z * z * ( 2 * a[3] - ( 4 * z * a[3] ) / 3 ) +
                y * ( -2 * a[2] + z * z * ( -2 * a[3] + ( 4 * z * a[3] ) / 3 ) ) ) +
          x * ( x * y *
                    ( -2 * a[2] - 2 * a[6] +
                      z * z *
                          ( -2 * a[3] - 2 * a[7] + z * ( ( 4 * a[3] ) / 3 + ( 4 * a[7] ) / 3 ) ) +
                      y * ( 2 * a[2] + 2 * a[6] +
                            z * z *
                                ( 2 * a[3] + z * ( ( -4 * a[3] ) / 3 - ( 4 * a[7] ) / 3 ) +
                                  2 * a[7] ) ) ) +
                y * ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
                      y * ( -2 * a[6] + z * z * ( -2 * a[7] + ( 4 * z * a[7] ) / 3 ) ) ) ) );
    result[3] =
        x *
        ( z * ( 2 * a[1] - 2 * z * a[1] ) +
          y * y *
              ( z * ( 2 * a[3] - 2 * z * a[3] ) +
                y * z * ( ( -4 * a[3] ) / 3 + ( 4 * z * a[3] ) / 3 ) ) +
          x * ( z * ( 2 * a[5] - 2 * z * a[5] ) +
                y * y *
                    ( z * ( 2 * a[7] - 2 * z * a[7] ) +
                      y * z * ( ( -4 * a[7] ) / 3 + ( 4 * z * a[7] ) / 3 ) ) +
                x * ( z * ( -2 * a[1] - 2 * a[5] + z * ( 2 * a[1] + 2 * a[5] ) ) +
                      y * y *
                          ( y * z *
                                ( ( 4 * a[3] ) / 3 + z * ( ( -4 * a[3] ) / 3 - ( 4 * a[7] ) / 3 ) +
                                  ( 4 * a[7] ) / 3 ) +
                            z * ( -2 * a[3] - 2 * a[7] + z * ( 2 * a[3] + 2 * a[7] ) ) ) ) ) );
    result[4] =
        2 * a[4] + z * z * ( 2 * a[5] - ( 4 * z * a[5] ) / 3 ) +
        y * y *
            ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
              y * ( ( -4 * a[6] ) / 3 + z * z * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 9 ) ) ) +
        x * ( -6 * a[0] - 6 * a[4] +
              z * z * ( -6 * a[1] - 6 * a[5] + z * ( 4 * a[1] + 4 * a[5] ) ) +
              y * y *
                  ( -6 * a[2] - 6 * a[6] +
                    z * z * ( -6 * a[3] - 6 * a[7] + z * ( 4 * a[3] + 4 * a[7] ) ) +
                    y * ( 4 * a[2] + 4 * a[6] +
                          z * z *
                              ( 4 * a[3] + z * ( ( -8 * a[3] ) / 3 - ( 8 * a[7] ) / 3 ) +
                                4 * a[7] ) ) ) -
              6 * c[0] + 6 * c[1] );
    result[5] =
        y * ( 2 * a[2] + z * z * ( 2 * a[3] - ( 4 * z * a[3] ) / 3 ) +
              y * ( -2 * a[2] + z * z * ( -2 * a[3] + ( 4 * z * a[3] ) / 3 ) ) ) +
        x * ( x * y *
                  ( -6 * a[2] - 6 * a[6] +
                    z * z * ( -6 * a[3] - 6 * a[7] + z * ( 4 * a[3] + 4 * a[7] ) ) +
                    y * ( 6 * a[2] + 6 * a[6] +
                          z * z * ( 6 * a[3] + z * ( -4 * a[3] - 4 * a[7] ) + 6 * a[7] ) ) ) +
              y * ( 4 * a[6] + z * z * ( 4 * a[7] - ( 8 * z * a[7] ) / 3 ) +
                    y * ( -4 * a[6] + z * z * ( -4 * a[7] + ( 8 * z * a[7] ) / 3 ) ) ) );
    result[6] = z * ( 2 * a[1] - 2 * z * a[1] ) +
                y * y *
                    ( z * ( 2 * a[3] - 2 * z * a[3] ) +
                      y * z * ( ( -4 * a[3] ) / 3 + ( 4 * z * a[3] ) / 3 ) ) +
                x * ( z * ( 4 * a[5] - 4 * z * a[5] ) +
                      y * y *
                          ( z * ( 4 * a[7] - 4 * z * a[7] ) +
                            y * z * ( ( -8 * a[7] ) / 3 + ( 8 * z * a[7] ) / 3 ) ) +
                      x * ( z * ( -6 * a[1] - 6 * a[5] + z * ( 6 * a[1] + 6 * a[5] ) ) +
                            y * y *
                                ( y * z * ( 4 * a[3] + z * ( -4 * a[3] - 4 * a[7] ) + 4 * a[7] ) +
                                  z * ( -6 * a[3] - 6 * a[7] + z * ( 6 * a[3] + 6 * a[7] ) ) ) ) );
    result[7] =
        x *
        ( 2 * a[2] + z * z * ( 2 * a[3] - ( 4 * z * a[3] ) / 3 ) +
          y * ( -4 * a[2] + z * z * ( -4 * a[3] + ( 8 * z * a[3] ) / 3 ) ) +
          x * ( 2 * a[6] + z * z * ( 2 * a[7] - ( 4 * z * a[7] ) / 3 ) +
                y * ( -4 * a[6] + z * z * ( -4 * a[7] + ( 8 * z * a[7] ) / 3 ) ) +
                x * ( -2 * a[2] - 2 * a[6] +
                      z * z *
                          ( -2 * a[3] - 2 * a[7] + z * ( ( 4 * a[3] ) / 3 + ( 4 * a[7] ) / 3 ) ) +
                      y * ( 4 * a[2] + 4 * a[6] +
                            z * z *
                                ( 4 * a[3] + z * ( ( -8 * a[3] ) / 3 - ( 8 * a[7] ) / 3 ) +
                                  4 * a[7] ) ) ) ) );
    result[8] =
        x * ( y * ( z * ( 4 * a[3] - 4 * z * a[3] ) + y * z * ( -4 * a[3] + 4 * z * a[3] ) ) +
              x * ( y * ( z * ( 4 * a[7] - 4 * z * a[7] ) + y * z * ( -4 * a[7] + 4 * z * a[7] ) ) +
                    x * y *
                        ( y * z * ( 4 * a[3] + z * ( -4 * a[3] - 4 * a[7] ) + 4 * a[7] ) +
                          z * ( -4 * a[3] - 4 * a[7] + z * ( 4 * a[3] + 4 * a[7] ) ) ) ) );
    result[9] =
        x *
        ( 2 * a[1] - 4 * z * a[1] +
          y * y * ( 2 * a[3] - 4 * z * a[3] + y * ( ( -4 * a[3] ) / 3 + ( 8 * z * a[3] ) / 3 ) ) +
          x * ( 2 * a[5] - 4 * z * a[5] +
                y * y *
                    ( 2 * a[7] - 4 * z * a[7] + y * ( ( -4 * a[7] ) / 3 + ( 8 * z * a[7] ) / 3 ) ) +
                x * ( -2 * a[1] - 2 * a[5] + z * ( 4 * a[1] + 4 * a[5] ) +
                      y * y *
                          ( -2 * a[3] - 2 * a[7] +
                            y * ( ( 4 * a[3] ) / 3 + z * ( ( -8 * a[3] ) / 3 - ( 8 * a[7] ) / 3 ) +
                                  ( 4 * a[7] ) / 3 ) +
                            z * ( 4 * a[3] + 4 * a[7] ) ) ) ) );
    return result;
}


static std::array<double, 10>
cubic_none( double x, double y, double z, const ManufacturedSolution *mfs )
{
    const auto &a = mfs->geta();
    AMP_INSIST( a.size() >= 20, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    result[0] = a[0] + z * ( a[1] + z * ( a[2] + z * a[3] ) ) +
                y * ( a[4] + z * ( a[5] + z * a[6] ) + y * ( a[7] + z * a[8] + y * a[9] ) ) +
                x * ( a[10] + z * ( a[11] + z * a[12] ) + y * ( a[13] + z * a[14] + y * a[15] ) +
                      x * ( a[16] + z * a[17] + y * a[18] + x * a[19] ) );
    result[1] = a[10] + z * ( a[11] + z * a[12] + y * a[14] + 2 * x * a[17] ) +
                y * ( a[13] + y * a[15] + 2 * x * a[18] ) + x * ( 2 * a[16] + 3 * x * a[19] );
    result[2] = a[4] + z * ( a[5] + z * a[6] + 2 * y * a[8] + x * a[14] ) +
                y * ( 2 * a[7] + 3 * y * a[9] + 2 * x * a[15] ) + x * ( a[13] + x * a[18] );
    result[3] = a[1] + z * ( 2 * a[2] + 3 * z * a[3] + 2 * y * a[6] + 2 * x * a[12] ) +
                y * ( a[5] + y * a[8] + x * a[14] ) + x * ( a[11] + x * a[17] );
    result[4] = 2 * a[16] + 2 * z * a[17] + 2 * y * a[18] + 6 * x * a[19];
    result[5] = a[13] + z * a[14] + 2 * y * a[15] + 2 * x * a[18];
    result[6] = a[11] + 2 * z * a[12] + y * a[14] + 2 * x * a[17];
    result[7] = 2 * a[7] + 2 * z * a[8] + 6 * y * a[9] + 2 * x * a[15];
    result[8] = a[5] + 2 * z * a[6] + 2 * y * a[8] + x * a[14];
    result[9] = 2 * a[2] + 6 * z * a[3] + 2 * y * a[6] + 2 * x * a[12];
    return result;
}


static std::array<double, 10>
quad_cyl_rod_none( double r, double th, double z, const ManufacturedSolution *mfs )
{
    const auto &a = mfs->geta();
    AMP_INSIST( a.size() >= 15, "incorrect number of parameters specified" );

    double sth = sin( th ), cth = cos( th );
    std::array<double, 10> result;
    result[0] = a[0] + cth * a[0] + cth * cth * a[0] + sth * a[0] + cth * sth * a[0] +
                sth * sth * a[0] + z * ( a[1] + cth * a[1] + sth * a[1] + z * a[2] ) +
                r * ( a[10] + cth * a[10] + sth * a[10] + z * a[11] + r * a[14] );
    result[1] = a[10] + cth * a[10] + sth * a[10] + z * a[11] + 2 * r * a[14];
    result[2] = sth * ( -a[0] - sth * a[0] - z * a[1] ) + cth * ( a[0] + cth * a[0] + z * a[1] ) +
                r * ( cth * a[10] - sth * a[10] );
    result[3] = a[1] + cth * a[1] + sth * a[1] + 2 * z * a[2] + r * a[11];
    result[4] = 2 * a[14];
    result[5] = cth * a[10] - sth * a[10];
    result[6] = a[11];
    result[7] = cth * ( -a[0] - z * a[1] ) + sth * ( -a[0] - 4 * cth * a[0] - z * a[1] ) +
                r * ( -( cth * a[10] ) - sth * a[10] );
    result[8] = cth * a[1] - sth * a[1];
    result[9] = 2 * a[2];
    return result;
}


static std::array<double, 10>
cubic_cyl_shell_neumann( double r, double th, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 4, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 9, "incorrect number of parameters specified" );
    double sth = sin( th ), cth = cos( th );
    std::array<double, 10> result;
    result[0] = a[0] + cth * ( a[1] + cth * a[2] ) +
                sth * ( a[3] + cth * ( a[4] + cth * a[5] ) +
                        sth * ( a[6] + cth * ( a[7] + cth * a[8] ) ) ) +
                r * ( -2 * c[0] - c[1] + r * ( c[0] + c[1] ) ) +
                z * ( -c[2] + z * ( c[2] / 2 + c[3] / 2 ) );
    result[1] = -2 * c[0] - c[1] + r * ( 2 * c[0] + 2 * c[1] );
    result[2] =
        cth * ( a[3] + cth * ( a[4] + cth * a[5] ) ) +
        sth * ( -a[1] + sth * ( -a[4] - 2 * cth * a[5] + sth * ( -a[7] - 2 * cth * a[8] ) ) +
                cth * ( -2 * a[2] + 2 * a[6] + cth * ( 2 * a[7] + 2 * cth * a[8] ) ) );
    result[3] = -c[2] + z * ( c[2] + c[3] );
    result[4] = 2 * c[0] + 2 * c[1];
    result[5] = 0;
    result[6] = 0;
    result[7] =
        cth * ( -a[1] + cth * ( -2 * a[2] + 2 * a[6] + cth * ( 2 * a[7] + 2 * cth * a[8] ) ) ) +
        sth * ( -a[3] + cth * ( -4 * a[4] - 7 * cth * a[5] ) +
                sth * ( 2 * a[2] - 2 * a[6] + cth * ( -7 * a[7] - 12 * cth * a[8] ) +
                        sth * ( 2 * a[5] + 2 * sth * a[8] ) ) );
    result[8] = 0;
    result[9] = c[2] + c[3];
    return result;
}


static std::array<double, 10>
cubic_cyl_rod_dirichletz2( double r, double th, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 2, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 96, "incorrect number of parameters specified" );

    double sth = sin( th ), cth = cos( th );
    std::array<double, 10> result;
    result[0] =
        cth * ( z * ( a[2] + z * ( z * ( -a[2] - a[3] ) + a[3] ) ) +
                cth * ( z * ( a[4] + z * ( z * ( -a[4] - a[5] ) + a[5] ) ) +
                        cth * z * ( a[6] + z * ( z * ( -a[6] - a[7] ) + a[7] ) ) ) ) +
        sth *
            ( z * ( a[8] + z * ( z * ( -a[8] - a[9] ) + a[9] ) ) +
              cth * ( z * ( a[10] + z * ( z * ( -a[10] - a[11] ) + a[11] ) ) +
                      cth * ( z * ( a[12] + z * ( z * ( -a[12] - a[13] ) + a[13] ) ) +
                              cth * z * ( a[14] + z * ( z * ( -a[14] - a[15] ) + a[15] ) ) ) ) +
              sth *
                  ( z * ( a[16] + z * ( z * ( -a[16] - a[17] ) + a[17] ) ) +
                    cth * ( z * ( a[18] + z * ( z * ( -a[18] - a[19] ) + a[19] ) ) +
                            cth * ( z * ( a[20] + z * ( z * ( -a[20] - a[21] ) + a[21] ) ) +
                                    cth * z *
                                        ( a[22] + z * ( z * ( -a[22] - a[23] ) + a[23] ) ) ) ) +
                    sth * ( z * ( a[24] + z * ( z * ( -a[24] - a[25] ) + a[25] ) ) +
                            cth * ( z * ( a[26] + z * ( z * ( -a[26] - a[27] ) + a[27] ) ) +
                                    cth * ( z * ( a[28] + z * ( z * ( -a[28] - a[29] ) + a[29] ) ) +
                                            cth * z *
                                                ( a[30] + z * ( z * ( -a[30] - a[31] ) +
                                                                a[31] ) ) ) ) ) ) ) +
        r * ( z * ( a[32] + z * ( z * ( -a[32] - a[33] ) + a[33] ) ) +
              cth * ( z * ( a[34] + z * ( z * ( -a[34] - a[35] ) + a[35] ) ) +
                      cth * ( z * ( a[36] + z * ( z * ( -a[36] - a[37] ) + a[37] ) ) +
                              cth * z * ( a[38] + z * ( z * ( -a[38] - a[39] ) + a[39] ) ) ) ) +
              sth *
                  ( z * ( a[40] + z * ( z * ( -a[40] - a[41] ) + a[41] ) ) +
                    cth *
                        ( z * ( a[42] + z * ( z * ( -a[42] - a[43] ) + a[43] ) ) +
                          cth * ( z * ( a[44] + z * ( z * ( -a[44] - a[45] ) + a[45] ) ) +
                                  cth * z * ( a[46] + z * ( z * ( -a[46] - a[47] ) + a[47] ) ) ) ) +
                    sth * ( z * ( a[48] + z * ( z * ( -a[48] - a[49] ) + a[49] ) ) +
                            cth * ( z * ( a[50] + z * ( z * ( -a[50] - a[51] ) + a[51] ) ) +
                                    cth * ( z * ( a[52] + z * ( z * ( -a[52] - a[53] ) + a[53] ) ) +
                                            cth * z *
                                                ( a[54] +
                                                  z * ( z * ( -a[54] - a[55] ) + a[55] ) ) ) ) +
                            sth * ( z * ( a[56] + z * ( z * ( -a[56] - a[57] ) + a[57] ) ) +
                                    cth * ( z * ( a[58] + z * ( z * ( -a[58] - a[59] ) + a[59] ) ) +
                                            cth * ( z * ( a[60] +
                                                          z * ( z * ( -a[60] - a[61] ) + a[61] ) ) +
                                                    cth * z *
                                                        ( a[62] + z * ( z * ( -a[62] - a[63] ) +
                                                                        a[63] ) ) ) ) ) ) ) +
              r * ( z * ( a[64] + z * ( z * ( -a[64] - a[65] ) + a[65] ) ) +
                    cth *
                        ( z * ( a[66] + z * ( z * ( -a[66] - a[67] ) + a[67] ) ) +
                          cth * ( z * ( a[68] + z * ( z * ( -a[68] - a[69] ) + a[69] ) ) +
                                  cth * z * ( a[70] + z * ( z * ( -a[70] - a[71] ) + a[71] ) ) ) ) +
                    r * ( z * ( -a[32] / 3 - ( 2 * a[64] ) / 3 +
                                z * ( -a[33] / 3 +
                                      z * ( a[32] / 3 + a[33] / 3 + ( 2 * a[64] ) / 3 +
                                            ( 2 * a[65] ) / 3 ) -
                                      ( 2 * a[65] ) / 3 ) ) +
                          cth *
                              ( z * ( -a[34] / 3 - ( 2 * a[66] ) / 3 +
                                      z * ( -a[35] / 3 +
                                            z * ( a[34] / 3 + a[35] / 3 + ( 2 * a[66] ) / 3 +
                                                  ( 2 * a[67] ) / 3 ) -
                                            ( 2 * a[67] ) / 3 ) ) +
                                cth * ( z * ( -a[36] / 3 - ( 2 * a[68] ) / 3 +
                                              z * ( -a[37] / 3 +
                                                    z * ( a[36] / 3 + a[37] / 3 +
                                                          ( 2 * a[68] ) / 3 + ( 2 * a[69] ) / 3 ) -
                                                    ( 2 * a[69] ) / 3 ) ) +
                                        cth * z *
                                            ( -a[38] / 3 - ( 2 * a[70] ) / 3 +
                                              z * ( -a[39] / 3 +
                                                    z * ( a[38] / 3 + a[39] / 3 +
                                                          ( 2 * a[70] ) / 3 + ( 2 * a[71] ) / 3 ) -
                                                    ( 2 * a[71] ) / 3 ) ) ) ) +
                          sth *
                              ( z * ( -a[40] / 3 - ( 2 * a[72] ) / 3 +
                                      z * ( -a[41] / 3 +
                                            z * ( a[40] / 3 + a[41] / 3 + ( 2 * a[72] ) / 3 +
                                                  ( 2 * a[73] ) / 3 ) -
                                            ( 2 * a[73] ) / 3 ) ) +
                                cth * ( z * ( -a[42] / 3 - ( 2 * a[74] ) / 3 +
                                              z * ( -a[43] / 3 +
                                                    z * ( a[42] / 3 + a[43] / 3 +
                                                          ( 2 * a[74] ) / 3 + ( 2 * a[75] ) / 3 ) -
                                                    ( 2 * a[75] ) / 3 ) ) +
                                        cth * ( z * ( -a[44] / 3 - ( 2 * a[76] ) / 3 +
                                                      z * ( -a[45] / 3 +
                                                            z * ( a[44] / 3 + a[45] / 3 +
                                                                  ( 2 * a[76] ) / 3 +
                                                                  ( 2 * a[77] ) / 3 ) -
                                                            ( 2 * a[77] ) / 3 ) ) +
                                                cth * z *
                                                    ( -a[46] / 3 - ( 2 * a[78] ) / 3 +
                                                      z * ( -a[47] / 3 +
                                                            z * ( a[46] / 3 + a[47] / 3 +
                                                                  ( 2 * a[78] ) / 3 +
                                                                  ( 2 * a[79] ) / 3 ) -
                                                            ( 2 * a[79] ) / 3 ) ) ) ) +
                                sth *
                                    ( z * ( -a[48] / 3 - ( 2 * a[80] ) / 3 +
                                            z * ( -a[49] / 3 +
                                                  z * ( a[48] / 3 + a[49] / 3 + ( 2 * a[80] ) / 3 +
                                                        ( 2 * a[81] ) / 3 ) -
                                                  ( 2 * a[81] ) / 3 ) ) +
                                      cth * ( z * ( -a[50] / 3 - ( 2 * a[82] ) / 3 +
                                                    z * ( -a[51] / 3 +
                                                          z * ( a[50] / 3 + a[51] / 3 +
                                                                ( 2 * a[82] ) / 3 +
                                                                ( 2 * a[83] ) / 3 ) -
                                                          ( 2 * a[83] ) / 3 ) ) +
                                              cth * ( z * ( -a[52] / 3 - ( 2 * a[84] ) / 3 +
                                                            z * ( -a[53] / 3 +
                                                                  z * ( a[52] / 3 + a[53] / 3 +
                                                                        ( 2 * a[84] ) / 3 +
                                                                        ( 2 * a[85] ) / 3 ) -
                                                                  ( 2 * a[85] ) / 3 ) ) +
                                                      cth * z *
                                                          ( -a[54] / 3 - ( 2 * a[86] ) / 3 +
                                                            z * ( -a[55] / 3 +
                                                                  z * ( a[54] / 3 + a[55] / 3 +
                                                                        ( 2 * a[86] ) / 3 +
                                                                        ( 2 * a[87] ) / 3 ) -
                                                                  ( 2 * a[87] ) / 3 ) ) ) ) +
                                      sth *
                                          ( z * ( -a[56] / 3 - ( 2 * a[88] ) / 3 +
                                                  z * ( -a[57] / 3 +
                                                        z * ( a[56] / 3 + a[57] / 3 +
                                                              ( 2 * a[88] ) / 3 +
                                                              ( 2 * a[89] ) / 3 ) -
                                                        ( 2 * a[89] ) / 3 ) ) +
                                            cth *
                                                ( z * ( -a[58] / 3 - ( 2 * a[90] ) / 3 +
                                                        z * ( -a[59] / 3 +
                                                              z * ( a[58] / 3 + a[59] / 3 +
                                                                    ( 2 * a[90] ) / 3 +
                                                                    ( 2 * a[91] ) / 3 ) -
                                                              ( 2 * a[91] ) / 3 ) ) +
                                                  cth * ( z * ( -a[60] / 3 - ( 2 * a[92] ) / 3 +
                                                                z * ( -a[61] / 3 +
                                                                      z * ( a[60] / 3 + a[61] / 3 +
                                                                            ( 2 * a[92] ) / 3 +
                                                                            ( 2 * a[93] ) / 3 ) -
                                                                      ( 2 * a[93] ) / 3 ) ) +
                                                          cth * z *
                                                              ( -a[62] / 3 - ( 2 * a[94] ) / 3 +
                                                                z * ( -a[63] / 3 +
                                                                      z * ( a[62] / 3 + a[63] / 3 +
                                                                            ( 2 * a[94] ) / 3 +
                                                                            ( 2 * a[95] ) / 3 ) -
                                                                      ( 2 * a[95] ) /
                                                                          3 ) ) ) ) ) ) ) ) +
                    sth *
                        ( z * ( a[72] + z * ( z * ( -a[72] - a[73] ) + a[73] ) ) +
                          cth *
                              ( z * ( a[74] + z * ( z * ( -a[74] - a[75] ) + a[75] ) ) +
                                cth * ( z * ( a[76] + z * ( z * ( -a[76] - a[77] ) + a[77] ) ) +
                                        cth * z *
                                            ( a[78] + z * ( z * ( -a[78] - a[79] ) + a[79] ) ) ) ) +
                          sth *
                              ( z * ( a[80] + z * ( z * ( -a[80] - a[81] ) + a[81] ) ) +
                                cth * ( z * ( a[82] + z * ( z * ( -a[82] - a[83] ) + a[83] ) ) +
                                        cth * ( z * ( a[84] +
                                                      z * ( z * ( -a[84] - a[85] ) + a[85] ) ) +
                                                cth * z *
                                                    ( a[86] +
                                                      z * ( z * ( -a[86] - a[87] ) + a[87] ) ) ) ) +
                                sth *
                                    ( z * ( a[88] + z * ( z * ( -a[88] - a[89] ) + a[89] ) ) +
                                      cth *
                                          ( z * ( a[90] + z * ( z * ( -a[90] - a[91] ) + a[91] ) ) +
                                            cth * ( z * ( a[92] +
                                                          z * ( z * ( -a[92] - a[93] ) + a[93] ) ) +
                                                    cth * z *
                                                        ( a[94] + z * ( z * ( -a[94] - a[95] ) +
                                                                        a[95] ) ) ) ) ) ) ) ) ) +
        c[0] + z * ( a[0] + z * ( a[1] + z * ( -a[0] - a[1] - c[0] + c[1] ) ) );
    result[1] =
        z * ( a[32] + z * ( z * ( -a[32] - a[33] ) + a[33] ) ) +
        cth * ( z * ( a[34] + z * ( z * ( -a[34] - a[35] ) + a[35] ) ) +
                cth * ( z * ( a[36] + z * ( z * ( -a[36] - a[37] ) + a[37] ) ) +
                        cth * z * ( a[38] + z * ( z * ( -a[38] - a[39] ) + a[39] ) ) ) ) +
        sth *
            ( z * ( a[40] + z * ( z * ( -a[40] - a[41] ) + a[41] ) ) +
              cth * ( z * ( a[42] + z * ( z * ( -a[42] - a[43] ) + a[43] ) ) +
                      cth * ( z * ( a[44] + z * ( z * ( -a[44] - a[45] ) + a[45] ) ) +
                              cth * z * ( a[46] + z * ( z * ( -a[46] - a[47] ) + a[47] ) ) ) ) +
              sth *
                  ( z * ( a[48] + z * ( z * ( -a[48] - a[49] ) + a[49] ) ) +
                    cth * ( z * ( a[50] + z * ( z * ( -a[50] - a[51] ) + a[51] ) ) +
                            cth * ( z * ( a[52] + z * ( z * ( -a[52] - a[53] ) + a[53] ) ) +
                                    cth * z *
                                        ( a[54] + z * ( z * ( -a[54] - a[55] ) + a[55] ) ) ) ) +
                    sth * ( z * ( a[56] + z * ( z * ( -a[56] - a[57] ) + a[57] ) ) +
                            cth * ( z * ( a[58] + z * ( z * ( -a[58] - a[59] ) + a[59] ) ) +
                                    cth * ( z * ( a[60] + z * ( z * ( -a[60] - a[61] ) + a[61] ) ) +
                                            cth * z *
                                                ( a[62] + z * ( z * ( -a[62] - a[63] ) +
                                                                a[63] ) ) ) ) ) ) ) +
        r * ( z * ( 2 * a[64] + z * ( z * ( -2 * a[64] - 2 * a[65] ) + 2 * a[65] ) ) +
              cth *
                  ( z * ( 2 * a[66] + z * ( z * ( -2 * a[66] - 2 * a[67] ) + 2 * a[67] ) ) +
                    cth * ( z * ( 2 * a[68] + z * ( z * ( -2 * a[68] - 2 * a[69] ) + 2 * a[69] ) ) +
                            cth * z *
                                ( 2 * a[70] +
                                  z * ( z * ( -2 * a[70] - 2 * a[71] ) + 2 * a[71] ) ) ) ) +
              sth *
                  ( z * ( 2 * a[72] + z * ( z * ( -2 * a[72] - 2 * a[73] ) + 2 * a[73] ) ) +
                    cth * ( z * ( 2 * a[74] + z * ( z * ( -2 * a[74] - 2 * a[75] ) + 2 * a[75] ) ) +
                            cth * ( z * ( 2 * a[76] +
                                          z * ( z * ( -2 * a[76] - 2 * a[77] ) + 2 * a[77] ) ) +
                                    cth * z *
                                        ( 2 * a[78] +
                                          z * ( z * ( -2 * a[78] - 2 * a[79] ) + 2 * a[79] ) ) ) ) +
                    sth * ( z * ( 2 * a[80] + z * ( z * ( -2 * a[80] - 2 * a[81] ) + 2 * a[81] ) ) +
                            cth * ( z * ( 2 * a[82] +
                                          z * ( z * ( -2 * a[82] - 2 * a[83] ) + 2 * a[83] ) ) +
                                    cth * ( z * ( 2 * a[84] + z * ( z * ( -2 * a[84] - 2 * a[85] ) +
                                                                    2 * a[85] ) ) +
                                            cth * z *
                                                ( 2 * a[86] + z * ( z * ( -2 * a[86] - 2 * a[87] ) +
                                                                    2 * a[87] ) ) ) ) +
                            sth * ( z * ( 2 * a[88] +
                                          z * ( z * ( -2 * a[88] - 2 * a[89] ) + 2 * a[89] ) ) +
                                    cth * ( z * ( 2 * a[90] + z * ( z * ( -2 * a[90] - 2 * a[91] ) +
                                                                    2 * a[91] ) ) +
                                            cth * ( z * ( 2 * a[92] +
                                                          z * ( z * ( -2 * a[92] - 2 * a[93] ) +
                                                                2 * a[93] ) ) +
                                                    cth * z *
                                                        ( 2 * a[94] +
                                                          z * ( z * ( -2 * a[94] - 2 * a[95] ) +
                                                                2 * a[95] ) ) ) ) ) ) ) +
              r * ( z * ( -a[32] - 2 * a[64] +
                          z * ( -a[33] - 2 * a[65] +
                                z * ( a[32] + a[33] + 2 * a[64] + 2 * a[65] ) ) ) +
                    cth * ( z * ( -a[34] - 2 * a[66] +
                                  z * ( -a[35] - 2 * a[67] +
                                        z * ( a[34] + a[35] + 2 * a[66] + 2 * a[67] ) ) ) +
                            cth * ( z * ( -a[36] - 2 * a[68] +
                                          z * ( -a[37] - 2 * a[69] +
                                                z * ( a[36] + a[37] + 2 * a[68] + 2 * a[69] ) ) ) +
                                    cth * z *
                                        ( -a[38] - 2 * a[70] +
                                          z * ( -a[39] - 2 * a[71] +
                                                z * ( a[38] + a[39] + 2 * a[70] +
                                                      2 * a[71] ) ) ) ) ) +
                    sth *
                        ( z * ( -a[40] - 2 * a[72] +
                                z * ( -a[41] - 2 * a[73] +
                                      z * ( a[40] + a[41] + 2 * a[72] + 2 * a[73] ) ) ) +
                          cth * ( z * ( -a[42] - 2 * a[74] +
                                        z * ( -a[43] - 2 * a[75] +
                                              z * ( a[42] + a[43] + 2 * a[74] + 2 * a[75] ) ) ) +
                                  cth * ( z * ( -a[44] - 2 * a[76] +
                                                z * ( -a[45] - 2 * a[77] +
                                                      z * ( a[44] + a[45] + 2 * a[76] +
                                                            2 * a[77] ) ) ) +
                                          cth * z *
                                              ( -a[46] - 2 * a[78] +
                                                z * ( -a[47] - 2 * a[79] +
                                                      z * ( a[46] + a[47] + 2 * a[78] +
                                                            2 * a[79] ) ) ) ) ) +
                          sth *
                              ( z * ( -a[48] - 2 * a[80] +
                                      z * ( -a[49] - 2 * a[81] +
                                            z * ( a[48] + a[49] + 2 * a[80] + 2 * a[81] ) ) ) +
                                cth * ( z * ( -a[50] - 2 * a[82] +
                                              z * ( -a[51] - 2 * a[83] +
                                                    z * ( a[50] + a[51] + 2 * a[82] +
                                                          2 * a[83] ) ) ) +
                                        cth * ( z * ( -a[52] - 2 * a[84] +
                                                      z * ( -a[53] - 2 * a[85] +
                                                            z * ( a[52] + a[53] + 2 * a[84] +
                                                                  2 * a[85] ) ) ) +
                                                cth * z *
                                                    ( -a[54] - 2 * a[86] +
                                                      z * ( -a[55] - 2 * a[87] +
                                                            z * ( a[54] + a[55] + 2 * a[86] +
                                                                  2 * a[87] ) ) ) ) ) +
                                sth *
                                    ( z * ( -a[56] - 2 * a[88] +
                                            z * ( -a[57] - 2 * a[89] +
                                                  z * ( a[56] + a[57] + 2 * a[88] +
                                                        2 * a[89] ) ) ) +
                                      cth *
                                          ( z * ( -a[58] - 2 * a[90] +
                                                  z * ( -a[59] - 2 * a[91] +
                                                        z * ( a[58] + a[59] + 2 * a[90] +
                                                              2 * a[91] ) ) ) +
                                            cth * ( z * ( -a[60] - 2 * a[92] +
                                                          z * ( -a[61] - 2 * a[93] +
                                                                z * ( a[60] + a[61] + 2 * a[92] +
                                                                      2 * a[93] ) ) ) +
                                                    cth * z *
                                                        ( -a[62] - 2 * a[94] +
                                                          z * ( -a[63] - 2 * a[95] +
                                                                z * ( a[62] + a[63] + 2 * a[94] +
                                                                      2 * a[95] ) ) ) ) ) ) ) ) ) );
    result[2] =
        cth * ( z * ( a[8] + z * ( z * ( -a[8] - a[9] ) + a[9] ) ) +
                cth * ( z * ( a[10] + z * ( z * ( -a[10] - a[11] ) + a[11] ) ) +
                        cth * ( z * ( a[12] + z * ( z * ( -a[12] - a[13] ) + a[13] ) ) +
                                cth * z * ( a[14] + z * ( z * ( -a[14] - a[15] ) + a[15] ) ) ) ) ) +
        sth *
            ( z * ( -a[2] + z * ( -a[3] + z * ( a[2] + a[3] ) ) ) +
              cth * ( z * ( -2 * a[4] + 2 * a[16] +
                            z * ( -2 * a[5] + z * ( 2 * a[4] + 2 * a[5] - 2 * a[16] - 2 * a[17] ) +
                                  2 * a[17] ) ) +
                      cth * ( z * ( -3 * a[6] + 2 * a[18] +
                                    z * ( -3 * a[7] +
                                          z * ( 3 * a[6] + 3 * a[7] - 2 * a[18] - 2 * a[19] ) +
                                          2 * a[19] ) ) +
                              cth * ( z * ( 2 * a[20] +
                                            z * ( z * ( -2 * a[20] - 2 * a[21] ) + 2 * a[21] ) ) +
                                      cth * z *
                                          ( 2 * a[22] + z * ( z * ( -2 * a[22] - 2 * a[23] ) +
                                                              2 * a[23] ) ) ) ) ) +
              sth *
                  ( z * ( -a[10] + z * ( -a[11] + z * ( a[10] + a[11] ) ) ) +
                    cth *
                        ( z * ( -2 * a[12] + 3 * a[24] +
                                z * ( -2 * a[13] +
                                      z * ( 2 * a[12] + 2 * a[13] - 3 * a[24] - 3 * a[25] ) +
                                      3 * a[25] ) ) +
                          cth *
                              ( z * ( -3 * a[14] + 3 * a[26] +
                                      z * ( -3 * a[15] +
                                            z * ( 3 * a[14] + 3 * a[15] - 3 * a[26] - 3 * a[27] ) +
                                            3 * a[27] ) ) +
                                cth * ( z * ( 3 * a[28] +
                                              z * ( z * ( -3 * a[28] - 3 * a[29] ) + 3 * a[29] ) ) +
                                        cth * z *
                                            ( 3 * a[30] + z * ( z * ( -3 * a[30] - 3 * a[31] ) +
                                                                3 * a[31] ) ) ) ) ) +
                    sth *
                        ( z * ( -a[18] + z * ( -a[19] + z * ( a[18] + a[19] ) ) ) +
                          cth * ( z * ( -2 * a[20] +
                                        z * ( -2 * a[21] + z * ( 2 * a[20] + 2 * a[21] ) ) ) +
                                  cth * z *
                                      ( -3 * a[22] +
                                        z * ( -3 * a[23] + z * ( 3 * a[22] + 3 * a[23] ) ) ) ) +
                          sth *
                              ( z * ( -a[26] + z * ( -a[27] + z * ( a[26] + a[27] ) ) ) +
                                cth * ( z * ( -2 * a[28] +
                                              z * ( -2 * a[29] + z * ( 2 * a[28] + 2 * a[29] ) ) ) +
                                        cth * z *
                                            ( -3 * a[30] +
                                              z * ( -3 * a[31] +
                                                    z * ( 3 * a[30] + 3 * a[31] ) ) ) ) ) ) ) ) +
        r * ( cth * ( z * ( a[40] + z * ( z * ( -a[40] - a[41] ) + a[41] ) ) +
                      cth * ( z * ( a[42] + z * ( z * ( -a[42] - a[43] ) + a[43] ) ) +
                              cth * ( z * ( a[44] + z * ( z * ( -a[44] - a[45] ) + a[45] ) ) +
                                      cth * z *
                                          ( a[46] + z * ( z * ( -a[46] - a[47] ) + a[47] ) ) ) ) ) +
              sth *
                  ( z * ( -a[34] + z * ( -a[35] + z * ( a[34] + a[35] ) ) ) +
                    cth *
                        ( z * ( -2 * a[36] + 2 * a[48] +
                                z * ( -2 * a[37] +
                                      z * ( 2 * a[36] + 2 * a[37] - 2 * a[48] - 2 * a[49] ) +
                                      2 * a[49] ) ) +
                          cth *
                              ( z * ( -3 * a[38] + 2 * a[50] +
                                      z * ( -3 * a[39] +
                                            z * ( 3 * a[38] + 3 * a[39] - 2 * a[50] - 2 * a[51] ) +
                                            2 * a[51] ) ) +
                                cth * ( z * ( 2 * a[52] +
                                              z * ( z * ( -2 * a[52] - 2 * a[53] ) + 2 * a[53] ) ) +
                                        cth * z *
                                            ( 2 * a[54] + z * ( z * ( -2 * a[54] - 2 * a[55] ) +
                                                                2 * a[55] ) ) ) ) ) +
                    sth *
                        ( z * ( -a[42] + z * ( -a[43] + z * ( a[42] + a[43] ) ) ) +
                          cth *
                              ( z * ( -2 * a[44] + 3 * a[56] +
                                      z * ( -2 * a[45] +
                                            z * ( 2 * a[44] + 2 * a[45] - 3 * a[56] - 3 * a[57] ) +
                                            3 * a[57] ) ) +
                                cth *
                                    ( z * ( -3 * a[46] + 3 * a[58] +
                                            z * ( -3 * a[47] +
                                                  z * ( 3 * a[46] + 3 * a[47] - 3 * a[58] -
                                                        3 * a[59] ) +
                                                  3 * a[59] ) ) +
                                      cth *
                                          ( z * ( 3 * a[60] + z * ( z * ( -3 * a[60] - 3 * a[61] ) +
                                                                    3 * a[61] ) ) +
                                            cth * z *
                                                ( 3 * a[62] + z * ( z * ( -3 * a[62] - 3 * a[63] ) +
                                                                    3 * a[63] ) ) ) ) ) +
                          sth *
                              ( z * ( -a[50] + z * ( -a[51] + z * ( a[50] + a[51] ) ) ) +
                                cth *
                                    ( z * ( -2 * a[52] +
                                            z * ( -2 * a[53] + z * ( 2 * a[52] + 2 * a[53] ) ) ) +
                                      cth * z *
                                          ( -3 * a[54] +
                                            z * ( -3 * a[55] + z * ( 3 * a[54] + 3 * a[55] ) ) ) ) +
                                sth * ( z * ( -a[58] + z * ( -a[59] + z * ( a[58] + a[59] ) ) ) +
                                        cth * ( z * ( -2 * a[60] +
                                                      z * ( -2 * a[61] +
                                                            z * ( 2 * a[60] + 2 * a[61] ) ) ) +
                                                cth * z *
                                                    ( -3 * a[62] +
                                                      z * ( -3 * a[63] +
                                                            z * ( 3 * a[62] +
                                                                  3 * a[63] ) ) ) ) ) ) ) ) +
              r * ( cth * ( z * ( a[72] + z * ( z * ( -a[72] - a[73] ) + a[73] ) ) +
                            cth * ( z * ( a[74] + z * ( z * ( -a[74] - a[75] ) + a[75] ) ) +
                                    cth * ( z * ( a[76] + z * ( z * ( -a[76] - a[77] ) + a[77] ) ) +
                                            cth * z *
                                                ( a[78] +
                                                  z * ( z * ( -a[78] - a[79] ) + a[79] ) ) ) ) ) +
                    sth *
                        ( z * ( -a[66] + z * ( -a[67] + z * ( a[66] + a[67] ) ) ) +
                          cth *
                              ( z * ( -2 * a[68] + 2 * a[80] +
                                      z * ( -2 * a[69] +
                                            z * ( 2 * a[68] + 2 * a[69] - 2 * a[80] - 2 * a[81] ) +
                                            2 * a[81] ) ) +
                                cth *
                                    ( z * ( -3 * a[70] + 2 * a[82] +
                                            z * ( -3 * a[71] +
                                                  z * ( 3 * a[70] + 3 * a[71] - 2 * a[82] -
                                                        2 * a[83] ) +
                                                  2 * a[83] ) ) +
                                      cth *
                                          ( z * ( 2 * a[84] + z * ( z * ( -2 * a[84] - 2 * a[85] ) +
                                                                    2 * a[85] ) ) +
                                            cth * z *
                                                ( 2 * a[86] + z * ( z * ( -2 * a[86] - 2 * a[87] ) +
                                                                    2 * a[87] ) ) ) ) ) +
                          sth *
                              ( z * ( -a[74] + z * ( -a[75] + z * ( a[74] + a[75] ) ) ) +
                                cth * ( z * ( -2 * a[76] + 3 * a[88] +
                                              z * ( -2 * a[77] +
                                                    z * ( 2 * a[76] + 2 * a[77] - 3 * a[88] -
                                                          3 * a[89] ) +
                                                    3 * a[89] ) ) +
                                        cth * ( z * ( -3 * a[78] + 3 * a[90] +
                                                      z * ( -3 * a[79] +
                                                            z * ( 3 * a[78] + 3 * a[79] -
                                                                  3 * a[90] - 3 * a[91] ) +
                                                            3 * a[91] ) ) +
                                                cth * ( z * ( 3 * a[92] +
                                                              z * ( z * ( -3 * a[92] - 3 * a[93] ) +
                                                                    3 * a[93] ) ) +
                                                        cth * z *
                                                            ( 3 * a[94] +
                                                              z * ( z * ( -3 * a[94] - 3 * a[95] ) +
                                                                    3 * a[95] ) ) ) ) ) +
                                sth *
                                    ( z * ( -a[82] + z * ( -a[83] + z * ( a[82] + a[83] ) ) ) +
                                      cth * ( z * ( -2 * a[84] +
                                                    z * ( -2 * a[85] +
                                                          z * ( 2 * a[84] + 2 * a[85] ) ) ) +
                                              cth * z *
                                                  ( -3 * a[86] +
                                                    z * ( -3 * a[87] +
                                                          z * ( 3 * a[86] + 3 * a[87] ) ) ) ) +
                                      sth *
                                          ( z * ( -a[90] +
                                                  z * ( -a[91] + z * ( a[90] + a[91] ) ) ) +
                                            cth * ( z * ( -2 * a[92] +
                                                          z * ( -2 * a[93] +
                                                                z * ( 2 * a[92] + 2 * a[93] ) ) ) +
                                                    cth * z *
                                                        ( -3 * a[94] +
                                                          z * ( -3 * a[95] +
                                                                z * ( 3 * a[94] +
                                                                      3 * a[95] ) ) ) ) ) ) ) ) +
                    r * ( cth *
                              ( z * ( -a[40] / 3 - ( 2 * a[72] ) / 3 +
                                      z * ( -a[41] / 3 +
                                            z * ( a[40] / 3 + a[41] / 3 + ( 2 * a[72] ) / 3 +
                                                  ( 2 * a[73] ) / 3 ) -
                                            ( 2 * a[73] ) / 3 ) ) +
                                cth * ( z * ( -a[42] / 3 - ( 2 * a[74] ) / 3 +
                                              z * ( -a[43] / 3 +
                                                    z * ( a[42] / 3 + a[43] / 3 +
                                                          ( 2 * a[74] ) / 3 + ( 2 * a[75] ) / 3 ) -
                                                    ( 2 * a[75] ) / 3 ) ) +
                                        cth * ( z * ( -a[44] / 3 - ( 2 * a[76] ) / 3 +
                                                      z * ( -a[45] / 3 +
                                                            z * ( a[44] / 3 + a[45] / 3 +
                                                                  ( 2 * a[76] ) / 3 +
                                                                  ( 2 * a[77] ) / 3 ) -
                                                            ( 2 * a[77] ) / 3 ) ) +
                                                cth * z *
                                                    ( -a[46] / 3 - ( 2 * a[78] ) / 3 +
                                                      z * ( -a[47] / 3 +
                                                            z * ( a[46] / 3 + a[47] / 3 +
                                                                  ( 2 * a[78] ) / 3 +
                                                                  ( 2 * a[79] ) / 3 ) -
                                                            ( 2 * a[79] ) / 3 ) ) ) ) ) +
                          sth *
                              ( z * ( a[34] / 3 + ( 2 * a[66] ) / 3 +
                                      z * ( a[35] / 3 +
                                            z * ( -a[34] / 3 - a[35] / 3 - ( 2 * a[66] ) / 3 -
                                                  ( 2 * a[67] ) / 3 ) +
                                            ( 2 * a[67] ) / 3 ) ) +
                                cth *
                                    ( z * ( ( 2 * a[36] ) / 3 - ( 2 * a[48] ) / 3 +
                                            ( 4 * a[68] ) / 3 - ( 4 * a[80] ) / 3 +
                                            z * ( ( 2 * a[37] ) / 3 - ( 2 * a[49] ) / 3 +
                                                  ( 4 * a[69] ) / 3 - ( 4 * a[81] ) / 3 +
                                                  z * ( ( -2 * a[36] ) / 3 - ( 2 * a[37] ) / 3 +
                                                        ( 2 * a[48] ) / 3 + ( 2 * a[49] ) / 3 -
                                                        ( 4 * a[68] ) / 3 - ( 4 * a[69] ) / 3 +
                                                        ( 4 * a[80] ) / 3 +
                                                        ( 4 * a[81] ) / 3 ) ) ) +
                                      cth * ( z * ( a[38] - ( 2 * a[50] ) / 3 + 2 * a[70] -
                                                    ( 4 * a[82] ) / 3 +
                                                    z * ( a[39] - ( 2 * a[51] ) / 3 + 2 * a[71] -
                                                          ( 4 * a[83] ) / 3 +
                                                          z * ( -a[38] - a[39] + ( 2 * a[50] ) / 3 +
                                                                ( 2 * a[51] ) / 3 - 2 * a[70] -
                                                                2 * a[71] + ( 4 * a[82] ) / 3 +
                                                                ( 4 * a[83] ) / 3 ) ) ) +
                                              cth * ( z * ( ( -2 * a[52] ) / 3 - ( 4 * a[84] ) / 3 +
                                                            z * ( ( -2 * a[53] ) / 3 -
                                                                  ( 4 * a[85] ) / 3 +
                                                                  z * ( ( 2 * a[52] ) / 3 +
                                                                        ( 2 * a[53] ) / 3 +
                                                                        ( 4 * a[84] ) / 3 +
                                                                        ( 4 * a[85] ) / 3 ) ) ) +
                                                      cth * z *
                                                          ( ( -2 * a[54] ) / 3 - ( 4 * a[86] ) / 3 +
                                                            z * ( ( -2 * a[55] ) / 3 -
                                                                  ( 4 * a[87] ) / 3 +
                                                                  z * ( ( 2 * a[54] ) / 3 +
                                                                        ( 2 * a[55] ) / 3 +
                                                                        ( 4 * a[86] ) / 3 +
                                                                        ( 4 * a[87] ) /
                                                                            3 ) ) ) ) ) ) +
                                sth *
                                    ( z * ( a[42] / 3 + ( 2 * a[74] ) / 3 +
                                            z * ( a[43] / 3 +
                                                  z * ( -a[42] / 3 - a[43] / 3 - ( 2 * a[74] ) / 3 -
                                                        ( 2 * a[75] ) / 3 ) +
                                                  ( 2 * a[75] ) / 3 ) ) +
                                      sth *
                                          ( z * ( a[50] / 3 + ( 2 * a[82] ) / 3 +
                                                  z * ( a[51] / 3 +
                                                        z * ( -a[50] / 3 - a[51] / 3 -
                                                              ( 2 * a[82] ) / 3 -
                                                              ( 2 * a[83] ) / 3 ) +
                                                        ( 2 * a[83] ) / 3 ) ) +
                                            cth * ( z * ( ( 2 * a[52] ) / 3 + ( 4 * a[84] ) / 3 +
                                                          z * ( ( 2 * a[53] ) / 3 +
                                                                z * ( ( -2 * a[52] ) / 3 -
                                                                      ( 2 * a[53] ) / 3 -
                                                                      ( 4 * a[84] ) / 3 -
                                                                      ( 4 * a[85] ) / 3 ) +
                                                                ( 4 * a[85] ) / 3 ) ) +
                                                    cth * z *
                                                        ( a[54] + 2 * a[86] +
                                                          z * ( a[55] +
                                                                z * ( -a[54] - a[55] - 2 * a[86] -
                                                                      2 * a[87] ) +
                                                                2 * a[87] ) ) ) +
                                            sth * ( z * ( a[58] / 3 + ( 2 * a[90] ) / 3 +
                                                          z * ( a[59] / 3 +
                                                                z * ( -a[58] / 3 - a[59] / 3 -
                                                                      ( 2 * a[90] ) / 3 -
                                                                      ( 2 * a[91] ) / 3 ) +
                                                                ( 2 * a[91] ) / 3 ) ) +
                                                    cth * ( z * ( ( 2 * a[60] ) / 3 +
                                                                  ( 4 * a[92] ) / 3 +
                                                                  z * ( ( 2 * a[61] ) / 3 +
                                                                        z * ( ( -2 * a[60] ) / 3 -
                                                                              ( 2 * a[61] ) / 3 -
                                                                              ( 4 * a[92] ) / 3 -
                                                                              ( 4 * a[93] ) / 3 ) +
                                                                        ( 4 * a[93] ) / 3 ) ) +
                                                            cth * z *
                                                                ( a[62] + 2 * a[94] +
                                                                  z * ( a[63] +
                                                                        z * ( -a[62] - a[63] -
                                                                              2 * a[94] -
                                                                              2 * a[95] ) +
                                                                        2 * a[95] ) ) ) ) ) +
                                      cth *
                                          ( z * ( ( 2 * a[44] ) / 3 - a[56] + ( 4 * a[76] ) / 3 -
                                                  2 * a[88] +
                                                  z * ( ( 2 * a[45] ) / 3 - a[57] +
                                                        ( 4 * a[77] ) / 3 - 2 * a[89] +
                                                        z * ( ( -2 * a[44] ) / 3 -
                                                              ( 2 * a[45] ) / 3 + a[56] + a[57] -
                                                              ( 4 * a[76] ) / 3 -
                                                              ( 4 * a[77] ) / 3 + 2 * a[88] +
                                                              2 * a[89] ) ) ) +
                                            cth *
                                                ( z * ( a[46] - a[58] + 2 * a[78] - 2 * a[90] +
                                                        z * ( a[47] - a[59] + 2 * a[79] -
                                                              2 * a[91] +
                                                              z * ( -a[46] - a[47] + a[58] + a[59] -
                                                                    2 * a[78] - 2 * a[79] +
                                                                    2 * a[90] + 2 * a[91] ) ) ) +
                                                  cth *
                                                      ( z * ( -a[60] - 2 * a[92] +
                                                              z * ( -a[61] - 2 * a[93] +
                                                                    z * ( a[60] + a[61] +
                                                                          2 * a[92] +
                                                                          2 * a[93] ) ) ) +
                                                        cth * z *
                                                            ( -a[62] - 2 * a[94] +
                                                              z * ( -a[63] - 2 * a[95] +
                                                                    z * ( a[62] + a[63] +
                                                                          2 * a[94] +
                                                                          2 * a[95] ) ) ) ) ) ) ) ) ) ) );
    result[3] =
        a[0] +
        cth * ( a[2] + z * ( z * ( -3 * a[2] - 3 * a[3] ) + 2 * a[3] ) +
                cth * ( a[4] + z * ( z * ( -3 * a[4] - 3 * a[5] ) + 2 * a[5] ) +
                        cth * ( a[6] + z * ( z * ( -3 * a[6] - 3 * a[7] ) + 2 * a[7] ) ) ) ) +
        sth *
            ( a[8] + z * ( z * ( -3 * a[8] - 3 * a[9] ) + 2 * a[9] ) +
              cth * ( a[10] + z * ( z * ( -3 * a[10] - 3 * a[11] ) + 2 * a[11] ) +
                      cth * ( a[12] + z * ( z * ( -3 * a[12] - 3 * a[13] ) + 2 * a[13] ) +
                              cth * ( a[14] +
                                      z * ( z * ( -3 * a[14] - 3 * a[15] ) + 2 * a[15] ) ) ) ) +
              sth * ( a[16] + z * ( z * ( -3 * a[16] - 3 * a[17] ) + 2 * a[17] ) +
                      cth * ( a[18] + z * ( z * ( -3 * a[18] - 3 * a[19] ) + 2 * a[19] ) +
                              cth * ( a[20] + z * ( z * ( -3 * a[20] - 3 * a[21] ) + 2 * a[21] ) +
                                      cth * ( a[22] + z * ( z * ( -3 * a[22] - 3 * a[23] ) +
                                                            2 * a[23] ) ) ) ) +
                      sth * ( a[24] + z * ( z * ( -3 * a[24] - 3 * a[25] ) + 2 * a[25] ) +
                              cth * ( a[26] + z * ( z * ( -3 * a[26] - 3 * a[27] ) + 2 * a[27] ) +
                                      cth * ( a[28] +
                                              z * ( z * ( -3 * a[28] - 3 * a[29] ) + 2 * a[29] ) +
                                              cth * ( a[30] + z * ( z * ( -3 * a[30] - 3 * a[31] ) +
                                                                    2 * a[31] ) ) ) ) ) ) ) +
        r * ( a[32] + z * ( z * ( -3 * a[32] - 3 * a[33] ) + 2 * a[33] ) +
              cth * ( a[34] + z * ( z * ( -3 * a[34] - 3 * a[35] ) + 2 * a[35] ) +
                      cth * ( a[36] + z * ( z * ( -3 * a[36] - 3 * a[37] ) + 2 * a[37] ) +
                              cth * ( a[38] +
                                      z * ( z * ( -3 * a[38] - 3 * a[39] ) + 2 * a[39] ) ) ) ) +
              sth *
                  ( a[40] + z * ( z * ( -3 * a[40] - 3 * a[41] ) + 2 * a[41] ) +
                    cth * ( a[42] + z * ( z * ( -3 * a[42] - 3 * a[43] ) + 2 * a[43] ) +
                            cth * ( a[44] + z * ( z * ( -3 * a[44] - 3 * a[45] ) + 2 * a[45] ) +
                                    cth * ( a[46] + z * ( z * ( -3 * a[46] - 3 * a[47] ) +
                                                          2 * a[47] ) ) ) ) +
                    sth *
                        ( a[48] + z * ( z * ( -3 * a[48] - 3 * a[49] ) + 2 * a[49] ) +
                          cth *
                              ( a[50] + z * ( z * ( -3 * a[50] - 3 * a[51] ) + 2 * a[51] ) +
                                cth * ( a[52] + z * ( z * ( -3 * a[52] - 3 * a[53] ) + 2 * a[53] ) +
                                        cth * ( a[54] + z * ( z * ( -3 * a[54] - 3 * a[55] ) +
                                                              2 * a[55] ) ) ) ) +
                          sth *
                              ( a[56] + z * ( z * ( -3 * a[56] - 3 * a[57] ) + 2 * a[57] ) +
                                cth *
                                    ( a[58] + z * ( z * ( -3 * a[58] - 3 * a[59] ) + 2 * a[59] ) +
                                      cth * ( a[60] +
                                              z * ( z * ( -3 * a[60] - 3 * a[61] ) + 2 * a[61] ) +
                                              cth * ( a[62] + z * ( z * ( -3 * a[62] - 3 * a[63] ) +
                                                                    2 * a[63] ) ) ) ) ) ) ) +
              r * ( a[64] + z * ( z * ( -3 * a[64] - 3 * a[65] ) + 2 * a[65] ) +
                    cth * ( a[66] + z * ( z * ( -3 * a[66] - 3 * a[67] ) + 2 * a[67] ) +
                            cth * ( a[68] + z * ( z * ( -3 * a[68] - 3 * a[69] ) + 2 * a[69] ) +
                                    cth * ( a[70] + z * ( z * ( -3 * a[70] - 3 * a[71] ) +
                                                          2 * a[71] ) ) ) ) +
                    sth *
                        ( a[72] + z * ( z * ( -3 * a[72] - 3 * a[73] ) + 2 * a[73] ) +
                          cth *
                              ( a[74] + z * ( z * ( -3 * a[74] - 3 * a[75] ) + 2 * a[75] ) +
                                cth * ( a[76] + z * ( z * ( -3 * a[76] - 3 * a[77] ) + 2 * a[77] ) +
                                        cth * ( a[78] + z * ( z * ( -3 * a[78] - 3 * a[79] ) +
                                                              2 * a[79] ) ) ) ) +
                          sth *
                              ( a[80] + z * ( z * ( -3 * a[80] - 3 * a[81] ) + 2 * a[81] ) +
                                cth *
                                    ( a[82] + z * ( z * ( -3 * a[82] - 3 * a[83] ) + 2 * a[83] ) +
                                      cth * ( a[84] +
                                              z * ( z * ( -3 * a[84] - 3 * a[85] ) + 2 * a[85] ) +
                                              cth * ( a[86] + z * ( z * ( -3 * a[86] - 3 * a[87] ) +
                                                                    2 * a[87] ) ) ) ) +
                                sth *
                                    ( a[88] + z * ( z * ( -3 * a[88] - 3 * a[89] ) + 2 * a[89] ) +
                                      cth * ( a[90] +
                                              z * ( z * ( -3 * a[90] - 3 * a[91] ) + 2 * a[91] ) +
                                              cth * ( a[92] +
                                                      z * ( z * ( -3 * a[92] - 3 * a[93] ) +
                                                            2 * a[93] ) +
                                                      cth * ( a[94] +
                                                              z * ( z * ( -3 * a[94] - 3 * a[95] ) +
                                                                    2 * a[95] ) ) ) ) ) ) ) +
                    r * ( -a[32] / 3 - ( 2 * a[64] ) / 3 +
                          z * ( ( -2 * a[33] ) / 3 - ( 4 * a[65] ) / 3 +
                                z * ( a[32] + a[33] + 2 * a[64] + 2 * a[65] ) ) +
                          cth * ( -a[34] / 3 - ( 2 * a[66] ) / 3 +
                                  z * ( ( -2 * a[35] ) / 3 - ( 4 * a[67] ) / 3 +
                                        z * ( a[34] + a[35] + 2 * a[66] + 2 * a[67] ) ) +
                                  cth * ( -a[36] / 3 - ( 2 * a[68] ) / 3 +
                                          z * ( ( -2 * a[37] ) / 3 - ( 4 * a[69] ) / 3 +
                                                z * ( a[36] + a[37] + 2 * a[68] + 2 * a[69] ) ) +
                                          cth * ( -a[38] / 3 - ( 2 * a[70] ) / 3 +
                                                  z * ( ( -2 * a[39] ) / 3 - ( 4 * a[71] ) / 3 +
                                                        z * ( a[38] + a[39] + 2 * a[70] +
                                                              2 * a[71] ) ) ) ) ) +
                          sth *
                              ( -a[40] / 3 - ( 2 * a[72] ) / 3 +
                                z * ( ( -2 * a[41] ) / 3 - ( 4 * a[73] ) / 3 +
                                      z * ( a[40] + a[41] + 2 * a[72] + 2 * a[73] ) ) +
                                cth *
                                    ( -a[42] / 3 - ( 2 * a[74] ) / 3 +
                                      z * ( ( -2 * a[43] ) / 3 - ( 4 * a[75] ) / 3 +
                                            z * ( a[42] + a[43] + 2 * a[74] + 2 * a[75] ) ) +
                                      cth *
                                          ( -a[44] / 3 - ( 2 * a[76] ) / 3 +
                                            z * ( ( -2 * a[45] ) / 3 - ( 4 * a[77] ) / 3 +
                                                  z * ( a[44] + a[45] + 2 * a[76] + 2 * a[77] ) ) +
                                            cth * ( -a[46] / 3 - ( 2 * a[78] ) / 3 +
                                                    z * ( ( -2 * a[47] ) / 3 - ( 4 * a[79] ) / 3 +
                                                          z * ( a[46] + a[47] + 2 * a[78] +
                                                                2 * a[79] ) ) ) ) ) +
                                sth *
                                    ( -a[48] / 3 - ( 2 * a[80] ) / 3 +
                                      z * ( ( -2 * a[49] ) / 3 - ( 4 * a[81] ) / 3 +
                                            z * ( a[48] + a[49] + 2 * a[80] + 2 * a[81] ) ) +
                                      cth *
                                          ( -a[50] / 3 - ( 2 * a[82] ) / 3 +
                                            z * ( ( -2 * a[51] ) / 3 - ( 4 * a[83] ) / 3 +
                                                  z * ( a[50] + a[51] + 2 * a[82] + 2 * a[83] ) ) +
                                            cth * ( -a[52] / 3 - ( 2 * a[84] ) / 3 +
                                                    z * ( ( -2 * a[53] ) / 3 - ( 4 * a[85] ) / 3 +
                                                          z * ( a[52] + a[53] + 2 * a[84] +
                                                                2 * a[85] ) ) +
                                                    cth * ( -a[54] / 3 - ( 2 * a[86] ) / 3 +
                                                            z * ( ( -2 * a[55] ) / 3 -
                                                                  ( 4 * a[87] ) / 3 +
                                                                  z * ( a[54] + a[55] + 2 * a[86] +
                                                                        2 * a[87] ) ) ) ) ) +
                                      sth *
                                          ( -a[56] / 3 - ( 2 * a[88] ) / 3 +
                                            z * ( ( -2 * a[57] ) / 3 - ( 4 * a[89] ) / 3 +
                                                  z * ( a[56] + a[57] + 2 * a[88] + 2 * a[89] ) ) +
                                            cth *
                                                ( -a[58] / 3 - ( 2 * a[90] ) / 3 +
                                                  z * ( ( -2 * a[59] ) / 3 - ( 4 * a[91] ) / 3 +
                                                        z * ( a[58] + a[59] + 2 * a[90] +
                                                              2 * a[91] ) ) +
                                                  cth *
                                                      ( -a[60] / 3 - ( 2 * a[92] ) / 3 +
                                                        z * ( ( -2 * a[61] ) / 3 -
                                                              ( 4 * a[93] ) / 3 +
                                                              z * ( a[60] + a[61] + 2 * a[92] +
                                                                    2 * a[93] ) ) +
                                                        cth *
                                                            ( -a[62] / 3 - ( 2 * a[94] ) / 3 +
                                                              z * ( ( -2 * a[63] ) / 3 -
                                                                    ( 4 * a[95] ) / 3 +
                                                                    z * ( a[62] + a[63] +
                                                                          2 * a[94] +
                                                                          2 * a[95] ) ) ) ) ) ) ) ) ) ) ) +
        z * ( 2 * a[1] + z * ( -3 * a[0] - 3 * a[1] - 3 * c[0] + 3 * c[1] ) );
    result[4] =
        z * ( 2 * a[64] + z * ( z * ( -2 * a[64] - 2 * a[65] ) + 2 * a[65] ) ) +
        cth * ( z * ( 2 * a[66] + z * ( z * ( -2 * a[66] - 2 * a[67] ) + 2 * a[67] ) ) +
                cth * ( z * ( 2 * a[68] + z * ( z * ( -2 * a[68] - 2 * a[69] ) + 2 * a[69] ) ) +
                        cth * z *
                            ( 2 * a[70] + z * ( z * ( -2 * a[70] - 2 * a[71] ) + 2 * a[71] ) ) ) ) +
        sth *
            ( z * ( 2 * a[72] + z * ( z * ( -2 * a[72] - 2 * a[73] ) + 2 * a[73] ) ) +
              cth *
                  ( z * ( 2 * a[74] + z * ( z * ( -2 * a[74] - 2 * a[75] ) + 2 * a[75] ) ) +
                    cth * ( z * ( 2 * a[76] + z * ( z * ( -2 * a[76] - 2 * a[77] ) + 2 * a[77] ) ) +
                            cth * z *
                                ( 2 * a[78] +
                                  z * ( z * ( -2 * a[78] - 2 * a[79] ) + 2 * a[79] ) ) ) ) +
              sth *
                  ( z * ( 2 * a[80] + z * ( z * ( -2 * a[80] - 2 * a[81] ) + 2 * a[81] ) ) +
                    cth * ( z * ( 2 * a[82] + z * ( z * ( -2 * a[82] - 2 * a[83] ) + 2 * a[83] ) ) +
                            cth * ( z * ( 2 * a[84] +
                                          z * ( z * ( -2 * a[84] - 2 * a[85] ) + 2 * a[85] ) ) +
                                    cth * z *
                                        ( 2 * a[86] +
                                          z * ( z * ( -2 * a[86] - 2 * a[87] ) + 2 * a[87] ) ) ) ) +
                    sth * ( z * ( 2 * a[88] + z * ( z * ( -2 * a[88] - 2 * a[89] ) + 2 * a[89] ) ) +
                            cth * ( z * ( 2 * a[90] +
                                          z * ( z * ( -2 * a[90] - 2 * a[91] ) + 2 * a[91] ) ) +
                                    cth * ( z * ( 2 * a[92] + z * ( z * ( -2 * a[92] - 2 * a[93] ) +
                                                                    2 * a[93] ) ) +
                                            cth * z *
                                                ( 2 * a[94] + z * ( z * ( -2 * a[94] - 2 * a[95] ) +
                                                                    2 * a[95] ) ) ) ) ) ) ) +
        r * ( z * ( -2 * a[32] - 4 * a[64] +
                    z * ( -2 * a[33] - 4 * a[65] +
                          z * ( 2 * a[32] + 2 * a[33] + 4 * a[64] + 4 * a[65] ) ) ) +
              cth *
                  ( z * ( -2 * a[34] - 4 * a[66] +
                          z * ( -2 * a[35] - 4 * a[67] +
                                z * ( 2 * a[34] + 2 * a[35] + 4 * a[66] + 4 * a[67] ) ) ) +
                    cth * ( z * ( -2 * a[36] - 4 * a[68] +
                                  z * ( -2 * a[37] - 4 * a[69] +
                                        z * ( 2 * a[36] + 2 * a[37] + 4 * a[68] + 4 * a[69] ) ) ) +
                            cth * z *
                                ( -2 * a[38] - 4 * a[70] +
                                  z * ( -2 * a[39] - 4 * a[71] +
                                        z * ( 2 * a[38] + 2 * a[39] + 4 * a[70] +
                                              4 * a[71] ) ) ) ) ) +
              sth *
                  ( z * ( -2 * a[40] - 4 * a[72] +
                          z * ( -2 * a[41] - 4 * a[73] +
                                z * ( 2 * a[40] + 2 * a[41] + 4 * a[72] + 4 * a[73] ) ) ) +
                    cth * ( z * ( -2 * a[42] - 4 * a[74] +
                                  z * ( -2 * a[43] - 4 * a[75] +
                                        z * ( 2 * a[42] + 2 * a[43] + 4 * a[74] + 4 * a[75] ) ) ) +
                            cth * ( z * ( -2 * a[44] - 4 * a[76] +
                                          z * ( -2 * a[45] - 4 * a[77] +
                                                z * ( 2 * a[44] + 2 * a[45] + 4 * a[76] +
                                                      4 * a[77] ) ) ) +
                                    cth * z *
                                        ( -2 * a[46] - 4 * a[78] +
                                          z * ( -2 * a[47] - 4 * a[79] +
                                                z * ( 2 * a[46] + 2 * a[47] + 4 * a[78] +
                                                      4 * a[79] ) ) ) ) ) +
                    sth * ( z * ( -2 * a[48] - 4 * a[80] +
                                  z * ( -2 * a[49] - 4 * a[81] +
                                        z * ( 2 * a[48] + 2 * a[49] + 4 * a[80] + 4 * a[81] ) ) ) +
                            cth * ( z * ( -2 * a[50] - 4 * a[82] +
                                          z * ( -2 * a[51] - 4 * a[83] +
                                                z * ( 2 * a[50] + 2 * a[51] + 4 * a[82] +
                                                      4 * a[83] ) ) ) +
                                    cth * ( z * ( -2 * a[52] - 4 * a[84] +
                                                  z * ( -2 * a[53] - 4 * a[85] +
                                                        z * ( 2 * a[52] + 2 * a[53] + 4 * a[84] +
                                                              4 * a[85] ) ) ) +
                                            cth * z *
                                                ( -2 * a[54] - 4 * a[86] +
                                                  z * ( -2 * a[55] - 4 * a[87] +
                                                        z * ( 2 * a[54] + 2 * a[55] + 4 * a[86] +
                                                              4 * a[87] ) ) ) ) ) +
                            sth * ( z * ( -2 * a[56] - 4 * a[88] +
                                          z * ( -2 * a[57] - 4 * a[89] +
                                                z * ( 2 * a[56] + 2 * a[57] + 4 * a[88] +
                                                      4 * a[89] ) ) ) +
                                    cth * ( z * ( -2 * a[58] - 4 * a[90] +
                                                  z * ( -2 * a[59] - 4 * a[91] +
                                                        z * ( 2 * a[58] + 2 * a[59] + 4 * a[90] +
                                                              4 * a[91] ) ) ) +
                                            cth * ( z * ( -2 * a[60] - 4 * a[92] +
                                                          z * ( -2 * a[61] - 4 * a[93] +
                                                                z * ( 2 * a[60] + 2 * a[61] +
                                                                      4 * a[92] + 4 * a[93] ) ) ) +
                                                    cth * z *
                                                        ( -2 * a[62] - 4 * a[94] +
                                                          z * ( -2 * a[63] - 4 * a[95] +
                                                                z * ( 2 * a[62] + 2 * a[63] +
                                                                      4 * a[94] +
                                                                      4 * a[95] ) ) ) ) ) ) ) ) );
    result[5] =
        cth * ( z * ( a[40] + z * ( z * ( -a[40] - a[41] ) + a[41] ) ) +
                cth * ( z * ( a[42] + z * ( z * ( -a[42] - a[43] ) + a[43] ) ) +
                        cth * ( z * ( a[44] + z * ( z * ( -a[44] - a[45] ) + a[45] ) ) +
                                cth * z * ( a[46] + z * ( z * ( -a[46] - a[47] ) + a[47] ) ) ) ) ) +
        sth *
            ( z * ( -a[34] + z * ( -a[35] + z * ( a[34] + a[35] ) ) ) +
              cth *
                  ( z * ( -2 * a[36] + 2 * a[48] +
                          z * ( -2 * a[37] + z * ( 2 * a[36] + 2 * a[37] - 2 * a[48] - 2 * a[49] ) +
                                2 * a[49] ) ) +
                    cth * ( z * ( -3 * a[38] + 2 * a[50] +
                                  z * ( -3 * a[39] +
                                        z * ( 3 * a[38] + 3 * a[39] - 2 * a[50] - 2 * a[51] ) +
                                        2 * a[51] ) ) +
                            cth * ( z * ( 2 * a[52] +
                                          z * ( z * ( -2 * a[52] - 2 * a[53] ) + 2 * a[53] ) ) +
                                    cth * z *
                                        ( 2 * a[54] + z * ( z * ( -2 * a[54] - 2 * a[55] ) +
                                                            2 * a[55] ) ) ) ) ) +
              sth *
                  ( z * ( -a[42] + z * ( -a[43] + z * ( a[42] + a[43] ) ) ) +
                    cth *
                        ( z * ( -2 * a[44] + 3 * a[56] +
                                z * ( -2 * a[45] +
                                      z * ( 2 * a[44] + 2 * a[45] - 3 * a[56] - 3 * a[57] ) +
                                      3 * a[57] ) ) +
                          cth *
                              ( z * ( -3 * a[46] + 3 * a[58] +
                                      z * ( -3 * a[47] +
                                            z * ( 3 * a[46] + 3 * a[47] - 3 * a[58] - 3 * a[59] ) +
                                            3 * a[59] ) ) +
                                cth * ( z * ( 3 * a[60] +
                                              z * ( z * ( -3 * a[60] - 3 * a[61] ) + 3 * a[61] ) ) +
                                        cth * z *
                                            ( 3 * a[62] + z * ( z * ( -3 * a[62] - 3 * a[63] ) +
                                                                3 * a[63] ) ) ) ) ) +
                    sth *
                        ( z * ( -a[50] + z * ( -a[51] + z * ( a[50] + a[51] ) ) ) +
                          cth * ( z * ( -2 * a[52] +
                                        z * ( -2 * a[53] + z * ( 2 * a[52] + 2 * a[53] ) ) ) +
                                  cth * z *
                                      ( -3 * a[54] +
                                        z * ( -3 * a[55] + z * ( 3 * a[54] + 3 * a[55] ) ) ) ) +
                          sth *
                              ( z * ( -a[58] + z * ( -a[59] + z * ( a[58] + a[59] ) ) ) +
                                cth * ( z * ( -2 * a[60] +
                                              z * ( -2 * a[61] + z * ( 2 * a[60] + 2 * a[61] ) ) ) +
                                        cth * z *
                                            ( -3 * a[62] +
                                              z * ( -3 * a[63] +
                                                    z * ( 3 * a[62] + 3 * a[63] ) ) ) ) ) ) ) ) +
        r * ( cth *
                  ( z * ( 2 * a[72] + z * ( z * ( -2 * a[72] - 2 * a[73] ) + 2 * a[73] ) ) +
                    cth * ( z * ( 2 * a[74] + z * ( z * ( -2 * a[74] - 2 * a[75] ) + 2 * a[75] ) ) +
                            cth * ( z * ( 2 * a[76] +
                                          z * ( z * ( -2 * a[76] - 2 * a[77] ) + 2 * a[77] ) ) +
                                    cth * z *
                                        ( 2 * a[78] + z * ( z * ( -2 * a[78] - 2 * a[79] ) +
                                                            2 * a[79] ) ) ) ) ) +
              sth *
                  ( z * ( -2 * a[66] + z * ( -2 * a[67] + z * ( 2 * a[66] + 2 * a[67] ) ) ) +
                    cth *
                        ( z * ( -4 * a[68] + 4 * a[80] +
                                z * ( -4 * a[69] +
                                      z * ( 4 * a[68] + 4 * a[69] - 4 * a[80] - 4 * a[81] ) +
                                      4 * a[81] ) ) +
                          cth *
                              ( z * ( -6 * a[70] + 4 * a[82] +
                                      z * ( -6 * a[71] +
                                            z * ( 6 * a[70] + 6 * a[71] - 4 * a[82] - 4 * a[83] ) +
                                            4 * a[83] ) ) +
                                cth * ( z * ( 4 * a[84] +
                                              z * ( z * ( -4 * a[84] - 4 * a[85] ) + 4 * a[85] ) ) +
                                        cth * z *
                                            ( 4 * a[86] + z * ( z * ( -4 * a[86] - 4 * a[87] ) +
                                                                4 * a[87] ) ) ) ) ) +
                    sth *
                        ( z * ( -2 * a[74] + z * ( -2 * a[75] + z * ( 2 * a[74] + 2 * a[75] ) ) ) +
                          cth *
                              ( z * ( -4 * a[76] + 6 * a[88] +
                                      z * ( -4 * a[77] +
                                            z * ( 4 * a[76] + 4 * a[77] - 6 * a[88] - 6 * a[89] ) +
                                            6 * a[89] ) ) +
                                cth *
                                    ( z * ( -6 * a[78] + 6 * a[90] +
                                            z * ( -6 * a[79] +
                                                  z * ( 6 * a[78] + 6 * a[79] - 6 * a[90] -
                                                        6 * a[91] ) +
                                                  6 * a[91] ) ) +
                                      cth *
                                          ( z * ( 6 * a[92] + z * ( z * ( -6 * a[92] - 6 * a[93] ) +
                                                                    6 * a[93] ) ) +
                                            cth * z *
                                                ( 6 * a[94] + z * ( z * ( -6 * a[94] - 6 * a[95] ) +
                                                                    6 * a[95] ) ) ) ) ) +
                          sth *
                              ( z * ( -2 * a[82] +
                                      z * ( -2 * a[83] + z * ( 2 * a[82] + 2 * a[83] ) ) ) +
                                cth *
                                    ( z * ( -4 * a[84] +
                                            z * ( -4 * a[85] + z * ( 4 * a[84] + 4 * a[85] ) ) ) +
                                      cth * z *
                                          ( -6 * a[86] +
                                            z * ( -6 * a[87] + z * ( 6 * a[86] + 6 * a[87] ) ) ) ) +
                                sth * ( z * ( -2 * a[90] +
                                              z * ( -2 * a[91] + z * ( 2 * a[90] + 2 * a[91] ) ) ) +
                                        cth * ( z * ( -4 * a[92] +
                                                      z * ( -4 * a[93] +
                                                            z * ( 4 * a[92] + 4 * a[93] ) ) ) +
                                                cth * z *
                                                    ( -6 * a[94] +
                                                      z * ( -6 * a[95] +
                                                            z * ( 6 * a[94] +
                                                                  6 * a[95] ) ) ) ) ) ) ) ) +
              r * ( cth * ( z * ( -a[40] - 2 * a[72] +
                                  z * ( -a[41] - 2 * a[73] +
                                        z * ( a[40] + a[41] + 2 * a[72] + 2 * a[73] ) ) ) +
                            cth * ( z * ( -a[42] - 2 * a[74] +
                                          z * ( -a[43] - 2 * a[75] +
                                                z * ( a[42] + a[43] + 2 * a[74] + 2 * a[75] ) ) ) +
                                    cth * ( z * ( -a[44] - 2 * a[76] +
                                                  z * ( -a[45] - 2 * a[77] +
                                                        z * ( a[44] + a[45] + 2 * a[76] +
                                                              2 * a[77] ) ) ) +
                                            cth * z *
                                                ( -a[46] - 2 * a[78] +
                                                  z * ( -a[47] - 2 * a[79] +
                                                        z * ( a[46] + a[47] + 2 * a[78] +
                                                              2 * a[79] ) ) ) ) ) ) +
                    sth *
                        ( z * ( a[34] + 2 * a[66] +
                                z * ( a[35] + z * ( -a[34] - a[35] - 2 * a[66] - 2 * a[67] ) +
                                      2 * a[67] ) ) +
                          cth *
                              ( z * ( 2 * a[36] - 2 * a[48] + 4 * a[68] - 4 * a[80] +
                                      z * ( 2 * a[37] - 2 * a[49] + 4 * a[69] - 4 * a[81] +
                                            z * ( -2 * a[36] - 2 * a[37] + 2 * a[48] + 2 * a[49] -
                                                  4 * a[68] - 4 * a[69] + 4 * a[80] +
                                                  4 * a[81] ) ) ) +
                                cth *
                                    ( z * ( 3 * a[38] - 2 * a[50] + 6 * a[70] - 4 * a[82] +
                                            z * ( 3 * a[39] - 2 * a[51] + 6 * a[71] - 4 * a[83] +
                                                  z * ( -3 * a[38] - 3 * a[39] + 2 * a[50] +
                                                        2 * a[51] - 6 * a[70] - 6 * a[71] +
                                                        4 * a[82] + 4 * a[83] ) ) ) +
                                      cth * ( z * ( -2 * a[52] - 4 * a[84] +
                                                    z * ( -2 * a[53] - 4 * a[85] +
                                                          z * ( 2 * a[52] + 2 * a[53] + 4 * a[84] +
                                                                4 * a[85] ) ) ) +
                                              cth * z *
                                                  ( -2 * a[54] - 4 * a[86] +
                                                    z * ( -2 * a[55] - 4 * a[87] +
                                                          z * ( 2 * a[54] + 2 * a[55] + 4 * a[86] +
                                                                4 * a[87] ) ) ) ) ) ) +
                          sth *
                              ( z * ( a[42] + 2 * a[74] +
                                      z * ( a[43] + z * ( -a[42] - a[43] - 2 * a[74] - 2 * a[75] ) +
                                            2 * a[75] ) ) +
                                sth * ( z * ( a[50] + 2 * a[82] +
                                              z * ( a[51] +
                                                    z * ( -a[50] - a[51] - 2 * a[82] - 2 * a[83] ) +
                                                    2 * a[83] ) ) +
                                        cth * ( z * ( 2 * a[52] + 4 * a[84] +
                                                      z * ( 2 * a[53] +
                                                            z * ( -2 * a[52] - 2 * a[53] -
                                                                  4 * a[84] - 4 * a[85] ) +
                                                            4 * a[85] ) ) +
                                                cth * z *
                                                    ( 3 * a[54] + 6 * a[86] +
                                                      z * ( 3 * a[55] +
                                                            z * ( -3 * a[54] - 3 * a[55] -
                                                                  6 * a[86] - 6 * a[87] ) +
                                                            6 * a[87] ) ) ) +
                                        sth * ( z * ( a[58] + 2 * a[90] +
                                                      z * ( a[59] +
                                                            z * ( -a[58] - a[59] - 2 * a[90] -
                                                                  2 * a[91] ) +
                                                            2 * a[91] ) ) +
                                                cth * ( z * ( 2 * a[60] + 4 * a[92] +
                                                              z * ( 2 * a[61] +
                                                                    z * ( -2 * a[60] - 2 * a[61] -
                                                                          4 * a[92] - 4 * a[93] ) +
                                                                    4 * a[93] ) ) +
                                                        cth * z *
                                                            ( 3 * a[62] + 6 * a[94] +
                                                              z * ( 3 * a[63] +
                                                                    z * ( -3 * a[62] - 3 * a[63] -
                                                                          6 * a[94] - 6 * a[95] ) +
                                                                    6 * a[95] ) ) ) ) ) +
                                cth *
                                    ( z * ( 2 * a[44] - 3 * a[56] + 4 * a[76] - 6 * a[88] +
                                            z * ( 2 * a[45] - 3 * a[57] + 4 * a[77] - 6 * a[89] +
                                                  z * ( -2 * a[44] - 2 * a[45] + 3 * a[56] +
                                                        3 * a[57] - 4 * a[76] - 4 * a[77] +
                                                        6 * a[88] + 6 * a[89] ) ) ) +
                                      cth *
                                          ( z * ( 3 * a[46] - 3 * a[58] + 6 * a[78] - 6 * a[90] +
                                                  z * ( 3 * a[47] - 3 * a[59] + 6 * a[79] -
                                                        6 * a[91] +
                                                        z * ( -3 * a[46] - 3 * a[47] + 3 * a[58] +
                                                              3 * a[59] - 6 * a[78] - 6 * a[79] +
                                                              6 * a[90] + 6 * a[91] ) ) ) +
                                            cth * ( z * ( -3 * a[60] - 6 * a[92] +
                                                          z * ( -3 * a[61] - 6 * a[93] +
                                                                z * ( 3 * a[60] + 3 * a[61] +
                                                                      6 * a[92] + 6 * a[93] ) ) ) +
                                                    cth * z *
                                                        ( -3 * a[62] - 6 * a[94] +
                                                          z * ( -3 * a[63] - 6 * a[95] +
                                                                z * ( 3 * a[62] + 3 * a[63] +
                                                                      6 * a[94] +
                                                                      6 * a[95] ) ) ) ) ) ) ) ) ) );
    result[6] =
        a[32] + z * ( z * ( -3 * a[32] - 3 * a[33] ) + 2 * a[33] ) +
        cth * ( a[34] + z * ( z * ( -3 * a[34] - 3 * a[35] ) + 2 * a[35] ) +
                cth * ( a[36] + z * ( z * ( -3 * a[36] - 3 * a[37] ) + 2 * a[37] ) +
                        cth * ( a[38] + z * ( z * ( -3 * a[38] - 3 * a[39] ) + 2 * a[39] ) ) ) ) +
        sth *
            ( a[40] + z * ( z * ( -3 * a[40] - 3 * a[41] ) + 2 * a[41] ) +
              cth * ( a[42] + z * ( z * ( -3 * a[42] - 3 * a[43] ) + 2 * a[43] ) +
                      cth * ( a[44] + z * ( z * ( -3 * a[44] - 3 * a[45] ) + 2 * a[45] ) +
                              cth * ( a[46] +
                                      z * ( z * ( -3 * a[46] - 3 * a[47] ) + 2 * a[47] ) ) ) ) +
              sth * ( a[48] + z * ( z * ( -3 * a[48] - 3 * a[49] ) + 2 * a[49] ) +
                      cth * ( a[50] + z * ( z * ( -3 * a[50] - 3 * a[51] ) + 2 * a[51] ) +
                              cth * ( a[52] + z * ( z * ( -3 * a[52] - 3 * a[53] ) + 2 * a[53] ) +
                                      cth * ( a[54] + z * ( z * ( -3 * a[54] - 3 * a[55] ) +
                                                            2 * a[55] ) ) ) ) +
                      sth * ( a[56] + z * ( z * ( -3 * a[56] - 3 * a[57] ) + 2 * a[57] ) +
                              cth * ( a[58] + z * ( z * ( -3 * a[58] - 3 * a[59] ) + 2 * a[59] ) +
                                      cth * ( a[60] +
                                              z * ( z * ( -3 * a[60] - 3 * a[61] ) + 2 * a[61] ) +
                                              cth * ( a[62] + z * ( z * ( -3 * a[62] - 3 * a[63] ) +
                                                                    2 * a[63] ) ) ) ) ) ) ) +
        r * ( 2 * a[64] + z * ( z * ( -6 * a[64] - 6 * a[65] ) + 4 * a[65] ) +
              cth * ( 2 * a[66] + z * ( z * ( -6 * a[66] - 6 * a[67] ) + 4 * a[67] ) +
                      cth * ( 2 * a[68] + z * ( z * ( -6 * a[68] - 6 * a[69] ) + 4 * a[69] ) +
                              cth * ( 2 * a[70] +
                                      z * ( z * ( -6 * a[70] - 6 * a[71] ) + 4 * a[71] ) ) ) ) +
              sth *
                  ( 2 * a[72] + z * ( z * ( -6 * a[72] - 6 * a[73] ) + 4 * a[73] ) +
                    cth * ( 2 * a[74] + z * ( z * ( -6 * a[74] - 6 * a[75] ) + 4 * a[75] ) +
                            cth * ( 2 * a[76] + z * ( z * ( -6 * a[76] - 6 * a[77] ) + 4 * a[77] ) +
                                    cth * ( 2 * a[78] + z * ( z * ( -6 * a[78] - 6 * a[79] ) +
                                                              4 * a[79] ) ) ) ) +
                    sth *
                        ( 2 * a[80] + z * ( z * ( -6 * a[80] - 6 * a[81] ) + 4 * a[81] ) +
                          cth * ( 2 * a[82] + z * ( z * ( -6 * a[82] - 6 * a[83] ) + 4 * a[83] ) +
                                  cth * ( 2 * a[84] +
                                          z * ( z * ( -6 * a[84] - 6 * a[85] ) + 4 * a[85] ) +
                                          cth * ( 2 * a[86] + z * ( z * ( -6 * a[86] - 6 * a[87] ) +
                                                                    4 * a[87] ) ) ) ) +
                          sth *
                              ( 2 * a[88] + z * ( z * ( -6 * a[88] - 6 * a[89] ) + 4 * a[89] ) +
                                cth * ( 2 * a[90] +
                                        z * ( z * ( -6 * a[90] - 6 * a[91] ) + 4 * a[91] ) +
                                        cth * ( 2 * a[92] +
                                                z * ( z * ( -6 * a[92] - 6 * a[93] ) + 4 * a[93] ) +
                                                cth * ( 2 * a[94] +
                                                        z * ( z * ( -6 * a[94] - 6 * a[95] ) +
                                                              4 * a[95] ) ) ) ) ) ) ) +
              r * ( -a[32] - 2 * a[64] +
                    z * ( -2 * a[33] - 4 * a[65] +
                          z * ( 3 * a[32] + 3 * a[33] + 6 * a[64] + 6 * a[65] ) ) +
                    cth * ( -a[34] - 2 * a[66] +
                            z * ( -2 * a[35] - 4 * a[67] +
                                  z * ( 3 * a[34] + 3 * a[35] + 6 * a[66] + 6 * a[67] ) ) +
                            cth * ( -a[36] - 2 * a[68] +
                                    z * ( -2 * a[37] - 4 * a[69] +
                                          z * ( 3 * a[36] + 3 * a[37] + 6 * a[68] + 6 * a[69] ) ) +
                                    cth * ( -a[38] - 2 * a[70] +
                                            z * ( -2 * a[39] - 4 * a[71] +
                                                  z * ( 3 * a[38] + 3 * a[39] + 6 * a[70] +
                                                        6 * a[71] ) ) ) ) ) +
                    sth *
                        ( -a[40] - 2 * a[72] +
                          z * ( -2 * a[41] - 4 * a[73] +
                                z * ( 3 * a[40] + 3 * a[41] + 6 * a[72] + 6 * a[73] ) ) +
                          cth * ( -a[42] - 2 * a[74] +
                                  z * ( -2 * a[43] - 4 * a[75] +
                                        z * ( 3 * a[42] + 3 * a[43] + 6 * a[74] + 6 * a[75] ) ) +
                                  cth * ( -a[44] - 2 * a[76] +
                                          z * ( -2 * a[45] - 4 * a[77] +
                                                z * ( 3 * a[44] + 3 * a[45] + 6 * a[76] +
                                                      6 * a[77] ) ) +
                                          cth * ( -a[46] - 2 * a[78] +
                                                  z * ( -2 * a[47] - 4 * a[79] +
                                                        z * ( 3 * a[46] + 3 * a[47] + 6 * a[78] +
                                                              6 * a[79] ) ) ) ) ) +
                          sth *
                              ( -a[48] - 2 * a[80] +
                                z * ( -2 * a[49] - 4 * a[81] +
                                      z * ( 3 * a[48] + 3 * a[49] + 6 * a[80] + 6 * a[81] ) ) +
                                cth *
                                    ( -a[50] - 2 * a[82] +
                                      z * ( -2 * a[51] - 4 * a[83] +
                                            z * ( 3 * a[50] + 3 * a[51] + 6 * a[82] +
                                                  6 * a[83] ) ) +
                                      cth * ( -a[52] - 2 * a[84] +
                                              z * ( -2 * a[53] - 4 * a[85] +
                                                    z * ( 3 * a[52] + 3 * a[53] + 6 * a[84] +
                                                          6 * a[85] ) ) +
                                              cth * ( -a[54] - 2 * a[86] +
                                                      z * ( -2 * a[55] - 4 * a[87] +
                                                            z * ( 3 * a[54] + 3 * a[55] +
                                                                  6 * a[86] + 6 * a[87] ) ) ) ) ) +
                                sth *
                                    ( -a[56] - 2 * a[88] +
                                      z * ( -2 * a[57] - 4 * a[89] +
                                            z * ( 3 * a[56] + 3 * a[57] + 6 * a[88] +
                                                  6 * a[89] ) ) +
                                      cth *
                                          ( -a[58] - 2 * a[90] +
                                            z * ( -2 * a[59] - 4 * a[91] +
                                                  z * ( 3 * a[58] + 3 * a[59] + 6 * a[90] +
                                                        6 * a[91] ) ) +
                                            cth *
                                                ( -a[60] - 2 * a[92] +
                                                  z * ( -2 * a[61] - 4 * a[93] +
                                                        z * ( 3 * a[60] + 3 * a[61] + 6 * a[92] +
                                                              6 * a[93] ) ) +
                                                  cth * ( -a[62] - 2 * a[94] +
                                                          z * ( -2 * a[63] - 4 * a[95] +
                                                                z * ( 3 * a[62] + 3 * a[63] +
                                                                      6 * a[94] +
                                                                      6 * a[95] ) ) ) ) ) ) ) ) ) );
    result[7] =
        cth *
            ( z * ( -a[2] + z * ( -a[3] + z * ( a[2] + a[3] ) ) ) +
              cth * ( z * ( -2 * a[4] + 2 * a[16] +
                            z * ( -2 * a[5] + z * ( 2 * a[4] + 2 * a[5] - 2 * a[16] - 2 * a[17] ) +
                                  2 * a[17] ) ) +
                      cth * ( z * ( -3 * a[6] + 2 * a[18] +
                                    z * ( -3 * a[7] +
                                          z * ( 3 * a[6] + 3 * a[7] - 2 * a[18] - 2 * a[19] ) +
                                          2 * a[19] ) ) +
                              cth * ( z * ( 2 * a[20] +
                                            z * ( z * ( -2 * a[20] - 2 * a[21] ) + 2 * a[21] ) ) +
                                      cth * z *
                                          ( 2 * a[22] + z * ( z * ( -2 * a[22] - 2 * a[23] ) +
                                                              2 * a[23] ) ) ) ) ) ) +
        sth *
            ( z * ( -a[8] + z * ( -a[9] + z * ( a[8] + a[9] ) ) ) +
              cth *
                  ( z * ( -4 * a[10] + z * ( -4 * a[11] + z * ( 4 * a[10] + 4 * a[11] ) ) ) +
                    cth * ( z * ( -7 * a[12] + 6 * a[24] +
                                  z * ( -7 * a[13] +
                                        z * ( 7 * a[12] + 7 * a[13] - 6 * a[24] - 6 * a[25] ) +
                                        6 * a[25] ) ) +
                            cth * ( z * ( -10 * a[14] + 6 * a[26] +
                                          z * ( -10 * a[15] +
                                                z * ( 10 * a[14] + 10 * a[15] - 6 * a[26] -
                                                      6 * a[27] ) +
                                                6 * a[27] ) ) +
                                    cth * ( z * ( 6 * a[28] + z * ( z * ( -6 * a[28] - 6 * a[29] ) +
                                                                    6 * a[29] ) ) +
                                            cth * z *
                                                ( 6 * a[30] + z * ( z * ( -6 * a[30] - 6 * a[31] ) +
                                                                    6 * a[31] ) ) ) ) ) ) +
              sth *
                  ( z * ( 2 * a[4] - 2 * a[16] +
                          z * ( 2 * a[5] - 2 * a[17] +
                                z * ( -2 * a[4] - 2 * a[5] + 2 * a[16] + 2 * a[17] ) ) ) +
                    cth * ( z * ( 6 * a[6] - 7 * a[18] +
                                  z * ( 6 * a[7] - 7 * a[19] +
                                        z * ( -6 * a[6] - 6 * a[7] + 7 * a[18] + 7 * a[19] ) ) ) +
                            cth * ( z * ( -12 * a[20] +
                                          z * ( -12 * a[21] + z * ( 12 * a[20] + 12 * a[21] ) ) ) +
                                    cth * z *
                                        ( -17 * a[22] +
                                          z * ( -17 * a[23] +
                                                z * ( 17 * a[22] + 17 * a[23] ) ) ) ) ) +
                    sth *
                        ( z * ( 2 * a[12] - 3 * a[24] +
                                z * ( 2 * a[13] - 3 * a[25] +
                                      z * ( -2 * a[12] - 2 * a[13] + 3 * a[24] + 3 * a[25] ) ) ) +
                          sth * ( z * ( 2 * a[20] +
                                        z * ( z * ( -2 * a[20] - 2 * a[21] ) + 2 * a[21] ) ) +
                                  cth * z *
                                      ( 6 * a[22] +
                                        z * ( z * ( -6 * a[22] - 6 * a[23] ) + 6 * a[23] ) ) +
                                  sth * ( z * ( 2 * a[28] + z * ( z * ( -2 * a[28] - 2 * a[29] ) +
                                                                  2 * a[29] ) ) +
                                          cth * z *
                                              ( 6 * a[30] + z * ( z * ( -6 * a[30] - 6 * a[31] ) +
                                                                  6 * a[31] ) ) ) ) +
                          cth *
                              ( z * ( 6 * a[14] - 10 * a[26] +
                                      z * ( 6 * a[15] - 10 * a[27] +
                                            z * ( -6 * a[14] - 6 * a[15] + 10 * a[26] +
                                                  10 * a[27] ) ) ) +
                                cth * ( z * ( -17 * a[28] +
                                              z * ( -17 * a[29] +
                                                    z * ( 17 * a[28] + 17 * a[29] ) ) ) +
                                        cth * z *
                                            ( -24 * a[30] +
                                              z * ( -24 * a[31] +
                                                    z * ( 24 * a[30] + 24 * a[31] ) ) ) ) ) ) ) ) +
        r * ( cth *
                  ( z * ( -a[34] + z * ( -a[35] + z * ( a[34] + a[35] ) ) ) +
                    cth *
                        ( z * ( -2 * a[36] + 2 * a[48] +
                                z * ( -2 * a[37] +
                                      z * ( 2 * a[36] + 2 * a[37] - 2 * a[48] - 2 * a[49] ) +
                                      2 * a[49] ) ) +
                          cth *
                              ( z * ( -3 * a[38] + 2 * a[50] +
                                      z * ( -3 * a[39] +
                                            z * ( 3 * a[38] + 3 * a[39] - 2 * a[50] - 2 * a[51] ) +
                                            2 * a[51] ) ) +
                                cth * ( z * ( 2 * a[52] +
                                              z * ( z * ( -2 * a[52] - 2 * a[53] ) + 2 * a[53] ) ) +
                                        cth * z *
                                            ( 2 * a[54] + z * ( z * ( -2 * a[54] - 2 * a[55] ) +
                                                                2 * a[55] ) ) ) ) ) ) +
              sth *
                  ( z * ( -a[40] + z * ( -a[41] + z * ( a[40] + a[41] ) ) ) +
                    cth *
                        ( z * ( -4 * a[42] + z * ( -4 * a[43] + z * ( 4 * a[42] + 4 * a[43] ) ) ) +
                          cth *
                              ( z * ( -7 * a[44] + 6 * a[56] +
                                      z * ( -7 * a[45] +
                                            z * ( 7 * a[44] + 7 * a[45] - 6 * a[56] - 6 * a[57] ) +
                                            6 * a[57] ) ) +
                                cth *
                                    ( z * ( -10 * a[46] + 6 * a[58] +
                                            z * ( -10 * a[47] +
                                                  z * ( 10 * a[46] + 10 * a[47] - 6 * a[58] -
                                                        6 * a[59] ) +
                                                  6 * a[59] ) ) +
                                      cth *
                                          ( z * ( 6 * a[60] + z * ( z * ( -6 * a[60] - 6 * a[61] ) +
                                                                    6 * a[61] ) ) +
                                            cth * z *
                                                ( 6 * a[62] + z * ( z * ( -6 * a[62] - 6 * a[63] ) +
                                                                    6 * a[63] ) ) ) ) ) ) +
                    sth *
                        ( z * ( 2 * a[36] - 2 * a[48] +
                                z * ( 2 * a[37] - 2 * a[49] +
                                      z * ( -2 * a[36] - 2 * a[37] + 2 * a[48] + 2 * a[49] ) ) ) +
                          cth * ( z * ( 6 * a[38] - 7 * a[50] +
                                        z * ( 6 * a[39] - 7 * a[51] +
                                              z * ( -6 * a[38] - 6 * a[39] + 7 * a[50] +
                                                    7 * a[51] ) ) ) +
                                  cth * ( z * ( -12 * a[52] +
                                                z * ( -12 * a[53] +
                                                      z * ( 12 * a[52] + 12 * a[53] ) ) ) +
                                          cth * z *
                                              ( -17 * a[54] +
                                                z * ( -17 * a[55] +
                                                      z * ( 17 * a[54] + 17 * a[55] ) ) ) ) ) +
                          sth *
                              ( z * ( 2 * a[44] - 3 * a[56] +
                                      z * ( 2 * a[45] - 3 * a[57] +
                                            z * ( -2 * a[44] - 2 * a[45] + 3 * a[56] +
                                                  3 * a[57] ) ) ) +
                                sth *
                                    ( z * ( 2 * a[52] +
                                            z * ( z * ( -2 * a[52] - 2 * a[53] ) + 2 * a[53] ) ) +
                                      cth * z *
                                          ( 6 * a[54] +
                                            z * ( z * ( -6 * a[54] - 6 * a[55] ) + 6 * a[55] ) ) +
                                      sth *
                                          ( z * ( 2 * a[60] + z * ( z * ( -2 * a[60] - 2 * a[61] ) +
                                                                    2 * a[61] ) ) +
                                            cth * z *
                                                ( 6 * a[62] + z * ( z * ( -6 * a[62] - 6 * a[63] ) +
                                                                    6 * a[63] ) ) ) ) +
                                cth * ( z * ( 6 * a[46] - 10 * a[58] +
                                              z * ( 6 * a[47] - 10 * a[59] +
                                                    z * ( -6 * a[46] - 6 * a[47] + 10 * a[58] +
                                                          10 * a[59] ) ) ) +
                                        cth * ( z * ( -17 * a[60] +
                                                      z * ( -17 * a[61] +
                                                            z * ( 17 * a[60] + 17 * a[61] ) ) ) +
                                                cth * z *
                                                    ( -24 * a[62] +
                                                      z * ( -24 * a[63] +
                                                            z * ( 24 * a[62] +
                                                                  24 * a[63] ) ) ) ) ) ) ) ) +
              r * ( cth *
                        ( z * ( -a[66] + z * ( -a[67] + z * ( a[66] + a[67] ) ) ) +
                          cth *
                              ( z * ( -2 * a[68] + 2 * a[80] +
                                      z * ( -2 * a[69] +
                                            z * ( 2 * a[68] + 2 * a[69] - 2 * a[80] - 2 * a[81] ) +
                                            2 * a[81] ) ) +
                                cth *
                                    ( z * ( -3 * a[70] + 2 * a[82] +
                                            z * ( -3 * a[71] +
                                                  z * ( 3 * a[70] + 3 * a[71] - 2 * a[82] -
                                                        2 * a[83] ) +
                                                  2 * a[83] ) ) +
                                      cth *
                                          ( z * ( 2 * a[84] + z * ( z * ( -2 * a[84] - 2 * a[85] ) +
                                                                    2 * a[85] ) ) +
                                            cth * z *
                                                ( 2 * a[86] + z * ( z * ( -2 * a[86] - 2 * a[87] ) +
                                                                    2 * a[87] ) ) ) ) ) ) +
                    sth *
                        ( z * ( -a[72] + z * ( -a[73] + z * ( a[72] + a[73] ) ) ) +
                          cth *
                              ( z * ( -4 * a[74] +
                                      z * ( -4 * a[75] + z * ( 4 * a[74] + 4 * a[75] ) ) ) +
                                cth * ( z * ( -7 * a[76] + 6 * a[88] +
                                              z * ( -7 * a[77] +
                                                    z * ( 7 * a[76] + 7 * a[77] - 6 * a[88] -
                                                          6 * a[89] ) +
                                                    6 * a[89] ) ) +
                                        cth * ( z * ( -10 * a[78] + 6 * a[90] +
                                                      z * ( -10 * a[79] +
                                                            z * ( 10 * a[78] + 10 * a[79] -
                                                                  6 * a[90] - 6 * a[91] ) +
                                                            6 * a[91] ) ) +
                                                cth * ( z * ( 6 * a[92] +
                                                              z * ( z * ( -6 * a[92] - 6 * a[93] ) +
                                                                    6 * a[93] ) ) +
                                                        cth * z *
                                                            ( 6 * a[94] +
                                                              z * ( z * ( -6 * a[94] - 6 * a[95] ) +
                                                                    6 * a[95] ) ) ) ) ) ) +
                          sth *
                              ( z * ( 2 * a[68] - 2 * a[80] +
                                      z * ( 2 * a[69] - 2 * a[81] +
                                            z * ( -2 * a[68] - 2 * a[69] + 2 * a[80] +
                                                  2 * a[81] ) ) ) +
                                cth *
                                    ( z * ( 6 * a[70] - 7 * a[82] +
                                            z * ( 6 * a[71] - 7 * a[83] +
                                                  z * ( -6 * a[70] - 6 * a[71] + 7 * a[82] +
                                                        7 * a[83] ) ) ) +
                                      cth * ( z * ( -12 * a[84] +
                                                    z * ( -12 * a[85] +
                                                          z * ( 12 * a[84] + 12 * a[85] ) ) ) +
                                              cth * z *
                                                  ( -17 * a[86] +
                                                    z * ( -17 * a[87] +
                                                          z * ( 17 * a[86] + 17 * a[87] ) ) ) ) ) +
                                sth *
                                    ( z * ( 2 * a[76] - 3 * a[88] +
                                            z * ( 2 * a[77] - 3 * a[89] +
                                                  z * ( -2 * a[76] - 2 * a[77] + 3 * a[88] +
                                                        3 * a[89] ) ) ) +
                                      sth *
                                          ( z * ( 2 * a[84] + z * ( z * ( -2 * a[84] - 2 * a[85] ) +
                                                                    2 * a[85] ) ) +
                                            cth * z *
                                                ( 6 * a[86] + z * ( z * ( -6 * a[86] - 6 * a[87] ) +
                                                                    6 * a[87] ) ) +
                                            sth * ( z * ( 2 * a[92] +
                                                          z * ( z * ( -2 * a[92] - 2 * a[93] ) +
                                                                2 * a[93] ) ) +
                                                    cth * z *
                                                        ( 6 * a[94] +
                                                          z * ( z * ( -6 * a[94] - 6 * a[95] ) +
                                                                6 * a[95] ) ) ) ) +
                                      cth * ( z * ( 6 * a[78] - 10 * a[90] +
                                                    z * ( 6 * a[79] - 10 * a[91] +
                                                          z * ( -6 * a[78] - 6 * a[79] +
                                                                10 * a[90] + 10 * a[91] ) ) ) +
                                              cth * ( z * ( -17 * a[92] +
                                                            z * ( -17 * a[93] +
                                                                  z * ( 17 * a[92] +
                                                                        17 * a[93] ) ) ) +
                                                      cth * z *
                                                          ( -24 * a[94] +
                                                            z * ( -24 * a[95] +
                                                                  z * ( 24 * a[94] +
                                                                        24 * a[95] ) ) ) ) ) ) ) ) +
                    r * ( cth *
                              ( z * ( a[34] / 3 + ( 2 * a[66] ) / 3 +
                                      z * ( a[35] / 3 +
                                            z * ( -a[34] / 3 - a[35] / 3 - ( 2 * a[66] ) / 3 -
                                                  ( 2 * a[67] ) / 3 ) +
                                            ( 2 * a[67] ) / 3 ) ) +
                                cth *
                                    ( z * ( ( 2 * a[36] ) / 3 - ( 2 * a[48] ) / 3 +
                                            ( 4 * a[68] ) / 3 - ( 4 * a[80] ) / 3 +
                                            z * ( ( 2 * a[37] ) / 3 - ( 2 * a[49] ) / 3 +
                                                  ( 4 * a[69] ) / 3 - ( 4 * a[81] ) / 3 +
                                                  z * ( ( -2 * a[36] ) / 3 - ( 2 * a[37] ) / 3 +
                                                        ( 2 * a[48] ) / 3 + ( 2 * a[49] ) / 3 -
                                                        ( 4 * a[68] ) / 3 - ( 4 * a[69] ) / 3 +
                                                        ( 4 * a[80] ) / 3 +
                                                        ( 4 * a[81] ) / 3 ) ) ) +
                                      cth * ( z * ( a[38] - ( 2 * a[50] ) / 3 + 2 * a[70] -
                                                    ( 4 * a[82] ) / 3 +
                                                    z * ( a[39] - ( 2 * a[51] ) / 3 + 2 * a[71] -
                                                          ( 4 * a[83] ) / 3 +
                                                          z * ( -a[38] - a[39] + ( 2 * a[50] ) / 3 +
                                                                ( 2 * a[51] ) / 3 - 2 * a[70] -
                                                                2 * a[71] + ( 4 * a[82] ) / 3 +
                                                                ( 4 * a[83] ) / 3 ) ) ) +
                                              cth * ( z * ( ( -2 * a[52] ) / 3 - ( 4 * a[84] ) / 3 +
                                                            z * ( ( -2 * a[53] ) / 3 -
                                                                  ( 4 * a[85] ) / 3 +
                                                                  z * ( ( 2 * a[52] ) / 3 +
                                                                        ( 2 * a[53] ) / 3 +
                                                                        ( 4 * a[84] ) / 3 +
                                                                        ( 4 * a[85] ) / 3 ) ) ) +
                                                      cth * z *
                                                          ( ( -2 * a[54] ) / 3 - ( 4 * a[86] ) / 3 +
                                                            z * ( ( -2 * a[55] ) / 3 -
                                                                  ( 4 * a[87] ) / 3 +
                                                                  z * ( ( 2 * a[54] ) / 3 +
                                                                        ( 2 * a[55] ) / 3 +
                                                                        ( 4 * a[86] ) / 3 +
                                                                        ( 4 * a[87] ) /
                                                                            3 ) ) ) ) ) ) ) +
                          sth *
                              ( z * ( a[40] / 3 + ( 2 * a[72] ) / 3 +
                                      z * ( a[41] / 3 +
                                            z * ( -a[40] / 3 - a[41] / 3 - ( 2 * a[72] ) / 3 -
                                                  ( 2 * a[73] ) / 3 ) +
                                            ( 2 * a[73] ) / 3 ) ) +
                                sth *
                                    ( z * ( ( -2 * a[36] ) / 3 + ( 2 * a[48] ) / 3 -
                                            ( 4 * a[68] ) / 3 + ( 4 * a[80] ) / 3 +
                                            z * ( ( -2 * a[37] ) / 3 + ( 2 * a[49] ) / 3 -
                                                  ( 4 * a[69] ) / 3 +
                                                  z * ( ( 2 * a[36] ) / 3 + ( 2 * a[37] ) / 3 -
                                                        ( 2 * a[48] ) / 3 - ( 2 * a[49] ) / 3 +
                                                        ( 4 * a[68] ) / 3 + ( 4 * a[69] ) / 3 -
                                                        ( 4 * a[80] ) / 3 - ( 4 * a[81] ) / 3 ) +
                                                  ( 4 * a[81] ) / 3 ) ) +
                                      cth *
                                          ( z * ( -2 * a[38] + ( 7 * a[50] ) / 3 - 4 * a[70] +
                                                  ( 14 * a[82] ) / 3 +
                                                  z * ( -2 * a[39] + ( 7 * a[51] ) / 3 - 4 * a[71] +
                                                        z * ( 2 * a[38] + 2 * a[39] -
                                                              ( 7 * a[50] ) / 3 -
                                                              ( 7 * a[51] ) / 3 + 4 * a[70] +
                                                              4 * a[71] - ( 14 * a[82] ) / 3 -
                                                              ( 14 * a[83] ) / 3 ) +
                                                        ( 14 * a[83] ) / 3 ) ) +
                                            cth * ( z * ( 4 * a[52] + 8 * a[84] +
                                                          z * ( 4 * a[53] +
                                                                z * ( -4 * a[52] - 4 * a[53] -
                                                                      8 * a[84] - 8 * a[85] ) +
                                                                8 * a[85] ) ) +
                                                    cth * z *
                                                        ( ( 17 * a[54] ) / 3 + ( 34 * a[86] ) / 3 +
                                                          z * ( ( 17 * a[55] ) / 3 +
                                                                z * ( ( -17 * a[54] ) / 3 -
                                                                      ( 17 * a[55] ) / 3 -
                                                                      ( 34 * a[86] ) / 3 -
                                                                      ( 34 * a[87] ) / 3 ) +
                                                                ( 34 * a[87] ) / 3 ) ) ) ) +
                                      sth *
                                          ( z * ( ( -2 * a[44] ) / 3 + a[56] - ( 4 * a[76] ) / 3 +
                                                  2 * a[88] +
                                                  z * ( ( -2 * a[45] ) / 3 + a[57] -
                                                        ( 4 * a[77] ) / 3 +
                                                        z * ( ( 2 * a[44] ) / 3 +
                                                              ( 2 * a[45] ) / 3 - a[56] - a[57] +
                                                              ( 4 * a[76] ) / 3 +
                                                              ( 4 * a[77] ) / 3 - 2 * a[88] -
                                                              2 * a[89] ) +
                                                        2 * a[89] ) ) +
                                            cth *
                                                ( z * ( -2 * a[46] + ( 10 * a[58] ) / 3 -
                                                        4 * a[78] + ( 20 * a[90] ) / 3 +
                                                        z * ( -2 * a[47] + ( 10 * a[59] ) / 3 -
                                                              4 * a[79] +
                                                              z * ( 2 * a[46] + 2 * a[47] -
                                                                    ( 10 * a[58] ) / 3 -
                                                                    ( 10 * a[59] ) / 3 + 4 * a[78] +
                                                                    4 * a[79] - ( 20 * a[90] ) / 3 -
                                                                    ( 20 * a[91] ) / 3 ) +
                                                              ( 20 * a[91] ) / 3 ) ) +
                                                  cth * ( z * ( ( 17 * a[60] ) / 3 +
                                                                ( 34 * a[92] ) / 3 +
                                                                z * ( ( 17 * a[61] ) / 3 +
                                                                      z * ( ( -17 * a[60] ) / 3 -
                                                                            ( 17 * a[61] ) / 3 -
                                                                            ( 34 * a[92] ) / 3 -
                                                                            ( 34 * a[93] ) / 3 ) +
                                                                      ( 34 * a[93] ) / 3 ) ) +
                                                          cth * z *
                                                              ( 8 * a[62] + 16 * a[94] +
                                                                z * ( 8 * a[63] +
                                                                      z * ( -8 * a[62] - 8 * a[63] -
                                                                            16 * a[94] -
                                                                            16 * a[95] ) +
                                                                      16 * a[95] ) ) ) ) +
                                            sth *
                                                ( z * ( ( -2 * a[52] ) / 3 - ( 4 * a[84] ) / 3 +
                                                        z * ( ( -2 * a[53] ) / 3 -
                                                              ( 4 * a[85] ) / 3 +
                                                              z * ( ( 2 * a[52] ) / 3 +
                                                                    ( 2 * a[53] ) / 3 +
                                                                    ( 4 * a[84] ) / 3 +
                                                                    ( 4 * a[85] ) / 3 ) ) ) +
                                                  cth * z *
                                                      ( -2 * a[54] - 4 * a[86] +
                                                        z * ( -2 * a[55] - 4 * a[87] +
                                                              z * ( 2 * a[54] + 2 * a[55] +
                                                                    4 * a[86] + 4 * a[87] ) ) ) +
                                                  sth *
                                                      ( z * ( ( -2 * a[60] ) / 3 -
                                                              ( 4 * a[92] ) / 3 +
                                                              z * ( ( -2 * a[61] ) / 3 -
                                                                    ( 4 * a[93] ) / 3 +
                                                                    z * ( ( 2 * a[60] ) / 3 +
                                                                          ( 2 * a[61] ) / 3 +
                                                                          ( 4 * a[92] ) / 3 +
                                                                          ( 4 * a[93] ) / 3 ) ) ) +
                                                        cth * z *
                                                            ( -2 * a[62] - 4 * a[94] +
                                                              z * ( -2 * a[63] - 4 * a[95] +
                                                                    z * ( 2 * a[62] + 2 * a[63] +
                                                                          4 * a[94] +
                                                                          4 * a[95] ) ) ) ) ) ) ) +
                                cth *
                                    ( z * ( ( 4 * a[42] ) / 3 + ( 8 * a[74] ) / 3 +
                                            z * ( ( 4 * a[43] ) / 3 +
                                                  z * ( ( -4 * a[42] ) / 3 - ( 4 * a[43] ) / 3 -
                                                        ( 8 * a[74] ) / 3 - ( 8 * a[75] ) / 3 ) +
                                                  ( 8 * a[75] ) / 3 ) ) +
                                      cth *
                                          ( z * ( ( 7 * a[44] ) / 3 - 2 * a[56] +
                                                  ( 14 * a[76] ) / 3 - 4 * a[88] +
                                                  z * ( ( 7 * a[45] ) / 3 - 2 * a[57] +
                                                        ( 14 * a[77] ) / 3 - 4 * a[89] +
                                                        z * ( ( -7 * a[44] ) / 3 -
                                                              ( 7 * a[45] ) / 3 + 2 * a[56] +
                                                              2 * a[57] - ( 14 * a[76] ) / 3 -
                                                              ( 14 * a[77] ) / 3 + 4 * a[88] +
                                                              4 * a[89] ) ) ) +
                                            cth *
                                                ( z * ( ( 10 * a[46] ) / 3 - 2 * a[58] +
                                                        ( 20 * a[78] ) / 3 - 4 * a[90] +
                                                        z * ( ( 10 * a[47] ) / 3 - 2 * a[59] +
                                                              ( 20 * a[79] ) / 3 - 4 * a[91] +
                                                              z * ( ( -10 * a[46] ) / 3 -
                                                                    ( 10 * a[47] ) / 3 + 2 * a[58] +
                                                                    2 * a[59] - ( 20 * a[78] ) / 3 -
                                                                    ( 20 * a[79] ) / 3 + 4 * a[90] +
                                                                    4 * a[91] ) ) ) +
                                                  cth *
                                                      ( z * ( -2 * a[60] - 4 * a[92] +
                                                              z * ( -2 * a[61] - 4 * a[93] +
                                                                    z * ( 2 * a[60] + 2 * a[61] +
                                                                          4 * a[92] +
                                                                          4 * a[93] ) ) ) +
                                                        cth * z *
                                                            ( -2 * a[62] - 4 * a[94] +
                                                              z * ( -2 * a[63] - 4 * a[95] +
                                                                    z * ( 2 * a[62] + 2 * a[63] +
                                                                          4 * a[94] +
                                                                          4 * a[95] ) ) ) ) ) ) ) ) ) ) );
    result[8] =
        cth * ( a[8] + z * ( z * ( -3 * a[8] - 3 * a[9] ) + 2 * a[9] ) +
                cth * ( a[10] + z * ( z * ( -3 * a[10] - 3 * a[11] ) + 2 * a[11] ) +
                        cth * ( a[12] + z * ( z * ( -3 * a[12] - 3 * a[13] ) + 2 * a[13] ) +
                                cth * ( a[14] +
                                        z * ( z * ( -3 * a[14] - 3 * a[15] ) + 2 * a[15] ) ) ) ) ) +
        sth *
            ( -a[2] + z * ( -2 * a[3] + z * ( 3 * a[2] + 3 * a[3] ) ) +
              cth *
                  ( -2 * a[4] + 2 * a[16] +
                    z * ( -4 * a[5] + z * ( 6 * a[4] + 6 * a[5] - 6 * a[16] - 6 * a[17] ) +
                          4 * a[17] ) +
                    cth * ( -3 * a[6] + 2 * a[18] +
                            z * ( -6 * a[7] + z * ( 9 * a[6] + 9 * a[7] - 6 * a[18] - 6 * a[19] ) +
                                  4 * a[19] ) +
                            cth * ( 2 * a[20] + z * ( z * ( -6 * a[20] - 6 * a[21] ) + 4 * a[21] ) +
                                    cth * ( 2 * a[22] + z * ( z * ( -6 * a[22] - 6 * a[23] ) +
                                                              4 * a[23] ) ) ) ) ) +
              sth *
                  ( -a[10] + z * ( -2 * a[11] + z * ( 3 * a[10] + 3 * a[11] ) ) +
                    cth *
                        ( -2 * a[12] + 3 * a[24] +
                          z * ( -4 * a[13] + z * ( 6 * a[12] + 6 * a[13] - 9 * a[24] - 9 * a[25] ) +
                                6 * a[25] ) +
                          cth * ( -3 * a[14] + 3 * a[26] +
                                  z * ( -6 * a[15] +
                                        z * ( 9 * a[14] + 9 * a[15] - 9 * a[26] - 9 * a[27] ) +
                                        6 * a[27] ) +
                                  cth * ( 3 * a[28] +
                                          z * ( z * ( -9 * a[28] - 9 * a[29] ) + 6 * a[29] ) +
                                          cth * ( 3 * a[30] + z * ( z * ( -9 * a[30] - 9 * a[31] ) +
                                                                    6 * a[31] ) ) ) ) ) +
                    sth *
                        ( -a[18] + z * ( -2 * a[19] + z * ( 3 * a[18] + 3 * a[19] ) ) +
                          cth * ( -2 * a[20] + z * ( -4 * a[21] + z * ( 6 * a[20] + 6 * a[21] ) ) +
                                  cth * ( -3 * a[22] +
                                          z * ( -6 * a[23] + z * ( 9 * a[22] + 9 * a[23] ) ) ) ) +
                          sth *
                              ( -a[26] + z * ( -2 * a[27] + z * ( 3 * a[26] + 3 * a[27] ) ) +
                                cth * ( -2 * a[28] +
                                        z * ( -4 * a[29] + z * ( 6 * a[28] + 6 * a[29] ) ) +
                                        cth * ( -3 * a[30] +
                                                z * ( -6 * a[31] +
                                                      z * ( 9 * a[30] + 9 * a[31] ) ) ) ) ) ) ) ) +
        r * ( cth * ( a[40] + z * ( z * ( -3 * a[40] - 3 * a[41] ) + 2 * a[41] ) +
                      cth * ( a[42] + z * ( z * ( -3 * a[42] - 3 * a[43] ) + 2 * a[43] ) +
                              cth * ( a[44] + z * ( z * ( -3 * a[44] - 3 * a[45] ) + 2 * a[45] ) +
                                      cth * ( a[46] + z * ( z * ( -3 * a[46] - 3 * a[47] ) +
                                                            2 * a[47] ) ) ) ) ) +
              sth *
                  ( -a[34] + z * ( -2 * a[35] + z * ( 3 * a[34] + 3 * a[35] ) ) +
                    cth *
                        ( -2 * a[36] + 2 * a[48] +
                          z * ( -4 * a[37] + z * ( 6 * a[36] + 6 * a[37] - 6 * a[48] - 6 * a[49] ) +
                                4 * a[49] ) +
                          cth * ( -3 * a[38] + 2 * a[50] +
                                  z * ( -6 * a[39] +
                                        z * ( 9 * a[38] + 9 * a[39] - 6 * a[50] - 6 * a[51] ) +
                                        4 * a[51] ) +
                                  cth * ( 2 * a[52] +
                                          z * ( z * ( -6 * a[52] - 6 * a[53] ) + 4 * a[53] ) +
                                          cth * ( 2 * a[54] + z * ( z * ( -6 * a[54] - 6 * a[55] ) +
                                                                    4 * a[55] ) ) ) ) ) +
                    sth *
                        ( -a[42] + z * ( -2 * a[43] + z * ( 3 * a[42] + 3 * a[43] ) ) +
                          cth *
                              ( -2 * a[44] + 3 * a[56] +
                                z * ( -4 * a[45] +
                                      z * ( 6 * a[44] + 6 * a[45] - 9 * a[56] - 9 * a[57] ) +
                                      6 * a[57] ) +
                                cth *
                                    ( -3 * a[46] + 3 * a[58] +
                                      z * ( -6 * a[47] +
                                            z * ( 9 * a[46] + 9 * a[47] - 9 * a[58] - 9 * a[59] ) +
                                            6 * a[59] ) +
                                      cth * ( 3 * a[60] +
                                              z * ( z * ( -9 * a[60] - 9 * a[61] ) + 6 * a[61] ) +
                                              cth * ( 3 * a[62] +
                                                      z * ( z * ( -9 * a[62] - 9 * a[63] ) +
                                                            6 * a[63] ) ) ) ) ) +
                          sth *
                              ( -a[50] + z * ( -2 * a[51] + z * ( 3 * a[50] + 3 * a[51] ) ) +
                                cth * ( -2 * a[52] +
                                        z * ( -4 * a[53] + z * ( 6 * a[52] + 6 * a[53] ) ) +
                                        cth * ( -3 * a[54] +
                                                z * ( -6 * a[55] +
                                                      z * ( 9 * a[54] + 9 * a[55] ) ) ) ) +
                                sth *
                                    ( -a[58] + z * ( -2 * a[59] + z * ( 3 * a[58] + 3 * a[59] ) ) +
                                      cth * ( -2 * a[60] +
                                              z * ( -4 * a[61] + z * ( 6 * a[60] + 6 * a[61] ) ) +
                                              cth * ( -3 * a[62] +
                                                      z * ( -6 * a[63] +
                                                            z * ( 9 * a[62] +
                                                                  9 * a[63] ) ) ) ) ) ) ) ) +
              r * ( cth * ( a[72] + z * ( z * ( -3 * a[72] - 3 * a[73] ) + 2 * a[73] ) +
                            cth * ( a[74] + z * ( z * ( -3 * a[74] - 3 * a[75] ) + 2 * a[75] ) +
                                    cth * ( a[76] +
                                            z * ( z * ( -3 * a[76] - 3 * a[77] ) + 2 * a[77] ) +
                                            cth * ( a[78] + z * ( z * ( -3 * a[78] - 3 * a[79] ) +
                                                                  2 * a[79] ) ) ) ) ) +
                    sth *
                        ( -a[66] + z * ( -2 * a[67] + z * ( 3 * a[66] + 3 * a[67] ) ) +
                          cth *
                              ( -2 * a[68] + 2 * a[80] +
                                z * ( -4 * a[69] +
                                      z * ( 6 * a[68] + 6 * a[69] - 6 * a[80] - 6 * a[81] ) +
                                      4 * a[81] ) +
                                cth *
                                    ( -3 * a[70] + 2 * a[82] +
                                      z * ( -6 * a[71] +
                                            z * ( 9 * a[70] + 9 * a[71] - 6 * a[82] - 6 * a[83] ) +
                                            4 * a[83] ) +
                                      cth * ( 2 * a[84] +
                                              z * ( z * ( -6 * a[84] - 6 * a[85] ) + 4 * a[85] ) +
                                              cth * ( 2 * a[86] +
                                                      z * ( z * ( -6 * a[86] - 6 * a[87] ) +
                                                            4 * a[87] ) ) ) ) ) +
                          sth *
                              ( -a[74] + z * ( -2 * a[75] + z * ( 3 * a[74] + 3 * a[75] ) ) +
                                cth *
                                    ( -2 * a[76] + 3 * a[88] +
                                      z * ( -4 * a[77] +
                                            z * ( 6 * a[76] + 6 * a[77] - 9 * a[88] - 9 * a[89] ) +
                                            6 * a[89] ) +
                                      cth * ( -3 * a[78] + 3 * a[90] +
                                              z * ( -6 * a[79] +
                                                    z * ( 9 * a[78] + 9 * a[79] - 9 * a[90] -
                                                          9 * a[91] ) +
                                                    6 * a[91] ) +
                                              cth * ( 3 * a[92] +
                                                      z * ( z * ( -9 * a[92] - 9 * a[93] ) +
                                                            6 * a[93] ) +
                                                      cth * ( 3 * a[94] +
                                                              z * ( z * ( -9 * a[94] - 9 * a[95] ) +
                                                                    6 * a[95] ) ) ) ) ) +
                                sth *
                                    ( -a[82] + z * ( -2 * a[83] + z * ( 3 * a[82] + 3 * a[83] ) ) +
                                      cth * ( -2 * a[84] +
                                              z * ( -4 * a[85] + z * ( 6 * a[84] + 6 * a[85] ) ) +
                                              cth * ( -3 * a[86] +
                                                      z * ( -6 * a[87] +
                                                            z * ( 9 * a[86] + 9 * a[87] ) ) ) ) +
                                      sth *
                                          ( -a[90] +
                                            z * ( -2 * a[91] + z * ( 3 * a[90] + 3 * a[91] ) ) +
                                            cth * ( -2 * a[92] +
                                                    z * ( -4 * a[93] +
                                                          z * ( 6 * a[92] + 6 * a[93] ) ) +
                                                    cth * ( -3 * a[94] +
                                                            z * ( -6 * a[95] +
                                                                  z * ( 9 * a[94] +
                                                                        9 * a[95] ) ) ) ) ) ) ) ) +
                    r * ( cth *
                              ( -a[40] / 3 - ( 2 * a[72] ) / 3 +
                                z * ( ( -2 * a[41] ) / 3 - ( 4 * a[73] ) / 3 +
                                      z * ( a[40] + a[41] + 2 * a[72] + 2 * a[73] ) ) +
                                cth *
                                    ( -a[42] / 3 - ( 2 * a[74] ) / 3 +
                                      z * ( ( -2 * a[43] ) / 3 - ( 4 * a[75] ) / 3 +
                                            z * ( a[42] + a[43] + 2 * a[74] + 2 * a[75] ) ) +
                                      cth *
                                          ( -a[44] / 3 - ( 2 * a[76] ) / 3 +
                                            z * ( ( -2 * a[45] ) / 3 - ( 4 * a[77] ) / 3 +
                                                  z * ( a[44] + a[45] + 2 * a[76] + 2 * a[77] ) ) +
                                            cth * ( -a[46] / 3 - ( 2 * a[78] ) / 3 +
                                                    z * ( ( -2 * a[47] ) / 3 - ( 4 * a[79] ) / 3 +
                                                          z * ( a[46] + a[47] + 2 * a[78] +
                                                                2 * a[79] ) ) ) ) ) ) +
                          sth *
                              ( a[34] / 3 + ( 2 * a[66] ) / 3 +
                                z * ( ( 2 * a[35] ) / 3 +
                                      z * ( -a[34] - a[35] - 2 * a[66] - 2 * a[67] ) +
                                      ( 4 * a[67] ) / 3 ) +
                                cth *
                                    ( ( 2 * a[36] ) / 3 - ( 2 * a[48] ) / 3 + ( 4 * a[68] ) / 3 -
                                      ( 4 * a[80] ) / 3 +
                                      z * ( ( 4 * a[37] ) / 3 - ( 4 * a[49] ) / 3 +
                                            ( 8 * a[69] ) / 3 - ( 8 * a[81] ) / 3 +
                                            z * ( -2 * a[36] - 2 * a[37] + 2 * a[48] + 2 * a[49] -
                                                  4 * a[68] - 4 * a[69] + 4 * a[80] +
                                                  4 * a[81] ) ) +
                                      cth *
                                          ( a[38] - ( 2 * a[50] ) / 3 + 2 * a[70] -
                                            ( 4 * a[82] ) / 3 +
                                            z * ( 2 * a[39] - ( 4 * a[51] ) / 3 + 4 * a[71] -
                                                  ( 8 * a[83] ) / 3 +
                                                  z * ( -3 * a[38] - 3 * a[39] + 2 * a[50] +
                                                        2 * a[51] - 6 * a[70] - 6 * a[71] +
                                                        4 * a[82] + 4 * a[83] ) ) +
                                            cth * ( ( -2 * a[52] ) / 3 - ( 4 * a[84] ) / 3 +
                                                    z * ( ( -4 * a[53] ) / 3 - ( 8 * a[85] ) / 3 +
                                                          z * ( 2 * a[52] + 2 * a[53] + 4 * a[84] +
                                                                4 * a[85] ) ) +
                                                    cth * ( ( -2 * a[54] ) / 3 - ( 4 * a[86] ) / 3 +
                                                            z * ( ( -4 * a[55] ) / 3 -
                                                                  ( 8 * a[87] ) / 3 +
                                                                  z * ( 2 * a[54] + 2 * a[55] +
                                                                        4 * a[86] +
                                                                        4 * a[87] ) ) ) ) ) ) +
                                sth *
                                    ( a[42] / 3 + ( 2 * a[74] ) / 3 +
                                      z * ( ( 2 * a[43] ) / 3 +
                                            z * ( -a[42] - a[43] - 2 * a[74] - 2 * a[75] ) +
                                            ( 4 * a[75] ) / 3 ) +
                                      sth *
                                          ( a[50] / 3 + ( 2 * a[82] ) / 3 +
                                            z * ( ( 2 * a[51] ) / 3 +
                                                  z * ( -a[50] - a[51] - 2 * a[82] - 2 * a[83] ) +
                                                  ( 4 * a[83] ) / 3 ) +
                                            cth * ( ( 2 * a[52] ) / 3 + ( 4 * a[84] ) / 3 +
                                                    z * ( ( 4 * a[53] ) / 3 +
                                                          z * ( -2 * a[52] - 2 * a[53] - 4 * a[84] -
                                                                4 * a[85] ) +
                                                          ( 8 * a[85] ) / 3 ) +
                                                    cth * ( a[54] + 2 * a[86] +
                                                            z * ( 2 * a[55] +
                                                                  z * ( -3 * a[54] - 3 * a[55] -
                                                                        6 * a[86] - 6 * a[87] ) +
                                                                  4 * a[87] ) ) ) +
                                            sth * ( a[58] / 3 + ( 2 * a[90] ) / 3 +
                                                    z * ( ( 2 * a[59] ) / 3 +
                                                          z * ( -a[58] - a[59] - 2 * a[90] -
                                                                2 * a[91] ) +
                                                          ( 4 * a[91] ) / 3 ) +
                                                    cth * ( ( 2 * a[60] ) / 3 + ( 4 * a[92] ) / 3 +
                                                            z * ( ( 4 * a[61] ) / 3 +
                                                                  z * ( -2 * a[60] - 2 * a[61] -
                                                                        4 * a[92] - 4 * a[93] ) +
                                                                  ( 8 * a[93] ) / 3 ) +
                                                            cth * ( a[62] + 2 * a[94] +
                                                                    z * ( 2 * a[63] +
                                                                          z * ( -3 * a[62] -
                                                                                3 * a[63] -
                                                                                6 * a[94] -
                                                                                6 * a[95] ) +
                                                                          4 * a[95] ) ) ) ) ) +
                                      cth *
                                          ( ( 2 * a[44] ) / 3 - a[56] + ( 4 * a[76] ) / 3 -
                                            2 * a[88] +
                                            z * ( ( 4 * a[45] ) / 3 - 2 * a[57] +
                                                  ( 8 * a[77] ) / 3 - 4 * a[89] +
                                                  z * ( -2 * a[44] - 2 * a[45] + 3 * a[56] +
                                                        3 * a[57] - 4 * a[76] - 4 * a[77] +
                                                        6 * a[88] + 6 * a[89] ) ) +
                                            cth *
                                                ( a[46] - a[58] + 2 * a[78] - 2 * a[90] +
                                                  z * ( 2 * a[47] - 2 * a[59] + 4 * a[79] -
                                                        4 * a[91] +
                                                        z * ( -3 * a[46] - 3 * a[47] + 3 * a[58] +
                                                              3 * a[59] - 6 * a[78] - 6 * a[79] +
                                                              6 * a[90] + 6 * a[91] ) ) +
                                                  cth *
                                                      ( -a[60] - 2 * a[92] +
                                                        z * ( -2 * a[61] - 4 * a[93] +
                                                              z * ( 3 * a[60] + 3 * a[61] +
                                                                    6 * a[92] + 6 * a[93] ) ) +
                                                        cth *
                                                            ( -a[62] - 2 * a[94] +
                                                              z * ( -2 * a[63] - 4 * a[95] +
                                                                    z * ( 3 * a[62] + 3 * a[63] +
                                                                          6 * a[94] +
                                                                          6 * a[95] ) ) ) ) ) ) ) ) ) ) );
    result[9] =
        2 * a[1] +
        cth * ( z * ( -6 * a[2] - 6 * a[3] ) + 2 * a[3] +
                cth * ( z * ( -6 * a[4] - 6 * a[5] ) + 2 * a[5] +
                        cth * ( z * ( -6 * a[6] - 6 * a[7] ) + 2 * a[7] ) ) ) +
        sth * ( z * ( -6 * a[8] - 6 * a[9] ) + 2 * a[9] +
                cth * ( z * ( -6 * a[10] - 6 * a[11] ) + 2 * a[11] +
                        cth * ( z * ( -6 * a[12] - 6 * a[13] ) + 2 * a[13] +
                                cth * ( z * ( -6 * a[14] - 6 * a[15] ) + 2 * a[15] ) ) ) +
                sth * ( z * ( -6 * a[16] - 6 * a[17] ) + 2 * a[17] +
                        cth * ( z * ( -6 * a[18] - 6 * a[19] ) + 2 * a[19] +
                                cth * ( z * ( -6 * a[20] - 6 * a[21] ) + 2 * a[21] +
                                        cth * ( z * ( -6 * a[22] - 6 * a[23] ) + 2 * a[23] ) ) ) +
                        sth * ( z * ( -6 * a[24] - 6 * a[25] ) + 2 * a[25] +
                                cth * ( z * ( -6 * a[26] - 6 * a[27] ) + 2 * a[27] +
                                        cth * ( z * ( -6 * a[28] - 6 * a[29] ) + 2 * a[29] +
                                                cth * ( z * ( -6 * a[30] - 6 * a[31] ) +
                                                        2 * a[31] ) ) ) ) ) ) +
        r * ( z * ( -6 * a[32] - 6 * a[33] ) + 2 * a[33] +
              cth * ( z * ( -6 * a[34] - 6 * a[35] ) + 2 * a[35] +
                      cth * ( z * ( -6 * a[36] - 6 * a[37] ) + 2 * a[37] +
                              cth * ( z * ( -6 * a[38] - 6 * a[39] ) + 2 * a[39] ) ) ) +
              sth *
                  ( z * ( -6 * a[40] - 6 * a[41] ) + 2 * a[41] +
                    cth * ( z * ( -6 * a[42] - 6 * a[43] ) + 2 * a[43] +
                            cth * ( z * ( -6 * a[44] - 6 * a[45] ) + 2 * a[45] +
                                    cth * ( z * ( -6 * a[46] - 6 * a[47] ) + 2 * a[47] ) ) ) +
                    sth *
                        ( z * ( -6 * a[48] - 6 * a[49] ) + 2 * a[49] +
                          cth * ( z * ( -6 * a[50] - 6 * a[51] ) + 2 * a[51] +
                                  cth * ( z * ( -6 * a[52] - 6 * a[53] ) + 2 * a[53] +
                                          cth * ( z * ( -6 * a[54] - 6 * a[55] ) + 2 * a[55] ) ) ) +
                          sth * ( z * ( -6 * a[56] - 6 * a[57] ) + 2 * a[57] +
                                  cth * ( z * ( -6 * a[58] - 6 * a[59] ) + 2 * a[59] +
                                          cth * ( z * ( -6 * a[60] - 6 * a[61] ) + 2 * a[61] +
                                                  cth * ( z * ( -6 * a[62] - 6 * a[63] ) +
                                                          2 * a[63] ) ) ) ) ) ) +
              r * ( z * ( -6 * a[64] - 6 * a[65] ) + 2 * a[65] +
                    cth * ( z * ( -6 * a[66] - 6 * a[67] ) + 2 * a[67] +
                            cth * ( z * ( -6 * a[68] - 6 * a[69] ) + 2 * a[69] +
                                    cth * ( z * ( -6 * a[70] - 6 * a[71] ) + 2 * a[71] ) ) ) +
                    sth *
                        ( z * ( -6 * a[72] - 6 * a[73] ) + 2 * a[73] +
                          cth * ( z * ( -6 * a[74] - 6 * a[75] ) + 2 * a[75] +
                                  cth * ( z * ( -6 * a[76] - 6 * a[77] ) + 2 * a[77] +
                                          cth * ( z * ( -6 * a[78] - 6 * a[79] ) + 2 * a[79] ) ) ) +
                          sth *
                              ( z * ( -6 * a[80] - 6 * a[81] ) + 2 * a[81] +
                                cth * ( z * ( -6 * a[82] - 6 * a[83] ) + 2 * a[83] +
                                        cth * ( z * ( -6 * a[84] - 6 * a[85] ) + 2 * a[85] +
                                                cth * ( z * ( -6 * a[86] - 6 * a[87] ) +
                                                        2 * a[87] ) ) ) +
                                sth * ( z * ( -6 * a[88] - 6 * a[89] ) + 2 * a[89] +
                                        cth * ( z * ( -6 * a[90] - 6 * a[91] ) + 2 * a[91] +
                                                cth * ( z * ( -6 * a[92] - 6 * a[93] ) + 2 * a[93] +
                                                        cth * ( z * ( -6 * a[94] - 6 * a[95] ) +
                                                                2 * a[95] ) ) ) ) ) ) +
                    r * ( ( -2 * a[33] ) / 3 - ( 4 * a[65] ) / 3 +
                          z * ( 2 * a[32] + 2 * a[33] + 4 * a[64] + 4 * a[65] ) +
                          cth * ( ( -2 * a[35] ) / 3 - ( 4 * a[67] ) / 3 +
                                  z * ( 2 * a[34] + 2 * a[35] + 4 * a[66] + 4 * a[67] ) +
                                  cth * ( ( -2 * a[37] ) / 3 - ( 4 * a[69] ) / 3 +
                                          z * ( 2 * a[36] + 2 * a[37] + 4 * a[68] + 4 * a[69] ) +
                                          cth * ( ( -2 * a[39] ) / 3 - ( 4 * a[71] ) / 3 +
                                                  z * ( 2 * a[38] + 2 * a[39] + 4 * a[70] +
                                                        4 * a[71] ) ) ) ) +
                          sth *
                              ( ( -2 * a[41] ) / 3 - ( 4 * a[73] ) / 3 +
                                z * ( 2 * a[40] + 2 * a[41] + 4 * a[72] + 4 * a[73] ) +
                                cth * ( ( -2 * a[43] ) / 3 - ( 4 * a[75] ) / 3 +
                                        z * ( 2 * a[42] + 2 * a[43] + 4 * a[74] + 4 * a[75] ) +
                                        cth * ( ( -2 * a[45] ) / 3 - ( 4 * a[77] ) / 3 +
                                                z * ( 2 * a[44] + 2 * a[45] + 4 * a[76] +
                                                      4 * a[77] ) +
                                                cth * ( ( -2 * a[47] ) / 3 - ( 4 * a[79] ) / 3 +
                                                        z * ( 2 * a[46] + 2 * a[47] + 4 * a[78] +
                                                              4 * a[79] ) ) ) ) +
                                sth *
                                    ( ( -2 * a[49] ) / 3 - ( 4 * a[81] ) / 3 +
                                      z * ( 2 * a[48] + 2 * a[49] + 4 * a[80] + 4 * a[81] ) +
                                      cth *
                                          ( ( -2 * a[51] ) / 3 - ( 4 * a[83] ) / 3 +
                                            z * ( 2 * a[50] + 2 * a[51] + 4 * a[82] + 4 * a[83] ) +
                                            cth * ( ( -2 * a[53] ) / 3 - ( 4 * a[85] ) / 3 +
                                                    z * ( 2 * a[52] + 2 * a[53] + 4 * a[84] +
                                                          4 * a[85] ) +
                                                    cth * ( ( -2 * a[55] ) / 3 - ( 4 * a[87] ) / 3 +
                                                            z * ( 2 * a[54] + 2 * a[55] +
                                                                  4 * a[86] + 4 * a[87] ) ) ) ) +
                                      sth *
                                          ( ( -2 * a[57] ) / 3 - ( 4 * a[89] ) / 3 +
                                            z * ( 2 * a[56] + 2 * a[57] + 4 * a[88] + 4 * a[89] ) +
                                            cth *
                                                ( ( -2 * a[59] ) / 3 - ( 4 * a[91] ) / 3 +
                                                  z * ( 2 * a[58] + 2 * a[59] + 4 * a[90] +
                                                        4 * a[91] ) +
                                                  cth *
                                                      ( ( -2 * a[61] ) / 3 - ( 4 * a[93] ) / 3 +
                                                        z * ( 2 * a[60] + 2 * a[61] + 4 * a[92] +
                                                              4 * a[93] ) +
                                                        cth *
                                                            ( ( -2 * a[63] ) / 3 -
                                                              ( 4 * a[95] ) / 3 +
                                                              z * ( 2 * a[62] + 2 * a[63] +
                                                                    4 * a[94] +
                                                                    4 * a[95] ) ) ) ) ) ) ) ) ) ) +
        z * ( -6 * a[0] - 6 * a[1] - 6 * c[0] + 6 * c[1] );
    return result;
}


static std::array<double, 10>
cubic_cyl_rod_rz_none( double r, const double, double z, const ManufacturedSolution *mfs )
{
    const auto &a = mfs->geta();
    AMP_INSIST( a.size() >= 16, "incorrect number of parameters specified" );
    std::array<double, 10> result;
    result[0] = a[0] + z * ( a[1] + z * ( a[2] + z * a[3] ) ) +
                r * ( a[4] + z * ( a[5] + z * ( a[6] + z * a[7] ) ) +
                      r * ( a[8] + z * ( a[9] + z * ( a[10] + z * a[11] ) ) +
                            r * ( a[12] + z * ( a[13] + z * ( a[14] + z * a[15] ) ) ) ) );
    result[1] = a[4] + z * ( a[5] + z * ( a[6] + z * a[7] ) ) +
                r * ( 2 * a[8] + z * ( 2 * a[9] + z * ( 2 * a[10] + 2 * z * a[11] ) ) +
                      r * ( 3 * a[12] + z * ( 3 * a[13] + z * ( 3 * a[14] + 3 * z * a[15] ) ) ) );
    result[2] = 0;
    result[3] = a[1] + z * ( 2 * a[2] + 3 * z * a[3] ) +
                r * ( a[5] + z * ( 2 * a[6] + 3 * z * a[7] ) +
                      r * ( a[9] + z * ( 2 * a[10] + 3 * z * a[11] ) +
                            r * ( a[13] + z * ( 2 * a[14] + 3 * z * a[15] ) ) ) );
    result[4] = 2 * a[8] + z * ( 2 * a[9] + z * ( 2 * a[10] + 2 * z * a[11] ) ) +
                r * ( 6 * a[12] + z * ( 6 * a[13] + z * ( 6 * a[14] + 6 * z * a[15] ) ) );
    result[5] = 0;
    result[6] = a[5] + z * ( 2 * a[6] + 3 * z * a[7] ) +
                r * ( 2 * a[9] + z * ( 4 * a[10] + 6 * z * a[11] ) +
                      r * ( 3 * a[13] + z * ( 6 * a[14] + 9 * z * a[15] ) ) );
    result[7] = 0;
    result[8] = 0;
    result[9] = 2 * a[2] + 6 * z * a[3] +
                r * ( 2 * a[6] + 6 * z * a[7] +
                      r * ( 2 * a[10] + 6 * z * a[11] + r * ( 2 * a[14] + 6 * z * a[15] ) ) );
    return result;
}


static std::array<double, 10>
cubic_cyl_rod_none( double r, double th, double z, const ManufacturedSolution *mfs )
{
    const auto &a = mfs->geta();
    AMP_INSIST( a.size() >= 64, "incorrect number of parameters specified" );
    double sth = sin( th ), s2th = sin( 2. * th ), s3th = sin( 3. * th );
    double cth = cos( th ), c2th = cos( 2. * th ), c3th = cos( 3. * th );
    std::array<double, 10> result;
    result[0] =
        a[0] + cth * ( a[4] + z * ( a[5] + z * ( a[6] + z * a[7] ) ) ) + c2th * a[8] +
        c3th * a[12] +
        z * ( a[1] + c2th * a[9] + c3th * a[13] +
              z * ( a[2] + c2th * a[10] + c3th * a[14] +
                    z * ( a[3] + c2th * a[11] + c3th * a[15] ) ) ) +
        r * ( a[16] + cth * ( a[20] + z * ( a[21] + z * ( a[22] + z * a[23] ) ) ) + c2th * a[24] +
              c3th * a[28] +
              z * ( a[17] + c2th * a[25] + c3th * a[29] +
                    z * ( a[18] + c2th * a[26] + c3th * a[30] +
                          z * ( a[19] + c2th * a[27] + c3th * a[31] ) ) ) +
              r * ( a[32] + cth * ( a[36] + z * ( a[37] + z * ( a[38] + z * a[39] ) ) ) +
                    c2th * a[40] + c3th * a[44] +
                    z * ( a[33] + c2th * a[41] + c3th * a[45] +
                          z * ( a[34] + c2th * a[42] + c3th * a[46] +
                                z * ( a[35] + c2th * a[43] + c3th * a[47] ) ) ) +
                    r * ( a[48] + cth * ( a[52] + z * ( a[53] + z * ( a[54] + z * a[55] ) ) ) +
                          c2th * a[56] + c3th * a[60] +
                          z * ( a[49] + c2th * a[57] + c3th * a[61] +
                                z * ( a[50] + c2th * a[58] + c3th * a[62] +
                                      z * ( a[51] + c2th * a[59] + c3th * a[63] ) ) ) ) ) );
    result[1] =
        a[16] + cth * ( a[20] + z * ( a[21] + z * ( a[22] + z * a[23] ) ) ) + c2th * a[24] +
        c3th * a[28] +
        z * ( a[17] + c2th * a[25] + c3th * a[29] +
              z * ( a[18] + c2th * a[26] + c3th * a[30] +
                    z * ( a[19] + c2th * a[27] + c3th * a[31] ) ) ) +
        r * ( 2 * a[32] +
              cth * ( 2 * a[36] + z * ( 2 * a[37] + z * ( 2 * a[38] + 2 * z * a[39] ) ) ) +
              2 * c2th * a[40] + 2 * c3th * a[44] +
              z * ( 2 * a[33] + 2 * c2th * a[41] + 2 * c3th * a[45] +
                    z * ( 2 * a[34] + 2 * c2th * a[42] + 2 * c3th * a[46] +
                          z * ( 2 * a[35] + 2 * c2th * a[43] + 2 * c3th * a[47] ) ) ) +
              r * ( 3 * a[48] +
                    cth * ( 3 * a[52] + z * ( 3 * a[53] + z * ( 3 * a[54] + 3 * z * a[55] ) ) ) +
                    3 * c2th * a[56] + 3 * c3th * a[60] +
                    z * ( 3 * a[49] + 3 * c2th * a[57] + 3 * c3th * a[61] +
                          z * ( 3 * a[50] + 3 * c2th * a[58] + 3 * c3th * a[62] +
                                z * ( 3 * a[51] + 3 * c2th * a[59] + 3 * c3th * a[63] ) ) ) ) );
    result[2] =
        sth * ( -a[4] + z * ( -a[5] + z * ( -a[6] - z * a[7] ) ) ) - 2 * s2th * a[8] -
        3 * s3th * a[12] +
        z * ( -2 * s2th * a[9] - 3 * s3th * a[13] +
              z * ( -2 * s2th * a[10] - 3 * s3th * a[14] +
                    z * ( -2 * s2th * a[11] - 3 * s3th * a[15] ) ) ) +
        r * ( sth * ( -a[20] + z * ( -a[21] + z * ( -a[22] - z * a[23] ) ) ) - 2 * s2th * a[24] -
              3 * s3th * a[28] +
              z * ( -2 * s2th * a[25] - 3 * s3th * a[29] +
                    z * ( -2 * s2th * a[26] - 3 * s3th * a[30] +
                          z * ( -2 * s2th * a[27] - 3 * s3th * a[31] ) ) ) +
              r * ( sth * ( -a[36] + z * ( -a[37] + z * ( -a[38] - z * a[39] ) ) ) -
                    2 * s2th * a[40] - 3 * s3th * a[44] +
                    z * ( -2 * s2th * a[41] - 3 * s3th * a[45] +
                          z * ( -2 * s2th * a[42] - 3 * s3th * a[46] +
                                z * ( -2 * s2th * a[43] - 3 * s3th * a[47] ) ) ) +
                    r * ( sth * ( -a[52] + z * ( -a[53] + z * ( -a[54] - z * a[55] ) ) ) -
                          2 * s2th * a[56] - 3 * s3th * a[60] +
                          z * ( -2 * s2th * a[57] - 3 * s3th * a[61] +
                                z * ( -2 * s2th * a[58] - 3 * s3th * a[62] +
                                      z * ( -2 * s2th * a[59] - 3 * s3th * a[63] ) ) ) ) ) );
    result[3] =
        a[1] + cth * ( a[5] + z * ( 2 * a[6] + 3 * z * a[7] ) ) + c2th * a[9] + c3th * a[13] +
        z * ( 2 * a[2] + 2 * c2th * a[10] + 2 * c3th * a[14] +
              z * ( 3 * a[3] + 3 * c2th * a[11] + 3 * c3th * a[15] ) ) +
        r * ( a[17] + cth * ( a[21] + z * ( 2 * a[22] + 3 * z * a[23] ) ) + c2th * a[25] +
              c3th * a[29] +
              z * ( 2 * a[18] + 2 * c2th * a[26] + 2 * c3th * a[30] +
                    z * ( 3 * a[19] + 3 * c2th * a[27] + 3 * c3th * a[31] ) ) +
              r * ( a[33] + cth * ( a[37] + z * ( 2 * a[38] + 3 * z * a[39] ) ) + c2th * a[41] +
                    c3th * a[45] +
                    z * ( 2 * a[34] + 2 * c2th * a[42] + 2 * c3th * a[46] +
                          z * ( 3 * a[35] + 3 * c2th * a[43] + 3 * c3th * a[47] ) ) +
                    r * ( a[49] + cth * ( a[53] + z * ( 2 * a[54] + 3 * z * a[55] ) ) +
                          c2th * a[57] + c3th * a[61] +
                          z * ( 2 * a[50] + 2 * c2th * a[58] + 2 * c3th * a[62] +
                                z * ( 3 * a[51] + 3 * c2th * a[59] + 3 * c3th * a[63] ) ) ) ) );
    result[4] = 2 * a[32] +
                cth * ( 2 * a[36] + z * ( 2 * a[37] + z * ( 2 * a[38] + 2 * z * a[39] ) ) ) +
                2 * c2th * a[40] + 2 * c3th * a[44] +
                z * ( 2 * a[33] + 2 * c2th * a[41] + 2 * c3th * a[45] +
                      z * ( 2 * a[34] + 2 * c2th * a[42] + 2 * c3th * a[46] +
                            z * ( 2 * a[35] + 2 * c2th * a[43] + 2 * c3th * a[47] ) ) ) +
                r * ( 6 * a[48] +
                      cth * ( 6 * a[52] + z * ( 6 * a[53] + z * ( 6 * a[54] + 6 * z * a[55] ) ) ) +
                      6 * c2th * a[56] + 6 * c3th * a[60] +
                      z * ( 6 * a[49] + 6 * c2th * a[57] + 6 * c3th * a[61] +
                            z * ( 6 * a[50] + 6 * c2th * a[58] + 6 * c3th * a[62] +
                                  z * ( 6 * a[51] + 6 * c2th * a[59] + 6 * c3th * a[63] ) ) ) );
    result[5] =
        sth * ( -a[20] + z * ( -a[21] + z * ( -a[22] - z * a[23] ) ) ) - 2 * s2th * a[24] -
        3 * s3th * a[28] +
        z * ( -2 * s2th * a[25] - 3 * s3th * a[29] +
              z * ( -2 * s2th * a[26] - 3 * s3th * a[30] +
                    z * ( -2 * s2th * a[27] - 3 * s3th * a[31] ) ) ) +
        r * ( sth * ( -2 * a[36] + z * ( -2 * a[37] + z * ( -2 * a[38] - 2 * z * a[39] ) ) ) -
              4 * s2th * a[40] - 6 * s3th * a[44] +
              z * ( -4 * s2th * a[41] - 6 * s3th * a[45] +
                    z * ( -4 * s2th * a[42] - 6 * s3th * a[46] +
                          z * ( -4 * s2th * a[43] - 6 * s3th * a[47] ) ) ) +
              r * ( sth * ( -3 * a[52] + z * ( -3 * a[53] + z * ( -3 * a[54] - 3 * z * a[55] ) ) ) -
                    6 * s2th * a[56] - 9 * s3th * a[60] +
                    z * ( -6 * s2th * a[57] - 9 * s3th * a[61] +
                          z * ( -6 * s2th * a[58] - 9 * s3th * a[62] +
                                z * ( -6 * s2th * a[59] - 9 * s3th * a[63] ) ) ) ) );
    result[6] = a[17] + cth * ( a[21] + z * ( 2 * a[22] + 3 * z * a[23] ) ) + c2th * a[25] +
                c3th * a[29] +
                z * ( 2 * a[18] + 2 * c2th * a[26] + 2 * c3th * a[30] +
                      z * ( 3 * a[19] + 3 * c2th * a[27] + 3 * c3th * a[31] ) ) +
                r * ( 2 * a[33] + cth * ( 2 * a[37] + z * ( 4 * a[38] + 6 * z * a[39] ) ) +
                      2 * c2th * a[41] + 2 * c3th * a[45] +
                      z * ( 4 * a[34] + 4 * c2th * a[42] + 4 * c3th * a[46] +
                            z * ( 6 * a[35] + 6 * c2th * a[43] + 6 * c3th * a[47] ) ) +
                      r * ( 3 * a[49] + cth * ( 3 * a[53] + z * ( 6 * a[54] + 9 * z * a[55] ) ) +
                            3 * c2th * a[57] + 3 * c3th * a[61] +
                            z * ( 6 * a[50] + 6 * c2th * a[58] + 6 * c3th * a[62] +
                                  z * ( 9 * a[51] + 9 * c2th * a[59] + 9 * c3th * a[63] ) ) ) );
    result[7] =
        cth * ( -a[4] + z * ( -a[5] + z * ( -a[6] - z * a[7] ) ) ) - 4 * c2th * a[8] -
        9 * c3th * a[12] +
        z * ( -4 * c2th * a[9] - 9 * c3th * a[13] +
              z * ( -4 * c2th * a[10] - 9 * c3th * a[14] +
                    z * ( -4 * c2th * a[11] - 9 * c3th * a[15] ) ) ) +
        r * ( cth * ( -a[20] + z * ( -a[21] + z * ( -a[22] - z * a[23] ) ) ) - 4 * c2th * a[24] -
              9 * c3th * a[28] +
              z * ( -4 * c2th * a[25] - 9 * c3th * a[29] +
                    z * ( -4 * c2th * a[26] - 9 * c3th * a[30] +
                          z * ( -4 * c2th * a[27] - 9 * c3th * a[31] ) ) ) +
              r * ( cth * ( -a[36] + z * ( -a[37] + z * ( -a[38] - z * a[39] ) ) ) -
                    4 * c2th * a[40] - 9 * c3th * a[44] +
                    z * ( -4 * c2th * a[41] - 9 * c3th * a[45] +
                          z * ( -4 * c2th * a[42] - 9 * c3th * a[46] +
                                z * ( -4 * c2th * a[43] - 9 * c3th * a[47] ) ) ) +
                    r * ( cth * ( -a[52] + z * ( -a[53] + z * ( -a[54] - z * a[55] ) ) ) -
                          4 * c2th * a[56] - 9 * c3th * a[60] +
                          z * ( -4 * c2th * a[57] - 9 * c3th * a[61] +
                                z * ( -4 * c2th * a[58] - 9 * c3th * a[62] +
                                      z * ( -4 * c2th * a[59] - 9 * c3th * a[63] ) ) ) ) ) );
    result[8] = sth * ( -a[5] + z * ( -2 * a[6] - 3 * z * a[7] ) ) - 2 * s2th * a[9] -
                3 * s3th * a[13] +
                z * ( -4 * s2th * a[10] - 6 * s3th * a[14] +
                      z * ( -6 * s2th * a[11] - 9 * s3th * a[15] ) ) +
                r * ( sth * ( -a[21] + z * ( -2 * a[22] - 3 * z * a[23] ) ) - 2 * s2th * a[25] -
                      3 * s3th * a[29] +
                      z * ( -4 * s2th * a[26] - 6 * s3th * a[30] +
                            z * ( -6 * s2th * a[27] - 9 * s3th * a[31] ) ) +
                      r * ( sth * ( -a[37] + z * ( -2 * a[38] - 3 * z * a[39] ) ) -
                            2 * s2th * a[41] - 3 * s3th * a[45] +
                            z * ( -4 * s2th * a[42] - 6 * s3th * a[46] +
                                  z * ( -6 * s2th * a[43] - 9 * s3th * a[47] ) ) +
                            r * ( sth * ( -a[53] + z * ( -2 * a[54] - 3 * z * a[55] ) ) -
                                  2 * s2th * a[57] - 3 * s3th * a[61] +
                                  z * ( -4 * s2th * a[58] - 6 * s3th * a[62] +
                                        z * ( -6 * s2th * a[59] - 9 * s3th * a[63] ) ) ) ) );
    result[9] =
        2 * a[2] + cth * ( 2 * a[6] + 6 * z * a[7] ) + 2 * c2th * a[10] + 2 * c3th * a[14] +
        z * ( 6 * a[3] + 6 * c2th * a[11] + 6 * c3th * a[15] ) +
        r * ( 2 * a[18] + cth * ( 2 * a[22] + 6 * z * a[23] ) + 2 * c2th * a[26] +
              2 * c3th * a[30] + z * ( 6 * a[19] + 6 * c2th * a[27] + 6 * c3th * a[31] ) +
              r * ( 2 * a[34] + cth * ( 2 * a[38] + 6 * z * a[39] ) + 2 * c2th * a[42] +
                    2 * c3th * a[46] + z * ( 6 * a[35] + 6 * c2th * a[43] + 6 * c3th * a[47] ) +
                    r * ( 2 * a[50] + cth * ( 2 * a[54] + 6 * z * a[55] ) + 2 * c2th * a[58] +
                          2 * c3th * a[62] +
                          z * ( 6 * a[51] + 6 * c2th * a[59] + 6 * c3th * a[63] ) ) ) );
    return result;
}


static std::array<double, 10>
quad_cyl_shell_neumann( double r, double th, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 4, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 9, "incorrect number of parameters specified" );

    double sth = sin( th ), cth = cos( th );
    std::array<double, 10> result;
    result[0] = a[0] + cth * ( a[1] + cth * a[2] ) +
                sth * ( a[3] + cth * ( a[4] + cth * a[5] ) +
                        sth * ( a[6] + cth * ( a[7] + cth * a[8] ) ) ) +
                r * ( -2 * c[0] - c[1] + r * ( c[0] + c[1] ) ) +
                z * ( -c[2] + z * ( c[2] / 2 + c[3] / 2 ) );
    result[1] = -2 * c[0] - c[1] + r * ( 2 * c[0] + 2 * c[1] );
    result[2] =
        cth * ( a[3] + cth * ( a[4] + cth * a[5] ) ) +
        sth * ( -a[1] + sth * ( -a[4] - 2 * cth * a[5] + sth * ( -a[7] - 2 * cth * a[8] ) ) +
                cth * ( -2 * a[2] + 2 * a[6] + cth * ( 2 * a[7] + 2 * cth * a[8] ) ) );
    result[3] = -c[2] + z * ( c[2] + c[3] );
    result[4] = 2 * c[0] + 2 * c[1];
    result[5] = 0;
    result[6] = 0;
    result[7] =
        cth * ( -a[1] + cth * ( -2 * a[2] + 2 * a[6] + cth * ( 2 * a[7] + 2 * cth * a[8] ) ) ) +
        sth * ( -a[3] + cth * ( -4 * a[4] - 7 * cth * a[5] ) +
                sth * ( 2 * a[2] - 2 * a[6] + cth * ( -7 * a[7] - 12 * cth * a[8] ) +
                        sth * ( 2 * a[5] + 2 * sth * a[8] ) ) );
    result[8] = 0;
    result[9] = c[2] + c[3];
    return result;
}


static std::array<double, 10>
quad_cyl_qtr_shell_neumann( double r, double th, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 4, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 7, "incorrect number of parameters specified" );

    double sth = sin( th ), cth = cos( th );
    std::array<double, 10> result;
    result[0] = a[0] + cth * ( a[1] + cth * a[2] ) +
                sth * ( a[3] + cth * ( cth * ( -a[3] - a[4] ) + a[4] ) +
                        sth * ( a[5] + cth * ( -a[1] - a[4] + cth * a[6] ) ) ) +
                r * ( -2 * c[0] - c[1] + r * ( c[0] + c[1] ) ) +
                z * ( -c[2] + z * ( c[2] / 2 + c[3] / 2 ) );
    result[1] = -2 * c[0] - c[1] + r * ( 2 * c[0] + 2 * c[1] );
    result[2] =
        cth * ( a[3] + cth * ( cth * ( -a[3] - a[4] ) + a[4] ) ) +
        sth * ( -a[1] +
                sth * ( -a[4] + cth * ( 2 * a[3] + 2 * a[4] ) +
                        sth * ( a[1] + a[4] - 2 * cth * a[6] ) ) +
                cth * ( -2 * a[2] + 2 * a[5] + cth * ( -2 * a[1] - 2 * a[4] + 2 * cth * a[6] ) ) );
    result[3] = -c[2] + z * ( c[2] + c[3] );
    result[4] = 2 * c[0] + 2 * c[1];
    result[5] = 0;
    result[6] = 0;
    result[7] =
        cth * ( -a[1] +
                cth * ( -2 * a[2] + 2 * a[5] + cth * ( -2 * a[1] - 2 * a[4] + 2 * cth * a[6] ) ) ) +
        sth * ( -a[3] + cth * ( -4 * a[4] + cth * ( 7 * a[3] + 7 * a[4] ) ) +
                sth * ( 2 * a[2] - 2 * a[5] + cth * ( 7 * a[1] + 7 * a[4] - 12 * cth * a[6] ) +
                        sth * ( -2 * a[3] - 2 * a[4] + 2 * sth * a[6] ) ) );
    result[8] = 0;
    result[9] = c[2] + c[3];
    return result;
}


static std::array<double, 10>
quad_cyl_qtr_shell_dirichlet2( double r, double th, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 4, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 7, "incorrect number of parameters specified" );

    double sth = sin( th ), cth = cos( th );
    std::array<double, 10> result;
    result[0] = a[0] + cth * ( a[1] + cth * a[2] ) +
                sth * ( a[3] + cth * ( cth * ( -a[3] - a[4] ) + a[4] ) +
                        sth * ( a[5] + cth * ( -a[1] - a[4] + cth * a[6] ) ) ) +
                r * ( -2 * c[0] - c[1] + r * ( c[0] + c[1] ) ) +
                z * ( -c[2] + z * ( c[2] / 2 + c[3] / 2 ) );
    result[1] = -2 * c[0] - c[1] + r * ( 2 * c[0] + 2 * c[1] );
    result[2] =
        cth * ( a[3] + cth * ( cth * ( -a[3] - a[4] ) + a[4] ) ) +
        sth * ( -a[1] +
                sth * ( -a[4] + cth * ( 2 * a[3] + 2 * a[4] ) +
                        sth * ( a[1] + a[4] - 2 * cth * a[6] ) ) +
                cth * ( -2 * a[2] + 2 * a[5] + cth * ( -2 * a[1] - 2 * a[4] + 2 * cth * a[6] ) ) );
    result[3] = -c[2] + z * ( c[2] + c[3] );
    result[4] = 2 * c[0] + 2 * c[1];
    result[5] = 0;
    result[6] = 0;
    result[7] =
        cth * ( -a[1] +
                cth * ( -2 * a[2] + 2 * a[5] + cth * ( -2 * a[1] - 2 * a[4] + 2 * cth * a[6] ) ) ) +
        sth * ( -a[3] + cth * ( -4 * a[4] + cth * ( 7 * a[3] + 7 * a[4] ) ) +
                sth * ( 2 * a[2] - 2 * a[5] + cth * ( 7 * a[1] + 7 * a[4] - 12 * cth * a[6] ) +
                        sth * ( -2 * a[3] - 2 * a[4] + 2 * sth * a[6] ) ) );
    result[8] = 0;
    result[9] = c[2] + c[3];
    return result;
}


static std::array<double, 10>
cubic_cyl_qtr_shell_neumann( double r, double th, double z, const ManufacturedSolution *mfs )
{
    const auto &c = mfs->getc();
    const auto &a = mfs->geta();
    AMP_INSIST( c.size() >= 4, "not enough boundary values specified" );
    AMP_INSIST( a.size() >= 56, "incorrect number of parameters specified" );

    double sth = sin( th ), cth = cos( th );
    std::array<double, 10> result;
    result[0] =
        a[0] +
        cth * ( a[2] + z * z * ( a[3] - ( 2 * z * a[3] ) / 3 ) +
                cth * ( a[4] + z * z * ( a[5] - ( 2 * z * a[5] ) / 3 ) +
                        cth * ( a[6] + z * z * ( a[7] - ( 2 * z * a[7] ) / 3 ) ) ) ) +
        sth *
            ( a[8] + z * z * ( a[9] - ( 2 * z * a[9] ) / 3 ) +
              cth * ( a[10] + z * z * ( a[11] - ( 2 * z * a[11] ) / 3 ) +
                      cth * ( a[12] +
                              cth * ( -a[8] - a[10] - a[12] +
                                      z * z *
                                          ( -a[9] - a[11] +
                                            z * ( ( 2 * a[9] ) / 3 + ( 2 * a[11] ) / 3 +
                                                  ( 2 * a[13] ) / 3 ) -
                                            a[13] ) ) +
                              z * z * ( a[13] - ( 2 * z * a[13] ) / 3 ) ) ) +
              sth *
                  ( a[14] + z * z * ( a[15] - ( 2 * z * a[15] ) / 3 ) +
                    cth *
                        ( a[16] + z * z * ( a[17] - ( 2 * z * a[17] ) / 3 ) +
                          cth * ( a[18] + z * z * ( a[19] - ( 2 * z * a[19] ) / 3 ) +
                                  cth * ( a[20] + z * z * ( a[21] - ( 2 * z * a[21] ) / 3 ) ) ) ) +
                    sth * ( a[22] + z * z * ( a[23] - ( 2 * z * a[23] ) / 3 ) +
                            cth * ( -a[2] - a[10] - a[16] +
                                    z * z *
                                        ( -a[3] - a[11] +
                                          z * ( ( 2 * a[3] ) / 3 + ( 2 * a[11] ) / 3 +
                                                ( 2 * a[17] ) / 3 ) -
                                          a[17] ) +
                                    cth * ( a[24] + z * z * ( a[25] - ( 2 * z * a[25] ) / 3 ) +
                                            cth * ( a[26] + z * z *
                                                                ( a[27] - ( 2 * z * a[27] ) /
                                                                              3 ) ) ) ) ) ) ) +
        r * ( a[28] + z * z * ( a[29] - ( 2 * z * a[29] ) / 3 ) +
              cth * ( a[30] + z * z * ( a[31] - ( 2 * z * a[31] ) / 3 ) +
                      cth * ( a[32] + z * z * ( a[33] - ( 2 * z * a[33] ) / 3 ) +
                              cth * ( a[34] + z * z * ( a[35] - ( 2 * z * a[35] ) / 3 ) ) ) ) +
              sth *
                  ( a[36] + z * z * ( a[37] - ( 2 * z * a[37] ) / 3 ) +
                    cth * ( a[38] + z * z * ( a[39] - ( 2 * z * a[39] ) / 3 ) +
                            cth * ( a[40] +
                                    cth * ( -a[36] - a[38] - a[40] +
                                            z * z *
                                                ( -a[37] - a[39] +
                                                  z * ( ( 2 * a[37] ) / 3 + ( 2 * a[39] ) / 3 +
                                                        ( 2 * a[41] ) / 3 ) -
                                                  a[41] ) ) +
                                    z * z * ( a[41] - ( 2 * z * a[41] ) / 3 ) ) ) +
                    sth *
                        ( a[42] + z * z * ( a[43] - ( 2 * z * a[43] ) / 3 ) +
                          cth * ( a[44] + z * z * ( a[45] - ( 2 * z * a[45] ) / 3 ) +
                                  cth * ( a[46] + z * z * ( a[47] - ( 2 * z * a[47] ) / 3 ) +
                                          cth * ( a[48] +
                                                  z * z * ( a[49] - ( 2 * z * a[49] ) / 3 ) ) ) ) +
                          sth *
                              ( a[50] + z * z * ( a[51] - ( 2 * z * a[51] ) / 3 ) +
                                cth * ( -a[30] - a[38] - a[44] +
                                        z * z *
                                            ( -a[31] - a[39] +
                                              z * ( ( 2 * a[31] ) / 3 + ( 2 * a[39] ) / 3 +
                                                    ( 2 * a[45] ) / 3 ) -
                                              a[45] ) +
                                        cth * ( a[52] + z * z * ( a[53] - ( 2 * z * a[53] ) / 3 ) +
                                                cth * ( a[54] + z * z *
                                                                    ( a[55] - ( 2 * z * a[55] ) /
                                                                                  3 ) ) ) ) ) ) ) +
              r * ( ( -3 * a[28] ) / 2 + z * z * ( ( -3 * a[29] ) / 2 + z * a[29] ) +
                    cth *
                        ( ( -3 * a[30] ) / 2 + z * z * ( ( -3 * a[31] ) / 2 + z * a[31] ) +
                          cth * ( ( -3 * a[32] ) / 2 + z * z * ( ( -3 * a[33] ) / 2 + z * a[33] ) +
                                  cth * ( ( -3 * a[34] ) / 2 +
                                          z * z * ( ( -3 * a[35] ) / 2 + z * a[35] ) ) ) ) +
                    sth *
                        ( ( -3 * a[36] ) / 2 + z * z * ( ( -3 * a[37] ) / 2 + z * a[37] ) +
                          cth * ( ( -3 * a[38] ) / 2 + z * z * ( ( -3 * a[39] ) / 2 + z * a[39] ) +
                                  cth * ( ( -3 * a[40] ) / 2 +
                                          z * z * ( ( -3 * a[41] ) / 2 + z * a[41] ) +
                                          cth * ( ( 3 * a[36] ) / 2 + ( 3 * a[38] ) / 2 +
                                                  ( 3 * a[40] ) / 2 +
                                                  z * z *
                                                      ( ( 3 * a[37] ) / 2 + ( 3 * a[39] ) / 2 +
                                                        z * ( -a[37] - a[39] - a[41] ) +
                                                        ( 3 * a[41] ) / 2 ) ) ) ) +
                          sth *
                              ( ( -3 * a[42] ) / 2 + z * z * ( ( -3 * a[43] ) / 2 + z * a[43] ) +
                                cth * ( ( -3 * a[44] ) / 2 +
                                        z * z * ( ( -3 * a[45] ) / 2 + z * a[45] ) +
                                        cth * ( ( -3 * a[46] ) / 2 +
                                                z * z * ( ( -3 * a[47] ) / 2 + z * a[47] ) +
                                                cth *
                                                    ( ( -3 * a[48] ) / 2 +
                                                      z * z *
                                                          ( ( -3 * a[49] ) / 2 + z * a[49] ) ) ) ) +
                                sth * ( ( -3 * a[50] ) / 2 +
                                        z * z * ( ( -3 * a[51] ) / 2 + z * a[51] ) +
                                        cth * ( ( 3 * a[30] ) / 2 + ( 3 * a[38] ) / 2 +
                                                ( 3 * a[44] ) / 2 +
                                                z * z *
                                                    ( ( 3 * a[31] ) / 2 + ( 3 * a[39] ) / 2 +
                                                      z * ( -a[31] - a[39] - a[45] ) +
                                                      ( 3 * a[45] ) / 2 ) +
                                                cth * ( ( -3 * a[52] ) / 2 +
                                                        z * z * ( ( -3 * a[53] ) / 2 + z * a[53] ) +
                                                        cth * ( ( -3 * a[54] ) / 2 +
                                                                z * z *
                                                                    ( ( -3 * a[55] ) / 2 +
                                                                      z * a[55] ) ) ) ) ) ) ) -
                    2 * c[0] +
                    r * ( ( 2 * a[28] ) / 3 +
                          z * z * ( ( 2 * a[29] ) / 3 - ( 4 * z * a[29] ) / 9 ) +
                          cth * ( ( 2 * a[30] ) / 3 +
                                  z * z * ( ( 2 * a[31] ) / 3 - ( 4 * z * a[31] ) / 9 ) +
                                  cth * ( ( 2 * a[32] ) / 3 +
                                          z * z * ( ( 2 * a[33] ) / 3 - ( 4 * z * a[33] ) / 9 ) +
                                          cth * ( ( 2 * a[34] ) / 3 +
                                                  z * z *
                                                      ( ( 2 * a[35] ) / 3 -
                                                        ( 4 * z * a[35] ) / 9 ) ) ) ) +
                          sth *
                              ( ( 2 * a[36] ) / 3 +
                                z * z * ( ( 2 * a[37] ) / 3 - ( 4 * z * a[37] ) / 9 ) +
                                cth * ( ( 2 * a[38] ) / 3 +
                                        z * z * ( ( 2 * a[39] ) / 3 - ( 4 * z * a[39] ) / 9 ) +
                                        cth *
                                            ( ( 2 * a[40] ) / 3 +
                                              cth * ( ( -2 * a[36] ) / 3 - ( 2 * a[38] ) / 3 -
                                                      ( 2 * a[40] ) / 3 +
                                                      z * z *
                                                          ( ( -2 * a[37] ) / 3 - ( 2 * a[39] ) / 3 +
                                                            z * ( ( 4 * a[37] ) / 9 +
                                                                  ( 4 * a[39] ) / 9 +
                                                                  ( 4 * a[41] ) / 9 ) -
                                                            ( 2 * a[41] ) / 3 ) ) +
                                              z * z *
                                                  ( ( 2 * a[41] ) / 3 -
                                                    ( 4 * z * a[41] ) / 9 ) ) ) +
                                sth *
                                    ( ( 2 * a[42] ) / 3 +
                                      z * z * ( ( 2 * a[43] ) / 3 - ( 4 * z * a[43] ) / 9 ) +
                                      cth * ( ( 2 * a[44] ) / 3 +
                                              z * z *
                                                  ( ( 2 * a[45] ) / 3 - ( 4 * z * a[45] ) / 9 ) +
                                              cth * ( ( 2 * a[46] ) / 3 +
                                                      z * z *
                                                          ( ( 2 * a[47] ) / 3 -
                                                            ( 4 * z * a[47] ) / 9 ) +
                                                      cth * ( ( 2 * a[48] ) / 3 +
                                                              z * z *
                                                                  ( ( 2 * a[49] ) / 3 -
                                                                    ( 4 * z * a[49] ) / 9 ) ) ) ) +
                                      sth *
                                          ( ( 2 * a[50] ) / 3 +
                                            z * z * ( ( 2 * a[51] ) / 3 - ( 4 * z * a[51] ) / 9 ) +
                                            cth * ( ( -2 * a[30] ) / 3 - ( 2 * a[38] ) / 3 -
                                                    ( 2 * a[44] ) / 3 +
                                                    z * z *
                                                        ( ( -2 * a[31] ) / 3 - ( 2 * a[39] ) / 3 +
                                                          z * ( ( 4 * a[31] ) / 9 +
                                                                ( 4 * a[39] ) / 9 +
                                                                ( 4 * a[45] ) / 9 ) -
                                                          ( 2 * a[45] ) / 3 ) +
                                                    cth * ( ( 2 * a[52] ) / 3 +
                                                            z * z *
                                                                ( ( 2 * a[53] ) / 3 -
                                                                  ( 4 * z * a[53] ) / 9 ) +
                                                            cth * ( ( 2 * a[54] ) / 3 +
                                                                    z * z *
                                                                        ( ( 2 * a[55] ) / 3 -
                                                                          ( 4 * z * a[55] ) /
                                                                              9 ) ) ) ) ) ) ) +
                          ( 4 * c[0] ) / 3 + ( 2 * c[1] ) / 3 ) -
                    c[1] / 2 ) ) +
        z * ( -c[2] + z * ( a[1] + z * ( ( -2 * a[1] ) / 3 + c[2] / 3 + c[3] / 3 ) ) );
    result[1] =
        a[28] + z * z * ( a[29] - ( 2 * z * a[29] ) / 3 ) +
        cth * ( a[30] + z * z * ( a[31] - ( 2 * z * a[31] ) / 3 ) +
                cth * ( a[32] + z * z * ( a[33] - ( 2 * z * a[33] ) / 3 ) +
                        cth * ( a[34] + z * z * ( a[35] - ( 2 * z * a[35] ) / 3 ) ) ) ) +
        sth *
            ( a[36] + z * z * ( a[37] - ( 2 * z * a[37] ) / 3 ) +
              cth * ( a[38] + z * z * ( a[39] - ( 2 * z * a[39] ) / 3 ) +
                      cth * ( a[40] +
                              cth * ( -a[36] - a[38] - a[40] +
                                      z * z *
                                          ( -a[37] - a[39] +
                                            z * ( ( 2 * a[37] ) / 3 + ( 2 * a[39] ) / 3 +
                                                  ( 2 * a[41] ) / 3 ) -
                                            a[41] ) ) +
                              z * z * ( a[41] - ( 2 * z * a[41] ) / 3 ) ) ) +
              sth *
                  ( a[42] + z * z * ( a[43] - ( 2 * z * a[43] ) / 3 ) +
                    cth *
                        ( a[44] + z * z * ( a[45] - ( 2 * z * a[45] ) / 3 ) +
                          cth * ( a[46] + z * z * ( a[47] - ( 2 * z * a[47] ) / 3 ) +
                                  cth * ( a[48] + z * z * ( a[49] - ( 2 * z * a[49] ) / 3 ) ) ) ) +
                    sth * ( a[50] + z * z * ( a[51] - ( 2 * z * a[51] ) / 3 ) +
                            cth * ( -a[30] - a[38] - a[44] +
                                    z * z *
                                        ( -a[31] - a[39] +
                                          z * ( ( 2 * a[31] ) / 3 + ( 2 * a[39] ) / 3 +
                                                ( 2 * a[45] ) / 3 ) -
                                          a[45] ) +
                                    cth * ( a[52] + z * z * ( a[53] - ( 2 * z * a[53] ) / 3 ) +
                                            cth * ( a[54] + z * z *
                                                                ( a[55] - ( 2 * z * a[55] ) /
                                                                              3 ) ) ) ) ) ) ) +
        r * ( -3 * a[28] + z * z * ( -3 * a[29] + 2 * z * a[29] ) +
              cth * ( -3 * a[30] + z * z * ( -3 * a[31] + 2 * z * a[31] ) +
                      cth * ( -3 * a[32] + z * z * ( -3 * a[33] + 2 * z * a[33] ) +
                              cth * ( -3 * a[34] + z * z * ( -3 * a[35] + 2 * z * a[35] ) ) ) ) +
              sth *
                  ( -3 * a[36] + z * z * ( -3 * a[37] + 2 * z * a[37] ) +
                    cth * ( -3 * a[38] + z * z * ( -3 * a[39] + 2 * z * a[39] ) +
                            cth * ( -3 * a[40] + z * z * ( -3 * a[41] + 2 * z * a[41] ) +
                                    cth * ( 3 * a[36] + 3 * a[38] + 3 * a[40] +
                                            z * z *
                                                ( 3 * a[37] + 3 * a[39] +
                                                  z * ( -2 * a[37] - 2 * a[39] - 2 * a[41] ) +
                                                  3 * a[41] ) ) ) ) +
                    sth * ( -3 * a[42] + z * z * ( -3 * a[43] + 2 * z * a[43] ) +
                            cth * ( -3 * a[44] + z * z * ( -3 * a[45] + 2 * z * a[45] ) +
                                    cth * ( -3 * a[46] + z * z * ( -3 * a[47] + 2 * z * a[47] ) +
                                            cth * ( -3 * a[48] +
                                                    z * z * ( -3 * a[49] + 2 * z * a[49] ) ) ) ) +
                            sth * ( -3 * a[50] + z * z * ( -3 * a[51] + 2 * z * a[51] ) +
                                    cth * ( 3 * a[30] + 3 * a[38] + 3 * a[44] +
                                            z * z *
                                                ( 3 * a[31] + 3 * a[39] +
                                                  z * ( -2 * a[31] - 2 * a[39] - 2 * a[45] ) +
                                                  3 * a[45] ) +
                                            cth * ( -3 * a[52] +
                                                    z * z * ( -3 * a[53] + 2 * z * a[53] ) +
                                                    cth * ( -3 * a[54] +
                                                            z * z *
                                                                ( -3 * a[55] +
                                                                  2 * z * a[55] ) ) ) ) ) ) ) -
              4 * c[0] - c[1] +
              r * ( 2 * a[28] + z * z * ( 2 * a[29] - ( 4 * z * a[29] ) / 3 ) +
                    cth * ( 2 * a[30] + z * z * ( 2 * a[31] - ( 4 * z * a[31] ) / 3 ) +
                            cth * ( 2 * a[32] + z * z * ( 2 * a[33] - ( 4 * z * a[33] ) / 3 ) +
                                    cth * ( 2 * a[34] +
                                            z * z * ( 2 * a[35] - ( 4 * z * a[35] ) / 3 ) ) ) ) +
                    sth *
                        ( 2 * a[36] + z * z * ( 2 * a[37] - ( 4 * z * a[37] ) / 3 ) +
                          cth *
                              ( 2 * a[38] + z * z * ( 2 * a[39] - ( 4 * z * a[39] ) / 3 ) +
                                cth * ( 2 * a[40] + z * z * ( 2 * a[41] - ( 4 * z * a[41] ) / 3 ) +
                                        cth * ( -2 * a[36] - 2 * a[38] - 2 * a[40] +
                                                z * z *
                                                    ( -2 * a[37] - 2 * a[39] - 2 * a[41] +
                                                      z * ( ( 4 * a[37] ) / 3 + ( 4 * a[39] ) / 3 +
                                                            ( 4 * a[41] ) / 3 ) ) ) ) ) +
                          sth *
                              ( 2 * a[42] + z * z * ( 2 * a[43] - ( 4 * z * a[43] ) / 3 ) +
                                cth * ( 2 * a[44] + z * z * ( 2 * a[45] - ( 4 * z * a[45] ) / 3 ) +
                                        cth * ( 2 * a[46] +
                                                z * z * ( 2 * a[47] - ( 4 * z * a[47] ) / 3 ) +
                                                cth * ( 2 * a[48] +
                                                        z * z *
                                                            ( 2 * a[49] -
                                                              ( 4 * z * a[49] ) / 3 ) ) ) ) +
                                sth * ( 2 * a[50] + z * z * ( 2 * a[51] - ( 4 * z * a[51] ) / 3 ) +
                                        cth * ( -2 * a[30] - 2 * a[38] - 2 * a[44] +
                                                z * z *
                                                    ( -2 * a[31] - 2 * a[39] - 2 * a[45] +
                                                      z * ( ( 4 * a[31] ) / 3 + ( 4 * a[39] ) / 3 +
                                                            ( 4 * a[45] ) / 3 ) ) +
                                                cth * ( 2 * a[52] +
                                                        z * z *
                                                            ( 2 * a[53] - ( 4 * z * a[53] ) / 3 ) +
                                                        cth * ( 2 * a[54] +
                                                                z * z *
                                                                    ( 2 * a[55] -
                                                                      ( 4 * z * a[55] ) /
                                                                          3 ) ) ) ) ) ) ) +
                    4 * c[0] + 2 * c[1] ) );
    result[2] =
        cth * ( a[8] + z * z * ( a[9] - ( 2 * z * a[9] ) / 3 ) +
                cth * ( a[10] + z * z * ( a[11] - ( 2 * z * a[11] ) / 3 ) +
                        cth * ( a[12] +
                                cth * ( -a[8] - a[10] - a[12] +
                                        z * z *
                                            ( -a[9] - a[11] +
                                              z * ( ( 2 * a[9] ) / 3 + ( 2 * a[11] ) / 3 +
                                                    ( 2 * a[13] ) / 3 ) -
                                              a[13] ) ) +
                                z * z * ( a[13] - ( 2 * z * a[13] ) / 3 ) ) ) ) +
        sth *
            ( -a[2] + z * z * ( -a[3] + ( 2 * z * a[3] ) / 3 ) +
              cth *
                  ( -2 * a[4] + 2 * a[14] +
                    z * z *
                        ( -2 * a[5] + z * ( ( 4 * a[5] ) / 3 - ( 4 * a[15] ) / 3 ) + 2 * a[15] ) +
                    cth *
                        ( -3 * a[6] + 2 * a[16] +
                          z * z * ( -3 * a[7] + z * ( 2 * a[7] - ( 4 * a[17] ) / 3 ) + 2 * a[17] ) +
                          cth * ( 2 * a[18] + z * z * ( 2 * a[19] - ( 4 * z * a[19] ) / 3 ) +
                                  cth * ( 2 * a[20] +
                                          z * z * ( 2 * a[21] - ( 4 * z * a[21] ) / 3 ) ) ) ) ) +
              sth *
                  ( -a[10] + z * z * ( -a[11] + ( 2 * z * a[11] ) / 3 ) +
                    cth * ( -2 * a[12] + 3 * a[22] +
                            z * z *
                                ( -2 * a[13] + z * ( ( 4 * a[13] ) / 3 - 2 * a[23] ) + 3 * a[23] ) +
                            cth * ( -3 * a[2] + 3 * a[8] + 3 * a[12] - 3 * a[16] +
                                    z * z *
                                        ( -3 * a[3] + 3 * a[9] + 3 * a[13] - 3 * a[17] +
                                          z * ( 2 * a[3] - 2 * a[9] - 2 * a[13] + 2 * a[17] ) ) +
                                    cth * ( 3 * a[24] + z * z * ( 3 * a[25] - 2 * z * a[25] ) +
                                            cth * ( 3 * a[26] +
                                                    z * z * ( 3 * a[27] - 2 * z * a[27] ) ) ) ) ) +
                    sth *
                        ( -a[16] + z * z * ( -a[17] + ( 2 * z * a[17] ) / 3 ) +
                          cth * ( -2 * a[18] + z * z * ( -2 * a[19] + ( 4 * z * a[19] ) / 3 ) +
                                  cth * ( -3 * a[20] + z * z * ( -3 * a[21] + 2 * z * a[21] ) ) ) +
                          sth *
                              ( a[2] + a[10] + a[16] +
                                z * z *
                                    ( a[3] + a[11] +
                                      z * ( ( -2 * a[3] ) / 3 - ( 2 * a[11] ) / 3 -
                                            ( 2 * a[17] ) / 3 ) +
                                      a[17] ) +
                                cth *
                                    ( -2 * a[24] + z * z * ( -2 * a[25] + ( 4 * z * a[25] ) / 3 ) +
                                      cth * ( -3 * a[26] +
                                              z * z * ( -3 * a[27] + 2 * z * a[27] ) ) ) ) ) ) ) +
        r * ( cth * ( a[36] + z * z * ( a[37] - ( 2 * z * a[37] ) / 3 ) +
                      cth * ( a[38] + z * z * ( a[39] - ( 2 * z * a[39] ) / 3 ) +
                              cth * ( a[40] +
                                      cth * ( -a[36] - a[38] - a[40] +
                                              z * z *
                                                  ( -a[37] - a[39] +
                                                    z * ( ( 2 * a[37] ) / 3 + ( 2 * a[39] ) / 3 +
                                                          ( 2 * a[41] ) / 3 ) -
                                                    a[41] ) ) +
                                      z * z * ( a[41] - ( 2 * z * a[41] ) / 3 ) ) ) ) +
              sth *
                  ( -a[30] + z * z * ( -a[31] + ( 2 * z * a[31] ) / 3 ) +
                    cth *
                        ( -2 * a[32] + 2 * a[42] +
                          z * z *
                              ( -2 * a[33] + z * ( ( 4 * a[33] ) / 3 - ( 4 * a[43] ) / 3 ) +
                                2 * a[43] ) +
                          cth *
                              ( -3 * a[34] + 2 * a[44] +
                                z * z *
                                    ( -3 * a[35] + z * ( 2 * a[35] - ( 4 * a[45] ) / 3 ) +
                                      2 * a[45] ) +
                                cth * ( 2 * a[46] + z * z * ( 2 * a[47] - ( 4 * z * a[47] ) / 3 ) +
                                        cth * ( 2 * a[48] +
                                                z * z *
                                                    ( 2 * a[49] - ( 4 * z * a[49] ) / 3 ) ) ) ) ) +
                    sth *
                        ( -a[38] + z * z * ( -a[39] + ( 2 * z * a[39] ) / 3 ) +
                          cth *
                              ( -2 * a[40] + 3 * a[50] +
                                z * z *
                                    ( -2 * a[41] + z * ( ( 4 * a[41] ) / 3 - 2 * a[51] ) +
                                      3 * a[51] ) +
                                cth * ( -3 * a[30] + 3 * a[36] + 3 * a[40] - 3 * a[44] +
                                        z * z *
                                            ( -3 * a[31] + 3 * a[37] + 3 * a[41] - 3 * a[45] +
                                              z * ( 2 * a[31] - 2 * a[37] - 2 * a[41] +
                                                    2 * a[45] ) ) +
                                        cth * ( 3 * a[52] + z * z * ( 3 * a[53] - 2 * z * a[53] ) +
                                                cth * ( 3 * a[54] +
                                                        z * z *
                                                            ( 3 * a[55] - 2 * z * a[55] ) ) ) ) ) +
                          sth * ( -a[44] + z * z * ( -a[45] + ( 2 * z * a[45] ) / 3 ) +
                                  cth * ( -2 * a[46] +
                                          z * z * ( -2 * a[47] + ( 4 * z * a[47] ) / 3 ) +
                                          cth * ( -3 * a[48] +
                                                  z * z * ( -3 * a[49] + 2 * z * a[49] ) ) ) +
                                  sth * ( a[30] + a[38] + a[44] +
                                          z * z *
                                              ( a[31] + a[39] +
                                                z * ( ( -2 * a[31] ) / 3 - ( 2 * a[39] ) / 3 -
                                                      ( 2 * a[45] ) / 3 ) +
                                                a[45] ) +
                                          cth * ( -2 * a[52] +
                                                  z * z * ( -2 * a[53] + ( 4 * z * a[53] ) / 3 ) +
                                                  cth * ( -3 * a[54] +
                                                          z * z *
                                                              ( -3 * a[55] +
                                                                2 * z * a[55] ) ) ) ) ) ) ) +
              r * ( cth *
                        ( ( -3 * a[36] ) / 2 + z * z * ( ( -3 * a[37] ) / 2 + z * a[37] ) +
                          cth * ( ( -3 * a[38] ) / 2 + z * z * ( ( -3 * a[39] ) / 2 + z * a[39] ) +
                                  cth * ( ( -3 * a[40] ) / 2 +
                                          z * z * ( ( -3 * a[41] ) / 2 + z * a[41] ) +
                                          cth * ( ( 3 * a[36] ) / 2 + ( 3 * a[38] ) / 2 +
                                                  ( 3 * a[40] ) / 2 +
                                                  z * z *
                                                      ( ( 3 * a[37] ) / 2 + ( 3 * a[39] ) / 2 +
                                                        z * ( -a[37] - a[39] - a[41] ) +
                                                        ( 3 * a[41] ) / 2 ) ) ) ) ) +
                    sth *
                        ( ( 3 * a[30] ) / 2 + z * z * ( ( 3 * a[31] ) / 2 - z * a[31] ) +
                          cth *
                              ( 3 * a[32] - 3 * a[42] +
                                z * z *
                                    ( 3 * a[33] - 3 * a[43] + z * ( -2 * a[33] + 2 * a[43] ) ) +
                                cth *
                                    ( ( 9 * a[34] ) / 2 - 3 * a[44] +
                                      z * z *
                                          ( ( 9 * a[35] ) / 2 - 3 * a[45] +
                                            z * ( -3 * a[35] + 2 * a[45] ) ) +
                                      cth *
                                          ( -3 * a[46] + z * z * ( -3 * a[47] + 2 * z * a[47] ) +
                                            cth * ( -3 * a[48] +
                                                    z * z * ( -3 * a[49] + 2 * z * a[49] ) ) ) ) ) +
                          sth *
                              ( ( 3 * a[38] ) / 2 + z * z * ( ( 3 * a[39] ) / 2 - z * a[39] ) +
                                sth * ( ( 3 * a[44] ) / 2 +
                                        z * z * ( ( 3 * a[45] ) / 2 - z * a[45] ) +
                                        cth * ( 3 * a[46] + z * z * ( 3 * a[47] - 2 * z * a[47] ) +
                                                cth * ( ( 9 * a[48] ) / 2 +
                                                        z * z *
                                                            ( ( 9 * a[49] ) / 2 -
                                                              3 * z * a[49] ) ) ) +
                                        sth * ( ( -3 * a[30] ) / 2 - ( 3 * a[38] ) / 2 -
                                                ( 3 * a[44] ) / 2 +
                                                z * z *
                                                    ( ( -3 * a[31] ) / 2 - ( 3 * a[39] ) / 2 -
                                                      ( 3 * a[45] ) / 2 +
                                                      z * ( a[31] + a[39] + a[45] ) ) +
                                                cth * ( 3 * a[52] +
                                                        z * z * ( 3 * a[53] - 2 * z * a[53] ) +
                                                        cth * ( ( 9 * a[54] ) / 2 +
                                                                z * z *
                                                                    ( ( 9 * a[55] ) / 2 -
                                                                      3 * z * a[55] ) ) ) ) ) +
                                cth * ( 3 * a[40] - ( 9 * a[50] ) / 2 +
                                        z * z *
                                            ( 3 * a[41] - ( 9 * a[51] ) / 2 +
                                              z * ( -2 * a[41] + 3 * a[51] ) ) +
                                        cth * ( ( 9 * a[30] ) / 2 - ( 9 * a[36] ) / 2 -
                                                ( 9 * a[40] ) / 2 + ( 9 * a[44] ) / 2 +
                                                z * z *
                                                    ( ( 9 * a[31] ) / 2 - ( 9 * a[37] ) / 2 -
                                                      ( 9 * a[41] ) / 2 +
                                                      z * ( -3 * a[31] + 3 * a[37] + 3 * a[41] -
                                                            3 * a[45] ) +
                                                      ( 9 * a[45] ) / 2 ) +
                                                cth * ( ( -9 * a[52] ) / 2 +
                                                        z * z *
                                                            ( ( -9 * a[53] ) / 2 + 3 * z * a[53] ) +
                                                        cth * ( ( -9 * a[54] ) / 2 +
                                                                z * z *
                                                                    ( ( -9 * a[55] ) / 2 +
                                                                      3 * z * a[55] ) ) ) ) ) ) ) +
                    r * ( cth *
                              ( ( 2 * a[36] ) / 3 +
                                z * z * ( ( 2 * a[37] ) / 3 - ( 4 * z * a[37] ) / 9 ) +
                                cth * ( ( 2 * a[38] ) / 3 +
                                        z * z * ( ( 2 * a[39] ) / 3 - ( 4 * z * a[39] ) / 9 ) +
                                        cth *
                                            ( ( 2 * a[40] ) / 3 +
                                              cth * ( ( -2 * a[36] ) / 3 - ( 2 * a[38] ) / 3 -
                                                      ( 2 * a[40] ) / 3 +
                                                      z * z *
                                                          ( ( -2 * a[37] ) / 3 - ( 2 * a[39] ) / 3 +
                                                            z * ( ( 4 * a[37] ) / 9 +
                                                                  ( 4 * a[39] ) / 9 +
                                                                  ( 4 * a[41] ) / 9 ) -
                                                            ( 2 * a[41] ) / 3 ) ) +
                                              z * z *
                                                  ( ( 2 * a[41] ) / 3 -
                                                    ( 4 * z * a[41] ) / 9 ) ) ) ) +
                          sth *
                              ( ( -2 * a[30] ) / 3 +
                                z * z * ( ( -2 * a[31] ) / 3 + ( 4 * z * a[31] ) / 9 ) +
                                cth *
                                    ( ( -4 * a[32] ) / 3 + ( 4 * a[42] ) / 3 +
                                      z * z *
                                          ( ( -4 * a[33] ) / 3 +
                                            z * ( ( 8 * a[33] ) / 9 - ( 8 * a[43] ) / 9 ) +
                                            ( 4 * a[43] ) / 3 ) +
                                      cth *
                                          ( -2 * a[34] + ( 4 * a[44] ) / 3 +
                                            z * z *
                                                ( -2 * a[35] +
                                                  z * ( ( 4 * a[35] ) / 3 - ( 8 * a[45] ) / 9 ) +
                                                  ( 4 * a[45] ) / 3 ) +
                                            cth * ( ( 4 * a[46] ) / 3 +
                                                    z * z *
                                                        ( ( 4 * a[47] ) / 3 -
                                                          ( 8 * z * a[47] ) / 9 ) +
                                                    cth * ( ( 4 * a[48] ) / 3 +
                                                            z * z *
                                                                ( ( 4 * a[49] ) / 3 -
                                                                  ( 8 * z * a[49] ) / 9 ) ) ) ) ) +
                                sth *
                                    ( ( -2 * a[38] ) / 3 +
                                      z * z * ( ( -2 * a[39] ) / 3 + ( 4 * z * a[39] ) / 9 ) +
                                      cth *
                                          ( ( -4 * a[40] ) / 3 + 2 * a[50] +
                                            z * z *
                                                ( ( -4 * a[41] ) / 3 +
                                                  z * ( ( 8 * a[41] ) / 9 - ( 4 * a[51] ) / 3 ) +
                                                  2 * a[51] ) +
                                            cth *
                                                ( -2 * a[30] + 2 * a[36] + 2 * a[40] - 2 * a[44] +
                                                  z * z *
                                                      ( -2 * a[31] + 2 * a[37] + 2 * a[41] -
                                                        2 * a[45] +
                                                        z * ( ( 4 * a[31] ) / 3 -
                                                              ( 4 * a[37] ) / 3 -
                                                              ( 4 * a[41] ) / 3 +
                                                              ( 4 * a[45] ) / 3 ) ) +
                                                  cth *
                                                      ( 2 * a[52] +
                                                        z * z *
                                                            ( 2 * a[53] - ( 4 * z * a[53] ) / 3 ) +
                                                        cth *
                                                            ( 2 * a[54] +
                                                              z * z *
                                                                  ( 2 * a[55] - ( 4 * z * a[55] ) /
                                                                                    3 ) ) ) ) ) +
                                      sth *
                                          ( ( -2 * a[44] ) / 3 +
                                            z * z * ( ( -2 * a[45] ) / 3 + ( 4 * z * a[45] ) / 9 ) +
                                            cth * ( ( -4 * a[46] ) / 3 +
                                                    z * z *
                                                        ( ( -4 * a[47] ) / 3 +
                                                          ( 8 * z * a[47] ) / 9 ) +
                                                    cth * ( -2 * a[48] +
                                                            z * z *
                                                                ( -2 * a[49] +
                                                                  ( 4 * z * a[49] ) / 3 ) ) ) +
                                            sth * ( ( 2 * a[30] ) / 3 + ( 2 * a[38] ) / 3 +
                                                    ( 2 * a[44] ) / 3 +
                                                    z * z *
                                                        ( ( 2 * a[31] ) / 3 + ( 2 * a[39] ) / 3 +
                                                          z * ( ( -4 * a[31] ) / 9 -
                                                                ( 4 * a[39] ) / 9 -
                                                                ( 4 * a[45] ) / 9 ) +
                                                          ( 2 * a[45] ) / 3 ) +
                                                    cth * ( ( -4 * a[52] ) / 3 +
                                                            z * z *
                                                                ( ( -4 * a[53] ) / 3 +
                                                                  ( 8 * z * a[53] ) / 9 ) +
                                                            cth * ( -2 * a[54] +
                                                                    z * z *
                                                                        ( -2 * a[55] +
                                                                          ( 4 * z * a[55] ) /
                                                                              3 ) ) ) ) ) ) ) ) ) );
    result[3] =
        cth *
            ( z * ( 2 * a[3] - 2 * z * a[3] ) +
              cth * ( z * ( 2 * a[5] - 2 * z * a[5] ) + cth * z * ( 2 * a[7] - 2 * z * a[7] ) ) ) +
        sth * ( z * ( 2 * a[9] - 2 * z * a[9] ) +
                cth * ( z * ( 2 * a[11] - 2 * z * a[11] ) +
                        cth * ( z * ( 2 * a[13] - 2 * z * a[13] ) +
                                cth * z *
                                    ( -2 * a[9] - 2 * a[11] - 2 * a[13] +
                                      z * ( 2 * a[9] + 2 * a[11] + 2 * a[13] ) ) ) ) +
                sth * ( z * ( 2 * a[15] - 2 * z * a[15] ) +
                        cth * ( z * ( 2 * a[17] - 2 * z * a[17] ) +
                                cth * ( z * ( 2 * a[19] - 2 * z * a[19] ) +
                                        cth * z * ( 2 * a[21] - 2 * z * a[21] ) ) ) +
                        sth * ( z * ( 2 * a[23] - 2 * z * a[23] ) +
                                cth * ( z * ( -2 * a[3] - 2 * a[11] - 2 * a[17] +
                                              z * ( 2 * a[3] + 2 * a[11] + 2 * a[17] ) ) +
                                        cth * ( z * ( 2 * a[25] - 2 * z * a[25] ) +
                                                cth * z * ( 2 * a[27] - 2 * z * a[27] ) ) ) ) ) ) +
        r * ( z * ( 2 * a[29] - 2 * z * a[29] ) +
              cth * ( z * ( 2 * a[31] - 2 * z * a[31] ) +
                      cth * ( z * ( 2 * a[33] - 2 * z * a[33] ) +
                              cth * z * ( 2 * a[35] - 2 * z * a[35] ) ) ) +
              sth * ( z * ( 2 * a[37] - 2 * z * a[37] ) +
                      cth * ( z * ( 2 * a[39] - 2 * z * a[39] ) +
                              cth * ( z * ( 2 * a[41] - 2 * z * a[41] ) +
                                      cth * z *
                                          ( -2 * a[37] - 2 * a[39] - 2 * a[41] +
                                            z * ( 2 * a[37] + 2 * a[39] + 2 * a[41] ) ) ) ) +
                      sth * ( z * ( 2 * a[43] - 2 * z * a[43] ) +
                              cth * ( z * ( 2 * a[45] - 2 * z * a[45] ) +
                                      cth * ( z * ( 2 * a[47] - 2 * z * a[47] ) +
                                              cth * z * ( 2 * a[49] - 2 * z * a[49] ) ) ) +
                              sth * ( z * ( 2 * a[51] - 2 * z * a[51] ) +
                                      cth * ( z * ( -2 * a[31] - 2 * a[39] - 2 * a[45] +
                                                    z * ( 2 * a[31] + 2 * a[39] + 2 * a[45] ) ) +
                                              cth * ( z * ( 2 * a[53] - 2 * z * a[53] ) +
                                                      cth * z *
                                                          ( 2 * a[55] - 2 * z * a[55] ) ) ) ) ) ) +
              r * ( z * ( -3 * a[29] + 3 * z * a[29] ) +
                    cth * ( z * ( -3 * a[31] + 3 * z * a[31] ) +
                            cth * ( z * ( -3 * a[33] + 3 * z * a[33] ) +
                                    cth * z * ( -3 * a[35] + 3 * z * a[35] ) ) ) +
                    sth *
                        ( z * ( -3 * a[37] + 3 * z * a[37] ) +
                          cth * ( z * ( -3 * a[39] + 3 * z * a[39] ) +
                                  cth * ( cth * z *
                                              ( 3 * a[37] + 3 * a[39] +
                                                z * ( -3 * a[37] - 3 * a[39] - 3 * a[41] ) +
                                                3 * a[41] ) +
                                          z * ( -3 * a[41] + 3 * z * a[41] ) ) ) +
                          sth * ( z * ( -3 * a[43] + 3 * z * a[43] ) +
                                  cth * ( z * ( -3 * a[45] + 3 * z * a[45] ) +
                                          cth * ( z * ( -3 * a[47] + 3 * z * a[47] ) +
                                                  cth * z * ( -3 * a[49] + 3 * z * a[49] ) ) ) +
                                  sth * ( z * ( -3 * a[51] + 3 * z * a[51] ) +
                                          cth * ( z * ( 3 * a[31] + 3 * a[39] +
                                                        z * ( -3 * a[31] - 3 * a[39] - 3 * a[45] ) +
                                                        3 * a[45] ) +
                                                  cth * ( z * ( -3 * a[53] + 3 * z * a[53] ) +
                                                          cth * z *
                                                              ( -3 * a[55] +
                                                                3 * z * a[55] ) ) ) ) ) ) +
                    r * ( z * ( ( 4 * a[29] ) / 3 - ( 4 * z * a[29] ) / 3 ) +
                          cth * ( z * ( ( 4 * a[31] ) / 3 - ( 4 * z * a[31] ) / 3 ) +
                                  cth * ( z * ( ( 4 * a[33] ) / 3 - ( 4 * z * a[33] ) / 3 ) +
                                          cth * z *
                                              ( ( 4 * a[35] ) / 3 - ( 4 * z * a[35] ) / 3 ) ) ) +
                          sth *
                              ( z * ( ( 4 * a[37] ) / 3 - ( 4 * z * a[37] ) / 3 ) +
                                cth * ( z * ( ( 4 * a[39] ) / 3 - ( 4 * z * a[39] ) / 3 ) +
                                        cth * ( z * ( ( 4 * a[41] ) / 3 - ( 4 * z * a[41] ) / 3 ) +
                                                cth * z *
                                                    ( ( -4 * a[37] ) / 3 - ( 4 * a[39] ) / 3 -
                                                      ( 4 * a[41] ) / 3 +
                                                      z * ( ( 4 * a[37] ) / 3 + ( 4 * a[39] ) / 3 +
                                                            ( 4 * a[41] ) / 3 ) ) ) ) +
                                sth *
                                    ( z * ( ( 4 * a[43] ) / 3 - ( 4 * z * a[43] ) / 3 ) +
                                      cth * ( z * ( ( 4 * a[45] ) / 3 - ( 4 * z * a[45] ) / 3 ) +
                                              cth * ( z * ( ( 4 * a[47] ) / 3 -
                                                            ( 4 * z * a[47] ) / 3 ) +
                                                      cth * z *
                                                          ( ( 4 * a[49] ) / 3 -
                                                            ( 4 * z * a[49] ) / 3 ) ) ) +
                                      sth * ( z * ( ( 4 * a[51] ) / 3 - ( 4 * z * a[51] ) / 3 ) +
                                              cth * ( z * ( ( -4 * a[31] ) / 3 - ( 4 * a[39] ) / 3 -
                                                            ( 4 * a[45] ) / 3 +
                                                            z * ( ( 4 * a[31] ) / 3 +
                                                                  ( 4 * a[39] ) / 3 +
                                                                  ( 4 * a[45] ) / 3 ) ) +
                                                      cth * ( z * ( ( 4 * a[53] ) / 3 -
                                                                    ( 4 * z * a[53] ) / 3 ) +
                                                              cth * z *
                                                                  ( ( 4 * a[55] ) / 3 -
                                                                    ( 4 * z * a[55] ) /
                                                                        3 ) ) ) ) ) ) ) ) ) -
        c[2] + z * ( 2 * a[1] + z * ( -2 * a[1] + c[2] + c[3] ) );
    result[4] =
        -3 * a[28] + z * z * ( -3 * a[29] + 2 * z * a[29] ) +
        cth * ( -3 * a[30] + z * z * ( -3 * a[31] + 2 * z * a[31] ) +
                cth * ( -3 * a[32] + z * z * ( -3 * a[33] + 2 * z * a[33] ) +
                        cth * ( -3 * a[34] + z * z * ( -3 * a[35] + 2 * z * a[35] ) ) ) ) +
        sth *
            ( -3 * a[36] + z * z * ( -3 * a[37] + 2 * z * a[37] ) +
              cth * ( -3 * a[38] + z * z * ( -3 * a[39] + 2 * z * a[39] ) +
                      cth * ( -3 * a[40] + z * z * ( -3 * a[41] + 2 * z * a[41] ) +
                              cth * ( 3 * a[36] + 3 * a[38] + 3 * a[40] +
                                      z * z *
                                          ( 3 * a[37] + 3 * a[39] +
                                            z * ( -2 * a[37] - 2 * a[39] - 2 * a[41] ) +
                                            3 * a[41] ) ) ) ) +
              sth *
                  ( -3 * a[42] + z * z * ( -3 * a[43] + 2 * z * a[43] ) +
                    cth * ( -3 * a[44] + z * z * ( -3 * a[45] + 2 * z * a[45] ) +
                            cth * ( -3 * a[46] + z * z * ( -3 * a[47] + 2 * z * a[47] ) +
                                    cth * ( -3 * a[48] +
                                            z * z * ( -3 * a[49] + 2 * z * a[49] ) ) ) ) +
                    sth * ( -3 * a[50] + z * z * ( -3 * a[51] + 2 * z * a[51] ) +
                            cth * ( 3 * a[30] + 3 * a[38] + 3 * a[44] +
                                    z * z *
                                        ( 3 * a[31] + 3 * a[39] +
                                          z * ( -2 * a[31] - 2 * a[39] - 2 * a[45] ) + 3 * a[45] ) +
                                    cth * ( -3 * a[52] + z * z * ( -3 * a[53] + 2 * z * a[53] ) +
                                            cth * ( -3 * a[54] +
                                                    z * z *
                                                        ( -3 * a[55] + 2 * z * a[55] ) ) ) ) ) ) ) -
        4 * c[0] - c[1] +
        r * ( 4 * a[28] + z * z * ( 4 * a[29] - ( 8 * z * a[29] ) / 3 ) +
              cth * ( 4 * a[30] + z * z * ( 4 * a[31] - ( 8 * z * a[31] ) / 3 ) +
                      cth * ( 4 * a[32] + z * z * ( 4 * a[33] - ( 8 * z * a[33] ) / 3 ) +
                              cth * ( 4 * a[34] +
                                      z * z * ( 4 * a[35] - ( 8 * z * a[35] ) / 3 ) ) ) ) +
              sth *
                  ( 4 * a[36] + z * z * ( 4 * a[37] - ( 8 * z * a[37] ) / 3 ) +
                    cth * ( 4 * a[38] + z * z * ( 4 * a[39] - ( 8 * z * a[39] ) / 3 ) +
                            cth * ( 4 * a[40] + z * z * ( 4 * a[41] - ( 8 * z * a[41] ) / 3 ) +
                                    cth * ( -4 * a[36] - 4 * a[38] - 4 * a[40] +
                                            z * z *
                                                ( -4 * a[37] - 4 * a[39] - 4 * a[41] +
                                                  z * ( ( 8 * a[37] ) / 3 + ( 8 * a[39] ) / 3 +
                                                        ( 8 * a[41] ) / 3 ) ) ) ) ) +
                    sth *
                        ( 4 * a[42] + z * z * ( 4 * a[43] - ( 8 * z * a[43] ) / 3 ) +
                          cth *
                              ( 4 * a[44] + z * z * ( 4 * a[45] - ( 8 * z * a[45] ) / 3 ) +
                                cth *
                                    ( 4 * a[46] + z * z * ( 4 * a[47] - ( 8 * z * a[47] ) / 3 ) +
                                      cth * ( 4 * a[48] +
                                              z * z * ( 4 * a[49] - ( 8 * z * a[49] ) / 3 ) ) ) ) +
                          sth * ( 4 * a[50] + z * z * ( 4 * a[51] - ( 8 * z * a[51] ) / 3 ) +
                                  cth * ( -4 * a[30] - 4 * a[38] - 4 * a[44] +
                                          z * z *
                                              ( -4 * a[31] - 4 * a[39] - 4 * a[45] +
                                                z * ( ( 8 * a[31] ) / 3 + ( 8 * a[39] ) / 3 +
                                                      ( 8 * a[45] ) / 3 ) ) +
                                          cth * ( 4 * a[52] +
                                                  z * z * ( 4 * a[53] - ( 8 * z * a[53] ) / 3 ) +
                                                  cth * ( 4 * a[54] +
                                                          z * z *
                                                              ( 4 * a[55] - ( 8 * z * a[55] ) /
                                                                                3 ) ) ) ) ) ) ) +
              8 * c[0] + 4 * c[1] );
    result[5] =
        cth * ( a[36] + z * z * ( a[37] - ( 2 * z * a[37] ) / 3 ) +
                cth * ( a[38] + z * z * ( a[39] - ( 2 * z * a[39] ) / 3 ) +
                        cth * ( a[40] +
                                cth * ( -a[36] - a[38] - a[40] +
                                        z * z *
                                            ( -a[37] - a[39] +
                                              z * ( ( 2 * a[37] ) / 3 + ( 2 * a[39] ) / 3 +
                                                    ( 2 * a[41] ) / 3 ) -
                                              a[41] ) ) +
                                z * z * ( a[41] - ( 2 * z * a[41] ) / 3 ) ) ) ) +
        sth *
            ( -a[30] + z * z * ( -a[31] + ( 2 * z * a[31] ) / 3 ) +
              cth *
                  ( -2 * a[32] + 2 * a[42] +
                    z * z *
                        ( -2 * a[33] + z * ( ( 4 * a[33] ) / 3 - ( 4 * a[43] ) / 3 ) + 2 * a[43] ) +
                    cth * ( -3 * a[34] + 2 * a[44] +
                            z * z *
                                ( -3 * a[35] + z * ( 2 * a[35] - ( 4 * a[45] ) / 3 ) + 2 * a[45] ) +
                            cth * ( 2 * a[46] + z * z * ( 2 * a[47] - ( 4 * z * a[47] ) / 3 ) +
                                    cth * ( 2 * a[48] +
                                            z * z * ( 2 * a[49] - ( 4 * z * a[49] ) / 3 ) ) ) ) ) +
              sth *
                  ( -a[38] + z * z * ( -a[39] + ( 2 * z * a[39] ) / 3 ) +
                    cth * ( -2 * a[40] + 3 * a[50] +
                            z * z *
                                ( -2 * a[41] + z * ( ( 4 * a[41] ) / 3 - 2 * a[51] ) + 3 * a[51] ) +
                            cth * ( -3 * a[30] + 3 * a[36] + 3 * a[40] - 3 * a[44] +
                                    z * z *
                                        ( -3 * a[31] + 3 * a[37] + 3 * a[41] - 3 * a[45] +
                                          z * ( 2 * a[31] - 2 * a[37] - 2 * a[41] + 2 * a[45] ) ) +
                                    cth * ( 3 * a[52] + z * z * ( 3 * a[53] - 2 * z * a[53] ) +
                                            cth * ( 3 * a[54] +
                                                    z * z * ( 3 * a[55] - 2 * z * a[55] ) ) ) ) ) +
                    sth *
                        ( -a[44] + z * z * ( -a[45] + ( 2 * z * a[45] ) / 3 ) +
                          cth * ( -2 * a[46] + z * z * ( -2 * a[47] + ( 4 * z * a[47] ) / 3 ) +
                                  cth * ( -3 * a[48] + z * z * ( -3 * a[49] + 2 * z * a[49] ) ) ) +
                          sth *
                              ( a[30] + a[38] + a[44] +
                                z * z *
                                    ( a[31] + a[39] +
                                      z * ( ( -2 * a[31] ) / 3 - ( 2 * a[39] ) / 3 -
                                            ( 2 * a[45] ) / 3 ) +
                                      a[45] ) +
                                cth *
                                    ( -2 * a[52] + z * z * ( -2 * a[53] + ( 4 * z * a[53] ) / 3 ) +
                                      cth * ( -3 * a[54] +
                                              z * z * ( -3 * a[55] + 2 * z * a[55] ) ) ) ) ) ) ) +
        r * ( cth * ( -3 * a[36] + z * z * ( -3 * a[37] + 2 * z * a[37] ) +
                      cth * ( -3 * a[38] + z * z * ( -3 * a[39] + 2 * z * a[39] ) +
                              cth * ( -3 * a[40] + z * z * ( -3 * a[41] + 2 * z * a[41] ) +
                                      cth * ( 3 * a[36] + 3 * a[38] + 3 * a[40] +
                                              z * z *
                                                  ( 3 * a[37] + 3 * a[39] +
                                                    z * ( -2 * a[37] - 2 * a[39] - 2 * a[41] ) +
                                                    3 * a[41] ) ) ) ) ) +
              sth *
                  ( 3 * a[30] + z * z * ( 3 * a[31] - 2 * z * a[31] ) +
                    cth * ( 6 * a[32] - 6 * a[42] +
                            z * z * ( 6 * a[33] - 6 * a[43] + z * ( -4 * a[33] + 4 * a[43] ) ) +
                            cth * ( 9 * a[34] - 6 * a[44] +
                                    z * z *
                                        ( 9 * a[35] - 6 * a[45] + z * ( -6 * a[35] + 4 * a[45] ) ) +
                                    cth * ( -6 * a[46] + z * z * ( -6 * a[47] + 4 * z * a[47] ) +
                                            cth * ( -6 * a[48] +
                                                    z * z * ( -6 * a[49] + 4 * z * a[49] ) ) ) ) ) +
                    sth *
                        ( 3 * a[38] + z * z * ( 3 * a[39] - 2 * z * a[39] ) +
                          sth *
                              ( 3 * a[44] + z * z * ( 3 * a[45] - 2 * z * a[45] ) +
                                cth * ( 6 * a[46] + z * z * ( 6 * a[47] - 4 * z * a[47] ) +
                                        cth * ( 9 * a[48] + z * z *
                                                                ( 9 * a[49] - 6 * z * a[49] ) ) ) +
                                sth * ( -3 * a[30] - 3 * a[38] - 3 * a[44] +
                                        z * z *
                                            ( -3 * a[31] - 3 * a[39] - 3 * a[45] +
                                              z * ( 2 * a[31] + 2 * a[39] + 2 * a[45] ) ) +
                                        cth * ( 6 * a[52] + z * z * ( 6 * a[53] - 4 * z * a[53] ) +
                                                cth * ( 9 * a[54] +
                                                        z * z *
                                                            ( 9 * a[55] - 6 * z * a[55] ) ) ) ) ) +
                          cth *
                              ( 6 * a[40] - 9 * a[50] +
                                z * z * ( 6 * a[41] - 9 * a[51] + z * ( -4 * a[41] + 6 * a[51] ) ) +
                                cth *
                                    ( 9 * a[30] - 9 * a[36] - 9 * a[40] + 9 * a[44] +
                                      z * z *
                                          ( 9 * a[31] - 9 * a[37] - 9 * a[41] +
                                            z * ( -6 * a[31] + 6 * a[37] + 6 * a[41] - 6 * a[45] ) +
                                            9 * a[45] ) +
                                      cth *
                                          ( -9 * a[52] + z * z * ( -9 * a[53] + 6 * z * a[53] ) +
                                            cth * ( -9 * a[54] +
                                                    z * z *
                                                        ( -9 * a[55] + 6 * z * a[55] ) ) ) ) ) ) ) +
              r * ( cth *
                        ( 2 * a[36] + z * z * ( 2 * a[37] - ( 4 * z * a[37] ) / 3 ) +
                          cth *
                              ( 2 * a[38] + z * z * ( 2 * a[39] - ( 4 * z * a[39] ) / 3 ) +
                                cth * ( 2 * a[40] + z * z * ( 2 * a[41] - ( 4 * z * a[41] ) / 3 ) +
                                        cth * ( -2 * a[36] - 2 * a[38] - 2 * a[40] +
                                                z * z *
                                                    ( -2 * a[37] - 2 * a[39] - 2 * a[41] +
                                                      z * ( ( 4 * a[37] ) / 3 + ( 4 * a[39] ) / 3 +
                                                            ( 4 * a[41] ) / 3 ) ) ) ) ) ) +
                    sth *
                        ( -2 * a[30] + z * z * ( -2 * a[31] + ( 4 * z * a[31] ) / 3 ) +
                          cth * ( -4 * a[32] + 4 * a[42] +
                                  z * z *
                                      ( -4 * a[33] + z * ( ( 8 * a[33] ) / 3 - ( 8 * a[43] ) / 3 ) +
                                        4 * a[43] ) +
                                  cth * ( -6 * a[34] + 4 * a[44] +
                                          z * z *
                                              ( -6 * a[35] + z * ( 4 * a[35] - ( 8 * a[45] ) / 3 ) +
                                                4 * a[45] ) +
                                          cth * ( 4 * a[46] +
                                                  z * z * ( 4 * a[47] - ( 8 * z * a[47] ) / 3 ) +
                                                  cth * ( 4 * a[48] +
                                                          z * z *
                                                              ( 4 * a[49] -
                                                                ( 8 * z * a[49] ) / 3 ) ) ) ) ) +
                          sth *
                              ( -2 * a[38] + z * z * ( -2 * a[39] + ( 4 * z * a[39] ) / 3 ) +
                                cth *
                                    ( -4 * a[40] + 6 * a[50] +
                                      z * z *
                                          ( -4 * a[41] + z * ( ( 8 * a[41] ) / 3 - 4 * a[51] ) +
                                            6 * a[51] ) +
                                      cth * ( -6 * a[30] + 6 * a[36] + 6 * a[40] - 6 * a[44] +
                                              z * z *
                                                  ( -6 * a[31] + 6 * a[37] + 6 * a[41] - 6 * a[45] +
                                                    z * ( 4 * a[31] - 4 * a[37] - 4 * a[41] +
                                                          4 * a[45] ) ) +
                                              cth * ( 6 * a[52] +
                                                      z * z * ( 6 * a[53] - 4 * z * a[53] ) +
                                                      cth * ( 6 * a[54] +
                                                              z * z *
                                                                  ( 6 * a[55] -
                                                                    4 * z * a[55] ) ) ) ) ) +
                                sth *
                                    ( -2 * a[44] + z * z * ( -2 * a[45] + ( 4 * z * a[45] ) / 3 ) +
                                      cth * ( -4 * a[46] +
                                              z * z * ( -4 * a[47] + ( 8 * z * a[47] ) / 3 ) +
                                              cth * ( -6 * a[48] +
                                                      z * z * ( -6 * a[49] + 4 * z * a[49] ) ) ) +
                                      sth *
                                          ( 2 * a[30] + 2 * a[38] + 2 * a[44] +
                                            z * z *
                                                ( 2 * a[31] + 2 * a[39] +
                                                  z * ( ( -4 * a[31] ) / 3 - ( 4 * a[39] ) / 3 -
                                                        ( 4 * a[45] ) / 3 ) +
                                                  2 * a[45] ) +
                                            cth * ( -4 * a[52] +
                                                    z * z * ( -4 * a[53] + ( 8 * z * a[53] ) / 3 ) +
                                                    cth * ( -6 * a[54] +
                                                            z * z *
                                                                ( -6 * a[55] +
                                                                  4 * z * a[55] ) ) ) ) ) ) ) ) );
    result[6] =
        z * ( 2 * a[29] - 2 * z * a[29] ) +
        cth * ( z * ( 2 * a[31] - 2 * z * a[31] ) +
                cth * ( z * ( 2 * a[33] - 2 * z * a[33] ) +
                        cth * z * ( 2 * a[35] - 2 * z * a[35] ) ) ) +
        sth * ( z * ( 2 * a[37] - 2 * z * a[37] ) +
                cth * ( z * ( 2 * a[39] - 2 * z * a[39] ) +
                        cth * ( z * ( 2 * a[41] - 2 * z * a[41] ) +
                                cth * z *
                                    ( -2 * a[37] - 2 * a[39] - 2 * a[41] +
                                      z * ( 2 * a[37] + 2 * a[39] + 2 * a[41] ) ) ) ) +
                sth * ( z * ( 2 * a[43] - 2 * z * a[43] ) +
                        cth * ( z * ( 2 * a[45] - 2 * z * a[45] ) +
                                cth * ( z * ( 2 * a[47] - 2 * z * a[47] ) +
                                        cth * z * ( 2 * a[49] - 2 * z * a[49] ) ) ) +
                        sth * ( z * ( 2 * a[51] - 2 * z * a[51] ) +
                                cth * ( z * ( -2 * a[31] - 2 * a[39] - 2 * a[45] +
                                              z * ( 2 * a[31] + 2 * a[39] + 2 * a[45] ) ) +
                                        cth * ( z * ( 2 * a[53] - 2 * z * a[53] ) +
                                                cth * z * ( 2 * a[55] - 2 * z * a[55] ) ) ) ) ) ) +
        r * ( z * ( -6 * a[29] + 6 * z * a[29] ) +
              cth * ( z * ( -6 * a[31] + 6 * z * a[31] ) +
                      cth * ( z * ( -6 * a[33] + 6 * z * a[33] ) +
                              cth * z * ( -6 * a[35] + 6 * z * a[35] ) ) ) +
              sth *
                  ( z * ( -6 * a[37] + 6 * z * a[37] ) +
                    cth * ( z * ( -6 * a[39] + 6 * z * a[39] ) +
                            cth * ( cth * z *
                                        ( 6 * a[37] + 6 * a[39] +
                                          z * ( -6 * a[37] - 6 * a[39] - 6 * a[41] ) + 6 * a[41] ) +
                                    z * ( -6 * a[41] + 6 * z * a[41] ) ) ) +
                    sth * ( z * ( -6 * a[43] + 6 * z * a[43] ) +
                            cth * ( z * ( -6 * a[45] + 6 * z * a[45] ) +
                                    cth * ( z * ( -6 * a[47] + 6 * z * a[47] ) +
                                            cth * z * ( -6 * a[49] + 6 * z * a[49] ) ) ) +
                            sth * ( z * ( -6 * a[51] + 6 * z * a[51] ) +
                                    cth * ( z * ( 6 * a[31] + 6 * a[39] +
                                                  z * ( -6 * a[31] - 6 * a[39] - 6 * a[45] ) +
                                                  6 * a[45] ) +
                                            cth * ( z * ( -6 * a[53] + 6 * z * a[53] ) +
                                                    cth * z *
                                                        ( -6 * a[55] + 6 * z * a[55] ) ) ) ) ) ) +
              r * ( z * ( 4 * a[29] - 4 * z * a[29] ) +
                    cth * ( z * ( 4 * a[31] - 4 * z * a[31] ) +
                            cth * ( z * ( 4 * a[33] - 4 * z * a[33] ) +
                                    cth * z * ( 4 * a[35] - 4 * z * a[35] ) ) ) +
                    sth *
                        ( z * ( 4 * a[37] - 4 * z * a[37] ) +
                          cth * ( z * ( 4 * a[39] - 4 * z * a[39] ) +
                                  cth * ( z * ( 4 * a[41] - 4 * z * a[41] ) +
                                          cth * z *
                                              ( -4 * a[37] - 4 * a[39] - 4 * a[41] +
                                                z * ( 4 * a[37] + 4 * a[39] + 4 * a[41] ) ) ) ) +
                          sth *
                              ( z * ( 4 * a[43] - 4 * z * a[43] ) +
                                cth * ( z * ( 4 * a[45] - 4 * z * a[45] ) +
                                        cth * ( z * ( 4 * a[47] - 4 * z * a[47] ) +
                                                cth * z * ( 4 * a[49] - 4 * z * a[49] ) ) ) +
                                sth * ( z * ( 4 * a[51] - 4 * z * a[51] ) +
                                        cth * ( z * ( -4 * a[31] - 4 * a[39] - 4 * a[45] +
                                                      z * ( 4 * a[31] + 4 * a[39] + 4 * a[45] ) ) +
                                                cth * ( z * ( 4 * a[53] - 4 * z * a[53] ) +
                                                        cth * z *
                                                            ( 4 * a[55] -
                                                              4 * z * a[55] ) ) ) ) ) ) ) );
    result[7] =
        cth *
            ( -a[2] + z * z * ( -a[3] + ( 2 * z * a[3] ) / 3 ) +
              cth *
                  ( -2 * a[4] + 2 * a[14] +
                    z * z *
                        ( -2 * a[5] + z * ( ( 4 * a[5] ) / 3 - ( 4 * a[15] ) / 3 ) + 2 * a[15] ) +
                    cth *
                        ( -3 * a[6] + 2 * a[16] +
                          z * z * ( -3 * a[7] + z * ( 2 * a[7] - ( 4 * a[17] ) / 3 ) + 2 * a[17] ) +
                          cth * ( 2 * a[18] + z * z * ( 2 * a[19] - ( 4 * z * a[19] ) / 3 ) +
                                  cth * ( 2 * a[20] +
                                          z * z * ( 2 * a[21] - ( 4 * z * a[21] ) / 3 ) ) ) ) ) ) +
        sth *
            ( -a[8] + z * z * ( -a[9] + ( 2 * z * a[9] ) / 3 ) +
              cth *
                  ( -4 * a[10] + z * z * ( -4 * a[11] + ( 8 * z * a[11] ) / 3 ) +
                    cth *
                        ( -7 * a[12] + 6 * a[22] +
                          z * z *
                              ( -7 * a[13] + z * ( ( 14 * a[13] ) / 3 - 4 * a[23] ) + 6 * a[23] ) +
                          cth * ( -6 * a[2] + 10 * a[8] + 4 * a[10] + 10 * a[12] - 6 * a[16] +
                                  z * z *
                                      ( -6 * a[3] + 10 * a[9] + 4 * a[11] + 10 * a[13] - 6 * a[17] +
                                        z * ( 4 * a[3] - ( 20 * a[9] ) / 3 - ( 8 * a[11] ) / 3 -
                                              ( 20 * a[13] ) / 3 + 4 * a[17] ) ) +
                                  cth * ( 6 * a[24] + z * z * ( 6 * a[25] - 4 * z * a[25] ) +
                                          cth * ( 6 * a[26] +
                                                  z * z * ( 6 * a[27] - 4 * z * a[27] ) ) ) ) ) ) +
              sth *
                  ( 2 * a[4] - 2 * a[14] +
                    z * z *
                        ( 2 * a[5] - 2 * a[15] + z * ( ( -4 * a[5] ) / 3 + ( 4 * a[15] ) / 3 ) ) +
                    cth * ( 6 * a[6] - 7 * a[16] +
                            z * z *
                                ( 6 * a[7] - 7 * a[17] + z * ( -4 * a[7] + ( 14 * a[17] ) / 3 ) ) +
                            cth * ( -12 * a[18] + z * z * ( -12 * a[19] + 8 * z * a[19] ) +
                                    cth * ( -17 * a[20] +
                                            z * z * ( -17 * a[21] + ( 34 * z * a[21] ) / 3 ) ) ) ) +
                    sth *
                        ( 2 * a[12] - 3 * a[22] +
                          z * z *
                              ( 2 * a[13] - 3 * a[23] + z * ( ( -4 * a[13] ) / 3 + 2 * a[23] ) ) +
                          sth *
                              ( 2 * a[18] + z * z * ( 2 * a[19] - ( 4 * z * a[19] ) / 3 ) +
                                cth * ( 6 * a[20] + z * z * ( 6 * a[21] - 4 * z * a[21] ) ) +
                                sth * ( 2 * a[24] + z * z * ( 2 * a[25] - ( 4 * z * a[25] ) / 3 ) +
                                        cth * ( 6 * a[26] +
                                                z * z * ( 6 * a[27] - 4 * z * a[27] ) ) ) ) +
                          cth * ( 10 * a[2] - 6 * a[8] + 4 * a[10] - 6 * a[12] + 10 * a[16] +
                                  z * z *
                                      ( 10 * a[3] - 6 * a[9] + 4 * a[11] - 6 * a[13] +
                                        z * ( ( -20 * a[3] ) / 3 + 4 * a[9] - ( 8 * a[11] ) / 3 +
                                              4 * a[13] - ( 20 * a[17] ) / 3 ) +
                                        10 * a[17] ) +
                                  cth * ( -17 * a[24] +
                                          z * z * ( -17 * a[25] + ( 34 * z * a[25] ) / 3 ) +
                                          cth * ( -24 * a[26] +
                                                  z * z *
                                                      ( -24 * a[27] + 16 * z * a[27] ) ) ) ) ) ) ) +
        r * ( cth *
                  ( -a[30] + z * z * ( -a[31] + ( 2 * z * a[31] ) / 3 ) +
                    cth *
                        ( -2 * a[32] + 2 * a[42] +
                          z * z *
                              ( -2 * a[33] + z * ( ( 4 * a[33] ) / 3 - ( 4 * a[43] ) / 3 ) +
                                2 * a[43] ) +
                          cth *
                              ( -3 * a[34] + 2 * a[44] +
                                z * z *
                                    ( -3 * a[35] + z * ( 2 * a[35] - ( 4 * a[45] ) / 3 ) +
                                      2 * a[45] ) +
                                cth * ( 2 * a[46] + z * z * ( 2 * a[47] - ( 4 * z * a[47] ) / 3 ) +
                                        cth * ( 2 * a[48] + z * z *
                                                                ( 2 * a[49] - ( 4 * z * a[49] ) /
                                                                                  3 ) ) ) ) ) ) +
              sth *
                  ( -a[36] + z * z * ( -a[37] + ( 2 * z * a[37] ) / 3 ) +
                    cth *
                        ( -4 * a[38] + z * z * ( -4 * a[39] + ( 8 * z * a[39] ) / 3 ) +
                          cth *
                              ( -7 * a[40] + 6 * a[50] +
                                z * z *
                                    ( -7 * a[41] + z * ( ( 14 * a[41] ) / 3 - 4 * a[51] ) +
                                      6 * a[51] ) +
                                cth *
                                    ( -6 * a[30] + 10 * a[36] + 4 * a[38] + 10 * a[40] - 6 * a[44] +
                                      z * z *
                                          ( -6 * a[31] + 10 * a[37] + 4 * a[39] + 10 * a[41] -
                                            6 * a[45] +
                                            z * ( 4 * a[31] - ( 20 * a[37] ) / 3 -
                                                  ( 8 * a[39] ) / 3 - ( 20 * a[41] ) / 3 +
                                                  4 * a[45] ) ) +
                                      cth * ( 6 * a[52] + z * z * ( 6 * a[53] - 4 * z * a[53] ) +
                                              cth * ( 6 * a[54] +
                                                      z * z *
                                                          ( 6 * a[55] - 4 * z * a[55] ) ) ) ) ) ) +
                    sth *
                        ( 2 * a[32] - 2 * a[42] +
                          z * z *
                              ( 2 * a[33] - 2 * a[43] +
                                z * ( ( -4 * a[33] ) / 3 + ( 4 * a[43] ) / 3 ) ) +
                          cth *
                              ( 6 * a[34] - 7 * a[44] +
                                z * z *
                                    ( 6 * a[35] - 7 * a[45] +
                                      z * ( -4 * a[35] + ( 14 * a[45] ) / 3 ) ) +
                                cth * ( -12 * a[46] + z * z * ( -12 * a[47] + 8 * z * a[47] ) +
                                        cth * ( -17 * a[48] +
                                                z * z *
                                                    ( -17 * a[49] + ( 34 * z * a[49] ) / 3 ) ) ) ) +
                          sth *
                              ( 2 * a[40] - 3 * a[50] +
                                z * z *
                                    ( 2 * a[41] - 3 * a[51] +
                                      z * ( ( -4 * a[41] ) / 3 + 2 * a[51] ) ) +
                                sth *
                                    ( 2 * a[46] + z * z * ( 2 * a[47] - ( 4 * z * a[47] ) / 3 ) +
                                      cth * ( 6 * a[48] + z * z * ( 6 * a[49] - 4 * z * a[49] ) ) +
                                      sth * ( 2 * a[52] +
                                              z * z * ( 2 * a[53] - ( 4 * z * a[53] ) / 3 ) +
                                              cth * ( 6 * a[54] +
                                                      z * z * ( 6 * a[55] - 4 * z * a[55] ) ) ) ) +
                                cth *
                                    ( 10 * a[30] - 6 * a[36] + 4 * a[38] - 6 * a[40] + 10 * a[44] +
                                      z * z *
                                          ( 10 * a[31] - 6 * a[37] + 4 * a[39] - 6 * a[41] +
                                            z * ( ( -20 * a[31] ) / 3 + 4 * a[37] -
                                                  ( 8 * a[39] ) / 3 + 4 * a[41] -
                                                  ( 20 * a[45] ) / 3 ) +
                                            10 * a[45] ) +
                                      cth * ( -17 * a[52] +
                                              z * z * ( -17 * a[53] + ( 34 * z * a[53] ) / 3 ) +
                                              cth * ( -24 * a[54] +
                                                      z * z *
                                                          ( -24 * a[55] +
                                                            16 * z * a[55] ) ) ) ) ) ) ) +
              r * ( cth *
                        ( ( 3 * a[30] ) / 2 + z * z * ( ( 3 * a[31] ) / 2 - z * a[31] ) +
                          cth *
                              ( 3 * a[32] - 3 * a[42] +
                                z * z *
                                    ( 3 * a[33] - 3 * a[43] + z * ( -2 * a[33] + 2 * a[43] ) ) +
                                cth *
                                    ( ( 9 * a[34] ) / 2 - 3 * a[44] +
                                      z * z *
                                          ( ( 9 * a[35] ) / 2 - 3 * a[45] +
                                            z * ( -3 * a[35] + 2 * a[45] ) ) +
                                      cth * ( -3 * a[46] + z * z * ( -3 * a[47] + 2 * z * a[47] ) +
                                              cth * ( -3 * a[48] +
                                                      z * z *
                                                          ( -3 * a[49] + 2 * z * a[49] ) ) ) ) ) ) +
                    sth *
                        ( ( 3 * a[36] ) / 2 + z * z * ( ( 3 * a[37] ) / 2 - z * a[37] ) +
                          sth *
                              ( -3 * a[32] + 3 * a[42] +
                                z * z *
                                    ( -3 * a[33] + z * ( 2 * a[33] - 2 * a[43] ) + 3 * a[43] ) +
                                cth *
                                    ( -9 * a[34] + ( 21 * a[44] ) / 2 +
                                      z * z *
                                          ( -9 * a[35] + z * ( 6 * a[35] - 7 * a[45] ) +
                                            ( 21 * a[45] ) / 2 ) +
                                      cth * ( 18 * a[46] + z * z * ( 18 * a[47] - 12 * z * a[47] ) +
                                              cth * ( ( 51 * a[48] ) / 2 +
                                                      z * z *
                                                          ( ( 51 * a[49] ) / 2 -
                                                            17 * z * a[49] ) ) ) ) +
                                sth *
                                    ( -3 * a[40] + ( 9 * a[50] ) / 2 +
                                      z * z *
                                          ( -3 * a[41] + z * ( 2 * a[41] - 3 * a[51] ) +
                                            ( 9 * a[51] ) / 2 ) +
                                      cth * ( -15 * a[30] + 9 * a[36] - 6 * a[38] + 9 * a[40] -
                                              15 * a[44] +
                                              z * z *
                                                  ( -15 * a[31] + 9 * a[37] - 6 * a[39] +
                                                    9 * a[41] - 15 * a[45] +
                                                    z * ( 10 * a[31] - 6 * a[37] + 4 * a[39] -
                                                          6 * a[41] + 10 * a[45] ) ) +
                                              cth * ( ( 51 * a[52] ) / 2 +
                                                      z * z *
                                                          ( ( 51 * a[53] ) / 2 - 17 * z * a[53] ) +
                                                      cth * ( 36 * a[54] +
                                                              z * z *
                                                                  ( 36 * a[55] -
                                                                    24 * z * a[55] ) ) ) ) +
                                      sth * ( -3 * a[46] + z * z * ( -3 * a[47] + 2 * z * a[47] ) +
                                              cth * ( -9 * a[48] +
                                                      z * z * ( -9 * a[49] + 6 * z * a[49] ) ) +
                                              sth * ( -3 * a[52] +
                                                      z * z * ( -3 * a[53] + 2 * z * a[53] ) +
                                                      cth * ( -9 * a[54] +
                                                              z * z *
                                                                  ( -9 * a[55] +
                                                                    6 * z * a[55] ) ) ) ) ) ) +
                          cth *
                              ( 6 * a[38] + z * z * ( 6 * a[39] - 4 * z * a[39] ) +
                                cth *
                                    ( ( 21 * a[40] ) / 2 - 9 * a[50] +
                                      z * z *
                                          ( ( 21 * a[41] ) / 2 - 9 * a[51] +
                                            z * ( -7 * a[41] + 6 * a[51] ) ) +
                                      cth *
                                          ( 9 * a[30] - 15 * a[36] - 6 * a[38] - 15 * a[40] +
                                            9 * a[44] +
                                            z * z *
                                                ( 9 * a[31] - 15 * a[37] - 6 * a[39] - 15 * a[41] +
                                                  z * ( -6 * a[31] + 10 * a[37] + 4 * a[39] +
                                                        10 * a[41] - 6 * a[45] ) +
                                                  9 * a[45] ) +
                                            cth * ( -9 * a[52] +
                                                    z * z * ( -9 * a[53] + 6 * z * a[53] ) +
                                                    cth * ( -9 * a[54] +
                                                            z * z *
                                                                ( -9 * a[55] +
                                                                  6 * z * a[55] ) ) ) ) ) ) ) +
                    r * ( cth *
                              ( ( -2 * a[30] ) / 3 +
                                z * z * ( ( -2 * a[31] ) / 3 + ( 4 * z * a[31] ) / 9 ) +
                                cth *
                                    ( ( -4 * a[32] ) / 3 + ( 4 * a[42] ) / 3 +
                                      z * z *
                                          ( ( -4 * a[33] ) / 3 +
                                            z * ( ( 8 * a[33] ) / 9 - ( 8 * a[43] ) / 9 ) +
                                            ( 4 * a[43] ) / 3 ) +
                                      cth * ( -2 * a[34] + ( 4 * a[44] ) / 3 +
                                              z * z *
                                                  ( -2 * a[35] +
                                                    z * ( ( 4 * a[35] ) / 3 - ( 8 * a[45] ) / 9 ) +
                                                    ( 4 * a[45] ) / 3 ) +
                                              cth * ( ( 4 * a[46] ) / 3 +
                                                      z * z *
                                                          ( ( 4 * a[47] ) / 3 -
                                                            ( 8 * z * a[47] ) / 9 ) +
                                                      cth * ( ( 4 * a[48] ) / 3 +
                                                              z * z *
                                                                  ( ( 4 * a[49] ) / 3 -
                                                                    ( 8 * z * a[49] ) /
                                                                        9 ) ) ) ) ) ) +
                          sth *
                              ( ( -2 * a[36] ) / 3 +
                                z * z * ( ( -2 * a[37] ) / 3 + ( 4 * z * a[37] ) / 9 ) +
                                cth * ( ( -8 * a[38] ) / 3 +
                                        z * z * ( ( -8 * a[39] ) / 3 + ( 16 * z * a[39] ) / 9 ) +
                                        cth *
                                            ( ( -14 * a[40] ) / 3 + 4 * a[50] +
                                              z * z *
                                                  ( ( -14 * a[41] ) / 3 +
                                                    z * ( ( 28 * a[41] ) / 9 - ( 8 * a[51] ) / 3 ) +
                                                    4 * a[51] ) +
                                              cth * ( -4 * a[30] + ( 20 * a[36] ) / 3 +
                                                      ( 8 * a[38] ) / 3 + ( 20 * a[40] ) / 3 -
                                                      4 * a[44] +
                                                      z * z *
                                                          ( -4 * a[31] + ( 20 * a[37] ) / 3 +
                                                            ( 8 * a[39] ) / 3 + ( 20 * a[41] ) / 3 -
                                                            4 * a[45] +
                                                            z * ( ( 8 * a[31] ) / 3 -
                                                                  ( 40 * a[37] ) / 9 -
                                                                  ( 16 * a[39] ) / 9 -
                                                                  ( 40 * a[41] ) / 9 +
                                                                  ( 8 * a[45] ) / 3 ) ) +
                                                      cth * ( 4 * a[52] +
                                                              z * z *
                                                                  ( 4 * a[53] -
                                                                    ( 8 * z * a[53] ) / 3 ) +
                                                              cth * ( 4 * a[54] +
                                                                      z * z *
                                                                          ( 4 * a[55] -
                                                                            ( 8 * z * a[55] ) /
                                                                                3 ) ) ) ) ) ) +
                                sth *
                                    ( ( 4 * a[32] ) / 3 - ( 4 * a[42] ) / 3 +
                                      z * z *
                                          ( ( 4 * a[33] ) / 3 +
                                            z * ( ( -8 * a[33] ) / 9 + ( 8 * a[43] ) / 9 ) -
                                            ( 4 * a[43] ) / 3 ) +
                                      cth * ( 4 * a[34] - ( 14 * a[44] ) / 3 +
                                              z * z *
                                                  ( 4 * a[35] - ( 14 * a[45] ) / 3 +
                                                    z * ( ( -8 * a[35] ) / 3 +
                                                          ( 28 * a[45] ) / 9 ) ) +
                                              cth * ( -8 * a[46] +
                                                      z * z *
                                                          ( -8 * a[47] + ( 16 * z * a[47] ) / 3 ) +
                                                      cth * ( ( -34 * a[48] ) / 3 +
                                                              z * z *
                                                                  ( ( -34 * a[49] ) / 3 +
                                                                    ( 68 * z * a[49] ) / 9 ) ) ) ) +
                                      sth *
                                          ( ( 4 * a[40] ) / 3 - 2 * a[50] +
                                            z * z *
                                                ( ( 4 * a[41] ) / 3 - 2 * a[51] +
                                                  z * ( ( -8 * a[41] ) / 9 + ( 4 * a[51] ) / 3 ) ) +
                                            sth * ( ( 4 * a[46] ) / 3 +
                                                    z * z *
                                                        ( ( 4 * a[47] ) / 3 -
                                                          ( 8 * z * a[47] ) / 9 ) +
                                                    cth * ( 4 * a[48] +
                                                            z * z *
                                                                ( 4 * a[49] -
                                                                  ( 8 * z * a[49] ) / 3 ) ) +
                                                    sth * ( ( 4 * a[52] ) / 3 +
                                                            z * z *
                                                                ( ( 4 * a[53] ) / 3 -
                                                                  ( 8 * z * a[53] ) / 9 ) +
                                                            cth * ( 4 * a[54] +
                                                                    z * z *
                                                                        ( 4 * a[55] -
                                                                          ( 8 * z * a[55] ) /
                                                                              3 ) ) ) ) +
                                            cth * ( ( 20 * a[30] ) / 3 - 4 * a[36] +
                                                    ( 8 * a[38] ) / 3 - 4 * a[40] +
                                                    ( 20 * a[44] ) / 3 +
                                                    z * z *
                                                        ( ( 20 * a[31] ) / 3 - 4 * a[37] +
                                                          ( 8 * a[39] ) / 3 - 4 * a[41] +
                                                          z * ( ( -40 * a[31] ) / 9 +
                                                                ( 8 * a[37] ) / 3 -
                                                                ( 16 * a[39] ) / 9 +
                                                                ( 8 * a[41] ) / 3 -
                                                                ( 40 * a[45] ) / 9 ) +
                                                          ( 20 * a[45] ) / 3 ) +
                                                    cth * ( ( -34 * a[52] ) / 3 +
                                                            z * z *
                                                                ( ( -34 * a[53] ) / 3 +
                                                                  ( 68 * z * a[53] ) / 9 ) +
                                                            cth * ( -16 * a[54] +
                                                                    z * z *
                                                                        ( -16 * a[55] +
                                                                          ( 32 * z * a[55] ) /
                                                                              3 ) ) ) ) ) ) ) ) ) );
    result[8] =
        cth * ( z * ( 2 * a[9] - 2 * z * a[9] ) +
                cth * ( z * ( 2 * a[11] - 2 * z * a[11] ) +
                        cth * ( z * ( 2 * a[13] - 2 * z * a[13] ) +
                                cth * z *
                                    ( -2 * a[9] - 2 * a[11] - 2 * a[13] +
                                      z * ( 2 * a[9] + 2 * a[11] + 2 * a[13] ) ) ) ) ) +
        sth *
            ( z * ( -2 * a[3] + 2 * z * a[3] ) +
              cth * ( z * ( -4 * a[5] + z * ( 4 * a[5] - 4 * a[15] ) + 4 * a[15] ) +
                      cth * ( z * ( -6 * a[7] + z * ( 6 * a[7] - 4 * a[17] ) + 4 * a[17] ) +
                              cth * ( z * ( 4 * a[19] - 4 * z * a[19] ) +
                                      cth * z * ( 4 * a[21] - 4 * z * a[21] ) ) ) ) +
              sth *
                  ( z * ( -2 * a[11] + 2 * z * a[11] ) +
                    cth * ( z * ( -4 * a[13] + z * ( 4 * a[13] - 6 * a[23] ) + 6 * a[23] ) +
                            cth * ( z * ( -6 * a[3] + 6 * a[9] + 6 * a[13] - 6 * a[17] +
                                          z * ( 6 * a[3] - 6 * a[9] - 6 * a[13] + 6 * a[17] ) ) +
                                    cth * ( z * ( 6 * a[25] - 6 * z * a[25] ) +
                                            cth * z * ( 6 * a[27] - 6 * z * a[27] ) ) ) ) +
                    sth * ( z * ( -2 * a[17] + 2 * z * a[17] ) +
                            cth * ( z * ( -4 * a[19] + 4 * z * a[19] ) +
                                    cth * z * ( -6 * a[21] + 6 * z * a[21] ) ) +
                            sth * ( z * ( 2 * a[3] + 2 * a[11] +
                                          z * ( -2 * a[3] - 2 * a[11] - 2 * a[17] ) + 2 * a[17] ) +
                                    cth * ( z * ( -4 * a[25] + 4 * z * a[25] ) +
                                            cth * z * ( -6 * a[27] + 6 * z * a[27] ) ) ) ) ) ) +
        r * ( cth * ( z * ( 2 * a[37] - 2 * z * a[37] ) +
                      cth * ( z * ( 2 * a[39] - 2 * z * a[39] ) +
                              cth * ( z * ( 2 * a[41] - 2 * z * a[41] ) +
                                      cth * z *
                                          ( -2 * a[37] - 2 * a[39] - 2 * a[41] +
                                            z * ( 2 * a[37] + 2 * a[39] + 2 * a[41] ) ) ) ) ) +
              sth *
                  ( z * ( -2 * a[31] + 2 * z * a[31] ) +
                    cth * ( z * ( -4 * a[33] + z * ( 4 * a[33] - 4 * a[43] ) + 4 * a[43] ) +
                            cth * ( z * ( -6 * a[35] + z * ( 6 * a[35] - 4 * a[45] ) + 4 * a[45] ) +
                                    cth * ( z * ( 4 * a[47] - 4 * z * a[47] ) +
                                            cth * z * ( 4 * a[49] - 4 * z * a[49] ) ) ) ) +
                    sth * ( z * ( -2 * a[39] + 2 * z * a[39] ) +
                            cth * ( z * ( -4 * a[41] + z * ( 4 * a[41] - 6 * a[51] ) + 6 * a[51] ) +
                                    cth * ( z * ( -6 * a[31] + 6 * a[37] + 6 * a[41] - 6 * a[45] +
                                                  z * ( 6 * a[31] - 6 * a[37] - 6 * a[41] +
                                                        6 * a[45] ) ) +
                                            cth * ( z * ( 6 * a[53] - 6 * z * a[53] ) +
                                                    cth * z * ( 6 * a[55] - 6 * z * a[55] ) ) ) ) +
                            sth * ( z * ( -2 * a[45] + 2 * z * a[45] ) +
                                    cth * ( z * ( -4 * a[47] + 4 * z * a[47] ) +
                                            cth * z * ( -6 * a[49] + 6 * z * a[49] ) ) +
                                    sth * ( z * ( 2 * a[31] + 2 * a[39] +
                                                  z * ( -2 * a[31] - 2 * a[39] - 2 * a[45] ) +
                                                  2 * a[45] ) +
                                            cth * ( z * ( -4 * a[53] + 4 * z * a[53] ) +
                                                    cth * z *
                                                        ( -6 * a[55] + 6 * z * a[55] ) ) ) ) ) ) +
              r * ( cth * ( z * ( -3 * a[37] + 3 * z * a[37] ) +
                            cth * ( z * ( -3 * a[39] + 3 * z * a[39] ) +
                                    cth * ( cth * z *
                                                ( 3 * a[37] + 3 * a[39] +
                                                  z * ( -3 * a[37] - 3 * a[39] - 3 * a[41] ) +
                                                  3 * a[41] ) +
                                            z * ( -3 * a[41] + 3 * z * a[41] ) ) ) ) +
                    sth *
                        ( z * ( 3 * a[31] - 3 * z * a[31] ) +
                          cth * ( z * ( 6 * a[33] - 6 * a[43] + z * ( -6 * a[33] + 6 * a[43] ) ) +
                                  cth * ( z * ( 9 * a[35] - 6 * a[45] +
                                                z * ( -9 * a[35] + 6 * a[45] ) ) +
                                          cth * ( z * ( -6 * a[47] + 6 * z * a[47] ) +
                                                  cth * z * ( -6 * a[49] + 6 * z * a[49] ) ) ) ) +
                          sth *
                              ( z * ( 3 * a[39] - 3 * z * a[39] ) +
                                sth * ( z * ( 3 * a[45] - 3 * z * a[45] ) +
                                        cth * ( z * ( 6 * a[47] - 6 * z * a[47] ) +
                                                cth * z * ( 9 * a[49] - 9 * z * a[49] ) ) +
                                        sth * ( z * ( -3 * a[31] - 3 * a[39] - 3 * a[45] +
                                                      z * ( 3 * a[31] + 3 * a[39] + 3 * a[45] ) ) +
                                                cth * ( z * ( 6 * a[53] - 6 * z * a[53] ) +
                                                        cth * z *
                                                            ( 9 * a[55] - 9 * z * a[55] ) ) ) ) +
                                cth *
                                    ( z * ( 6 * a[41] - 9 * a[51] +
                                            z * ( -6 * a[41] + 9 * a[51] ) ) +
                                      cth * ( z * ( 9 * a[31] - 9 * a[37] - 9 * a[41] +
                                                    z * ( -9 * a[31] + 9 * a[37] + 9 * a[41] -
                                                          9 * a[45] ) +
                                                    9 * a[45] ) +
                                              cth * ( z * ( -9 * a[53] + 9 * z * a[53] ) +
                                                      cth * z *
                                                          ( -9 * a[55] + 9 * z * a[55] ) ) ) ) ) ) +
                    r * ( cth *
                              ( z * ( ( 4 * a[37] ) / 3 - ( 4 * z * a[37] ) / 3 ) +
                                cth * ( z * ( ( 4 * a[39] ) / 3 - ( 4 * z * a[39] ) / 3 ) +
                                        cth * ( z * ( ( 4 * a[41] ) / 3 - ( 4 * z * a[41] ) / 3 ) +
                                                cth * z *
                                                    ( ( -4 * a[37] ) / 3 - ( 4 * a[39] ) / 3 -
                                                      ( 4 * a[41] ) / 3 +
                                                      z * ( ( 4 * a[37] ) / 3 + ( 4 * a[39] ) / 3 +
                                                            ( 4 * a[41] ) / 3 ) ) ) ) ) +
                          sth *
                              ( z * ( ( -4 * a[31] ) / 3 + ( 4 * z * a[31] ) / 3 ) +
                                cth * ( z * ( ( -8 * a[33] ) / 3 +
                                              z * ( ( 8 * a[33] ) / 3 - ( 8 * a[43] ) / 3 ) +
                                              ( 8 * a[43] ) / 3 ) +
                                        cth * ( z * ( -4 * a[35] +
                                                      z * ( 4 * a[35] - ( 8 * a[45] ) / 3 ) +
                                                      ( 8 * a[45] ) / 3 ) +
                                                cth * ( z * ( ( 8 * a[47] ) / 3 -
                                                              ( 8 * z * a[47] ) / 3 ) +
                                                        cth * z *
                                                            ( ( 8 * a[49] ) / 3 -
                                                              ( 8 * z * a[49] ) / 3 ) ) ) ) +
                                sth *
                                    ( z * ( ( -4 * a[39] ) / 3 + ( 4 * z * a[39] ) / 3 ) +
                                      cth * ( z * ( ( -8 * a[41] ) / 3 +
                                                    z * ( ( 8 * a[41] ) / 3 - 4 * a[51] ) +
                                                    4 * a[51] ) +
                                              cth * ( z * ( -4 * a[31] + 4 * a[37] + 4 * a[41] -
                                                            4 * a[45] +
                                                            z * ( 4 * a[31] - 4 * a[37] -
                                                                  4 * a[41] + 4 * a[45] ) ) +
                                                      cth * ( z * ( 4 * a[53] - 4 * z * a[53] ) +
                                                              cth * z *
                                                                  ( 4 * a[55] -
                                                                    4 * z * a[55] ) ) ) ) +
                                      sth * ( z * ( ( -4 * a[45] ) / 3 + ( 4 * z * a[45] ) / 3 ) +
                                              cth * ( z * ( ( -8 * a[47] ) / 3 +
                                                            ( 8 * z * a[47] ) / 3 ) +
                                                      cth * z * ( -4 * a[49] + 4 * z * a[49] ) ) +
                                              sth * ( z * ( ( 4 * a[31] ) / 3 + ( 4 * a[39] ) / 3 +
                                                            z * ( ( -4 * a[31] ) / 3 -
                                                                  ( 4 * a[39] ) / 3 -
                                                                  ( 4 * a[45] ) / 3 ) +
                                                            ( 4 * a[45] ) / 3 ) +
                                                      cth * ( z * ( ( -8 * a[53] ) / 3 +
                                                                    ( 8 * z * a[53] ) / 3 ) +
                                                              cth * z *
                                                                  ( -4 * a[55] +
                                                                    4 * z * a[55] ) ) ) ) ) ) ) ) );
    result[9] =
        2 * a[1] +
        cth * ( 2 * a[3] - 4 * z * a[3] +
                cth * ( 2 * a[5] - 4 * z * a[5] + cth * ( 2 * a[7] - 4 * z * a[7] ) ) ) +
        sth * ( 2 * a[9] - 4 * z * a[9] +
                cth * ( 2 * a[11] - 4 * z * a[11] +
                        cth * ( 2 * a[13] - 4 * z * a[13] +
                                cth * ( -2 * a[9] - 2 * a[11] - 2 * a[13] +
                                        z * ( 4 * a[9] + 4 * a[11] + 4 * a[13] ) ) ) ) +
                sth * ( 2 * a[15] - 4 * z * a[15] +
                        cth * ( 2 * a[17] - 4 * z * a[17] +
                                cth * ( 2 * a[19] - 4 * z * a[19] +
                                        cth * ( 2 * a[21] - 4 * z * a[21] ) ) ) +
                        sth * ( 2 * a[23] - 4 * z * a[23] +
                                cth * ( -2 * a[3] - 2 * a[11] - 2 * a[17] +
                                        z * ( 4 * a[3] + 4 * a[11] + 4 * a[17] ) +
                                        cth * ( 2 * a[25] - 4 * z * a[25] +
                                                cth * ( 2 * a[27] - 4 * z * a[27] ) ) ) ) ) ) +
        r * ( 2 * a[29] - 4 * z * a[29] +
              cth * ( 2 * a[31] - 4 * z * a[31] +
                      cth * ( 2 * a[33] - 4 * z * a[33] + cth * ( 2 * a[35] - 4 * z * a[35] ) ) ) +
              sth *
                  ( 2 * a[37] - 4 * z * a[37] +
                    cth * ( 2 * a[39] - 4 * z * a[39] +
                            cth * ( 2 * a[41] - 4 * z * a[41] +
                                    cth * ( -2 * a[37] - 2 * a[39] - 2 * a[41] +
                                            z * ( 4 * a[37] + 4 * a[39] + 4 * a[41] ) ) ) ) +
                    sth * ( 2 * a[43] - 4 * z * a[43] +
                            cth * ( 2 * a[45] - 4 * z * a[45] +
                                    cth * ( 2 * a[47] - 4 * z * a[47] +
                                            cth * ( 2 * a[49] - 4 * z * a[49] ) ) ) +
                            sth * ( 2 * a[51] - 4 * z * a[51] +
                                    cth * ( -2 * a[31] - 2 * a[39] - 2 * a[45] +
                                            z * ( 4 * a[31] + 4 * a[39] + 4 * a[45] ) +
                                            cth * ( 2 * a[53] - 4 * z * a[53] +
                                                    cth * ( 2 * a[55] - 4 * z * a[55] ) ) ) ) ) ) +
              r * ( -3 * a[29] + 6 * z * a[29] +
                    cth * ( -3 * a[31] + 6 * z * a[31] +
                            cth * ( -3 * a[33] + 6 * z * a[33] +
                                    cth * ( -3 * a[35] + 6 * z * a[35] ) ) ) +
                    sth * ( -3 * a[37] + 6 * z * a[37] +
                            cth * ( -3 * a[39] + 6 * z * a[39] +
                                    cth * ( -3 * a[41] + 6 * z * a[41] +
                                            cth * ( 3 * a[37] + 3 * a[39] +
                                                    z * ( -6 * a[37] - 6 * a[39] - 6 * a[41] ) +
                                                    3 * a[41] ) ) ) +
                            sth * ( -3 * a[43] + 6 * z * a[43] +
                                    cth * ( -3 * a[45] + 6 * z * a[45] +
                                            cth * ( -3 * a[47] + 6 * z * a[47] +
                                                    cth * ( -3 * a[49] + 6 * z * a[49] ) ) ) +
                                    sth * ( -3 * a[51] + 6 * z * a[51] +
                                            cth * ( 3 * a[31] + 3 * a[39] +
                                                    z * ( -6 * a[31] - 6 * a[39] - 6 * a[45] ) +
                                                    3 * a[45] +
                                                    cth * ( -3 * a[53] + 6 * z * a[53] +
                                                            cth * ( -3 * a[55] +
                                                                    6 * z * a[55] ) ) ) ) ) ) +
                    r * ( ( 4 * a[29] ) / 3 - ( 8 * z * a[29] ) / 3 +
                          cth * ( ( 4 * a[31] ) / 3 - ( 8 * z * a[31] ) / 3 +
                                  cth * ( ( 4 * a[33] ) / 3 - ( 8 * z * a[33] ) / 3 +
                                          cth * ( ( 4 * a[35] ) / 3 - ( 8 * z * a[35] ) / 3 ) ) ) +
                          sth *
                              ( ( 4 * a[37] ) / 3 - ( 8 * z * a[37] ) / 3 +
                                cth *
                                    ( ( 4 * a[39] ) / 3 - ( 8 * z * a[39] ) / 3 +
                                      cth * ( ( 4 * a[41] ) / 3 - ( 8 * z * a[41] ) / 3 +
                                              cth * ( ( -4 * a[37] ) / 3 - ( 4 * a[39] ) / 3 -
                                                      ( 4 * a[41] ) / 3 +
                                                      z * ( ( 8 * a[37] ) / 3 + ( 8 * a[39] ) / 3 +
                                                            ( 8 * a[41] ) / 3 ) ) ) ) +
                                sth *
                                    ( ( 4 * a[43] ) / 3 - ( 8 * z * a[43] ) / 3 +
                                      cth * ( ( 4 * a[45] ) / 3 - ( 8 * z * a[45] ) / 3 +
                                              cth * ( ( 4 * a[47] ) / 3 - ( 8 * z * a[47] ) / 3 +
                                                      cth * ( ( 4 * a[49] ) / 3 -
                                                              ( 8 * z * a[49] ) / 3 ) ) ) +
                                      sth * ( ( 4 * a[51] ) / 3 - ( 8 * z * a[51] ) / 3 +
                                              cth * ( ( -4 * a[31] ) / 3 - ( 4 * a[39] ) / 3 -
                                                      ( 4 * a[45] ) / 3 +
                                                      z * ( ( 8 * a[31] ) / 3 + ( 8 * a[39] ) / 3 +
                                                            ( 8 * a[45] ) / 3 ) +
                                                      cth * ( ( 4 * a[53] ) / 3 -
                                                              ( 8 * z * a[53] ) / 3 +
                                                              cth * ( ( 4 * a[55] ) / 3 -
                                                                      ( 8 * z * a[55] ) /
                                                                          3 ) ) ) ) ) ) ) ) ) +
        z * ( -4 * a[1] + 2 * c[2] + 2 * c[3] );
    return result;
}

static std::array<double, 10>
general_quadratic_exponential( double x, double y, double z, const ManufacturedSolution *mfs )
{
    std::array<double, 3> w    = { 0, 0, 0 };
    std::array<double, 3> darg = { 0, 0, 0 };
    const auto &h              = mfs->geth();
    const auto &hs             = mfs->geths();
    w[0]                       = x;
    w[1]                       = y;
    w[2]                       = z;
    double arg                 = 0.;
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            darg[i] += hs[i][j] * w[j];
            arg += w[i] * h[i][j] * w[j];
        }
    }
    double log10 = 2.3025850929940455;
    arg *= .5 * log10;
    double f   = exp( arg );
    double df  = log10 * f;
    double ddf = log10 * df;
    std::array<double, 10> result;
    result[0] = f;
    result[1] = darg[0] * df;
    result[2] = darg[1] * df;
    result[3] = darg[2] * df;
    result[4] = hs[0][0] * df + darg[0] * darg[0] * ddf;
    result[5] = hs[0][1] * df + darg[0] * darg[1] * ddf;
    result[6] = hs[0][2] * df + darg[0] * darg[2] * ddf;
    result[7] = hs[1][1] * df + darg[1] * darg[1] * ddf;
    result[8] = hs[1][2] * df + darg[1] * darg[2] * ddf;
    result[9] = hs[2][2] * df + darg[2] * darg[2] * ddf;
    return result;
}

static std::array<double, 10>
general_quadratic_sinusoid( double x, double y, double z, const ManufacturedSolution *mfs )
{
    std::array<double, 3> w    = { 0, 0, 0 };
    std::array<double, 3> darg = { 0, 0, 0 };
    const auto &h              = mfs->geth();
    const auto &hs             = mfs->geths();
    w[0]                       = x;
    w[1]                       = y;
    w[2]                       = z;
    double arg                 = 0.;
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            darg[i] += hs[i][j] * w[j];
            arg += w[i] * h[i][j] * w[j];
        }
    }
    arg *= .5;
    arg *= pi2;
    double f   = sin( arg );
    double df  = pi2 * cos( arg );
    double ddf = -pi2 * pi2 * f;
    std::array<double, 10> result;
    result[0] = f;
    result[1] = darg[0] * df;
    result[2] = darg[1] * df;
    result[3] = darg[2] * df;
    result[4] = hs[0][0] * df + darg[0] * darg[0] * ddf;
    result[5] = hs[0][1] * df + darg[0] * darg[1] * ddf;
    result[6] = hs[0][2] * df + darg[0] * darg[2] * ddf;
    result[7] = hs[1][1] * df + darg[1] * darg[1] * ddf;
    result[8] = hs[1][2] * df + darg[1] * darg[2] * ddf;
    result[9] = hs[2][2] * df + darg[2] * darg[2] * ddf;
    return result;
}

static std::array<double, 10> general_quadratic_exponential_sinusoid(
    double x, double y, double z, const ManufacturedSolution *mfs )
{
    std::array<double, 3> w    = { 0, 0, 0 };
    std::array<double, 3> darg = { 0, 0, 0 };
    const auto &h              = mfs->geth();
    const auto &hs             = mfs->geths();
    w[0]                       = x;
    w[1]                       = y;
    w[2]                       = z;
    double arg                 = 0.;
    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            darg[i] += hs[i][j] * w[j];
            arg += w[i] * h[i][j] * w[j];
        }
    }
    arg *= .5;
    double log10 = 2.3025850929940455;
    double s     = sin( pi2 * arg );
    double c     = cos( pi2 * arg );
    double e     = exp( log10 * arg );
    double f     = s * e;
    double df    = pi2 * c * e + s * log10 * e;
    double ddf   = -pi2 * pi2 * s * e + 2. * pi2 * c * log10 * e + s * log10 * log10 * e;
    std::array<double, 10> result;
    result[0] = f;
    result[1] = darg[0] * df;
    result[2] = darg[1] * df;
    result[3] = darg[2] * df;
    result[4] = hs[0][0] * df + darg[0] * darg[0] * ddf;
    result[5] = hs[0][1] * df + darg[0] * darg[1] * ddf;
    result[6] = hs[0][2] * df + darg[0] * darg[2] * ddf;
    result[7] = hs[1][1] * df + darg[1] * darg[1] * ddf;
    result[8] = hs[1][2] * df + darg[1] * darg[2] * ddf;
    result[9] = hs[2][2] * df + darg[2] * darg[2] * ddf;
    return result;
}


/************************************************************************
 *  ManufacturedSolution                                                 *
 ************************************************************************/
ManufacturedSolution::ManufacturedSolution( std::shared_ptr<Database> db )
    : d_c( 1 ), d_a( 1 ), d_CylindricalCoords( false )
{
    d_h.fill( { 0, 0, 0 } );
    d_hs.fill( { 0, 0, 0 } );
    d_MaximumTheta = 2. * pi;

    auto name = db->getName();
    AMP_INSIST( name == "ManufacturedSolution", "incorrect database name" );

    if ( db->keyExists( "Geometry" ) && db->keyExists( "Order" ) &&
         db->keyExists( "BoundaryType" ) ) {

        d_FunctionType = FunctionType::POLYNOMIAL;

        std::string geom   = db->getString( "Geometry" );
        std::string order  = db->getString( "Order" );
        std::string bctype = db->getString( "BoundaryType" );
        d_Name             = geom + order + bctype;

        if ( geom == "Brick" )
            d_geom = Geometry::BRICK;
        else if ( geom == "CylindricalRod" )
            d_geom = Geometry::CYLROD;
        else if ( geom == "CylindricalRodRZ" )
            d_geom = Geometry::CYLRODRZ;
        else if ( geom == "CylindricalShell" )
            d_geom = Geometry::CYLSHELL;
        else if ( geom == "CylindricalQuarterShell" )
            d_geom = Geometry::QTRCYLSHELL;
        else
            AMP_INSIST( false, "improper specification of geometry" );

        if ( order == "Quadratic" )
            d_order = Order::QUADRATIC;
        else if ( order == "Cubic" )
            d_order = Order::CUBIC;
        else
            AMP_INSIST( false, "improper specification of order" );

        if ( bctype == "Neumann" )
            d_bcType = BCType::NEUMANN;
        else if ( bctype == "Dirichlet-1" )
            d_bcType = BCType::DIRICHLET1;
        else if ( bctype == "Dirichlet-2" )
            d_bcType = BCType::DIRICHLET2;
        else if ( bctype == "Dirichlet-2-z" )
            d_bcType = BCType::DIRICHLETZ2;
        else if ( bctype == "None" )
            d_bcType = BCType::NONE;
        else
            AMP_INSIST( false, "improper specification of boundary condition type" );

        if ( d_geom == Geometry::BRICK ) {
            if ( d_order == Order::QUADRATIC ) {
                if ( d_bcType == BCType::NEUMANN ) {
                    d_functionPointer    = quad_neumann;
                    d_NumberOfParameters = 1;
                    d_NumberOfInputs     = 6;
                } else if ( d_bcType == BCType::DIRICHLET1 ) {
                    d_functionPointer    = quad_dirichlet1;
                    d_NumberOfParameters = 1;
                    d_NumberOfInputs     = 2;
                } else if ( d_bcType == BCType::DIRICHLET2 ) {
                    d_functionPointer    = quad_dirichlet2;
                    d_NumberOfParameters = 1;
                    d_NumberOfInputs     = 2;
                } else if ( d_bcType == BCType::NONE ) {
                    d_functionPointer    = quad_none;
                    d_NumberOfParameters = 10;
                    d_NumberOfInputs     = 0;
                } else {
                    AMP_INSIST( false, "manufactured solution combination is not available" );
                }
            } else if ( d_order == Order::CUBIC ) {
                if ( d_bcType == BCType::NEUMANN ) {
                    d_functionPointer    = cubic_neumann;
                    d_NumberOfParameters = 8;
                    d_NumberOfInputs     = 6;
                } else if ( d_bcType == BCType::DIRICHLET1 ) {
                    d_functionPointer    = cubic_dirichlet1;
                    d_NumberOfParameters = 8;
                    d_NumberOfInputs     = 2;
                } else if ( d_bcType == BCType::DIRICHLET2 ) {
                    d_functionPointer    = cubic_dirichlet2;
                    d_NumberOfParameters = 8;
                    d_NumberOfInputs     = 2;
                } else if ( d_bcType == BCType::NONE ) {
                    d_functionPointer    = cubic_none;
                    d_NumberOfParameters = 20;
                    d_NumberOfInputs     = 0;
                } else {
                    AMP_INSIST( false, "manufactured solution combination is not available" );
                }
            }
        } else if ( d_geom == Geometry::CYLROD ) {
            if ( d_order == Order::QUADRATIC ) {
                if ( d_bcType == BCType::NONE ) {
                    d_functionPointer    = quad_cyl_rod_none;
                    d_NumberOfParameters = 15;
                    d_NumberOfInputs     = 0;
                } else {
                    AMP_INSIST( false, "manufactured solution combination is not available" );
                }
            } else if ( d_order == Order::CUBIC ) {
                if ( d_bcType == BCType::DIRICHLETZ2 ) {
                    d_functionPointer    = cubic_cyl_rod_dirichletz2;
                    d_NumberOfParameters = 96;
                    d_NumberOfInputs     = 2;
                } else if ( d_bcType == BCType::NONE ) {
                    d_functionPointer    = cubic_cyl_rod_none;
                    d_NumberOfParameters = 64;
                    d_NumberOfInputs     = 0;
                } else {
                    AMP_INSIST( false, "manufactured solution combination is not available" );
                }
            } else {
                AMP_INSIST( false, "manufactured solution combination is not available" );
            }
            d_CylindricalCoords = true;
        } else if ( d_geom == Geometry::CYLRODRZ ) {
            if ( d_bcType == BCType::NONE ) {
                d_functionPointer    = cubic_cyl_rod_rz_none;
                d_NumberOfParameters = 16;
                d_NumberOfInputs     = 0;
            } else {
                AMP_INSIST( false, "manufactured solution combination is not available" );
            }
            d_CylindricalCoords = true;
        } else if ( d_geom == Geometry::CYLSHELL ) {
            if ( d_order == Order::QUADRATIC ) {
                if ( d_bcType == BCType::NEUMANN ) {
                    d_functionPointer    = quad_cyl_shell_neumann;
                    d_NumberOfParameters = 9;
                    d_NumberOfInputs     = 4;
                } else if ( d_bcType == BCType::NONE ) {
                    d_functionPointer =
                        quad_cyl_rod_none; // TODO: This should be: quad_cyl_shell_none; once that
                                           // function is completed.
                    d_NumberOfParameters = 15;
                    d_NumberOfInputs     = 0;
                } else {
                    AMP_INSIST( false, "manufactured solution combination is not available" );
                }
            } else if ( d_order == Order::CUBIC ) {
                if ( d_bcType == BCType::NONE ) {
                    d_functionPointer    = cubic_cyl_shell_neumann;
                    d_NumberOfParameters = 35;
                    d_NumberOfInputs     = 0;
                }
            } else {
                AMP_INSIST( false, "manufactured solution combination is not available" );
            }
            d_CylindricalCoords = true;
        } else if ( d_geom == Geometry::QTRCYLSHELL ) {
            if ( d_order == Order::QUADRATIC ) {
                if ( d_bcType == BCType::NEUMANN ) {
                    d_functionPointer    = quad_cyl_qtr_shell_neumann;
                    d_NumberOfParameters = 7;
                    d_NumberOfInputs     = 4;
                } else if ( d_bcType == BCType::DIRICHLET2 ) {
                    d_functionPointer    = quad_cyl_qtr_shell_dirichlet2;
                    d_NumberOfParameters = 7;
                    d_NumberOfInputs     = 4;
                } else if ( d_bcType == BCType::NONE ) {
                    d_functionPointer =
                        cubic_cyl_rod_none; // TODO: This should be: &quad_cyl_qtr_shell_none; once
                                            // that function is completed.
                    d_NumberOfParameters = 20;
                    d_NumberOfInputs     = 0;
                } else {
                    AMP_INSIST( false, "manufactured solution combination is not available" );
                }
            } else if ( d_order == Order::CUBIC ) {
                d_functionPointer    = cubic_cyl_qtr_shell_neumann;
                d_NumberOfParameters = 56;
                d_NumberOfInputs     = 4;
                if ( d_bcType == BCType::NEUMANN ) {
                } else if ( d_bcType == BCType::NONE ) {
                    d_functionPointer = cubic_cyl_rod_none; // TODO: This should be:
                                                            // &cubic_cyl_qtr_shell_none; once
                                                            // that function is completed.
                    d_NumberOfParameters = 20;
                    d_NumberOfInputs     = 0;
                } else {
                    AMP_INSIST( false, "manufactured solution combination is not available" );
                }
            }
            d_CylindricalCoords = true;
            d_MaximumTheta      = pi / 2.;
        }

        d_c.resize( d_NumberOfInputs );
        d_a.resize( d_NumberOfParameters );

        bool hasCoefficients = false, hasBoundaryData = false;
        if ( db->keyExists( "Coefficients" ) ) {
            hasCoefficients = true;
        }
        if ( db->keyExists( "BoundaryData" ) ) {
            hasBoundaryData = true;
        }
        if ( hasCoefficients && ( hasBoundaryData || ( d_bcType == BCType::NONE ) ) ) {
            auto array = db->getVector<double>( "Coefficients" );
            AMP_INSIST( array.size() >= d_NumberOfParameters, "wrong number of coefficients" );
            d_a.resize( d_NumberOfParameters );
            for ( size_t i = 0; i < d_NumberOfParameters; i++ )
                d_a[i] = array[i];
        }
        if ( hasCoefficients && hasBoundaryData ) {
            auto array = db->getVector<double>( "BoundaryData" );
            AMP_INSIST( array.size() >= d_NumberOfInputs, "wrong number of boundary data" );
            d_c.resize( d_NumberOfInputs );
            for ( size_t i = 0; i < d_NumberOfInputs; i++ )
                d_c[i] = array[i];
        }
    } else if ( db->keyExists( "QuadraticDistortion" ) && db->keyExists( "QuadraticFunction" ) ) {
        d_FunctionType       = FunctionType::GENERALQUADRATIC;
        std::string function = db->getString( "QuadraticFunction" );
        if ( function == "Exponential" )
            d_functionPointer = general_quadratic_exponential;
        else if ( function == "Sinusoid" )
            d_functionPointer = general_quadratic_sinusoid;
        else if ( function == "ExponentialSinusoid" )
            d_functionPointer = general_quadratic_exponential_sinusoid;
        else
            AMP_INSIST( false, "invalid value for QuadraticFunction" );
        auto distortion = db->getVector<double>( "QuadraticDistortion" );
        AMP_INSIST( distortion.size() == 9, "distortion array must have 9 elements" );
        for ( int i = 0; i < 3; i++ )
            for ( int j = 0; j < 3; j++ ) {
                d_h[i][j]  = distortion[i * 3 + j];
                d_hs[i][j] = .5 * ( distortion[i * 3 + j] + distortion[j * 3 + i] );
            }
        d_Name = "QuadraticDistortion" + function;
    } else {
        AMP_INSIST( false, "unrecognized manufactured solution type or missing keys" );
    }

    if ( !d_CylindricalCoords ) {
        d_MinX = db->getWithDefault<double>( "MinX", 0. );
        d_MinY = db->getWithDefault<double>( "MinY", 0. );
        d_MinZ = db->getWithDefault<double>( "MinZ", 0. );
        d_MaxX = db->getWithDefault<double>( "MaxX", 1. );
        d_MaxY = db->getWithDefault<double>( "MaxY", 1. );
        d_MaxZ = db->getWithDefault<double>( "MaxZ", 1. );
        AMP_ASSERT( d_MaxX >= d_MinX );
        AMP_ASSERT( d_MaxY >= d_MinY );
        AMP_ASSERT( d_MaxZ >= d_MinZ );
        d_ScaleX = 1. / ( d_MaxX - d_MinX + std::numeric_limits<double>::epsilon() );
        d_ScaleY = 1. / ( d_MaxY - d_MinY + std::numeric_limits<double>::epsilon() );
        d_ScaleZ = 1. / ( d_MaxZ - d_MinZ + std::numeric_limits<double>::epsilon() );
    } else {
        d_MinR  = db->getWithDefault<double>( "MinR", 0. );
        d_MinTh = 0.;
        d_MinZ  = db->getWithDefault<double>( "MinZ", 0. );
        d_MaxR  = db->getWithDefault<double>( "MaxR", 1. );
        d_MaxTh = 2. * pi;
        d_MaxZ  = db->getWithDefault<double>( "MaxZ", 1. );
        if ( d_MaximumTheta < 2. * pi ) {
            d_MinTh = db->getWithDefault<double>( "MinTh", 0. );
            d_MaxTh = db->getWithDefault<double>( "MaxTh", 2. * pi );
        }
        d_ScaleR  = 1. / ( d_MaxR - d_MinR + std::numeric_limits<double>::epsilon() );
        d_ScaleTh = 1. / ( d_MaxTh - d_MinTh + std::numeric_limits<double>::epsilon() );
        d_ScaleZ  = 1. / ( d_MaxZ - d_MinZ + std::numeric_limits<double>::epsilon() );
    }
}

std::array<double, 10> ManufacturedSolution::evaluate( double x, double y, double z ) const
{
    if ( !d_CylindricalCoords ) {
        AMP_ASSERT( x >= d_MinX && x <= d_MaxX );
        AMP_ASSERT( y >= d_MinY && y <= d_MaxY );
        AMP_ASSERT( z >= d_MinZ && z <= d_MaxZ );
        double xs, ys, zs;
        xs = ( x - d_MinX ) * d_ScaleX;
        ys = ( y - d_MinY ) * d_ScaleY;
        zs = ( z - d_MinZ ) * d_ScaleZ;

        auto result = d_functionPointer( xs, ys, zs, this );
        result[1] *= d_ScaleX;
        result[2] *= d_ScaleY;
        result[3] *= d_ScaleZ;
        result[4] *= d_ScaleX * d_ScaleX;
        result[5] *= d_ScaleX * d_ScaleY;
        result[6] *= d_ScaleX * d_ScaleZ;
        result[7] *= d_ScaleY * d_ScaleY;
        result[8] *= d_ScaleY * d_ScaleZ;
        result[9] *= d_ScaleZ * d_ScaleZ;
        return result;
    } else {
        double r = x, th = y;
        AMP_ASSERT( r >= d_MinR && r <= d_MaxR );
        AMP_ASSERT( th >= d_MinTh && th <= d_MaxTh );
        AMP_ASSERT( z >= d_MinZ && z <= d_MaxZ );
        double rs, ths = th, zs;
        rs = ( r - d_MinR ) * d_ScaleR;
        zs = ( z - d_MinZ ) * d_ScaleZ;
        if ( d_MaximumTheta < 2. * pi )
            ths = ( th - d_MinTh ) * d_ScaleTh;

        auto result = d_functionPointer( rs, ths, zs, this );
        result[1] *= d_ScaleR;
        result[2] *= d_ScaleTh;
        result[3] *= d_ScaleZ;
        result[4] *= d_ScaleR * d_ScaleR;
        result[5] *= d_ScaleR * d_ScaleTh;
        result[6] *= d_ScaleR * d_ScaleZ;
        result[7] *= d_ScaleTh * d_ScaleTh;
        result[8] *= d_ScaleTh * d_ScaleZ;
        result[9] *= d_ScaleZ * d_ScaleZ;
        return result;
    }
}


} // namespace AMP
