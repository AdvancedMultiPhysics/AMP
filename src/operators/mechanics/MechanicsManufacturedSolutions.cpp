#include "AMP/operators/mechanics/MechanicsManufacturedSolutions.h"
#include "AMP/IO/PIO.h"
#include "AMP/utils/Constants.h"


constexpr double pi = AMP::Constants::pi;


namespace AMP::MechanicsManufacturedSolution {


// Default constructor
MMS::MMS( double YM, double PR )
    : E( YM ),
      nu( PR ),
      Lx( 1.0 ),
      Ly( 1.0 ),
      Lz( 1.0 ),
      Mx( 1.0 ),
      My( 1.0 ),
      Mz( 1.0 ),
      axx( 0.0 ),
      bxx( 1.0 ),
      axy( 0.0 ),
      bxy( 1.0 ),
      axz( 0.0 ),
      bxz( 1.0 ),
      ayx( 0.0 ),
      byx( 1.0 ),
      ayy( 0.0 ),
      byy( 1.0 ),
      ayz( 0.0 ),
      byz( 1.0 ),
      azx( 0.0 ),
      bzx( 1.0 ),
      azy( 0.0 ),
      bzy( 1.0 ),
      azz( 0.0 ),
      bzz( 1.0 ),
      name( "unnamed" )
{
}


// MMSLinear
double MMSLinear::getExactSolutionX( double x, double y, double z ) const
{
    return Mx * ( bxx + ( axx * x ) / Lx ) * ( bxy + ( axy * y ) / Ly ) *
           ( bxz + ( axz * z ) / Lz );
}
double MMSLinear::getExactSolutionY( double x, double y, double z ) const
{
    return My * ( byx + ( ayx * x ) / Lx ) * ( byy + ( ayy * y ) / Ly ) *
           ( byz + ( ayz * z ) / Lz );
}
double MMSLinear::getExactSolutionZ( double x, double y, double z ) const
{
    return Mz * ( bzx + ( azx * x ) / Lx ) * ( bzy + ( azy * y ) / Ly ) *
           ( bzz + ( azz * z ) / Lz );
}
double MMSLinear::getForcingTermX( double /* x */, double y, double z ) const
{
    return ( ( Mz * azx * azz * ( bzy + ( azy * y ) / Ly ) ) / ( Lx * Lz ) +
             ( My * ayx * ayy * ( byz + ( ayz * z ) / Lz ) ) / ( Lx * Ly ) ) *
               ( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) -
           ( E * Mz * azx * azz * ( bzy + ( azy * y ) / Ly ) ) / ( Lx * Lz * ( nu * 2.0 + 2.0 ) ) -
           ( E * My * ayx * ayy * ( byz + ( ayz * z ) / Lz ) ) / ( Lx * Ly * ( nu * 2.0 + 2.0 ) );
}
double MMSLinear::getForcingTermY( double x, double /* y */, double z ) const
{
    return ( ( Mz * azy * azz * ( bzx + ( azx * x ) / Lx ) ) / ( Ly * Lz ) +
             ( Mx * axx * axy * ( bxz + ( axz * z ) / Lz ) ) / ( Lx * Ly ) ) *
               ( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) -
           ( E * Mz * azy * azz * ( bzx + ( azx * x ) / Lx ) ) / ( Ly * Lz * ( nu * 2.0 + 2.0 ) ) -
           ( E * Mx * axx * axy * ( bxz + ( axz * z ) / Lz ) ) / ( Lx * Ly * ( nu * 2.0 + 2.0 ) );
}
double MMSLinear::getForcingTermZ( double x, double y, double /* z */ ) const
{
    return ( ( My * ayy * ayz * ( byx + ( ayx * x ) / Lx ) ) / ( Ly * Lz ) +
             ( Mx * axx * axz * ( bxy + ( axy * y ) / Ly ) ) / ( Lx * Lz ) ) *
               ( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) -
           ( E * My * ayy * ayz * ( byx + ( ayx * x ) / Lx ) ) / ( Ly * Lz * ( nu * 2.0 + 2.0 ) ) -
           ( E * Mx * axx * axz * ( bxy + ( axy * y ) / Ly ) ) / ( Lx * Lz * ( nu * 2.0 + 2.0 ) );
}
std::vector<double> MMSLinear::getStressTensor( double x, double y, double z ) const
{
    std::vector<double> stress;
    stress.push_back(
        -( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) *
            ( ( Mz * azz * ( bzx + ( azx * x ) / Lx ) * ( bzy + ( azy * y ) / Ly ) ) / Lz +
              ( My * ayy * ( byx + ( ayx * x ) / Lx ) * ( byz + ( ayz * z ) / Lz ) ) / Ly +
              ( Mx * axx * ( bxy + ( axy * y ) / Ly ) * ( bxz + ( axz * z ) / Lz ) ) / Lx ) +
        ( E * Mx * axx * ( bxy + ( axy * y ) / Ly ) * ( bxz + ( axz * z ) / Lz ) * 2.0 ) /
            ( Lx * ( nu * 2.0 + 2.0 ) ) );
    stress.push_back(
        -( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) *
            ( ( Mz * azz * ( bzx + ( azx * x ) / Lx ) * ( bzy + ( azy * y ) / Ly ) ) / Lz +
              ( My * ayy * ( byx + ( ayx * x ) / Lx ) * ( byz + ( ayz * z ) / Lz ) ) / Ly +
              ( Mx * axx * ( bxy + ( axy * y ) / Ly ) * ( bxz + ( axz * z ) / Lz ) ) / Lx ) +
        ( E * My * ayy * ( byx + ( ayx * x ) / Lx ) * ( byz + ( ayz * z ) / Lz ) * 2.0 ) /
            ( Ly * ( nu * 2.0 + 2.0 ) ) );
    stress.push_back(
        -( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) *
            ( ( Mz * azz * ( bzx + ( azx * x ) / Lx ) * ( bzy + ( azy * y ) / Ly ) ) / Lz +
              ( My * ayy * ( byx + ( ayx * x ) / Lx ) * ( byz + ( ayz * z ) / Lz ) ) / Ly +
              ( Mx * axx * ( bxy + ( axy * y ) / Ly ) * ( bxz + ( axz * z ) / Lz ) ) / Lx ) +
        ( E * Mz * azz * ( bzx + ( azx * x ) / Lx ) * ( bzy + ( azy * y ) / Ly ) * 2.0 ) /
            ( Lz * ( nu * 2.0 + 2.0 ) ) );
    stress.push_back(
        ( E *
          ( ( My * ayz * ( byx + ( ayx * x ) / Lx ) * ( byy + ( ayy * y ) / Ly ) * 0.5 ) / Lz +
            ( Mz * azy * ( bzx + ( azx * x ) / Lx ) * ( bzz + ( azz * z ) / Lz ) * 0.5 ) / Ly ) *
          2.0 ) /
        ( nu * 2.0 + 2.0 ) );
    stress.push_back(
        ( E *
          ( ( Mx * axz * ( bxx + ( axx * x ) / Lx ) * ( bxy + ( axy * y ) / Ly ) * 0.5 ) / Lz +
            ( Mz * azx * ( bzy + ( azy * y ) / Ly ) * ( bzz + ( azz * z ) / Lz ) * 0.5 ) / Lx ) *
          2.0 ) /
        ( nu * 2.0 + 2.0 ) );
    stress.push_back(
        ( E *
          ( ( Mx * axy * ( bxx + ( axx * x ) / Lx ) * ( bxz + ( axz * z ) / Lz ) * 0.5 ) / Ly +
            ( My * ayx * ( byy + ( ayy * y ) / Ly ) * ( byz + ( ayz * z ) / Lz ) * 0.5 ) / Lx ) *
          2.0 ) /
        ( nu * 2.0 + 2.0 ) );
    return stress;
}


// MMSTrigonometric
double MMSTrigonometric::getExactSolutionX( double x, double y, double z ) const
{
    return Mx * sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
           sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
           sin( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 );
}
double MMSTrigonometric::getExactSolutionY( double /* x */, double y, double z ) const
{
    return My * sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
           sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
           sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 );
}
double MMSTrigonometric::getExactSolutionZ( double /* x */, double y, double z ) const
{
    return Mz * sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
           sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
           sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 );
}
double MMSTrigonometric::getForcingTermX( double x, double y, double z ) const
{
    return 1 / ( Lx * Lx ) * ( pi * pi ) * Mx * ( axx * axx ) *
               sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
               sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
               sin( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) *
               ( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) * ( -0.25 ) +
           ( E * 1 / ( Lx * Lx ) * ( pi * pi ) * Mx * ( axx * axx ) *
             sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
             sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
             sin( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.5 ) /
               ( nu * 2.0 + 2.0 ) +
           ( E * 1 / ( Ly * Ly ) * ( pi * pi * pi * pi ) * Mx * ( axy * axy ) *
             sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
             sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
             sin( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.25 ) /
               ( nu * 2.0 + 2.0 ) +
           ( E * 1 / ( Lz * Lz ) * ( pi * pi * pi * pi ) * Mx * ( axz * axz ) *
             sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
             sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
             sin( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.25 ) /
               ( nu * 2.0 + 2.0 );
}
double MMSTrigonometric::getForcingTermY( double x, double y, double z ) const
{
    return ( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) *
               ( 1 / ( Ly * Ly ) * ( pi * pi ) * My * ( ayx * ayx ) *
                     sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                     sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                     sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * ( -0.25 ) -
                 1 / ( Ly * Ly ) * ( pi * pi * pi * pi ) * My * ( ayy * ayy ) *
                     sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                     sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                     sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.25 +
                 1 / ( Ly * Ly ) * ( pi * pi * pi ) * My * ayx * ayy *
                     cos( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                     sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                     cos( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 +
                 ( ( pi * pi * pi ) * Mx * axx * axy *
                   cos( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
                   cos( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.25 ) /
                     ( Lx * Ly ) +
                 ( ( pi * pi * pi ) * Mz * azx * azy *
                   cos( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                   cos( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.25 ) /
                     ( Ly * Lz ) +
                 ( ( pi * pi * pi * pi ) * Mz * azy * azz *
                   cos( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                   cos( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.25 ) /
                     ( Ly * Lz ) ) +
           ( E *
             ( 1 / ( Ly * Ly ) * ( pi * pi ) * My * ( ayx * ayx ) *
                   sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.25 +
               1 / ( Ly * Ly ) * ( pi * pi * pi * pi ) * My * ( ayy * ayy ) *
                   sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.25 -
               1 / ( Ly * Ly ) * ( pi * pi * pi ) * My * ayx * ayy *
                   cos( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                   cos( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) *
             2.0 ) /
               ( nu * 2.0 + 2.0 ) -
           ( E *
             ( 1 / ( Lz * Lz ) * ( pi * pi * pi * pi ) * My * ( ayz * ayz ) *
                   sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * ( -1.0 / 8.0 ) +
               ( ( pi * pi * pi ) * Mz * azx * azy *
                 cos( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                 sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                 cos( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * ( 1.0 / 8.0 ) ) /
                   ( Ly * Lz ) +
               ( ( pi * pi * pi * pi ) * Mz * azy * azz *
                 cos( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                 cos( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                 sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * ( 1.0 / 8.0 ) ) /
                   ( Ly * Lz ) ) *
             2.0 ) /
               ( nu * 2.0 + 2.0 ) -
           ( E * ( pi * pi * pi ) * Mx * axx * axy *
             cos( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
             sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
             cos( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.25 ) /
               ( Lx * Ly * ( nu * 2.0 + 2.0 ) );
}
double MMSTrigonometric::getForcingTermZ( double x, double y, double z ) const
{
    return ( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) *
               ( 1 / ( Lz * Lz ) * ( pi * pi ) * Mz * ( azx * azx ) *
                     sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                     sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                     sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * ( -0.25 ) -
                 1 / ( Lz * Lz ) * ( pi * pi * pi * pi ) * Mz * ( azz * azz ) *
                     sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                     sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                     sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.25 +
                 1 / ( Lz * Lz ) * ( pi * pi * pi ) * Mz * azx * azz *
                     cos( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                     sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                     cos( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 +
                 ( ( pi * pi * pi ) * Mx * axx * axz *
                   cos( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
                   cos( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.25 ) /
                     ( Lx * Lz ) +
                 ( ( pi * pi * pi ) * My * ayx * ayz *
                   cos( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                   cos( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.25 ) /
                     ( Ly * Lz ) +
                 ( ( pi * pi * pi * pi ) * My * ayy * ayz *
                   cos( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                   cos( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.25 ) /
                     ( Ly * Lz ) ) +
           ( E *
             ( 1 / ( Lz * Lz ) * ( pi * pi ) * Mz * ( azx * azx ) *
                   sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.25 +
               1 / ( Lz * Lz ) * ( pi * pi * pi * pi ) * Mz * ( azz * azz ) *
                   sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.25 -
               1 / ( Lz * Lz ) * ( pi * pi * pi ) * Mz * azx * azz *
                   cos( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                   cos( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) *
             2.0 ) /
               ( nu * 2.0 + 2.0 ) -
           ( E *
             ( 1 / ( Ly * Ly ) * ( pi * pi * pi * pi ) * Mz * ( azy * azy ) *
                   sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                   sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                   sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * ( -1.0 / 8.0 ) +
               ( ( pi * pi * pi ) * My * ayx * ayz *
                 cos( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                 sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                 cos( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * ( 1.0 / 8.0 ) ) /
                   ( Ly * Lz ) +
               ( ( pi * pi * pi * pi ) * My * ayy * ayz *
                 cos( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                 cos( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                 sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * ( 1.0 / 8.0 ) ) /
                   ( Ly * Lz ) ) *
             2.0 ) /
               ( nu * 2.0 + 2.0 ) -
           ( E * ( pi * pi * pi ) * Mx * axx * axz *
             cos( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
             sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
             cos( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.25 ) /
               ( Lx * Lz * ( nu * 2.0 + 2.0 ) );
}
std::vector<double> MMSTrigonometric::getStressTensor( double x, double y, double z ) const
{
    std::vector<double> stress;
    stress.push_back(
        -( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) *
            ( ( ( pi * pi ) * My * ayy * cos( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) /
                  Ly +
              ( ( pi * pi ) * Mz * azz * cos( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) /
                  Lz +
              ( pi * Mx * axx * sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.5 ) /
                  Lx +
              ( pi * My * ayx * sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) /
                  Ly +
              ( pi * Mz * azx * sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) /
                  Lz ) +
        ( E * pi * Mx * axx * sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
          sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
          cos( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) ) /
            ( Lx * ( nu * 2.0 + 2.0 ) ) );
    stress.push_back(
        -( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) *
            ( ( ( pi * pi ) * My * ayy * cos( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) /
                  Ly +
              ( ( pi * pi ) * Mz * azz * cos( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) /
                  Lz +
              ( pi * Mx * axx * sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.5 ) /
                  Lx +
              ( pi * My * ayx * sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) /
                  Ly +
              ( pi * Mz * azx * sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) /
                  Lz ) +
        ( E *
          ( ( ( pi * pi ) * My * ayy * cos( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
              sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
              sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) /
                Ly +
            ( pi * My * ayx * sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
              sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
              cos( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) /
                Ly ) *
          2.0 ) /
            ( nu * 2.0 + 2.0 ) );
    stress.push_back(
        -( ( E * ( 2.0 / 3.0 ) ) / ( nu * 2.0 + 2.0 ) + E / ( nu * 6.0 - 3.0 ) ) *
            ( ( ( pi * pi ) * My * ayy * cos( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) /
                  Ly +
              ( ( pi * pi ) * Mz * azz * cos( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) /
                  Lz +
              ( pi * Mx * axx * sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.5 ) /
                  Lx +
              ( pi * My * ayx * sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.5 ) /
                  Ly +
              ( pi * Mz * azx * sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
                sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
                cos( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) /
                  Lz ) +
        ( E *
          ( ( ( pi * pi ) * Mz * azz * cos( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
              sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
              sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) /
                Lz +
            ( pi * Mz * azx * sin( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
              sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
              cos( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.5 ) /
                Lz ) *
          2.0 ) /
            ( nu * 2.0 + 2.0 ) );
    stress.push_back(
        ( E *
          ( ( ( pi * pi ) * My * ayz * cos( pi * ( byz + ( pi * ayz * z ) / Lz ) * 0.5 ) *
              sin( pi * ( byy + ( pi * ayy * y ) / Ly ) * 0.5 ) *
              sin( pi * ( byx + ( ayx * y ) / Ly ) * 0.5 ) * 0.25 ) /
                Lz +
            ( ( pi * pi ) * Mz * azy * cos( pi * ( bzy + ( pi * azy * y ) / Ly ) * 0.5 ) *
              sin( pi * ( bzz + ( pi * azz * z ) / Lz ) * 0.5 ) *
              sin( pi * ( bzx + ( azx * z ) / Lz ) * 0.5 ) * 0.25 ) /
                Ly ) *
          2.0 ) /
        ( nu * 2.0 + 2.0 ) );
    stress.push_back(
        ( E * ( pi * pi ) * Mx * axz * cos( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
          sin( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
          sin( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.5 ) /
        ( Lz * ( nu * 2.0 + 2.0 ) ) );
    stress.push_back(
        ( E * ( pi * pi ) * Mx * axy * cos( pi * ( bxy + ( pi * axy * y ) / Ly ) * 0.5 ) *
          sin( pi * ( bxz + ( pi * axz * z ) / Lz ) * 0.5 ) *
          sin( pi * ( bxx + ( axx * x ) / Lx ) * 0.5 ) * 0.5 ) /
        ( Ly * ( nu * 2.0 + 2.0 ) ) );
    return stress;
}


// MMSBuilder
std::shared_ptr<MMS> MMSBuilder::createMMS( std::shared_ptr<AMP::Database> mmsDatabase )
{
    std::shared_ptr<MMS> mms;
    auto name = mmsDatabase->getWithDefault<std::string>( "name", "One" );
    if ( name == "Trigonometric" ) {
        mms = std::shared_ptr<MMS>( new MMSTrigonometric );
    } else if ( name == "Linear" ) {
        mms = std::shared_ptr<MMS>( new MMSLinear );
    } else {
        std::cerr << "Error: AMP::MechanicsManufacturedSolution::MMS" << name << " is not defined"
                  << std::endl;
        AMP_ASSERT( false );
    }
    if ( mmsDatabase->keyExists( "scale_x" ) ) {
        mms->scaleX( mmsDatabase->getScalar<double>( "scale_x" ) );
    }
    if ( mmsDatabase->keyExists( "scale_y" ) ) {
        mms->scaleY( mmsDatabase->getScalar<double>( "scale_y" ) );
    }
    if ( mmsDatabase->keyExists( "scale_z" ) ) {
        mms->scaleZ( mmsDatabase->getScalar<double>( "scale_z" ) );
    }
    if ( mmsDatabase->keyExists( "scale_xyz" ) ) {
        mms->scaleXYZ( mmsDatabase->getScalar<double>( "scale_xyz" ) );
    }
    if ( mmsDatabase->keyExists( "mult_x" ) ) {
        mms->multX( mmsDatabase->getScalar<double>( "mult_x" ) );
    }
    if ( mmsDatabase->keyExists( "mult_y" ) ) {
        mms->multY( mmsDatabase->getScalar<double>( "mult_y" ) );
    }
    if ( mmsDatabase->keyExists( "mult_z" ) ) {
        mms->multZ( mmsDatabase->getScalar<double>( "mult_z" ) );
    }
    if ( mmsDatabase->keyExists( "mult_xyz" ) ) {
        mms->multXYZ( mmsDatabase->getScalar<double>( "mult_xyz" ) );
    }
    return mms;
}


} // namespace AMP::MechanicsManufacturedSolution
