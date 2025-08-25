#include "AMP/operators/diffusionFD/DiffusionRotatedAnisotropicModel.h"
#include "AMP/utils/Constants.h"

namespace AMP::Operator {

#define PI AMP::Constants::pi

/* ----------------------------------------------------------------------------------
    Implementation of a rotated anisotropic constant-coefficient diffusion equation
---------------------------------------------------------------------------------- */
RotatedAnisotropicDiffusionModel::RotatedAnisotropicDiffusionModel(
    std::shared_ptr<AMP::Database> input_db )
    : d_input_db( input_db )
{

    // Do some error checking on the input
    AMP_INSIST( d_input_db, "Non-null input database required" );
    AMP_INSIST( d_input_db->keyExists( "dim" ), "Key ''d_dim'' is missing!" );
    d_dim = d_input_db->getScalar<size_t>( "dim" );

    AMP_INSIST( d_dim == 1 || d_dim == 2 || d_dim == 3,
                "Invalid dimension: dim=" + std::to_string( d_dim ) +
                    std::string( " !in {1,2,3}" ) );

    // Set PDE coefficients
    setDiffusionCoefficients();
}

void RotatedAnisotropicDiffusionModel::setDiffusionCoefficients()
{
    d_c_db = std::make_shared<AMP::Database>( "DiffusionCoefficients" );
    if ( d_dim == 1 ) {
        auto c = getSecondOrderPDECoefficients1D();
        d_c_db->putScalar<double>( "cxx", c[0] );
    } else if ( d_dim == 2 ) {
        auto c = getSecondOrderPDECoefficients2D();
        d_c_db->putScalar<double>( "cxx", c[0] );
        d_c_db->putScalar<double>( "cyy", c[1] );
        d_c_db->putScalar<double>( "cyx", c[2] );
    } else if ( d_dim == 3 ) {
        auto c = getSecondOrderPDECoefficients3D();
        d_c_db->putScalar<double>( "cxx", c[0] );
        d_c_db->putScalar<double>( "cyy", c[1] );
        d_c_db->putScalar<double>( "czz", c[2] );
        d_c_db->putScalar<double>( "cyx", c[3] );
        d_c_db->putScalar<double>( "czx", c[4] );
        d_c_db->putScalar<double>( "czy", c[5] );
    }
}

std::vector<double> RotatedAnisotropicDiffusionModel::getSecondOrderPDECoefficients1D() const
{
    // PDE coefficients
    double cxx            = 1.0;
    std::vector<double> c = { cxx };
    return c;
}

std::vector<double> RotatedAnisotropicDiffusionModel::getSecondOrderPDECoefficients2D() const
{
    // Default constants correspond to the isotropic case
    auto eps   = d_input_db->getWithDefault<double>( "eps", 1.0 );
    auto theta = d_input_db->getWithDefault<double>( "theta", 0.0 );

    // Trig functions of angle
    double cth = cos( theta );
    double sth = sin( theta );
    // Diffusion tensor coefficients
    double d11 = std::pow( cth, 2 ) + eps * std::pow( sth, 2 );
    double d22 = std::pow( cth, 2 ) * eps + std::pow( sth, 2 );
    double d12 = cth * sth * ( 1 - eps );
    // PDE coefficients
    double cxx            = d11;
    double cyy            = d22;
    double cyx            = 2.0 * d12;
    std::vector<double> c = { cxx, cyy, cyx };
    return c;
}


std::vector<double> RotatedAnisotropicDiffusionModel::getSecondOrderPDECoefficients3D() const
{
    // Default constants correspond to the isotropic case
    auto epsy  = d_input_db->getWithDefault<double>( "epsy", 1.0 );
    auto epsz  = d_input_db->getWithDefault<double>( "epsz", 1.0 );
    auto alpha = d_input_db->getWithDefault<double>( "alpha", 0.0 );
    auto beta  = d_input_db->getWithDefault<double>( "beta", 0.0 );
    auto gamma = d_input_db->getWithDefault<double>( "gamma", 0.0 );

    // Trig functions of angles
    double ca = cos( alpha );
    double sa = sin( alpha );
    double cb = cos( beta );
    double sb = sin( beta );
    double cg = cos( gamma );
    double sg = sin( gamma );
    // Diffusion tensor coefficients
    double d11 = epsy * std::pow( ca * sg + cb * cg * sa, 2 ) +
                 epsz * std::pow( sa, 2 ) * std::pow( sb, 2 ) +
                 std::pow( ca * cg - cb * sa * sg, 2 );
    double d22 = std::pow( ca, 2 ) * epsz * std::pow( sb, 2 ) +
                 epsy * std::pow( ca * cb * cg - sa * sg, 2 ) +
                 std::pow( ca * cb * sg + cg * sa, 2 );
    double d33 = std::pow( cb, 2 ) * epsz + std::pow( cg, 2 ) * epsy * std::pow( sb, 2 ) +
                 std::pow( sb, 2 ) * std::pow( sg, 2 );
    double d12 = -ca * epsz * sa * std::pow( sb, 2 ) -
                 epsy * ( ca * sg + cb * cg * sa ) * ( ca * cb * cg - sa * sg ) +
                 ( ca * cg - cb * sa * sg ) * ( ca * cb * sg + cg * sa );
    double d13 = sb * ( cb * epsz * sa - cg * epsy * ( ca * sg + cb * cg * sa ) +
                        sg * ( ca * cg - cb * sa * sg ) );
    double d23 = sb * ( -ca * cb * epsz + cg * epsy * ( ca * cb * cg - sa * sg ) +
                        sg * ( ca * cb * sg + cg * sa ) );
    // PDE coefficients
    double cxx            = d11;
    double cyy            = d22;
    double czz            = d33;
    double cyx            = 2.0 * d12;
    double czx            = 2.0 * d13;
    double czy            = 2.0 * d23;
    std::vector<double> c = { cxx, cyy, czz, cyx, czx, czy };
    return c;
}


/* ----------------------------------------------------------------------------------------------
    Implementation of a MANUFACTURED rotated anisotropic constant-coefficient diffusion equation
---------------------------------------------------------------------------------------------- */
// Implementation of pure virtual function
// Dimension-agnostic wrapper around the exact source term functions
double
ManufacturedRotatedAnisotropicDiffusionModel::sourceTerm( AMP::Mesh::MeshElement &node ) const
{
    if ( d_dim == 1 ) {
        double x = ( node.coord() )[0];
        return sourceTerm_( x );
    } else if ( d_dim == 2 ) {
        double x = ( node.coord() )[0];
        double y = ( node.coord() )[1];
        return sourceTerm_( x, y );
    } else if ( d_dim == 3 ) {
        double x = ( node.coord() )[0];
        double y = ( node.coord() )[1];
        double z = ( node.coord() )[2];
        return sourceTerm_( x, y, z );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}

// Dimension-agnostic wrapper around the exact solution functions
double
ManufacturedRotatedAnisotropicDiffusionModel::exactSolution( AMP::Mesh::MeshElement &node ) const
{
    if ( d_dim == 1 ) {
        double x = ( node.coord() )[0];
        return exactSolution_( x );
    } else if ( d_dim == 2 ) {
        double x = ( node.coord() )[0];
        double y = ( node.coord() )[1];
        return exactSolution_( x, y );
    } else if ( d_dim == 3 ) {
        double x = ( node.coord() )[0];
        double y = ( node.coord() )[1];
        double z = ( node.coord() )[2];
        return exactSolution_( x, y, z );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}

// Exact solution, and corresponding source term
// 1D
double ManufacturedRotatedAnisotropicDiffusionModel::exactSolution_( double x ) const
{
    return std::sin( d_X_SHIFT + 2 * PI * x );
}
double ManufacturedRotatedAnisotropicDiffusionModel::sourceTerm_( double x ) const
{
    double cxx = d_c_db->getScalar<double>( "cxx" );
    return 4 * std::pow( PI, 2 ) * cxx * std::sin( d_X_SHIFT + 2 * PI * x );
}
// 2D
double ManufacturedRotatedAnisotropicDiffusionModel::exactSolution_( double x, double y ) const
{
    return std::sin( d_X_SHIFT + 2 * PI * x ) * std::sin( d_Y_SHIFT + 4 * PI * y );
}
double ManufacturedRotatedAnisotropicDiffusionModel::sourceTerm_( double x, double y ) const
{
    double cxx = d_c_db->getScalar<double>( "cxx" );
    double cyy = d_c_db->getScalar<double>( "cyy" );
    double cyx = d_c_db->getScalar<double>( "cyx" );
    return 4 * std::pow( PI, 2 ) *
           ( cxx * std::sin( d_X_SHIFT + 2 * PI * x ) * std::sin( d_Y_SHIFT + 4 * PI * y ) -
             2 * cyx * std::cos( d_X_SHIFT + 2 * PI * x ) * std::cos( d_Y_SHIFT + 4 * PI * y ) +
             4 * cyy * std::sin( d_X_SHIFT + 2 * PI * x ) * std::sin( d_Y_SHIFT + 4 * PI * y ) );
}
// 3D
double
ManufacturedRotatedAnisotropicDiffusionModel::exactSolution_( double x, double y, double z ) const
{
    return std::sin( d_X_SHIFT + 2 * PI * x ) * std::sin( d_Y_SHIFT + 4 * PI * y ) *
           std::sin( d_Z_SHIFT + 6 * PI * z );
}
double
ManufacturedRotatedAnisotropicDiffusionModel::sourceTerm_( double x, double y, double z ) const
{
    double cxx = d_c_db->getScalar<double>( "cxx" );
    double cyy = d_c_db->getScalar<double>( "cyy" );
    double czz = d_c_db->getScalar<double>( "czz" );
    double cyx = d_c_db->getScalar<double>( "cyx" );
    double czx = d_c_db->getScalar<double>( "czx" );
    double czy = d_c_db->getScalar<double>( "czy" );
    return 4 * std::pow( PI, 2 ) *
           ( cxx * std::sin( d_X_SHIFT + 2 * PI * x ) * std::sin( d_Y_SHIFT + 4 * PI * y ) *
                 std::sin( d_Z_SHIFT + 6 * PI * z ) -
             2 * cyx * std::sin( d_Z_SHIFT + 6 * PI * z ) * std::cos( d_X_SHIFT + 2 * PI * x ) *
                 std::cos( d_Y_SHIFT + 4 * PI * y ) -
             3 * czx * std::sin( d_Y_SHIFT + 4 * PI * y ) * std::cos( d_X_SHIFT + 2 * PI * x ) *
                 std::cos( d_Z_SHIFT + 6 * PI * z ) +
             4 * cyy * std::sin( d_X_SHIFT + 2 * PI * x ) * std::sin( d_Y_SHIFT + 4 * PI * y ) *
                 std::sin( d_Z_SHIFT + 6 * PI * z ) -
             6 * czy * std::sin( d_X_SHIFT + 2 * PI * x ) * std::cos( d_Y_SHIFT + 4 * PI * y ) *
                 std::cos( d_Z_SHIFT + 6 * PI * z ) +
             9 * czz * std::sin( d_X_SHIFT + 2 * PI * x ) * std::sin( d_Y_SHIFT + 4 * PI * y ) *
                 std::sin( d_Z_SHIFT + 6 * PI * z ) );
}
} // namespace AMP::Operator
