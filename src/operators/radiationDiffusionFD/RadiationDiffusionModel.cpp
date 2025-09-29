#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionModel.h"

namespace AMP::Operator {

/* -------------------------------------------------------------------- *
 * -------------------- Implementation of RadDifModel ----------------- *
 * -------------------------------------------------------------------- */

RadDifModel::RadDifModel( std::shared_ptr<AMP::Database> basic_db_,
                          std::shared_ptr<AMP::Database> mspecific_db_ )
    : d_basic_db( basic_db_ ), d_mspecific_db( mspecific_db_ )
{

    AMP_INSIST( basic_db_, "Non-null input database required!" );
    AMP_INSIST( mspecific_db_, "Non-null input database required!" );

    d_dim = d_basic_db->getScalar<size_t>( "dim" );

    // Start the construction of d_RadiationDiffusionFD_input_db; it must be finished by the derived
    // class Pack basic_db parameters into general_db
    d_RadiationDiffusionFD_input_db = std::make_shared<AMP::Database>( "GeneralPDEParams" );
    d_RadiationDiffusionFD_input_db->putScalar<int>( "dim", d_basic_db->getScalar<int>( "dim" ) );
    d_RadiationDiffusionFD_input_db->putScalar<bool>(
        "fluxLimited", d_basic_db->getScalar<bool>( "fluxLimited" ) );
    d_RadiationDiffusionFD_input_db->putScalar<bool>(
        "print_info_level", d_basic_db->getScalar<int>( "print_info_level" ) );
}

std::shared_ptr<AMP::Database> RadDifModel::getRadiationDiffusionFD_input_db() const
{
    AMP_INSIST( d_RadiationDiffusionFD_input_db_completed,
                "The derived class has not completed the construction of this database." );
    return d_RadiationDiffusionFD_input_db;
};

void RadDifModel::setCurrentTime( double currentTime_ ) { d_currentTime = currentTime_; };

double RadDifModel::getCurrentTime() const { return d_currentTime; };

double RadDifModel::exactSolution( size_t, const AMP::Mesh::Point & ) const
{
    AMP_ERROR( "Base class cannot provide a meaningful implementation of this function" );
}

void RadDifModel::getLHSRobinConstantsFromDB( size_t boundaryID, double &ak, double &bk ) const
{
    if ( boundaryID == 1 ) {
        ak = d_RadiationDiffusionFD_input_db->getScalar<double>( "a1" );
        bk = d_RadiationDiffusionFD_input_db->getScalar<double>( "b1" );
    } else if ( boundaryID == 2 ) {
        ak = d_RadiationDiffusionFD_input_db->getScalar<double>( "a2" );
        bk = d_RadiationDiffusionFD_input_db->getScalar<double>( "b2" );
    } else if ( boundaryID == 3 ) {
        ak = d_RadiationDiffusionFD_input_db->getScalar<double>( "a3" );
        bk = d_RadiationDiffusionFD_input_db->getScalar<double>( "b3" );
    } else if ( boundaryID == 4 ) {
        ak = d_RadiationDiffusionFD_input_db->getScalar<double>( "a4" );
        bk = d_RadiationDiffusionFD_input_db->getScalar<double>( "b4" );
    } else if ( boundaryID == 5 ) {
        ak = d_RadiationDiffusionFD_input_db->getScalar<double>( "a5" );
        bk = d_RadiationDiffusionFD_input_db->getScalar<double>( "b5" );
    } else if ( boundaryID == 6 ) {
        ak = d_RadiationDiffusionFD_input_db->getScalar<double>( "a6" );
        bk = d_RadiationDiffusionFD_input_db->getScalar<double>( "b6" );
    } else {
        AMP_ERROR( "Invalid boundaryID" );
    }
};

double RadDifModel::diffusionCoefficientE( double T, double zatom ) const
{
    if ( d_RadiationDiffusionFD_input_db->getScalar<std::string>( "model" ) == "nonlinear" ) {
        double sigma = std::pow( zatom / T, 3.0 );
        return 1.0 / ( 3 * sigma );
    } else if ( d_RadiationDiffusionFD_input_db->getScalar<std::string>( "model" ) == "linear" ) {
        return 1.0;
    } else {
        AMP_ERROR( "Invalid model" );
    }
}

bool RadDifModel::exactSolutionAvailable() const { return d_exactSolutionAvailable; }

/* -------------------------------------------------------------------- *
 * -------- Implementation of Mousseau_etal_2000_RadDifModel ---------- *
 * -------------------------------------------------------------------- */

Mousseau_etal_2000_RadDifModel::Mousseau_etal_2000_RadDifModel(
    std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> mspecific_db_ )
    : RadDifModel( basic_db_, mspecific_db_ )
{
    AMP_INSIST( d_dim < 3, "Mousseau_etal_2000_RadDifModel only implemented in 1D and 2D" );
    finalizeGeneralPDEModel_db();
}

double Mousseau_etal_2000_RadDifModel::sourceTerm( size_t, const AMP::Mesh::Point & ) const
{
    return 0.0;
}
double Mousseau_etal_2000_RadDifModel::initialCondition( size_t component,
                                                         const AMP::Mesh::Point & ) const
{
    double E0 = 1e-5;
    double T0 = std::pow( E0, 0.25 );
    if ( component == 0 ) {
        return E0;
    } else if ( component == 1 ) {
        return T0;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

void Mousseau_etal_2000_RadDifModel::finalizeGeneralPDEModel_db()
{
    /* The mpecific_db will have:
        z
        k
    */

    // Unpack parameters from model-specific database
    // Material atomic number
    double zatom = d_mspecific_db->getScalar<double>( "zatom" );
    // The temperature diffusion flux is k*T^{2.5}; eq. (16)
    double k21 = d_mspecific_db->getScalar<double>( "k" );

    // These constants do not appear
    double k11 = 1.0;
    double k12 = 1.0;
    double k22 = 1.0;

    double a1 = 0.25, b1 = 0.5, r1 = 1;
    double n1 = 0.0;
    double a2 = 0.25, b2 = 0.5, r2 = 0;
    double n2 = 0.0;
    // Set b constants to 1; they're arbitrary non-zero since only Neumann BCs are imposed on the y
    // boundaries
    double a3 = 0.0, b3 = 1.0, r3 = 0.0;
    double n3 = 0.0;
    double a4 = 0.0, b4 = 1.0, r4 = 0.0;
    double n4 = 0.0;

    // This PDE is nonlinear
    std::string model = "nonlinear";

    // Package into database
    d_RadiationDiffusionFD_input_db->putScalar<double>( "a1", a1 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "a2", a2 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "b1", b1 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "b2", b2 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "r1", r1 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "r2", r2 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "n1", n1 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "n2", n2 );
    if ( d_dim >= 2 ) {
        d_RadiationDiffusionFD_input_db->putScalar<double>( "a3", a3 );
        d_RadiationDiffusionFD_input_db->putScalar<double>( "a4", a4 );
        d_RadiationDiffusionFD_input_db->putScalar<double>( "b3", b3 );
        d_RadiationDiffusionFD_input_db->putScalar<double>( "b4", b4 );
        d_RadiationDiffusionFD_input_db->putScalar<double>( "r3", r3 );
        d_RadiationDiffusionFD_input_db->putScalar<double>( "r4", r4 );
        d_RadiationDiffusionFD_input_db->putScalar<double>( "n3", n3 );
        d_RadiationDiffusionFD_input_db->putScalar<double>( "n4", n4 );
    }

    d_RadiationDiffusionFD_input_db->putScalar<double>( "zatom", zatom );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "k11", k11 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "k12", k12 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "k21", k21 );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "k22", k22 );

    d_RadiationDiffusionFD_input_db->putScalar<std::string>( "model", model );
    // Flag that we've finalized this database
    d_RadiationDiffusionFD_input_db_completed = true;
}


/* -------------------------------------------------------------------- *
 * ------------- Implementation of Manufactured_RadDifModel ----------- *
 * -------------------------------------------------------------------- */

Manufactured_RadDifModel::Manufactured_RadDifModel( std::shared_ptr<AMP::Database> basic_db_,
                                                    std::shared_ptr<AMP::Database> specific_db_ )
    : RadDifModel( basic_db_, specific_db_ )
{
    // Set flag indicating this class does provide an implementation of exactSolution
    d_exactSolutionAvailable = true;

    finalizeGeneralPDEModel_db();
}

void Manufactured_RadDifModel::finalizeGeneralPDEModel_db()
{
    // Package into database
    d_RadiationDiffusionFD_input_db->putScalar<double>( "a1",
                                                        d_mspecific_db->getScalar<double>( "a1" ) );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "a2",
                                                        d_mspecific_db->getScalar<double>( "a2" ) );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "b1",
                                                        d_mspecific_db->getScalar<double>( "b1" ) );
    d_RadiationDiffusionFD_input_db->putScalar<double>( "b2",
                                                        d_mspecific_db->getScalar<double>( "b2" ) );
    if ( d_dim >= 2 ) {
        d_RadiationDiffusionFD_input_db->putScalar<double>(
            "a3", d_mspecific_db->getScalar<double>( "a3" ) );
        d_RadiationDiffusionFD_input_db->putScalar<double>(
            "a4", d_mspecific_db->getScalar<double>( "a4" ) );
        d_RadiationDiffusionFD_input_db->putScalar<double>(
            "b3", d_mspecific_db->getScalar<double>( "b3" ) );
        d_RadiationDiffusionFD_input_db->putScalar<double>(
            "b4", d_mspecific_db->getScalar<double>( "b4" ) );
    }
    if ( d_dim >= 3 ) {
        d_RadiationDiffusionFD_input_db->putScalar<double>(
            "a5", d_mspecific_db->getScalar<double>( "a5" ) );
        d_RadiationDiffusionFD_input_db->putScalar<double>(
            "a6", d_mspecific_db->getScalar<double>( "a6" ) );
        d_RadiationDiffusionFD_input_db->putScalar<double>(
            "b5", d_mspecific_db->getScalar<double>( "b5" ) );
        d_RadiationDiffusionFD_input_db->putScalar<double>(
            "b6", d_mspecific_db->getScalar<double>( "b6" ) );
    }

    d_RadiationDiffusionFD_input_db->putScalar<double>(
        "zatom", d_mspecific_db->getScalar<double>( "zatom" ) );
    d_RadiationDiffusionFD_input_db->putScalar<double>(
        "k11", d_mspecific_db->getScalar<double>( "k11" ) );
    d_RadiationDiffusionFD_input_db->putScalar<double>(
        "k12", d_mspecific_db->getScalar<double>( "k12" ) );
    d_RadiationDiffusionFD_input_db->putScalar<double>(
        "k21", d_mspecific_db->getScalar<double>( "k21" ) );
    d_RadiationDiffusionFD_input_db->putScalar<double>(
        "k22", d_mspecific_db->getScalar<double>( "k22" ) );

    d_RadiationDiffusionFD_input_db->putScalar<std::string>(
        "model", d_mspecific_db->getScalar<std::string>( "model" ) );

    // Flag that we've finalized this database
    d_RadiationDiffusionFD_input_db_completed = true;
}


// Dimension-agnostic wrapper around the sourceTerm_ functions
double Manufactured_RadDifModel::sourceTerm( size_t component, const AMP::Mesh::Point &point ) const
{
    if ( d_dim == 1 ) {
        return sourceTerm1D( component, point[0] );
    } else if ( d_dim == 2 ) {
        return sourceTerm2D( component, point[0], point[1] );
    } else if ( d_dim == 3 ) {
        return sourceTerm3D( component, point[0], point[1], point[2] );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}

double Manufactured_RadDifModel::initialCondition( size_t component,
                                                   const AMP::Mesh::Point &point ) const
{
    // We must set turn on this flag and then turn it off
    d_settingInitialCondition = true;
    double ic                 = exactSolution( component, point );
    d_settingInitialCondition = false;
    return ic;
}


double Manufactured_RadDifModel::exactSolution( size_t component,
                                                const AMP::Mesh::Point &point ) const
{
    if ( d_dim == 1 ) {
        return exactSolution1D( component, point[0] );
    } else if ( d_dim == 2 ) {
        return exactSolution2D( component, point[0], point[1] );
    } else if ( d_dim == 3 ) {
        return exactSolution3D( component, point[0], point[1], point[2] );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}


double Manufactured_RadDifModel::exactSolutionGradient( size_t component,
                                                        const AMP::Mesh::Point &point,
                                                        size_t gradComponent ) const
{
    if ( d_dim == 1 ) {
        return exactSolutionGradient1D( component, point[0] );
    } else if ( d_dim == 2 ) {
        return exactSolutionGradient2D( component, point[0], point[1], gradComponent );
    } else if ( d_dim == 3 ) {
        return exactSolutionGradient3D( component, point[0], point[1], point[2], gradComponent );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}

void Manufactured_RadDifModel::getNormalVector( size_t boundaryID,
                                                size_t &normalComponent,
                                                double &normalSign ) const
{

    AMP_INSIST( boundaryID >= 1 && boundaryID <= 6, "Invalid boundaryID" );
    // 1,2=x==0, 3,4=y==1, 5,6=z==2
    normalComponent = ( boundaryID - 1 ) / 2;
    // Odd boundaries are -1, even boundaries are +1
    normalSign = ( boundaryID % 2 == 0 ) ? +1.0 : -1.0;
}


double Manufactured_RadDifModel::getBoundaryFunctionValueE( size_t boundaryID,
                                                            const AMP::Mesh::Point &point ) const
{

    // Get sign and component direction of the normal vector
    double normalSign;
    size_t normalComponent;
    getNormalVector( boundaryID, normalComponent, normalSign );

    // Compute E on boundary
    double E = exactSolution( 0, point );

    // Compute diffusive flux D_E on boundary
    double zatom = d_RadiationDiffusionFD_input_db->getScalar<double>( "zatom" );
    double T     = exactSolution( 1, point );
    auto D_E     = diffusionCoefficientE( T, zatom );

    // Compute relevant component of gradient on boundary
    double dEdn = exactSolutionGradient( 0, point, normalComponent );

    // Unpack constants
    double k11 = d_RadiationDiffusionFD_input_db->getScalar<double>( "k11" );
    double ak, bk;
    getLHSRobinConstantsFromDB( boundaryID, ak, bk );

    return ak * E + bk * k11 * D_E * normalSign * dEdn;
}

double Manufactured_RadDifModel::getBoundaryFunctionValueT( size_t boundaryID,
                                                            const AMP::Mesh::Point &point ) const
{

    // Get sign and component direction of the normal vector
    double normalSign;
    size_t normalComponent;
    getNormalVector( boundaryID, normalComponent, normalSign );

    // Compute relevant component of gradient on boundary
    double dTdn = exactSolutionGradient( 1, point, normalComponent );
    return normalSign * dTdn;
}


// Implementation of 1D functions
double Manufactured_RadDifModel::exactSolution1D( size_t component, double x ) const
{

    double t = d_settingInitialCondition ? 0.0 : this->getCurrentTime();
    if ( component == 0 ) {
        double E = kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t );
        return E;
    } else if ( component == 1 ) {
        double T = std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) );
        return T;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::exactSolutionGradient1D( size_t component, double x ) const
{
    double t = this->getCurrentTime();
    if ( component == 0 ) {
        double dEdx = PI * kX * std::cos( PI * kT * t ) * std::cos( PI * kX * x + kXPhi );
        return dEdx;
    } else if ( component == 1 ) {
        double dTdx =
            ( 1.0 / 3.0 ) * PI * kX * std::cos( PI * kT * t ) * std::cos( PI * kX * x + kXPhi ) /
            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ), 2.0 / 3.0 );
        return dTdx;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::sourceTerm1D( size_t component, double x ) const
{

    double t = this->getCurrentTime();

    // Unpack parameters
    double k11   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k11" );
    double k12   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k12" );
    double k21   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k21" );
    double k22   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k22" );
    double zatom = d_RadiationDiffusionFD_input_db->getScalar<double>( "zatom" );

    if ( d_RadiationDiffusionFD_input_db->getScalar<std::string>( "model" ) == "linear" ) {
        if ( component == 0 ) {
            double sE =
                std::pow( PI, 2 ) * k11 * std::pow( kX, 2 ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kT * t ) -
                PI * kT * std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) -
                k12 *
                    ( -kE0 +
                      std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) ) -
                      std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) );
            return sE;
        } else if ( component == 1 ) {
            double sT =
                -1.0 / 3.0 * PI * kT * std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) /
                    std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ),
                              2.0 / 3.0 ) -
                k21 *
                    ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kX, 2 ) *
                          std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) /
                          std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ),
                                    2.0 / 3.0 ) -
                      2.0 / 9.0 * std::pow( PI, 2 ) * std::pow( kX, 2 ) *
                          std::pow( std::cos( PI * kT * t ), 2 ) *
                          std::pow( std::cos( PI * kX * x + kXPhi ), 2 ) /
                          std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ),
                                    5.0 / 3.0 ) ) +
                k22 *
                    ( -kE0 +
                      std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) ) -
                      std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) );
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( d_RadiationDiffusionFD_input_db->getScalar<std::string>( "model" ) ==
                "nonlinear" ) {
        if ( component == 0 ) {
            double sE =
                ( ( 1.0 / 3.0 ) * std::pow( PI, 2 ) * k11 * kE0 * std::pow( kX, 2 ) *
                      std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) +
                  ( 1.0 / 3.0 ) * std::pow( PI, 2 ) * k11 * std::pow( kX, 2 ) *
                      std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                      std::pow( std::cos( PI * kT * t ), 2 ) -
                  1.0 / 3.0 * std::pow( PI, 2 ) * k11 * std::pow( kX, 2 ) *
                      std::pow( std::cos( PI * kT * t ), 2 ) *
                      std::pow( std::cos( PI * kX * x + kXPhi ), 2 ) -
                  PI * kT * std::pow( zatom, 3 ) * std::sin( PI * kT * t ) *
                      std::sin( PI * kX * x + kXPhi ) -
                  k12 * std::pow( zatom, 6 ) *
                      std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) ) +
                  k12 * std::pow( zatom, 6 ) ) /
                std::pow( zatom, 3 );
            return sE;
        } else if ( component == 1 ) {
            double sT =
                ( ( 1.0 / 72.0 ) * std::pow( PI, 2 ) * k21 * std::pow( kX, 2 ) *
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ),
                                5.0 / 3.0 ) *
                      ( 24 * kE0 * std::sin( PI * kX * x + kXPhi ) + 10 * std::cos( PI * kT * t ) -
                        7 * std::cos( -PI * kT * t + 2 * PI * kX * x + 2 * kXPhi ) -
                        7 * std::cos( PI * kT * t + 2 * PI * kX * x + 2 * kXPhi ) ) *
                      std::cos( PI * kT * t ) -
                  1.0 / 3.0 * PI * kT *
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ),
                                11.0 / 6.0 ) *
                      std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) -
                  k22 * std::pow( zatom, 3 ) *
                      ( -std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ),
                                   17.0 / 6.0 ) +
                        std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ),
                                  5.0 / 2.0 ) ) ) /
                std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ),
                          5.0 / 2.0 );
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }
    } else {
        AMP_ERROR( "Invalid model" );
    }
}

// Implementation of 2D functions
double Manufactured_RadDifModel::exactSolution2D( size_t component, double x, double y ) const
{
    double t = d_settingInitialCondition ? 0.0 : this->getCurrentTime();
    if ( component == 0 ) {
        double E = kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                             std::cos( PI * kY * y + kYPhi );
        return E;
    } else if ( component == 1 ) {
        double T = std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                        std::cos( PI * kY * y + kYPhi ) );
        return T;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}


double Manufactured_RadDifModel::exactSolutionGradient2D( size_t component,
                                                          double x,
                                                          double y,
                                                          size_t gradComponent ) const
{

    double t = this->getCurrentTime();
    if ( component == 0 ) {
        if ( gradComponent == 0 ) {
            double dEdx = PI * kX * std::cos( PI * kT * t ) * std::cos( PI * kX * x + kXPhi ) *
                          std::cos( PI * kY * y + kYPhi );
            return dEdx;
        } else if ( gradComponent == 1 ) {
            double dEdy = -PI * kY * std::sin( PI * kX * x + kXPhi ) *
                          std::sin( PI * kY * y + kYPhi ) * std::cos( PI * kT * t );
            return dEdy;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( component == 1 ) {
        if ( gradComponent == 0 ) {
            double dTdx =
                ( 1.0 / 3.0 ) * PI * kX * std::cos( PI * kT * t ) *
                std::cos( PI * kX * x + kXPhi ) * std::cos( PI * kY * y + kYPhi ) /
                std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                    std::cos( PI * kY * y + kYPhi ),
                          2.0 / 3.0 );
            return dTdx;
        } else if ( gradComponent == 1 ) {
            double dTdy =
                -1.0 / 3.0 * PI * kY * std::sin( PI * kX * x + kXPhi ) *
                std::sin( PI * kY * y + kYPhi ) * std::cos( PI * kT * t ) /
                std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                    std::cos( PI * kY * y + kYPhi ),
                          2.0 / 3.0 );
            return dTdy;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::sourceTerm2D( size_t component, double x, double y ) const
{

    double t = this->getCurrentTime();

    // Unpack parameters
    double k11   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k11" );
    double k12   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k12" );
    double k21   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k21" );
    double k22   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k22" );
    double zatom = d_RadiationDiffusionFD_input_db->getScalar<double>( "zatom" );

    if ( d_RadiationDiffusionFD_input_db->getScalar<std::string>( "model" ) == "linear" ) {
        if ( component == 0 ) {
            double sE =
                std::pow( PI, 2 ) * k11 * ( std::pow( kX, 2 ) + std::pow( kY, 2 ) ) *
                    std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                    std::cos( PI * kY * y + kYPhi ) -
                PI * kT * std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kY * y + kYPhi ) +
                k12 * ( kE0 -
                        std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                             std::cos( PI * kY * y + kYPhi ) ) +
                        std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) );
            return sE;
        } else if ( component == 1 ) {
            double sT =
                ( ( 1.0 / 9.0 ) * std::pow( PI, 2 ) * k21 *
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                          std::cos( PI * kY * y + kYPhi ),
                                4.0 / 3.0 ) *
                      ( 3 *
                            ( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                        std::cos( PI * kY * y + kYPhi ) ) *
                            ( std::pow( kX, 2 ) + std::pow( kY, 2 ) ) *
                            std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kY * y + kYPhi ) +
                        2 *
                            ( std::pow( kX, 2 ) * std::pow( std::cos( PI * kX * x + kXPhi ), 2 ) *
                                  std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) +
                              std::pow( kY, 2 ) * std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                                  std::pow( std::sin( PI * kY * y + kYPhi ), 2 ) ) *
                            std::cos( PI * kT * t ) ) *
                      std::cos( PI * kT * t ) -
                  1.0 / 3.0 * PI * kT *
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                          std::cos( PI * kY * y + kYPhi ),
                                7.0 / 3.0 ) *
                      std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) *
                      std::cos( PI * kY * y + kYPhi ) -
                  k22 *
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                          std::cos( PI * kY * y + kYPhi ),
                                3 ) *
                      ( kE0 -
                        std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                             std::cos( PI * kY * y + kYPhi ) ) +
                        std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) ) ) /
                std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                    std::cos( PI * kY * y + kYPhi ),
                          3 );
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( d_RadiationDiffusionFD_input_db->getScalar<std::string>( "model" ) ==
                "nonlinear" ) {
        if ( component == 0 ) {
            double sE =
                ( 1.0 / 3.0 ) *
                ( std::pow( PI, 2 ) * k11 * kE0 * std::pow( kX, 2 ) *
                      std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                      std::cos( PI * kY * y + kYPhi ) +
                  std::pow( PI, 2 ) * k11 * kE0 * std::pow( kY, 2 ) *
                      std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                      std::cos( PI * kY * y + kYPhi ) +
                  std::pow( PI, 2 ) * k11 * std::pow( kX, 2 ) *
                      std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                      std::pow( std::cos( PI * kT * t ), 2 ) *
                      std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) -
                  std::pow( PI, 2 ) * k11 * std::pow( kX, 2 ) *
                      std::pow( std::cos( PI * kT * t ), 2 ) *
                      std::pow( std::cos( PI * kX * x + kXPhi ), 2 ) *
                      std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) -
                  std::pow( PI, 2 ) * k11 * std::pow( kY, 2 ) *
                      std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                      std::pow( std::sin( PI * kY * y + kYPhi ), 2 ) *
                      std::pow( std::cos( PI * kT * t ), 2 ) +
                  std::pow( PI, 2 ) * k11 * std::pow( kY, 2 ) *
                      std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                      std::pow( std::cos( PI * kT * t ), 2 ) *
                      std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) -
                  3 * PI * kT * std::pow( zatom, 3 ) * std::sin( PI * kT * t ) *
                      std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kY * y + kYPhi ) -
                  3 * k12 * std::pow( zatom, 6 ) *
                      std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                           std::cos( PI * kY * y + kYPhi ) ) +
                  3 * k12 * std::pow( zatom, 6 ) ) /
                std::pow( zatom, 3 );
            return sE;
        } else if ( component == 1 ) {
            double sT =
                -std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                     std::cos( PI * kY * y + kYPhi ),
                           -0.75 ) *
                ( std::pow( PI, 2 ) * k21 *
                      std::sqrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                           std::cos( PI * kY * y + kYPhi ) ) *
                      ( 0.375 * std::pow( kX, 2 ) * std::cos( PI * kT * t ) *
                            std::pow( std::cos( PI * kX * x + kXPhi ), 2 ) *
                            std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) +
                        0.375 * std::pow( kY, 2 ) * std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                            std::pow( std::sin( PI * kY * y + kYPhi ), 2 ) *
                            std::cos( PI * kT * t ) -
                        0.5 *
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ),
                                      1.0 ) *
                            ( std::pow( kX, 2 ) + std::pow( kY, 2 ) ) *
                            std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kY * y + kYPhi ) ) *
                      std::cos( PI * kT * t ) +
                  0.5 * PI * kT *
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                          std::cos( PI * kY * y + kYPhi ),
                                0.25 ) *
                      std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) *
                      std::cos( PI * kY * y + kYPhi ) +
                  k22 *
                      std::pow( zatom * std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                            std::cos( PI * kT * t ) *
                                                            std::cos( PI * kY * y + kYPhi ),
                                                  -0.5 ),
                                3.0 ) *
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                          std::cos( PI * kY * y + kYPhi ),
                                0.75 ) *
                      ( kE0 -
                        std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                            std::cos( PI * kY * y + kYPhi ),
                                  2.0 ) +
                        std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) ) );
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }
    } else {
        AMP_ERROR( "Invalid model" );
    }
}


// Implementation of 3D functions
double
Manufactured_RadDifModel::exactSolution3D( size_t component, double x, double y, double z ) const
{
    double t = d_settingInitialCondition ? 0.0 : this->getCurrentTime();
    if ( component == 0 ) {
        double E = kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                             std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi );
        return E;
    } else if ( component == 1 ) {
        double T = std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                        std::cos( PI * kY * y + kYPhi ) *
                                        std::cos( PI * kZ * z + kZPhi ) );
        return T;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::exactSolutionGradient3D(
    size_t component, double x, double y, double z, size_t gradComponent ) const
{

    double t = this->getCurrentTime();
    if ( component == 0 ) {
        if ( gradComponent == 0 ) {
            double dEdx = PI * kX * std::cos( PI * kT * t ) * std::cos( PI * kX * x + kXPhi ) *
                          std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi );
            return dEdx;
        } else if ( gradComponent == 1 ) {
            double dEdy = -PI * kY * std::sin( PI * kX * x + kXPhi ) *
                          std::sin( PI * kY * y + kYPhi ) * std::cos( PI * kT * t ) *
                          std::cos( PI * kZ * z + kZPhi );
            return dEdy;
        } else if ( gradComponent == 2 ) {
            double dEdz = -PI * kZ * std::sin( PI * kX * x + kXPhi ) *
                          std::sin( PI * kZ * z + kZPhi ) * std::cos( PI * kT * t ) *
                          std::cos( PI * kY * y + kYPhi );
            return dEdz;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( component == 1 ) {
        if ( gradComponent == 0 ) {
            double dTdx =
                ( 1.0 / 3.0 ) * PI * kX * std::cos( PI * kT * t ) *
                std::cos( PI * kX * x + kXPhi ) * std::cos( PI * kY * y + kYPhi ) *
                std::cos( PI * kZ * z + kZPhi ) /
                std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                    std::cos( PI * kY * y + kYPhi ) *
                                    std::cos( PI * kZ * z + kZPhi ),
                          2.0 / 3.0 );
            return dTdx;
        } else if ( gradComponent == 1 ) {
            double dTdy =
                -1.0 / 3.0 * PI * kY * std::sin( PI * kX * x + kXPhi ) *
                std::sin( PI * kY * y + kYPhi ) * std::cos( PI * kT * t ) *
                std::cos( PI * kZ * z + kZPhi ) /
                std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                    std::cos( PI * kY * y + kYPhi ) *
                                    std::cos( PI * kZ * z + kZPhi ),
                          2.0 / 3.0 );
            return dTdy;
        } else if ( gradComponent == 2 ) {
            double dTdz =
                -1.0 / 3.0 * PI * kZ * std::sin( PI * kX * x + kXPhi ) *
                std::sin( PI * kZ * z + kZPhi ) * std::cos( PI * kT * t ) *
                std::cos( PI * kY * y + kYPhi ) /
                std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                    std::cos( PI * kY * y + kYPhi ) *
                                    std::cos( PI * kZ * z + kZPhi ),
                          2.0 / 3.0 );
            return dTdz;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double
Manufactured_RadDifModel::sourceTerm3D( size_t component, double x, double y, double z ) const
{

    double t = this->getCurrentTime();

    // Unpack parameters
    double k11   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k11" );
    double k12   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k12" );
    double k21   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k21" );
    double k22   = d_RadiationDiffusionFD_input_db->getScalar<double>( "k22" );
    double zatom = d_RadiationDiffusionFD_input_db->getScalar<double>( "zatom" );

    if ( d_RadiationDiffusionFD_input_db->getScalar<std::string>( "model" ) == "linear" ) {
        if ( component == 0 ) {
            double sE =
                std::pow( PI, 2 ) * k11 * std::pow( kX, 2 ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kT * t ) * std::cos( PI * kY * y + kYPhi ) *
                    std::cos( PI * kZ * z + kZPhi ) +
                std::pow( PI, 2 ) * k11 * std::pow( kY, 2 ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kT * t ) * std::cos( PI * kY * y + kYPhi ) *
                    std::cos( PI * kZ * z + kZPhi ) +
                std::pow( PI, 2 ) * k11 * std::pow( kZ, 2 ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kT * t ) * std::cos( PI * kY * y + kYPhi ) *
                    std::cos( PI * kZ * z + kZPhi ) -
                PI * kT * std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) +
                k12 * ( kE0 -
                        std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                             std::cos( PI * kY * y + kYPhi ) *
                                             std::cos( PI * kZ * z + kZPhi ) ) +
                        std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) );
            return sE;
        } else if ( component == 1 ) {
            double sT =
                -1.0 / 3.0 * PI * kT * std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) /
                    std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                        std::cos( PI * kY * y + kYPhi ) *
                                        std::cos( PI * kZ * z + kZPhi ),
                              2.0 / 3.0 ) -
                k21 * ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kX, 2 ) *
                            std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      2.0 / 3.0 ) -
                        2.0 / 9.0 * std::pow( PI, 2 ) * std::pow( kX, 2 ) *
                            std::pow( std::cos( PI * kT * t ), 2 ) *
                            std::pow( std::cos( PI * kX * x + kXPhi ), 2 ) *
                            std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) *
                            std::pow( std::cos( PI * kZ * z + kZPhi ), 2 ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      5.0 / 3.0 ) ) -
                k21 * ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kY, 2 ) *
                            std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      2.0 / 3.0 ) -
                        2.0 / 9.0 * std::pow( PI, 2 ) * std::pow( kY, 2 ) *
                            std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                            std::pow( std::sin( PI * kY * y + kYPhi ), 2 ) *
                            std::pow( std::cos( PI * kT * t ), 2 ) *
                            std::pow( std::cos( PI * kZ * z + kZPhi ), 2 ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      5.0 / 3.0 ) ) -
                k21 * ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kZ, 2 ) *
                            std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      2.0 / 3.0 ) -
                        2.0 / 9.0 * std::pow( PI, 2 ) * std::pow( kZ, 2 ) *
                            std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                            std::pow( std::sin( PI * kZ * z + kZPhi ), 2 ) *
                            std::pow( std::cos( PI * kT * t ), 2 ) *
                            std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      5.0 / 3.0 ) ) +
                k22 * ( -kE0 +
                        std::cbrt( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                             std::cos( PI * kY * y + kYPhi ) *
                                             std::cos( PI * kZ * z + kZPhi ) ) -
                        std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) );
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( d_RadiationDiffusionFD_input_db->getScalar<std::string>( "model" ) ==
                "nonlinear" ) {
        if ( component == 0 ) {
            double sE =
                -PI * kT * std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) -
                k11 *
                    ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kX, 2 ) *
                          ( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                      std::cos( PI * kY * y + kYPhi ) *
                                      std::cos( PI * kZ * z + kZPhi ) ) *
                          std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                          std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) /
                          std::pow( zatom, 3 ) +
                      ( 1.0 / 3.0 ) * std::pow( PI, 2 ) * std::pow( kX, 2 ) *
                          std::pow( std::cos( PI * kT * t ), 2 ) *
                          std::pow( std::cos( PI * kX * x + kXPhi ), 2 ) *
                          std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) *
                          std::pow( std::cos( PI * kZ * z + kZPhi ), 2 ) / std::pow( zatom, 3 ) ) -
                k11 *
                    ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kY, 2 ) *
                          ( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                      std::cos( PI * kY * y + kYPhi ) *
                                      std::cos( PI * kZ * z + kZPhi ) ) *
                          std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                          std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) /
                          std::pow( zatom, 3 ) +
                      ( 1.0 / 3.0 ) * std::pow( PI, 2 ) * std::pow( kY, 2 ) *
                          std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                          std::pow( std::sin( PI * kY * y + kYPhi ), 2 ) *
                          std::pow( std::cos( PI * kT * t ), 2 ) *
                          std::pow( std::cos( PI * kZ * z + kZPhi ), 2 ) / std::pow( zatom, 3 ) ) -
                k11 *
                    ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kZ, 2 ) *
                          ( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                      std::cos( PI * kY * y + kYPhi ) *
                                      std::cos( PI * kZ * z + kZPhi ) ) *
                          std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                          std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) /
                          std::pow( zatom, 3 ) +
                      ( 1.0 / 3.0 ) * std::pow( PI, 2 ) * std::pow( kZ, 2 ) *
                          std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                          std::pow( std::sin( PI * kZ * z + kZPhi ), 2 ) *
                          std::pow( std::cos( PI * kT * t ), 2 ) *
                          std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) / std::pow( zatom, 3 ) ) -
                k12 * std::pow( zatom, 3 ) *
                    ( -kE0 +
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                          std::cos( PI * kY * y + kYPhi ) *
                                          std::cos( PI * kZ * z + kZPhi ),
                                4.0 / 3.0 ) -
                      std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                          std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) ) /
                    ( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) );
            return sE;
        } else if ( component == 1 ) {
            double sT =
                -1.0 / 3.0 * PI * kT * std::sin( PI * kT * t ) * std::sin( PI * kX * x + kXPhi ) *
                    std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) /
                    std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                        std::cos( PI * kY * y + kYPhi ) *
                                        std::cos( PI * kZ * z + kZPhi ),
                              2.0 / 3.0 ) -
                k21 * ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kX, 2 ) *
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      1.0 / 6.0 ) *
                            std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) +
                        ( 1.0 / 18.0 ) * std::pow( PI, 2 ) * std::pow( kX, 2 ) *
                            std::pow( std::cos( PI * kT * t ), 2 ) *
                            std::pow( std::cos( PI * kX * x + kXPhi ), 2 ) *
                            std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) *
                            std::pow( std::cos( PI * kZ * z + kZPhi ), 2 ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      5.0 / 6.0 ) ) -
                k21 * ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kY, 2 ) *
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      1.0 / 6.0 ) *
                            std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) +
                        ( 1.0 / 18.0 ) * std::pow( PI, 2 ) * std::pow( kY, 2 ) *
                            std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                            std::pow( std::sin( PI * kY * y + kYPhi ), 2 ) *
                            std::pow( std::cos( PI * kT * t ), 2 ) *
                            std::pow( std::cos( PI * kZ * z + kZPhi ), 2 ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      5.0 / 6.0 ) ) -
                k21 * ( -1.0 / 3.0 * std::pow( PI, 2 ) * std::pow( kZ, 2 ) *
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      1.0 / 6.0 ) *
                            std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                            std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) +
                        ( 1.0 / 18.0 ) * std::pow( PI, 2 ) * std::pow( kZ, 2 ) *
                            std::pow( std::sin( PI * kX * x + kXPhi ), 2 ) *
                            std::pow( std::sin( PI * kZ * z + kZPhi ), 2 ) *
                            std::pow( std::cos( PI * kT * t ), 2 ) *
                            std::pow( std::cos( PI * kY * y + kYPhi ), 2 ) /
                            std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) *
                                                std::cos( PI * kT * t ) *
                                                std::cos( PI * kY * y + kYPhi ) *
                                                std::cos( PI * kZ * z + kZPhi ),
                                      5.0 / 6.0 ) ) +
                k22 * std::pow( zatom, 3 ) *
                    ( -kE0 +
                      std::pow( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                          std::cos( PI * kY * y + kYPhi ) *
                                          std::cos( PI * kZ * z + kZPhi ),
                                4.0 / 3.0 ) -
                      std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                          std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) ) /
                    ( kE0 + std::sin( PI * kX * x + kXPhi ) * std::cos( PI * kT * t ) *
                                std::cos( PI * kY * y + kYPhi ) * std::cos( PI * kZ * z + kZPhi ) );
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }
    } else {
        AMP_ERROR( "Invalid model" );
    }
}

} // namespace AMP::Operator
