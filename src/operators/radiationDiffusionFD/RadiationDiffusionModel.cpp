#include "RadiationDiffusionModel.h"

// Constructor
RadDifModel::RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> mspecific_db_ ) : d_basic_db( basic_db_ ), d_mspecific_db( mspecific_db_ ) { 

    AMP_INSIST( basic_db_,     "Non-null input database required!" );
    AMP_INSIST( mspecific_db_, "Non-null input database required!" );

    d_dim = d_basic_db->getScalar<int>( "dim" );

    // Start the construction of d_general_db; it must be finished by the derived class
    // Pack basic_db parameters into general_db
    d_general_db = std::make_shared<AMP::Database>( "GeneralPDEParams" );
    d_general_db->putScalar<int>( "dim", d_basic_db->getScalar<int>( "dim" ) );
    d_general_db->putScalar<bool>( "fluxLimited", d_basic_db->getScalar<bool>( "fluxLimited" ) );
    d_general_db->putScalar<bool>( "print_info_level", d_basic_db->getScalar<int>( "print_info_level" ) );

    // Robin and Pseudo-Neumann values not necessarily used, so set to dummy values
    d_general_db->putScalar<double>( "r1", std::nan("") );
    d_general_db->putScalar<double>( "r2", std::nan("") );
    d_general_db->putScalar<double>( "r3", std::nan("") );
    d_general_db->putScalar<double>( "r4", std::nan("") );
    d_general_db->putScalar<double>( "n1", std::nan("") );
    d_general_db->putScalar<double>( "n2", std::nan("") );
    d_general_db->putScalar<double>( "n3", std::nan("") );
    d_general_db->putScalar<double>( "n4", std::nan("") );

    d_general_db->setDefaultAddKeyBehavior( AMP::Database::Check::Overwrite, false ); 
}

/* --------------------------------------------------------------------
-----------------------------------------------------------------------
-------------------------------------------------------------------- */

Mousseau_etal_2000_RadDifModel::Mousseau_etal_2000_RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> specific_db_ ) : RadDifModel( basic_db_, specific_db_ )  { 
    finalizeGeneralPDEModel_db();
}


double Mousseau_etal_2000_RadDifModel::sourceTerm( int , AMP::Mesh::MeshElement & ) const { return 0.0; }
double Mousseau_etal_2000_RadDifModel::initialCondition( int component, AMP::Mesh::MeshElement & ) const { 
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

void Mousseau_etal_2000_RadDifModel::finalizeGeneralPDEModel_db( ) {
    /*
    The specific_db will have:
        z
        k
    */
    
    // Unpack parameters from model-specific database
    // Material atomic number
    double z   = d_mspecific_db->getScalar<double>( "z" );
    // The temperature diffusion flux is k*T^{2.5}; eq. (16)
    double k21 = d_mspecific_db->getScalar<double>( "k" );
    
    double k11 = 1.0;
    double k12 = 1.0;
    double k22 = 1.0;

    double a1 = 0.25, b1 = 0.5, r1 = 1;
    double n1 = 0.0;
    double a2 = 0.25, b2 = 0.5, r2 = 0;
    double n2 = 0.0;
    // Set b constants to 1; they're arbitrary non-zero
    double a3 = 0.0, b3 = 1.0, r3 = 0.0;
    double n3 = 0.0;
    double a4 = 0.0, b4 = 1.0, r4 = 0.0;
    double n4 = 0.0;

    // This PDE is nonlinear 
    std::string model = "nonlinear";

    // Package into database
    d_general_db->putScalar<double>( "a1", a1 );
    d_general_db->putScalar<double>( "a2", a2 );
    d_general_db->putScalar<double>( "a3", a3 );
    d_general_db->putScalar<double>( "a4", a4 );
    d_general_db->putScalar<double>( "b1", b1 );
    d_general_db->putScalar<double>( "b2", b2 );
    d_general_db->putScalar<double>( "b3", b3 );
    d_general_db->putScalar<double>( "b4", b4 );
    d_general_db->putScalar<double>( "r1", r1 );
    d_general_db->putScalar<double>( "r2", r2 );
    d_general_db->putScalar<double>( "r3", r3 );
    d_general_db->putScalar<double>( "r4", r4 );
    d_general_db->putScalar<double>( "n1", n1 );
    d_general_db->putScalar<double>( "n2", n2 );
    d_general_db->putScalar<double>( "n3", n3 );
    d_general_db->putScalar<double>( "n4", n4 );

    d_general_db->putScalar<double>( "z",   z   );
    d_general_db->putScalar<double>( "k11", k11 );
    d_general_db->putScalar<double>( "k12", k12 );
    d_general_db->putScalar<double>( "k21", k21 );
    d_general_db->putScalar<double>( "k22", k22 );

    d_general_db->putScalar<std::string>( "model", model );

    // Restore default behavior
    d_general_db->setDefaultAddKeyBehavior( AMP::Database::Check::Error, false ); 
    // Flag that we've finalized this database
    d_general_db_completed = true;
}

/* --------------------------------------------------------------------
-----------------------------------------------------------------------
-------------------------------------------------------------------- */

Manufactured_RadDifModel::Manufactured_RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> specific_db_ ) : RadDifModel( basic_db_, specific_db_ ) { 
    // Set flag indicating this class does provide an implementation of exactSolution
    d_exactSolutionAvailable = true;

    finalizeGeneralPDEModel_db( );
}

void Manufactured_RadDifModel::finalizeGeneralPDEModel_db( ) {  
    // Package into database
    d_general_db->putScalar<double>( "a1", d_mspecific_db->getScalar<double>( "a1" ) );
    d_general_db->putScalar<double>( "a2", d_mspecific_db->getScalar<double>( "a2" ) );
    d_general_db->putScalar<double>( "a3", d_mspecific_db->getScalar<double>( "a3" ) );
    d_general_db->putScalar<double>( "a4", d_mspecific_db->getScalar<double>( "a4" ) );
    d_general_db->putScalar<double>( "b1", d_mspecific_db->getScalar<double>( "b1" ) );
    d_general_db->putScalar<double>( "b2", d_mspecific_db->getScalar<double>( "b2" ) );
    d_general_db->putScalar<double>( "b3", d_mspecific_db->getScalar<double>( "b3" ) );
    d_general_db->putScalar<double>( "b4", d_mspecific_db->getScalar<double>( "b4" ) );

    d_general_db->putScalar<double>( "z",   d_mspecific_db->getScalar<double>( "z" )   );
    d_general_db->putScalar<double>( "k11", d_mspecific_db->getScalar<double>( "k11" ) );
    d_general_db->putScalar<double>( "k12", d_mspecific_db->getScalar<double>( "k12" ) );
    d_general_db->putScalar<double>( "k21", d_mspecific_db->getScalar<double>( "k21" ) );
    d_general_db->putScalar<double>( "k22", d_mspecific_db->getScalar<double>( "k22" ) );

    d_general_db->putScalar<std::string>( "model", d_mspecific_db->getScalar<std::string>( "model" ) );

    // Restore default behavior
    d_general_db->setDefaultAddKeyBehavior( AMP::Database::Check::Error, false ); 
    // Flag that we've finalized this database
    d_general_db_completed = true;
}

// Implementation of pure virtual function
// Dimension-agnostic wrapper around the exact source term functions
double Manufactured_RadDifModel::sourceTerm( int component, AMP::Mesh::MeshElement &node ) const {
    if ( d_dim == 1 ) {
        double x = ( node.coord() )[0];
        return sourceTerm_( component, x );
    } else if ( d_dim == 2 ) {
        double x = ( node.coord() )[0];
        double y = ( node.coord() )[1];
        return sourceTerm_( component, x, y );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}

// Implementation of pure virtual function
double Manufactured_RadDifModel::initialCondition( int component, AMP::Mesh::MeshElement &node ) const {
    // We must set turn on this flag and then turn it off
    d_settingInitialCondition = true;
    double ic = exactSolution( component, node );
    d_settingInitialCondition = false;
    return ic;
}

// Dimension-agnostic wrapper around the exact solution functions
double Manufactured_RadDifModel::exactSolution( int component, AMP::Mesh::MeshElement &node ) const {
    if ( d_dim == 1 ) {
        double x = ( node.coord() )[0];
        return exactSolution_( component, x );
    } else if ( d_dim == 2 ) {
        double x = ( node.coord() )[0];
        double y = ( node.coord() )[1];
        return exactSolution_( component, x, y );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}

// Dimension-agnostic wrapper around the Robin functions
double Manufactured_RadDifModel::getRobinValueE( int boundary, double a, double b, AMP::Mesh::MeshElement & node ) const {
    if ( d_dim == 1 ) {
        return getRobinValueE1D_( boundary, a, b );
    } else if ( d_dim == 2 ) {
        double x = ( node.coord() )[0];
        double y = ( node.coord() )[1];
        return getRobinValueE2D_( boundary, a, b, x, y );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}

// Dimension-agnostic wrapper around the pseudo-Neumann functions
double Manufactured_RadDifModel::getPseudoNeumannValueT( int boundary, AMP::Mesh::MeshElement & node ) const {
    if ( d_dim == 1 ) {
        return getPseudoNeumannValueT1D_( boundary );
    } else if ( d_dim == 2 ) {
        double x = ( node.coord() )[0];
        double y = ( node.coord() )[1];
        return getPseudoNeumannValueT2D_( boundary, x, y );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
}

/* 1D. The boundaries of 0 and 1 are hard coded here. 
    Robin on E:
        a1 * E + b1 * - k11*D_E * dE/dx = r1 at x = 0...
        a2 * E + b2 * + k11*D_E * dE/dx = r2 at x = 1...
*/
double Manufactured_RadDifModel::getRobinValueE1D_( int boundary, double a, double b ) const {

    // Get value of x on boundary and sign of normal vector
    double x           = -1.0;
    double normal_sign = 0.0;
    if ( boundary == 1 ) { // West. Normal vector is -hat{x}
        x           = 0.0;
        normal_sign = -1.0;
    } else if ( boundary == 2 ) { // East. Normal vector is +hat{x}
        x           = 1.0;
        normal_sign = +1.0;
    }

    // Unpack parameters
    double k11   = d_general_db->getScalar<double>( "k11" );
    double z     = d_general_db->getScalar<double>( "z" );
    // Compute diffusive flux term D_E
    double D_E = 1.0; // For the linear model this is just constant
    if ( d_general_db->getScalar<std::string>( "model" ) == "nonlinear" ) {
        double T     = exactSolution_( 1, x );
        double sigma = std::pow( z/T, 3.0 );
        D_E = 1.0/(3*sigma);
    }

    double E    = exactSolution_( 0, x );
    double dEdx = exactSolutionGradient_( 0, x );

    return a * E + b * normal_sign * k11 * D_E * dEdx;
}

/* 1D. The boundaries of 0 and 1 are hard coded here. 
Pseudo-Neumann on T:
    -dT/dx = n1 at x = 0...
    +dT/dx = n2 at x = 1...
*/
double Manufactured_RadDifModel::getPseudoNeumannValueT1D_( int boundary ) const {

    // Get value of x on boundary and sign of normal vector
    double x = -1.0;
    double normal_sign = 0.0;
    if ( boundary == 1 ) { // West
        x = 0.0;
        // Normal vector is -hat{x}
        normal_sign      = -1.0;
    } else if ( boundary == 2 ) { // East
        x = 1.0;
        // Normal vector is +hat{x}
        normal_sign      = +1.0;
    }

    double dTdx = exactSolutionGradient_( 1, x );
    return normal_sign * dTdx;
}

/* 2D. The boundaries of 0 and 1 are hard coded here, so that the value (either x or y) that is parsed is ignored depending on the value of "boundary" (this just makes calling this function easier) */
double Manufactured_RadDifModel::getRobinValueE2D_( int boundary, double a, double b, double x, double y ) const {

    std::string normal_direction = "";
    double      normal_sign      = 0.0;
    if ( boundary == 1 ) { // West. Normal vector is -hat{x}
        x = 0.0;
        normal_direction = "x";
        normal_sign      = -1.0;
    } else if ( boundary == 2 ) { // East. Normal vector is +hat{x}
        x = 1.0;
        normal_direction = "x";
        normal_sign      = +1.0;
    } else if ( boundary == 3 ) { // South. Normal vector is -hat{y}
        y = 0.0;
        normal_direction = "y";
        normal_sign      = -1.0;
    } else if ( boundary == 4 ) { // North. Normal vector is +hat{y}
        y = 1.0;
        normal_direction = "y";
        normal_sign      = +1.0;
    }

    double E    = exactSolution_( 0, x, y );
    double dEdn = exactSolutionGradient_( 0, x, y, normal_direction );

    // Unpack parameters
    double k11   = d_general_db->getScalar<double>( "k11" );
    double z     = d_general_db->getScalar<double>( "z" );
    // Compute diffusive flux term D_E
    double D_E = 1.0; // For the linear model this is just constant
    if ( d_general_db->getScalar<std::string>( "model" ) == "nonlinear" ) {
        double T     = exactSolution_( 1, x, y );
        double sigma = std::pow( z/T, 3.0 );
        D_E          = 1.0/(3*sigma);
    }

    return a * E + b * normal_sign * k11 * D_E * dEdn;
}


/* 2D. The boundaries of 0 and 1 are hard coded here, so that the value (either x or y) that is parsed is ignored depending on the value of "boundary" (this just makes calling this function easier) */
double Manufactured_RadDifModel::getPseudoNeumannValueT2D_( int boundary, double x, double y ) const {

    std::string normal_direction = "";
    double      normal_sign      = 0.0;
    if ( boundary == 1 ) { // West. Normal vector is -hat{x}
        x = 0.0;
        normal_direction = "x";
        normal_sign      = -1.0;
    } else if ( boundary == 2 ) { // East. Normal vector is +hat{x}
        x = 1.0;
        normal_direction = "x";
        normal_sign      = +1.0;
    } else if ( boundary == 3 ) { // South. Normal vector is -hat{y}
        y = 0.0;
        normal_direction = "y";
        normal_sign      = -1.0;
    } else if ( boundary == 4 ) { // North. Normal vector is +hat{y}
        y = 1.0;
        normal_direction = "y";
        normal_sign      = +1.0;
    }

    double dTdn = exactSolutionGradient_( 1, x, y, normal_direction );
    return normal_sign * dTdn;
}

// Implementation of 1D functions
double Manufactured_RadDifModel::exactSolution_( int component, double x ) const {

    double t = d_settingInitialCondition ? 0.0 : this->getCurrentTime();
    if ( component == 0 ) {
        double E = kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t);
        return E;
    } else if ( component == 1 ) {
        double T = std::cbrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t));
        return T;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::exactSolutionGradient_( int component, double x ) const {
    double t = this->getCurrentTime();
    if ( component == 0 ) {
        double dEdx = M_PI*kX*std::cos(M_PI*kT*t)*std::cos(M_PI*kX*x + kXPhi);
        return dEdx;
    } else if ( component == 1 ) {
        double dTdx = (1.0/3.0)*M_PI*kX*std::cos(M_PI*kT*t)*std::cos(M_PI*kX*x + kXPhi)/std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t), 2.0/3.0);
        return dTdx;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::sourceTerm_( int component, double x ) const {
    
    double t    = this->getCurrentTime();

    // Unpack parameters
    double k11  = d_general_db->getScalar<double>( "k11" );
    double k12  = d_general_db->getScalar<double>( "k12" );
    double k21  = d_general_db->getScalar<double>( "k21" );
    double k22  = d_general_db->getScalar<double>( "k22" );
    double z    = d_general_db->getScalar<double>( "z" );

    if ( d_general_db->getScalar<std::string>( "model" ) == "linear" ) {
        if ( component == 0 ) {
            double sE = std::pow(M_PI, 2)*k11*std::pow(kX, 2)*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t) - M_PI*kT*std::sin(M_PI*kT*t)*std::sin(M_PI*kX*x + kXPhi) + k12*(kE0 - std::cbrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)) + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t));
            return sE;
        } else if ( component == 1 ) {
            double sT = ((2.0/9.0)*std::pow(M_PI, 2)*k21*std::pow(kX, 2)*std::pow(std::cos(M_PI*kT*t), 2)*std::pow(std::cos(M_PI*kX*x + kXPhi), 2) + (1.0/3.0)*M_PI*(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t))*(M_PI*k21*std::pow(kX, 2)*std::cos(M_PI*kT*t) - kT*std::sin(M_PI*kT*t))*std::sin(M_PI*kX*x + kXPhi) - k22*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t), 5.0/3.0)*(kE0 - std::cbrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)) + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)))/std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t), 5.0/3.0);
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( d_general_db->getScalar<std::string>( "model" ) == "nonlinear" ) {
        if ( component == 0 ) {
            double sE = ((1.0/3.0)*std::pow(M_PI, 2)*k11*kE0*std::pow(kX, 2)*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t) + (1.0/3.0)*std::pow(M_PI, 2)*k11*std::pow(kX, 2)*std::pow(std::sin(M_PI*kX*x + kXPhi), 2)*std::pow(std::cos(M_PI*kT*t), 2) - 1.0/3.0*std::pow(M_PI, 2)*k11*std::pow(kX, 2)*std::pow(std::cos(M_PI*kT*t), 2)*std::pow(std::cos(M_PI*kX*x + kXPhi), 2) - M_PI*kT*std::pow(z, 3)*std::sin(M_PI*kT*t)*std::sin(M_PI*kX*x + kXPhi) - k12*std::pow(z, 6)*std::cbrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)) + k12*std::pow(z, 6))/std::pow(z, 3);
            return sE;
        } else if ( component == 1 ) {
            double sT = ((1.0/72.0)*std::pow(M_PI, 2)*k21*std::pow(kX, 2)*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t), 5.0/3.0)*(24*kE0*std::sin(M_PI*kX*x + kXPhi) + 10*std::cos(M_PI*kT*t) - 7*std::cos(-M_PI*kT*t + 2*M_PI*kX*x + 2*kXPhi) - 7*std::cos(M_PI*kT*t + 2*M_PI*kX*x + 2*kXPhi))*std::cos(M_PI*kT*t) - 1.0/3.0*M_PI*kT*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t), 11.0/6.0)*std::sin(M_PI*kT*t)*std::sin(M_PI*kX*x + kXPhi) - k22*std::pow(z, 3)*(-std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t), 17.0/6.0) + std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t), 5.0/2.0)))/std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t), 5.0/2.0);
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }
    } else {
        AMP_ERROR( "Invalid model" );
    } 
}

// Implementation of 2D functions
double Manufactured_RadDifModel::exactSolution_( int component, double x, double y ) const {
    double t = d_settingInitialCondition ? 0.0 : this->getCurrentTime();
    if ( component == 0 ) {
        double E = kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi);
        return E;
    } else if ( component == 1 ) {
        double T = std::cbrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi));
        return T;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::exactSolutionGradient_( int component, double x, double y, const std::string & grad_component ) const {

    double t = this->getCurrentTime();
    if ( component == 0 ) {
        if ( grad_component == "x" ) {
            double dEdx = M_PI*kX*std::cos(M_PI*kT*t)*std::cos(M_PI*kX*x + kXPhi)*std::cos(M_PI*kY*y + kYPhi);
            return dEdx;
        } else if ( grad_component == "y" ) {
            double dEdy = -M_PI*kY*std::sin(M_PI*kX*x + kXPhi)*std::sin(M_PI*kY*y + kYPhi)*std::cos(M_PI*kT*t);
            return dEdy;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( component == 1 ) {
        if ( grad_component == "x" ) {
            double dTdx = (1.0/3.0)*M_PI*kX*std::cos(M_PI*kT*t)*std::cos(M_PI*kX*x + kXPhi)*std::cos(M_PI*kY*y + kYPhi)/std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 2.0/3.0);
            return dTdx;
        } else if ( grad_component == "y" ) {
            double dTdy = -1.0/3.0*M_PI*kY*std::sin(M_PI*kX*x + kXPhi)*std::sin(M_PI*kY*y + kYPhi)*std::cos(M_PI*kT*t)/std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 2.0/3.0);
            return dTdy;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::sourceTerm_( int component, double x, double y ) const {

    double t    = this->getCurrentTime();

    // Unpack parameters
    double k11  = d_general_db->getScalar<double>( "k11" );
    double k12  = d_general_db->getScalar<double>( "k12" );
    double k21  = d_general_db->getScalar<double>( "k21" );
    double k22  = d_general_db->getScalar<double>( "k22" );
    double z    = d_general_db->getScalar<double>( "z" );

    if ( d_general_db->getScalar<std::string>( "model" ) == "linear" ) {
        if ( component == 0 ) {
            double sE = std::pow(M_PI, 2)*k11*(std::pow(kX, 2) + std::pow(kY, 2))*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi) - M_PI*kT*std::sin(M_PI*kT*t)*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kY*y + kYPhi) + k12*(kE0 - std::cbrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi)) + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi));
            return sE;
        } else if ( component == 1 ) {
            double sT = ((1.0/9.0)*std::pow(M_PI, 2)*k21*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 4.0/3.0)*(3*(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi))*(std::pow(kX, 2) + std::pow(kY, 2))*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kY*y + kYPhi) + 2*(std::pow(kX, 2)*std::pow(std::cos(M_PI*kX*x + kXPhi), 2)*std::pow(std::cos(M_PI*kY*y + kYPhi), 2) + std::pow(kY, 2)*std::pow(std::sin(M_PI*kX*x + kXPhi), 2)*std::pow(std::sin(M_PI*kY*y + kYPhi), 2))*std::cos(M_PI*kT*t))*std::cos(M_PI*kT*t) - 1.0/3.0*M_PI*kT*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 7.0/3.0)*std::sin(M_PI*kT*t)*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kY*y + kYPhi) - k22*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 3)*(kE0 - std::cbrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi)) + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi)))/std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 3);
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( d_general_db->getScalar<std::string>( "model" ) == "nonlinear" ) {
        if ( component == 0 ) {
            double sE = (1.0/3.0)*(std::pow(M_PI, 2)*k11*kE0*std::pow(kX, 2)*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi) + std::pow(M_PI, 2)*k11*kE0*std::pow(kY, 2)*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi) + std::pow(M_PI, 2)*k11*std::pow(kX, 2)*std::pow(std::sin(M_PI*kX*x + kXPhi), 2)*std::pow(std::cos(M_PI*kT*t), 2)*std::pow(std::cos(M_PI*kY*y + kYPhi), 2) - std::pow(M_PI, 2)*k11*std::pow(kX, 2)*std::pow(std::cos(M_PI*kT*t), 2)*std::pow(std::cos(M_PI*kX*x + kXPhi), 2)*std::pow(std::cos(M_PI*kY*y + kYPhi), 2) - std::pow(M_PI, 2)*k11*std::pow(kY, 2)*std::pow(std::sin(M_PI*kX*x + kXPhi), 2)*std::pow(std::sin(M_PI*kY*y + kYPhi), 2)*std::pow(std::cos(M_PI*kT*t), 2) + std::pow(M_PI, 2)*k11*std::pow(kY, 2)*std::pow(std::sin(M_PI*kX*x + kXPhi), 2)*std::pow(std::cos(M_PI*kT*t), 2)*std::pow(std::cos(M_PI*kY*y + kYPhi), 2) - 3*M_PI*kT*std::pow(z, 3)*std::sin(M_PI*kT*t)*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kY*y + kYPhi) - 3*k12*std::pow(z, 6)*std::cbrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi)) + 3*k12*std::pow(z, 6))/std::pow(z, 3);
            return sE;
        } else if ( component == 1 ) {
            double sT = -std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), -0.75)*(std::pow(M_PI, 2)*k21*std::sqrt(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi))*(0.375*std::pow(kX, 2)*std::cos(M_PI*kT*t)*std::pow(std::cos(M_PI*kX*x + kXPhi), 2)*std::pow(std::cos(M_PI*kY*y + kYPhi), 2) + 0.375*std::pow(kY, 2)*std::pow(std::sin(M_PI*kX*x + kXPhi), 2)*std::pow(std::sin(M_PI*kY*y + kYPhi), 2)*std::cos(M_PI*kT*t) - 0.5*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 1.0)*(std::pow(kX, 2) + std::pow(kY, 2))*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kY*y + kYPhi))*std::cos(M_PI*kT*t) + 0.5*M_PI*kT*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 0.25)*std::sin(M_PI*kT*t)*std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kY*y + kYPhi) + k22*std::pow(z*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), -0.5), 3.0)*std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 0.75)*(kE0 - std::pow(kE0 + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi), 2.0) + std::sin(M_PI*kX*x + kXPhi)*std::cos(M_PI*kT*t)*std::cos(M_PI*kY*y + kYPhi)));
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }
    } else {
        AMP_ERROR( "Invalid model" );
    } 
}
