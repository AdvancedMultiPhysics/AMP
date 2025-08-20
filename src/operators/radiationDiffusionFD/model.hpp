#ifndef RAD_DIF_MODEL_
#define RAD_DIF_MODEL_

#include "discretization.hpp"



/* ------------------------------------------------
    Class radiation diffusion equation 
------------------------------------------------- */
/* Abstract base class representing a radiation-diffusion problem:
        u'(t) - L(u) - R(u)  = s(t), u(0) = u_0
    over the spatial domain [0,1]^d, for d = 1 or d = 2.

    where:
        1. L(u) = grad \dot ( D * \grad u ) is a nonlinear diffusion operator
        2. R(u) is a nonlinear reaction operator

    The vector u = [E, T] is a block vector, holding E and T.

    I call boundary conditions on T "Pseudo Neumann," which are not actually Neumann unless nk=0 since they don't involve the flux, except in the special case that the RHS is zero.

    Boundary conditions for the problem take the form:
    1D. 
    Robin on E:
        a1 * E + b1 * - k11*D_E * dE/dx = r1 at x = 0...
        a2 * E + b2 * + k11*D_E * dE/dx = r2 at x = 1...

    Pseudo Neumann on T 
                                -dT/dx = n1 at x = 0...
                                +dT/dx = n2 at x = 1...
        
    2D. 
    Robin on E:
        a1 * E + b1 * -k11*D_E * dE/dx = r1 at x = 0...
        a2 * E + b2 * +k11*D_E * dE/dx = r2 at x = 1...
        a3 * E + b3 * -k11*D_E * dE/dy = r3 at y = 0...
        a4 * E + b4 * +k11*D_E * dE/dy = r4 at y = 1...

    Pseudo Neumann on T:
                                -dT/dx = n1 at x = 0...
                                +dT/dx = n2 at x = 1...
                                -dT/dy = n3 at y = 0...
                                +dT/dy = n4 at y = 1...


    The class should be initialized with two databases: 
        1. A basic PDE database
        2. A model-specific database
*/
class RadDifModel {

public:

    // The current time of the solution
    double d_currentTime = 0.0;
    // Does the derived class implement an exact solution?
    bool d_exactSolutionAvailable = false;
    // Shorthand for spatial dimension since we reference it so much
    int  d_dim                    = -1;

    // Basic parameter database (with model-agnostic parameters)
    std::shared_ptr<AMP::Database> d_basic_db;
    // Parameters specific to a model
    std::shared_ptr<AMP::Database> d_mspecific_db;
    // Database of parameters used to describe general PDE formulation; this is a merger and possible re-interpretation of the above two
    std::shared_ptr<AMP::Database> d_general_db;

    RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> mspecific_db_ ) : d_basic_db( basic_db_ ), d_mspecific_db( mspecific_db_ ) { 

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

    inline double getCurrentTime() { return d_currentTime; };
    inline void setCurrentTime(double currentTime_) { d_currentTime = currentTime_; };

    /* Pure virtual functions */
    virtual double sourceTerm( int component, AMP::Mesh::MeshElement &node ) = 0;
    virtual double initialCondition( int component, AMP::Mesh::MeshElement &node ) = 0;

    /* Virtual functions */
    virtual double exactSolution( int component, AMP::Mesh::MeshElement & ) {
        AMP_ERROR( "Base class cannot provide an implementation of this function" );
    }

    // Get general database
    std::shared_ptr<AMP::Database> getGeneralPDEModelParameters( ) { 
        AMP_INSIST( d_general_db, "General PDE database is null; it should have been build during construction of derived class" );
        return d_general_db; };

private:

    /* Convert specific model parameters into parameters expected by the general formulation */ 
    virtual void finalizeGeneralPDEModel_db( ) = 0;
};


/* -------------------------------------------------------
    Class implementing basic radiation-diffusion equation 
---------------------------------------------------------- */
/* With a zero source s(t) = 0 and some specific initial condition 

The Energy diffusion flux is D_E = 1/3*sigma, so 1/6*sigma = 0.5*D_E

For the 1D problem:
-------------------
(58) at x=0: 1/4*E - 1/6*sigma * dE/dx = 1 -> a1 = 0.25, b1 = 0.5, r1 = 1
(59) at x=1: 1/4*E + 1/6*sigma * dE/dx = 0 -> a2 = 0.25, b2 = 0.5, r2 = 0
     at x=0: dT/dx = 0: -> n1 = 0
     at x=1: dT/dx = 0: -> n2 = 0 

For the 2D problem:
-------------------
(61) == (58) at x=0:       -> a1 = 0.25, b1 = 0.5, r1 = 1
(62) == (59) at x=1:       -> a2 = 0.25, b2 = 0.5, r2 = 0
(63.1) at y = 0: dE/dy = 0 -> a3 = 0,    b3 = anything nonzero, r3 = 0
(63.2) at y = 1: dE/dy = 0 -> a4 = 0,    b4 = anything nonzero, r4 = 0

(64) at y=0: dT/dy = 0: -> n3 = 0
     at y=1: dT/dy = 0: -> n4 = 0 

(65) at x=0: dT/dx = 0: -> n1 = 0
     at x=1: dT/dx = 0: -> n2 = 0 

*/

// See Mousseau et al. (2000), sec 4.1.
class Mousseau_etal_2000_RadDifModel : public RadDifModel {

public:

    Mousseau_etal_2000_RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> specific_db_ ) : RadDifModel( basic_db_, specific_db_ )  { 
        finalizeGeneralPDEModel_db();
    }
    double sourceTerm( int component, AMP::Mesh::MeshElement &node ) { return 0.0; }
    double initialCondition( int component, AMP::Mesh::MeshElement &node ) { 
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

    
private:

    void finalizeGeneralPDEModel_db( ) override {
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
        d_general_db->putScalar<double>( "n3", r3 );
        d_general_db->putScalar<double>( "n4", n4 );

        d_general_db->putScalar<double>( "z",   z   );
        d_general_db->putScalar<double>( "k11", k11 );
        d_general_db->putScalar<double>( "k12", k12 );
        d_general_db->putScalar<double>( "k21", k21 );
        d_general_db->putScalar<double>( "k22", k22 );

        d_general_db->putScalar<std::string>( "model", model );

        d_general_db->setDefaultAddKeyBehavior( AMP::Database::Check::Error, false ); 
    }
};


/* -------------------------------------------------------------------------
    Class implementing manufactured radiation-diffusion equation equation 
------------------------------------------------------------------------- */
/* A source term and exact solution are provided, as well as functions to get the Robin and pseudo Neumann BC values

*/
class Manufactured_RadDifModel : public RadDifModel {

public:

    // Call base class' constructor
    Manufactured_RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> specific_db_ ) : RadDifModel( basic_db_, specific_db_ ) { 
        // Set flag indicating this class does provide an implementation of exactSolution
        d_exactSolutionAvailable = true;

        finalizeGeneralPDEModel_db( );
    }
    
    // Implementation of pure virtual function
    // Dimesionless wrapper around the exact source term functions
    double sourceTerm( int component, AMP::Mesh::MeshElement &node ) override {
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
    double initialCondition( int component, AMP::Mesh::MeshElement &node ) override {
        double currentTime_ = this->getCurrentTime();
        this->setCurrentTime( 0.0 );
        double ic = exactSolution( component, node );
        this->setCurrentTime( currentTime_ );
        return ic;
    }

    // Dimesionless wrapper around the exact solution functions
    double exactSolution( int component, AMP::Mesh::MeshElement &node ) override {
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

    // Dimesionless wrapper around the Robin functions
    double getRobinValueE( int boundary, double a, double b, AMP::Mesh::MeshElement & node ) {
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

    // Dimesionless wrapper around the pseudo-Neumann functions
    double getPseudoNeumannValueT( int boundary, AMP::Mesh::MeshElement & node ) {
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

    
private:
    // Exact solution, its gradient and corresponding source term
    double exactSolution_( int component, double x );
    double exactSolutionGradient_( int component, double x );
    double sourceTerm_( int component, double x );
    //
    double exactSolution_( int component, double x, double y );
    double exactSolutionGradient_( int component, double x, double y, std::string grad_component );
    double sourceTerm_( int component, double x, double y );

    // Boundary-related functions
    double getRobinValueE1D_( int boundary, double a, double b );
    double getPseudoNeumannValueT1D_( int boundary );
    //
    double getRobinValueE2D_( int boundary, double a, double b, double x, double y );
    double getPseudoNeumannValueT2D_( int boundary, double x, double y );


    void finalizeGeneralPDEModel_db( ) override {  
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

        d_general_db->setDefaultAddKeyBehavior( AMP::Database::Check::Error, false ); 
    }
}; 

/* 1D. The boundaries of 0 and 1 are hard coded here. 
    Robin on E:
        a1 * E + b1 * - k11*D_E * dE/dx = r1 at x = 0...
        a2 * E + b2 * + k11*D_E * dE/dx = r2 at x = 1...
*/
double Manufactured_RadDifModel::getRobinValueE1D_( int boundary, double a, double b ) {

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
double Manufactured_RadDifModel::getPseudoNeumannValueT1D_( int boundary ) {

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
double Manufactured_RadDifModel::getRobinValueE2D_( int boundary, double a, double b, double x, double y ) {

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
double Manufactured_RadDifModel::getPseudoNeumannValueT2D_( int boundary, double x, double y ) {

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

    double T    = exactSolution_( 1, x, y );
    double dTdn = exactSolutionGradient_( 1, x, y, normal_direction );

    return normal_sign * dTdn;
}

// Implementation of 1D functions
double Manufactured_RadDifModel::exactSolution_( int component, double x ) {
    double t = this->getCurrentTime();
    if ( component == 0 ) {
        double E = std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2;
        return E;
    } else if ( component == 1 ) {
        double T = std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 0.25);
        return T;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::exactSolutionGradient_( int component, double x ) {
    double t = this->getCurrentTime();
    if ( component == 0 ) {
        double dEdx = 1.5*M_PI*std::cos(2*M_PI*t)*std::cos(1.5*M_PI*x);
        return dEdx;
    } else if ( component == 1 ) {
        double dTdx = 0.375*M_PI*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.75)*std::cos(2*M_PI*t)*std::cos(1.5*M_PI*x);
        return dTdx;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::sourceTerm_( int component, double x ) {
    
    double t    = this->getCurrentTime();

    // Unpack parameters
    double k11  = d_general_db->getScalar<double>( "k11" );
    double k12  = d_general_db->getScalar<double>( "k12" );
    double k21  = d_general_db->getScalar<double>( "k21" );
    double k22  = d_general_db->getScalar<double>( "k22" );
    double z    = d_general_db->getScalar<double>( "z" );

    if ( d_general_db->getScalar<std::string>( "model" ) == "linear" ) {
        if ( component == 0 ) {
            double sE = 2.25*std::pow(M_PI, 2)*k11*std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) - 2*M_PI*std::sin(2*M_PI*t)*std::sin(1.5*M_PI*x) + k12*(-std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 0.25) + std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2);
            return sE;
        } else if ( component == 1 ) {
            double sT = 0.421875*std::pow(M_PI, 2)*k21*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -1.75)*std::pow(std::cos(2*M_PI*t), 2)*std::pow(std::cos(1.5*M_PI*x), 2) + 0.5625*std::pow(M_PI, 2)*k21*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.75)*std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) - 0.5*M_PI*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.75)*std::sin(2*M_PI*t)*std::sin(1.5*M_PI*x) + k22*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 0.25) - k22*std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) - 2*k22;
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( d_general_db->getScalar<std::string>( "model" ) == "nonlinear" ) {
        if ( component == 0 ) {
            double sE = -0.5625*std::pow(M_PI, 2)*k11*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.25), -3.0)*1.0/(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2)*std::pow(std::cos(2*M_PI*t), 2)*std::pow(std::cos(1.5*M_PI*x), 2) + 0.75*std::pow(M_PI, 2)*k11*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.25), -3.0)*std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) - 2*M_PI*std::sin(2*M_PI*t)*std::sin(1.5*M_PI*x) - k12*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.25), 3.0)*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 1.0) + k12*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.25), 3.0)*std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2*k12*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.25), 3.0);
            return sE;
        } else if ( component == 1 ) {
            double sT = std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -2.0)*(std::pow(M_PI, 2)*k21*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 0.75)*(0.0703125*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 0.125)*std::cos(2*M_PI*t)*std::pow(std::cos(1.5*M_PI*x), 2) + 0.5625*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 1.125)*std::sin(1.5*M_PI*x))*std::cos(2*M_PI*t) - 0.5*M_PI*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 1.25)*std::sin(2*M_PI*t)*std::sin(1.5*M_PI*x) - k22*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, -0.25), 3.0)*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 2.0)*(-std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2, 1.0) + std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t) + 2));
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }
    } else {
        AMP_ERROR( "Invalid model" );
    } 
}

// Implementation of 2D functions
double Manufactured_RadDifModel::exactSolution_( int component, double x, double y ) {
    double t = this->getCurrentTime();
    if ( component == 0 ) {
        double E = std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2;
        return E;
    } else if ( component == 1 ) {
        double T = std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 0.25);
        return T;
    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::exactSolutionGradient_( int component, double x, double y, std::string grad_component ) {

    double t = this->getCurrentTime();
    if ( component == 0 ) {
        if ( grad_component == "x" ) {
            double dEdx = 1.5*M_PI*std::cos(2*M_PI*t)*std::cos(1.5*M_PI*x)*std::cos(3.5*M_PI*y);
            return dEdx;
        } else if ( grad_component == "y" ) {
            double dEdy = -3.5*M_PI*std::sin(1.5*M_PI*x)*std::sin(3.5*M_PI*y)*std::cos(2*M_PI*t);
            return dEdy;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( component == 1 ) {
        if ( grad_component == "x" ) {
            double dTdx = 0.375*M_PI*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.75)*std::cos(2*M_PI*t)*std::cos(1.5*M_PI*x)*std::cos(3.5*M_PI*y);
            return dTdx;
        } else if ( grad_component == "y" ) {
            double dTdy = -0.875*M_PI*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.75)*std::sin(1.5*M_PI*x)*std::sin(3.5*M_PI*y)*std::cos(2*M_PI*t);
            return dTdy;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else {
        AMP_ERROR( "Invalid component" );
    }
}

double Manufactured_RadDifModel::sourceTerm_( int component, double x, double y ) {

    double t    = this->getCurrentTime();

    // Unpack parameters
    double k11  = d_general_db->getScalar<double>( "k11" );
    double k12  = d_general_db->getScalar<double>( "k12" );
    double k21  = d_general_db->getScalar<double>( "k21" );
    double k22  = d_general_db->getScalar<double>( "k22" );
    double z    = d_general_db->getScalar<double>( "z" );

    if ( d_general_db->getScalar<std::string>( "model" ) == "linear" ) {
        if ( component == 0 ) {
            double sE = 14.5*std::pow(M_PI, 2)*k11*std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) - 2*M_PI*std::sin(2*M_PI*t)*std::sin(1.5*M_PI*x)*std::cos(3.5*M_PI*y) + k12*(-std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 0.25) + std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2);
            return sE;
        } else if ( component == 1 ) {
            double sT = 2.296875*std::pow(M_PI, 2)*k21*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -1.75)*std::pow(std::sin(1.5*M_PI*x), 2)*std::pow(std::sin(3.5*M_PI*y), 2)*std::pow(std::cos(2*M_PI*t), 2) + 0.421875*std::pow(M_PI, 2)*k21*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -1.75)*std::pow(std::cos(2*M_PI*t), 2)*std::pow(std::cos(1.5*M_PI*x), 2)*std::pow(std::cos(3.5*M_PI*y), 2) + 3.625*std::pow(M_PI, 2)*k21*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.75)*std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) - 0.5*M_PI*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.75)*std::sin(2*M_PI*t)*std::sin(1.5*M_PI*x)*std::cos(3.5*M_PI*y) + k22*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 0.25) - k22*std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) - 2*k22;
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }

    } else if ( d_general_db->getScalar<std::string>( "model" ) == "nonlinear" ) {
        if ( component == 0 ) {
            double sE = std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.25), -6.0)*1.0/(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2)*(-std::pow(M_PI, 2)*k11*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.25), 3.0)*((3.0624999999999996*std::pow(std::sin(1.5*M_PI*x), 2)*std::pow(std::sin(3.5*M_PI*y), 2) + 0.5625*std::pow(std::cos(1.5*M_PI*x), 2)*std::pow(std::cos(3.5*M_PI*y), 2))*std::cos(2*M_PI*t) - 4.833333333333333*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 1.0)*std::sin(1.5*M_PI*x)*std::cos(3.5*M_PI*y))*std::cos(2*M_PI*t) + std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.25), 6.0)*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 1.0)*(-2*M_PI*std::sin(2*M_PI*t)*std::sin(1.5*M_PI*x)*std::cos(3.5*M_PI*y) + k12*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.25), 3.0)*(-std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 1.0) + std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2)));
            return sE;
        } else if ( component == 1 ) {
            double sT = std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -2.0)*(std::pow(M_PI, 2)*k21*((0.3828125*std::pow(std::sin(1.5*M_PI*x), 2)*std::pow(std::sin(3.5*M_PI*y), 2) + 0.0703125*std::pow(std::cos(1.5*M_PI*x), 2)*std::pow(std::cos(3.5*M_PI*y), 2))*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 0.125)*std::cos(2*M_PI*t) + 3.625*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 1.125)*std::sin(1.5*M_PI*x)*std::cos(3.5*M_PI*y))*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 0.75)*std::cos(2*M_PI*t) - 0.5*M_PI*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 1.25)*std::sin(2*M_PI*t)*std::sin(1.5*M_PI*x)*std::cos(3.5*M_PI*y) - k22*std::pow(z*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, -0.25), 3.0)*std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 2.0)*(-std::pow(std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2, 1.0) + std::sin(1.5*M_PI*x)*std::cos(2*M_PI*t)*std::cos(3.5*M_PI*y) + 2));
            return sT;
        } else {
            AMP_ERROR( "Invalid component" );
        }
    } else {
        AMP_ERROR( "Invalid model" );
    } 
}


#endif