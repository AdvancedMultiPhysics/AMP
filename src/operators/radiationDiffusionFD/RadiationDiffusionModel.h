#ifndef RAD_DIF_MODEL
#define RAD_DIF_MODEL

#include "AMP/mesh/Mesh.h"
#include "AMP/utils/Database.h"
#include "AMP/mesh/MeshElement.h"


/* ----------------------------------------------------
    Class representing a Radiation diffusion equation 
---------------------------------------------------- */
/* Abstract base class representing a radiation-diffusion problem:
        u'(t) - L(u) - R(u)  = s(t), u(0) = u_0
    over the spatial domain [0,1]^d, for d = 1 or d = 2.

    where:
        1. L(u) = [grad \dot ( D0 * \grad u0 ), grad \dot ( D1 * \grad u1 )] is a nonlinear diffusion operator
        2. R(u) is a nonlinear reaction operator

    The vector u = [u0, u1] = [E, T] is a block vector, holding E and T.

    In more detail, the general PDE is of that discretizated by *TODO*:
        * L(u) = [grad \dot (k11*D_E \grad E), grad \dot (k21*D_T \grad T)]
            where:
                * if model == "linear": 
                    D_E = D_T = 1.0
                * if model == "nonlinear": 
                    D_E = 1/(3*sigma), D_T = T^2.5, and sigma = (z/T)^3
        
        * if model == "linear":
            R(u) = [k12*(T - E), -k22*(T - E)]
        * if model == "nonlinear":
            R(u) = [k12*simga*(T^4 - E), -k22*simga*(T^4 - E)]

    Boundary conditions come in two flavours:
        0. For E, Robin boundary conditions are specified on boundary k in the form of:
                ak * E + bk * n \dot k11*D_E * dE/dn = rk 
            for outward facing normal n
        1. For T, "Pseudo Neumann" boundary conditions are specified on boundary k in the form of:
                n \dot dT/dn = nk
            for outward facing normal n. Note these are not actually Neumann BCs unless nk=0 since they don't involve the flux, except in the special case of nk=0.
    The class is reponsible for providing the constants ak, bk, rk, nk

    In glorious detail, we have:
    1D. Robin on E:
        a1 * E + b1 * - k11*D_E * dE/dx = r1 at x = 0...
        a2 * E + b2 * + k11*D_E * dE/dx = r2 at x = 1...

        Pseudo Neumann on T 
                                -dT/dx = n1 at x = 0...
                                +dT/dx = n2 at x = 1...
        
    2D. Robin on E:
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

//
public:
    // The current time of the solution. 
    double d_currentTime = 0.0;
    // Does the derived class implement an exact solution?
    bool d_exactSolutionAvailable = false;
    // Shorthand for spatial dimension 
    int  d_dim                    = -1;

    // Basic parameter database (with model-agnostic parameters)
    std::shared_ptr<AMP::Database> d_basic_db;
    // Parameters specific to a model
    std::shared_ptr<AMP::Database> d_mspecific_db;
    // Database of parameters used to describe general PDE formulation; this is a merger and possible re-interpretation of the above two databases
    std::shared_ptr<AMP::Database> d_general_db = nullptr;
    // Flag derived classes must overwrite indicating they have constructed the above database
    bool d_general_db_completed = false;

    // Constructor
    RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> mspecific_db_ );

    // Destructor
    virtual ~RadDifModel() {};

    inline double getCurrentTime() const { return d_currentTime; };
    inline void setCurrentTime(double currentTime_) { d_currentTime = currentTime_; };

    /* Pure virtual functions */
    virtual double sourceTerm( int component, AMP::Mesh::MeshElement &node ) const = 0;
    virtual double initialCondition( int component, AMP::Mesh::MeshElement &node ) const = 0;

    /* Virtual functions */
    virtual double exactSolution( int , AMP::Mesh::MeshElement & ) const {
        AMP_ERROR( "Base class cannot provide a meaningful implementation of this function" );
    }

    // Get general database
    std::shared_ptr<AMP::Database> getGeneralPDEModelParameters( ) const { 
        AMP_INSIST( d_general_db_completed, "The derived class has not completed the construction of this database." );
        return d_general_db; };

//
private:

    /* Convert specific model parameters into parameters expected by the general formulation */ 
    virtual void finalizeGeneralPDEModel_db( ) = 0;
};


/* -----------------------------------------------------------------------------------------------
    Class representing certain Radiation diffusion equations considered by Mousseau et al. (2000)
------------------------------------------------------------------------------------------------ */
/* In particular: 
    1. The source term is zero
    2. There is a specific initial condition
    3. There are specific boundary conditions 

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

In working out the above constants, note that the energy diffusion flux is D_E = 1/3*sigma, so that 1/6*sigma = 0.5*D_E
*/
class Mousseau_etal_2000_RadDifModel : public RadDifModel {

//
public:

    // Constructor
    Mousseau_etal_2000_RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> specific_db_ );

    // Destructor
    virtual ~Mousseau_etal_2000_RadDifModel() {};
    
    double sourceTerm( int , AMP::Mesh::MeshElement & ) const override;
    double initialCondition( int component, AMP::Mesh::MeshElement & ) const override;

//
private:
    void finalizeGeneralPDEModel_db( ) override;
};


/* ------------------------------------------------------------------
    Class representing a manufactured radiation diffusion equation
------------------------------------------------------------------ */
/* In particular:
    1. An initial condition is provided
    2. An exact solution is provided
    3. A corresponding source term is provided
    4. Function handles for the corresponding E Robin values rk, and the T pseudo Neumann values nk are provided which accept the boundary id and spatial location on the boundary. (The manufactured solution is not constant along along any boundary) 

*/
class Manufactured_RadDifModel : public RadDifModel {

private:
    // Constants that the maufactured solutions depends on. 
    const double kE0   = 2.0;
    const double kT    = 1.7;
    const double kX    = 1.5;
    const double kXPhi = 0.245;
    const double kY    = 3.5;
    const double kYPhi = 0.784;
    const double kZ    = 2.5;
    const double kZPhi = 0.154;

//
public:

    // Constructor
    Manufactured_RadDifModel( std::shared_ptr<AMP::Database> basic_db_, std::shared_ptr<AMP::Database> specific_db_ );

    // Destructor
    virtual ~Manufactured_RadDifModel() {};

    double sourceTerm( int component, AMP::Mesh::MeshElement &node ) const override;
    double initialCondition( int component, AMP::Mesh::MeshElement &node ) const override;
    double exactSolution( int component, AMP::Mesh::MeshElement &node ) const override;
    double getRobinValueE( size_t boundaryID, double a, double b, AMP::Mesh::MeshElement & node ) const;
    double getPseudoNeumannValueT( size_t boundaryID, AMP::Mesh::MeshElement & node ) const;
    
//
private:
    // Flag used inside exactSolution_ functions to determine whether to use the currentTime or 0.0 when evaluating the function. This is a bit hacky, but allows for initialCondition to be a const function (without requiring currentTime to be mutable).
    mutable bool d_settingInitialCondition = false;

    void finalizeGeneralPDEModel_db( ) override;

    void getNormalVector( size_t boundaryID, std::string &direction, double &sign, double &x, double &y, double &z ) const;

    // Exact solution, its gradient and corresponding source term
    double exactSolution1D( int component, double x ) const;
    double exactSolutionGradient1D( int component, double x ) const;
    double sourceTerm1D( int component, double x ) const;
    //
    double exactSolution2D( int component, double x, double y ) const;
    double exactSolutionGradient2D( int component, double x, double y, const std::string & grad_component ) const;
    double sourceTerm2D( int component, double x, double y ) const;
    //
    double exactSolution3D( int component, double x, double y, double z ) const;
    double exactSolutionGradient3D( int component, double x, double y, double z, const std::string & grad_component ) const;
    double sourceTerm3D( int component, double x, double y, double z ) const;

    // Boundary-related functions
    double getRobinValueE1D_( size_t boundaryID, double a, double b ) const;
    double getPseudoNeumannValueT1D_( size_t boundaryID ) const;
    //
    double getRobinValueE2D_( size_t boundaryID, double a, double b, double x, double y ) const;
    double getPseudoNeumannValueT2D_( size_t boundaryID, double x, double y ) const;
    //
    double getRobinValueE3D_( size_t boundaryID, double a, double b, double x, double y, double z ) const;
    double getPseudoNeumannValueT3D_( size_t boundaryID, double x, double y, double z ) const;
}; 


#endif