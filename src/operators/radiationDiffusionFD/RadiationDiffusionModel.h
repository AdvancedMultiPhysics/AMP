#ifndef RAD_DIF_MODEL
#define RAD_DIF_MODEL

#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/utils/Constants.h"
#include "AMP/utils/Database.h"

namespace AMP::Operator {

#define PI AMP::Constants::pi

/** -------------------------------------------------------- *
 * --- Class representing a Radiation diffusion equation --- *
 * --------------------------------------------------------- */
/** Abstract base class representing the radiation-diffusion equation that's spatially discretized
 * by the class RadiationDiffusionFD.
 *
 * The class should be initialized with two databases:
 *  1. A basic PDE database, basic_db. This includes:
 *      "dim", "fluxLimited", "print_info_level".
 *  2. A model-specific database, mspecific_db. This should include any and all parameters that
 * when combined with basic_db provides sufficient context for the derived class to create a
 * RadiationDiffusionFD_input_db, which is a database suitable for creating an instance of a
 * RadiationDiffusionFD.
 *
 * The RadiationDiffusionFD_input_db database is obtained via getRadiationDiffusionFD_input_db()
 *
 * Derived classes can provide additional functionality such as providing exact solutions to the
 * PDE.
 */
class RadDifModel
{

public:
    //! Constructor
    RadDifModel( std::shared_ptr<AMP::Database> basic_db_,
                 std::shared_ptr<AMP::Database> mspecific_db_ );

    //! Destructor
    virtual ~RadDifModel(){};

    //! Get current time of class (e.g., as may be used in sourceTerm)
    double getCurrentTime() const;

    //! Set current time of class
    void setCurrentTime( double currentTime_ );

    //! Solution-independent source term in PDE at the given point
    virtual double sourceTerm( size_t component, const AMP::Mesh::Point &point ) const = 0;

    //! Initial condition of PDE at the given point
    virtual double initialCondition( size_t component, const AMP::Mesh::Point &point ) const = 0;

    //! Exact solution of PDE for given component at the given point
    virtual double exactSolution( size_t component, const AMP::Mesh::Point &point ) const;

    //! Get database suitable for creating an instance of a RadDifOp
    std::shared_ptr<AMP::Database> getRadiationDiffusionFD_input_db() const;

    //! Does the derived class implement an exact solution?
    bool exactSolutionAvailable() const;

    //
protected:
    //! The current time of the solution.
    double d_currentTime = 0.0;

    //! Shorthand for spatial dimension
    size_t d_dim = -1;

    //! Basic parameter database (with model-agnostic parameters)
    std::shared_ptr<AMP::Database> d_basic_db;

    //! Parameters specific to a model
    std::shared_ptr<AMP::Database> d_mspecific_db;

    //! Database of parameters required to create an instance of a RadiationDiffusionFD.
    std::shared_ptr<AMP::Database> d_RadiationDiffusionFD_input_db = nullptr;

    //! Flag derived classes must overwrite indicating they have constructed the above database
    bool d_RadiationDiffusionFD_input_db_completed = false;

    /* Convert specific model parameters into parameters expected by the general formulation */
    virtual void finalizeGeneralPDEModel_db() = 0;

    //! Does the derived class implement an exact solution?
    bool d_exactSolutionAvailable = false;

    //! Get the Robin constants ak and bk from the d_RadiationDiffusionFD_input_db for the given
    //! boundaryID
    void getLHSRobinConstantsFromDB( size_t boundaryID, double &ak, double &bk ) const;

    //! Energy diffusion coefficient D_E given temperature T
    double diffusionCoefficientE( double T, double zatom ) const;
};


/* -----------------------------------------------------------------------------------------------
    Class representing certain Radiation diffusion equations considered by Mousseau et al. (2000)
------------------------------------------------------------------------------------------------ */
/** In particular:
 * 1. The source term is zero
 * 2. There is a specific initial condition
 * 3. There are specific boundary conditions (see below)
 *
 * The boundary conditions are below, along with the calculations showing how they fit into the
 * general RadDifModel considered above.
 * For the 1D problem:
 * -------------------
 * (58) at x=0: 1/4*E - 1/6*sigma * dE/dx = 1 -> a1 = 0.25, b1 = 0.5, r1 = 1
 * (59) at x=1: 1/4*E + 1/6*sigma * dE/dx = 0 -> a2 = 0.25, b2 = 0.5, r2 = 0
 *  at x=0: dT/dx = 0: -> n1 = 0
 *  at x=1: dT/dx = 0: -> n2 = 0
 *
 * For the 2D problem:
 * -------------------
 * (61) == (58) at x=0:       -> a1 = 0.25, b1 = 0.5, r1 = 1
 * (62) == (59) at x=1:       -> a2 = 0.25, b2 = 0.5, r2 = 0
 * (63.1) at y = 0: dE/dy = 0 -> a3 = 0,    b3 = anything nonzero, r3 = 0
 * (63.2) at y = 1: dE/dy = 0 -> a4 = 0,    b4 = anything nonzero, r4 = 0
 *
 * (64) at y=0: dT/dy = 0: -> n3 = 0
 *      at y=1: dT/dy = 0: -> n4 = 0
 * (65) at x=0: dT/dx = 0: -> n1 = 0
 *      at x=1: dT/dx = 0: -> n2 = 0
 *
 * In working out the above constants, note that the energy diffusion flux is D_E = 1/3*sigma, so
 * that 1/6*sigma = 0.5*D_E
 * The incoming mspecific_db should have the two parameters:
 *  z -- atomic number
 *  k -- coefficient in the temperature diffusion flux
 *
 * See: Physics-Based Preconditioning and the Newton–Krylov Method for Non-equilibrium Radiation
 * Diffusion, V. A. Mousseau, D. A. Knoll, and W. J. Rider, Journal of Computational Physics 160,
 * 743–765 (2000)
 */
class Mousseau_etal_2000_RadDifModel : public RadDifModel
{

    //
public:
    Mousseau_etal_2000_RadDifModel( std::shared_ptr<AMP::Database> basic_db_,
                                    std::shared_ptr<AMP::Database> mspecific_db_ );

    virtual ~Mousseau_etal_2000_RadDifModel(){};

    double sourceTerm( size_t component, const AMP::Mesh::Point &point ) const override;

    double initialCondition( size_t component, const AMP::Mesh::Point &point ) const override;

    //
private:
    void finalizeGeneralPDEModel_db() override;
};


/* ------------------------------------------------------------------
    Class representing a manufactured radiation diffusion equation
------------------------------------------------------------------ */
/** In particular:
 * 1. An initial condition is provided
 * 2. An exact solution is provided
 * 3. A corresponding source term is provided
 * 4. Function handles for the corresponding E Robin values rk, and the T pseudo Neumann values nk
 * are provided which accept the boundary id and spatial location on the boundary. (The
 * manufactured solution is not constant along along any boundary)
 */
class Manufactured_RadDifModel : public RadDifModel
{

private:
    // Constants that the maufactured solutions depends on.
    constexpr static double kE0   = 2.0;
    constexpr static double kT    = 1.7;
    constexpr static double kX    = 1.5;
    constexpr static double kXPhi = 0.245;
    constexpr static double kY    = 3.5;
    constexpr static double kYPhi = 0.784;
    constexpr static double kZ    = 2.5;
    constexpr static double kZPhi = 0.154;

    //
public:
    Manufactured_RadDifModel( std::shared_ptr<AMP::Database> basic_db_,
                              std::shared_ptr<AMP::Database> specific_db_ );

    virtual ~Manufactured_RadDifModel(){};

    double sourceTerm( size_t component, const AMP::Mesh::Point &point ) const override;

    double initialCondition( size_t component, const AMP::Mesh::Point &point ) const override;

    double exactSolution( size_t component, const AMP::Mesh::Point &point ) const override;

    /** Return the value of the LHS of the Robin boundary equation on E. That is,
     *      ak*E + bk*k11*D_E * hat{nk}*grad(E)
     * @param[in] boundaryID ID of the boundary
     * @param[in] point point on the boundary where the expression is to be evaluated.
     * @note there is no redundancy here despite the boundaryID being specified because this class
     * does not know where a given boundary is located in space.
     */
    double getBoundaryFunctionValueE( size_t boundaryID, const AMP::Mesh::Point &point ) const;

    /** Return the value of the LHS of the pseudo Neumann boundary equation on T. That is,
     *      hat{nk}*grad(T)
     * @param[in] boundaryID ID of the boundary
     * @param[in] point point on the boundary where the expression is to be evaluated.
     * @note there is no redundancy here despite the boundaryID being specified because this class
     * does not know where a given boundary is located in space.
     */
    double getBoundaryFunctionValueT( size_t boundaryID, const AMP::Mesh::Point &point ) const;

    //
private:
    // Flag used inside exactSolution_ functions to determine whether to use the currentTime or 0.0
    // when evaluating the function. This is a bit hacky, but allows for initialCondition to be a
    // const function (without requiring currentTime to be mutable).
    mutable bool d_settingInitialCondition = false;

    void finalizeGeneralPDEModel_db() override;

    //! Get component and sign of normal vector given the boundaryID.
    void getNormalVector( size_t boundaryID, size_t &normalComponent, double &normalSign ) const;

    //! Dimension-agnostic wrapper around exactSolutionGradient_ functions
    double exactSolutionGradient( size_t component,
                                  const AMP::Mesh::Point &point,
                                  size_t gradComponent ) const;

    // Exact solution, its gradient and corresponding source term
    double exactSolution1D( size_t component, double x ) const;
    double exactSolutionGradient1D( size_t component, double x ) const;
    double sourceTerm1D( size_t component, double x ) const;
    //
    double exactSolution2D( size_t component, double x, double y ) const;
    double
    exactSolutionGradient2D( size_t component, double x, double y, size_t gradComponent ) const;
    double sourceTerm2D( size_t component, double x, double y ) const;
    //
    double exactSolution3D( size_t component, double x, double y, double z ) const;
    double exactSolutionGradient3D(
        size_t component, double x, double y, double z, size_t gradComponent ) const;
    double sourceTerm3D( size_t component, double x, double y, double z ) const;
};


} // namespace AMP::Operator

#endif