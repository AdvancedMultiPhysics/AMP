#ifndef included_AMP_DiffusionRotatedAnisotropicModel
#define included_AMP_DiffusionRotatedAnisotropicModel

#include "AMP/mesh/Mesh.h"
#include "AMP/operators/Operator.h"

namespace AMP::Operator {


/* ----------------------------------------------------------------------------------
    Implementation of a rotated anisotropic constant-coefficient diffusion equation
---------------------------------------------------------------------------------- */
/* Abstract base class representing a linear diffusion problem:
        - grad \dot ( D * grad u ) = s.
    where the diffusion tensor D represents rotations and anisotropies. Specifically, the PDEs are
   as follows:

    - In 1D, the PDE is -u_xx = f.

    - In 2D, the PDE is -u_xx -eps*u_yy = f, but rotated an angle of theta radians counter clockwise
   from the positive x axis.
    - In 2D, the input database should specify the constants "eps", and "theta" otherwise they are
   set to default values of eps=1 and theta=0 (the isotropic case).

    - In 3D, the PDE is -u_xx -epsy*u_yy - epsz*u_zz = f, but rotated according in 3D according the
   the so-called "extrinsic" 3-1-3 Euler angles (see
   https://en.wikipedia.org/wiki/Euler_angles#Definition_by_extrinsic_rotations) gamma, beta, and
   alpha. A new coordinate system XYZ is obtained from rotating the xyz coordinate system by: 1.
   first rotating gamma radians around the z axis, 2. then beta radians about the x axis, 3. then
   alpha radians about the z axis. 19-point finite differences are used.
    - In 3D, the input database should specify the constants "epsy", "epsz", "gamma", "beta", and
   "alpha" otherwise they are set to default values of epsy=epsz=1 and gamma=beta=alpha=0 (the
   isotropic case).

    - Note that for eps = epsy = epsz = 1, the PDEs are isotropic and grid aligned (the rotation
   angles have no impact in this case).

    This base class translates the operator "- grad \dot ( D * grad u )" into the form of that
   discretized by DiffusionFDOperator
*/
class RotatedAnisotropicDiffusionModel
{

public:
    // Flag indicating whether a derived class provides an exact solution
    bool d_exactSolutionAvailable = false;
    // Shorthand for dimension of PDE
    size_t d_dim;
    // Diffusion coefficients, cxx, etc.
    std::shared_ptr<AMP::Database> d_c_db;
    // Database used to construct class
    std::shared_ptr<AMP::Database> d_input_db;

    // Constructor
    RotatedAnisotropicDiffusionModel( std::shared_ptr<AMP::Database> input_db );

    // Destructor
    virtual ~RotatedAnisotropicDiffusionModel() {}

    /* Pure virtual functions */
    virtual double sourceTerm( AMP::Mesh::MeshElement &node ) const = 0;

    /* Virtual functions */
    virtual double exactSolution( AMP::Mesh::MeshElement & ) const
    {
        AMP_ERROR( "Base class cannot provide an implementation of this function" );
    }

private:
    void setDiffusionCoefficients();
    std::vector<double> getSecondOrderPDECoefficients1D() const;
    std::vector<double> getSecondOrderPDECoefficients2D() const;
    std::vector<double> getSecondOrderPDECoefficients3D() const;
};


/* ----------------------------------------------------------------------------------------------
    Implementation of a MANUFACTURED rotated anisotropic constant-coefficient diffusion equation
---------------------------------------------------------------------------------------------- */
/* A source term and corresponding exact solution are provided */
class ManufacturedRotatedAnisotropicDiffusionModel : public RotatedAnisotropicDiffusionModel
{

    //
public:
    // Constructor
    ManufacturedRotatedAnisotropicDiffusionModel( std::shared_ptr<AMP::Database> input_db )
        : RotatedAnisotropicDiffusionModel( input_db )
    {
        // Set flag indicating this class does provide an implementation of exactSolution
        d_exactSolutionAvailable = true;
    }

    // Destructor
    virtual ~ManufacturedRotatedAnisotropicDiffusionModel() {}

    double sourceTerm( AMP::Mesh::MeshElement &node ) const override;
    double exactSolution( AMP::Mesh::MeshElement &node ) const override;

    //
private:
    // Exact solution, and corresponding source term
    // 1D
    double exactSolution_( double x ) const;
    double sourceTerm_( double x ) const;
    // 2D
    double exactSolution_( double x, double y ) const;
    double sourceTerm_( double x, double y ) const;
    // 3D
    double exactSolution_( double x, double y, double z ) const;
    double sourceTerm_( double x, double y, double z ) const;

    // Constants to translate the exact solutions (trig functions), so as to avoid cases of these
    // functions being zero (and or constant) along boundaries (providing better opportunities for
    // identifying bugs in boundary computations).
    constexpr static double d_X_SHIFT = -0.325;
    constexpr static double d_Y_SHIFT = +0.987;
    constexpr static double d_Z_SHIFT = -0.478;
};
} // namespace AMP::Operator

#endif
