#ifndef included_AMP_PowerShape_h
#define included_AMP_PowerShape_h

// AMP files
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/libmesh/PowerShapeParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/Variable.h"
#include <memory>


// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature_gauss.h"
ENABLE_WARNINGS


#include <vector>


namespace AMP::Operator {


//===========================================================================//
/*!
 * \class PowerShape
 * \brief Provides the specific power distribution by node for a given mesh.
 */
//===========================================================================//

class PowerShape : public Operator
{
public:
    /*
     * A class for representing the neutronics source operator.
     */
    explicit PowerShape( std::shared_ptr<PowerShapeParameters> parameters );

    /**
     * Empty destructor for PowerShape
     */
    virtual ~PowerShape();

    //! Return the name of the operator
    std::string type() const override { return "PowerShape"; }

    /**
     * Print out all members of integrator instance to given output stream.
     */
    void printClassData( std::ostream &os ) const;

    /**
     * Write out state of object to given database.
     *
     * When assertion checking is active, the database pointer must be non-null.
     */
    void putToDatabase( std::shared_ptr<AMP::Database> db );

    /**
      The function that computes the residual.
     * @param u: multivector of the state.
     * @param f: specific power in Watts per gram
     The result of apply is
     * f = A(u)
     */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    /**
      A function to reinitialize this object.
      */
    void reset( std::shared_ptr<const OperatorParameters> parameters ) override;


    double evalFactorial( const int n );
    double choose( int, int );
    double evalZernike( const int m, const int n, const double rho, const double phi );
    double evalLegendre( const int n, const double x );
    double getVolumeIntegralSum( double rmax, double cx, double cy );
    double getFrapconFr( double radius, double rmax );
    double getZernikeRadial( double rho );
    double getZernike( double rho, double phi );
    double getGaussianF( double x, double y );

protected:
    std::shared_ptr<AMP::Database> d_db;

    std::shared_ptr<libMesh::FEType> d_feType;
    std::shared_ptr<libMesh::FEBase> d_fe;
    std::shared_ptr<libMesh::QBase> d_qrule;

    void createCurrentLibMeshElement();

    void destroyCurrentLibMeshElement();

    AMP::Mesh::MeshElementVectorPtr d_currNodes;

    libMesh::Elem *d_currElemPtr;

private:
    // Defines fission data types.
    enum class PowerShape_Types { LINEAR, QUADRATIC, CUBIC, NUM_POWER_SHAPES };

    bool d_useFixedPower;           // use a spatially constant power distribution
    std::string d_coordinateSystem; // Coordinate system
    std::string d_type;             // Cylindrical coordinate system
    unsigned int d_numXmoments;     // Number of moments in the x direction
    unsigned int d_numYmoments;     // Number of moments in the y direction.
    unsigned int d_numZmoments;     // Number of moments in the z direction.
    unsigned int d_numMoments;      // Number of moments in the Zernike basis function
    unsigned int d_numMNmoments;
    std::vector<double> d_Xmoments; // Moments in the x direction: size= d_numXmoments
    std::vector<double> d_Ymoments; // Moments in the y direction: size= d_numYmoments
    std::vector<double> d_Zmoments; // Moments in the z direction: size= d_numZmoments
    std::vector<double> d_Moments;  // Moments in the Zernike basis function
    double d_frapconConstant;       // Frapcon constant
    double d_angularConstant;       // Angular constan.

    // Radial Bounding Box
    // (0,1) = center (x,y)
    // (2,3) = radius (min, max)
    // (4,5) = height (min, max)
    std::vector<double> d_radialBoundingBox;

    // Gaussian constants
    double d_muX;
    double d_muY;
    double d_sigmaX;
    double d_sigmaY;
};
} // namespace AMP::Operator

#endif
