#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"

#include <memory>
#include <string>
#include <vector>


namespace AMP::MechanicsManufacturedSolution {

/**
 * This is a base class for mechanics manufactured solutions
 *
 * The MMS class provide functions to evaluate exact solution for the displacement @f$ u_i @f$
 * and corresponding forcing term @f$ f_i = - \sigma_{ij,j} @f$
 * where @f$ i \in \{x,y,z\}  @f$
 */
class MMS
{
public:
    /**
     * Default constructor
     * @param YM  Youngs modulus
     * @param PR  Poissons ratio
     *
     * Note: Youngs modulus E and Poissons ratio nu are assumed to be constant here but this can be
     * change later.
     * If the need for it arise shout me an email <dalg24@ne.tamu.edu>
     */
    MMS( double YM = 0.0, double PR = 0.0 );

    //! Destructor
    virtual ~MMS() {}

    /**
     * Mutator for Youngs modulus E
     */
    void setYoungsModulus( double YM ) { E = YM; }

    /**
     * Mutator for Poissons modulus nu
     */
    void setPoissonsRatio( double PR ) { nu = PR; }

    /**
     * Mutators to set scaling factor @f$ L_{i} @f$ and multiplying factor @f$ M_{i} @f$ for
     * displacement along axes @f$
     * i \in \{x,y,z\} @f$
     * Defaults values are @f$ L_{i} = 1 @f$ and @f$ M_{i} = 1 @f$
     */
    void scaleX( double L ) { Lx = L; }
    void scaleY( double L ) { Ly = L; }
    void scaleZ( double L ) { Lz = L; }
    void scaleXYZ( double L )
    {
        Lx = L;
        Ly = L;
        Lz = L;
    }
    void multX( double M ) { Mx = M; }
    void multY( double M ) { My = M; }
    void multZ( double M ) { Mz = M; }
    void multXYZ( double M )
    {
        Mx = M;
        My = M;
        Mz = M;
    }

    /**
     * Mutators for the coefficients @f$ a_{ij} @f$ and @f$ b_{ij} @f$ for all @f$ (i,j) \in
     * \{x,y,z\} @f$
     * Default values are @f$ a_{ij} = 0 @f$ and @f$ b_{ij} = 1 @f$
     * By setting one of the @f$ b_{ij} @f$ to zero you impose @f$ u_i = 0 @f$
     */
    void set_axx( double v ) { axx = v; }
    void set_bxx( double v ) { bxx = v; }
    void set_axy( double v ) { axy = v; }
    void set_bxy( double v ) { bxy = v; }
    void set_axz( double v ) { axz = v; }
    void set_bxz( double v ) { bxz = v; }
    void set_ayx( double v ) { ayx = v; }
    void set_byx( double v ) { byx = v; }
    void set_ayy( double v ) { ayy = v; }
    void set_byy( double v ) { byy = v; }
    void set_ayz( double v ) { ayz = v; }
    void set_byz( double v ) { byz = v; }
    void set_azx( double v ) { azx = v; }
    void set_bzx( double v ) { bzx = v; }
    void set_azy( double v ) { azy = v; }
    void set_bzy( double v ) { bzy = v; }
    void set_azz( double v ) { azz = v; }
    void set_bzz( double v ) { bzz = v; }

    /**
     * Accessor to the name/type of mechanics manufactured solution that have been instancied
     */
    const std::string &getName() const { return name; }

    /**
     * These are overridden in derived classes and constains exact solutions and forcing terms for
     * the displacement
     * along the i-axis
     */
    virtual double getExactSolutionX( double x, double y, double z ) const = 0;
    virtual double getExactSolutionY( double x, double y, double z ) const = 0;
    virtual double getExactSolutionZ( double x, double y, double z ) const = 0;
    virtual double getForcingTermX( double x, double y, double z ) const   = 0;
    virtual double getForcingTermY( double x, double y, double z ) const   = 0;
    virtual double getForcingTermZ( double x, double y, double z ) const   = 0;
    // I deactivated this because there was no need for it so far but I can provide these on request
    // virtual std::vector<double> getGradientX(double x, double y, double z) const = 0;
    // virtual std::vector<double> getGradientY(double x, double y, double z) const = 0;
    // virtual std::vector<double> getGradientZ(double x, double y, double z) const = 0;
    // virtual std::vector<double> getStrainTensor(double x, double y, double z) const = 0;

    /**
     * This is needed for traction boundary conditions
     * The function will return the stress tensor @f$ \sigma @f$ under the form of a vector
     *   @f$ (\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xz}, \sigma_{xy})^T @f$
     */
    virtual std::vector<double> getStressTensor( double x, double y, double z ) const = 0;

    /**
     * All the following points to the method described above and is provided for conveniance only
     * You may add as many as you want
     * For instance if you want to pass a Point directly as an argument or if you want to pass
     * vectors of integration
     * point (then do it by constant reference!!)
     */
    std::vector<double> getExactSolutions( double x, double y, double z ) const
    {
        std::vector<double> ExactSolutions;
        ExactSolutions.push_back( getExactSolutionX( x, y, z ) );
        ExactSolutions.push_back( getExactSolutionY( x, y, z ) );
        ExactSolutions.push_back( getExactSolutionZ( x, y, z ) );
        return ExactSolutions;
    }
    std::vector<double> getForcingTerms( double x, double y, double z ) const
    {
        std::vector<double> ForcingTerms;
        ForcingTerms.push_back( getForcingTermX( x, y, z ) );
        ForcingTerms.push_back( getForcingTermY( x, y, z ) );
        ForcingTerms.push_back( getForcingTermZ( x, y, z ) );
        return ForcingTerms;
    }
    std::vector<double> getExactSolutions( const std::vector<double> &xyz ) const
    {
        return getExactSolutions( xyz[0], xyz[1], xyz[2] );
    }
    std::vector<double> getForcingTerms( const std::vector<double> &xyz ) const
    {
        return getForcingTerms( xyz[0], xyz[1], xyz[2] );
    }
    double getExactSolutionX( const std::vector<double> &xyz ) const
    {
        AMP_ASSERT( xyz.size() != 3 );
        return getExactSolutionX( xyz[0], xyz[1], xyz[2] );
    }
    double getExactSolutionY( const std::vector<double> &xyz ) const
    {
        AMP_ASSERT( xyz.size() != 3 );
        return getExactSolutionY( xyz[0], xyz[1], xyz[2] );
    }
    double getExactSolutionZ( const std::vector<double> &xyz ) const
    {
        AMP_ASSERT( xyz.size() != 3 );
        return getExactSolutionZ( xyz[0], xyz[1], xyz[2] );
    }
    double getForcingTermX( const std::vector<double> &xyz ) const
    {
        AMP_ASSERT( xyz.size() != 3 );
        return getForcingTermX( xyz[0], xyz[1], xyz[2] );
    }
    double getForcingTermY( const std::vector<double> &xyz ) const
    {
        AMP_ASSERT( xyz.size() != 3 );
        return getForcingTermY( xyz[0], xyz[1], xyz[2] );
    }
    double getForcingTermZ( const std::vector<double> &xyz ) const
    {
        AMP_ASSERT( xyz.size() != 3 );
        return getForcingTermZ( xyz[0], xyz[1], xyz[2] );
    }
    // std::vector<double> getGradientX(const std::vector<double>& xyz) const {
    // AMP_ASSERT(xyz.size()!=3); return
    // getGradientX(xyz[0], xyz[1], xyz[2]); }
    // std::vector<double> getGradientY(const std::vector<double>& xyz) const {
    // AMP_ASSERT(xyz.size()!=3); return
    // getGradientY(xyz[0], xyz[1], xyz[2]); }
    // std::vector<double> getGradientZ(const std::vector<double>& xyz) const {
    // AMP_ASSERT(xyz.size()!=3); return
    // getGradientZ(xyz[0], xyz[1], xyz[2]); }
    // std::vector<double> getStrainTensor(const std::vector<double>& xyz) const {
    // AMP_ASSERT(xyz.size()!=3); return
    // getStrainTensor(xyz[0], xyz[1], xyz[2]); }
    std::vector<double> getStressTensor( const std::vector<double> &xyz ) const
    {
        AMP_ASSERT( xyz.size() != 3 );
        return getStressTensor( xyz[0], xyz[1], xyz[2] );
    }

protected:
    double E;          ///< Young's modulus
    double nu;         ///< Poisson's ratio
    double Lx, Ly, Lz; ///< scaling factors
    double Mx, My, Mz; ///< multiplying factors
                       /**
                        * a few extra coefficients :)
                        * Please refer to the derived classes to see how these coefficients are used
                        */
    double axx, bxx, axy, bxy, axz, bxz;
    double ayx, byx, ayy, byy, ayz, byz;
    double azx, bzx, azy, bzy, azz, bzz;
    std::string name; ///< name/type of mechanics manufactured solution
};                    // end class MMS


/**
 * The class MMSLinear implements the following manufactured solution for the displacement along
 * axis @f$ i @f$
 * @f$ \displaystyle{ u_i = M_i \prod_{j \in \{x,y,z\}} \left( a_{ij} \frac{j}{L_j} + b_{ij} \right)
 * \qquad \forall i
 * \in \{x, y, z\} } @f$
 *
 * The methods getExactSolutionI and getForcingtermI (where I is either X, Y, or Z) where generated
 * automatically using
 * a matlab script
 *
 * In principle you should get an exact answer to this kind of manufactured solution but it does not
 * hurt to verify it
 * actually does work
 */
class MMSLinear : public MMS
{
public:
    MMSLinear( double E = 0.0, double nu = 0.0 ) : MMS( E, nu ) { name = "Linear"; }
    double getExactSolutionX( double x, double y, double z ) const override;
    double getExactSolutionY( double x, double y, double z ) const override;
    double getExactSolutionZ( double x, double y, double z ) const override;
    double getForcingTermX( double /* x */, double y, double z ) const override;
    double getForcingTermY( double x, double /* y */, double z ) const override;
    double getForcingTermZ( double x, double y, double /* z */ ) const override;
    std::vector<double> getStressTensor( double x, double y, double z ) const override;
}; // end class MMSLinear


/**
 * The class MMSTrigonometric implements the following manufactured solution for the displacement
 * along axis @f$ i @f$
 * @f$ \displaystyle{ u_i = M_i \prod_{j \in \{x,y,z\}} \sin\left[ \frac{\pi}{2} \left(a_{ij}
 * \frac{j}{L_j} +
 * b_{ij}\right) \right] \qquad \forall i \in \{x, y, z\} } @f$
 *
 * The c++ code below is a direct copy and paste from the output of the matlab script I provided
 *
 * This manufactured solution should allow you to monitor the convergence rates for any order of
 * polynomial
 * approximation
 */
class MMSTrigonometric : public MMS
{
public:
    MMSTrigonometric( double E = 0.0, double nu = 0.0 ) : MMS( E, nu ) { name = "Trigonometric"; }
    double getExactSolutionX( double x, double y, double z ) const override;
    double getExactSolutionY( double /* x */, double y, double z ) const override;
    double getExactSolutionZ( double /* x */, double y, double z ) const override;
    double getForcingTermX( double x, double y, double z ) const override;
    double getForcingTermY( double x, double y, double z ) const override;
    double getForcingTermZ( double x, double y, double z ) const override;
    std::vector<double> getStressTensor( double x, double y, double z ) const override;
}; // end class MMSTrigonometric


/**
 * The class MMSBuilder is intended for choosing at run-time what mechanics manufactured solution to
 * work with
 *
 * You are welcome to extend this to other parameters
 */
class MMSBuilder
{
public:
    MMSBuilder() {}

    static std::shared_ptr<MMS> createMMS( std::shared_ptr<AMP::Database> mmsDatabase );
}; // end class MMSBuilder


} // namespace AMP::MechanicsManufacturedSolution
