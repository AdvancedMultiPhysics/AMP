#ifndef MMS_H
#define MMS_H

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <valarray>


namespace AMP {

class Database;


/**
 * Manufactured solution functions. These are the most general
 * 3d tri-quadratic and tri-cubic polynomials that satisfy the
 * following general boundary conditions on domains of the unit cube, unit cylindrical shell,
 * unit cylindrical quarter shell, and unit cylindrical rod, denoted by \f$V\f$:
 * \f{eqnarray*}
 * u(x,y,z) = c_i, \quad (x,y,z)\in \partial V_i, i \in {0,\dots,N_D} \\
 * \nabla u \cdot n_i = d_i, \quad (x,y,z) \in \partial V_i, i \in {0,\dots,N_N}
 * \f}
 * where \f$\bigcup_{i} \partial V_i = V\f$,
 * \f$\mathrm{int} \partial V_i \cap \mathrm{int} \partial V_j = \emptyset\f$,
 * \f$N_D + N_N = N_V\f$ is the number of surfaces in the boundary of \f$V\f$.
 * The volume \f$V\f$ is defined as follows
 *
 * \f{eqnarray*}
 * \mathrm{unit\:cube} \quad 0 \leq x \leq 1 \quad  0 \leq y \leq 1 \quad 0 \leq z \leq 1 \\
 * \mathrm{unit\:rod} \quad 0 \leq r \leq 1 \quad 0 \leq \theta \leq 2 \pi \quad 0 \leq z \leq 1 \\
 * \mathrm{unit\:shell} \quad  1/2 \leq r \leq 1 \quad 0 \leq \theta \leq 2 \pi \quad 0 \leq z \leq
 * 1 \\
 * \mathrm{unit\:quarter\:shell} \quad 1/2 \leq r \leq 1 \quad 0 \leq \theta \leq \pi / 2 \quad 0
 * \leq z \leq 1
 * \f}
 *
 * These polynomials were derived via symbolic computation, and the code
 * was automatically generated in efficient Horner form.
 *
 * Not all possibilities are non-trivial. The constructor will tell you which ones are available.
 *
 * Note: some of these are mislabeled at the moment. The cubes and rods are all good. The
 * quarter-shell
 * are suspect.
 */
class ManufacturedSolution
{
public:
    explicit ManufacturedSolution( std::shared_ptr<Database> db );

    /**
     * Evaluate the manufactured solution at a point.
     *  \param x x-coordinate
     *  \param y y-coordinate
     *  \param z z-coordinate
     */
    std::array<double, 10> evaluate( const double x, const double y, const double z ) const;

    size_t getNumberOfParameters() const { return d_NumberOfParameters; }

    size_t getNumberOfInputs() const { return d_NumberOfInputs; }

    void setTricubicParams( const std::valarray<double> &cin, const std::valarray<double> &ain )
    {
        d_c = cin;
        d_a = ain;
    }

    const std::string &get_name() const { return d_Name; }

private:
    /**
     *	\brief unit cube, quadratic, all Neumann BC's.
     *	The normal derivative \f$\frac{\partial u}{\partial n}\f$
     *	has the values \f$c_i, i=0,\dots,5\f$ on the planes \f$x=0, x=1, y=0, y=1, z=0, z=1\f$,
     *	respectively.
     *  Note that \f$\frac{\partial u}{\partial n} = - \frac{\partial u}{\partial x}\f$
     *	at \f$x=0\f$ and similarly for \f$y=0\f$ and \f$z=0\f$. The derivatives up to order
     *	two are returned in the order
     *	\f[ u,
     *	\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}, \frac{\partial u}{\partial z},
     *	\frac{\partial^2 u}{\partial x^2}, \frac{\partial^2 u}{\partial x \partial y},
     *  \frac{\partial^2 u}{\partial x \partial z},
     *	\frac{\partial^2 u}{\partial y^2},  \frac{\partial^2 u}{\partial y \partial z},
     *  \frac{\partial^2 u}{\partial z^2}, \f]
     *
     *	or more succinctly,
     *
     *	[0, x, y, z, xx, xy, xz, yy, yz, zz]
     *	 0  1  2  3   4   5   6   7   8   9
     *
     *	which is also true for cylindrical coordinates [x,y,z]->[r,th,z]
     *
     *  \param x x-coordinate
     *  \param y y-coordinate
     *  \param z z-coordinate
     *  \param mfs manufactured solution
     */
    static std::array<double, 10>
    quad_neumann( const double x, const double y, const double z, const ManufacturedSolution *mfs );

    static std::array<double, 10> quad_dirichlet1( const double x,
                                                   const double y,
                                                   const double z,
                                                   const ManufacturedSolution *mfs );

    static std::array<double, 10> quad_dirichlet2( const double x,
                                                   const double y,
                                                   const double z,
                                                   const ManufacturedSolution *mfs );

    static std::array<double, 10>
    quad_none( const double x, const double y, const double z, const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_neumann( const double x,
                                                 const double y,
                                                 const double z,
                                                 const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_dirichlet1( const double x,
                                                    const double y,
                                                    const double z,
                                                    const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_dirichlet2( const double x,
                                                    const double y,
                                                    const double z,
                                                    const ManufacturedSolution *mfs );

    static std::array<double, 10>
    cubic_none( const double x, const double y, const double z, const ManufacturedSolution *mfs );

    static std::array<double, 10> quad_cyl_rod_none( const double r,
                                                     const double th,
                                                     const double z,
                                                     const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_cyl_shell_neumann( const double r,
                                                           const double th,
                                                           const double z,
                                                           const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_cyl_rod_dirichletz2( const double r,
                                                             const double th,
                                                             const double z,
                                                             const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_cyl_rod_rz_none( const double r,
                                                         const double th,
                                                         const double z,
                                                         const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_cyl_rod_none( const double r,
                                                      const double th,
                                                      const double z,
                                                      const ManufacturedSolution *mfs );

    static std::array<double, 10> quad_cyl_shell_neumann( const double r,
                                                          const double th,
                                                          const double z,
                                                          const ManufacturedSolution *mfs );

    static std::array<double, 10> quad_cyl_qtr_shell_neumann( const double r,
                                                              const double th,
                                                              const double z,
                                                              const ManufacturedSolution *mfs );

    static std::array<double, 10> quad_cyl_qtr_shell_dirichlet2( const double r,
                                                                 const double th,
                                                                 const double z,
                                                                 const ManufacturedSolution *mfs );

    static std::array<double, 10> quad_cyl_qtr_shell_none( const double r,
                                                           const double th,
                                                           const double z,
                                                           const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_cyl_qtr_shell_neumann( const double r,
                                                               const double th,
                                                               const double z,
                                                               const ManufacturedSolution *mfs );

    static std::array<double, 10> cubic_cyl_qtr_shell_none( const double r,
                                                            const double th,
                                                            const double z,
                                                            ManufacturedSolution *mfs );

    static std::array<double, 10> general_quadratic_exponential( const double x,
                                                                 const double y,
                                                                 const double z,
                                                                 const ManufacturedSolution *mfs );

    static std::array<double, 10> general_quadratic_sinusoid( const double x,
                                                              const double y,
                                                              const double z,
                                                              const ManufacturedSolution *mfs );

    static std::array<double, 10> general_quadratic_exponential_sinusoid(
        const double x, const double y, const double z, const ManufacturedSolution *mfs );

    const std::valarray<double> &getc() const { return d_c; }
    const std::valarray<double> &geta() const { return d_a; }
    const std::array<std::array<double, 3>, 3> &geth() const { return d_h; }
    const std::array<std::array<double, 3>, 3> &geths() const { return d_hs; }

    enum class FunctionType { POLYNOMIAL, GENERALQUADRATIC };
    enum class Geometry { BRICK, CYLROD, CYLRODRZ, CYLSHELL, QTRCYLSHELL, LASTGeometry };
    enum class Order { QUADRATIC, CUBIC, FOURIER, GAUSSIAN, LASTOrder };
    enum class BCType { NEUMANN, DIRICHLET1, DIRICHLET2, DIRICHLETZ2, NONE, LASTType };

    FunctionType d_FunctionType;

    Geometry d_geom;
    Order d_order;
    BCType d_bcType;
    size_t d_NumberOfParameters;
    size_t d_NumberOfInputs;

    std::function<std::array<double, 10>(
        const double, const double, const double, const ManufacturedSolution * )>
        d_functionPointer;

    bool d_internalParameters;

    std::valarray<double> d_c;
    std::valarray<double> d_a;

    double d_MinX, d_MaxX, d_ScaleX;
    double d_MinY, d_MaxY, d_ScaleY;
    double d_MinZ, d_MaxZ, d_ScaleZ;
    double d_MinR, d_MaxR, d_ScaleR;
    double d_MinTh, d_MaxTh, d_ScaleTh;

    std::array<std::array<double, 3>, 3> d_h;
    std::array<std::array<double, 3>, 3> d_hs; // symmetrized h

    double d_Pi;
    double d_MaximumTheta;
    bool d_CylindricalCoords;

    std::string d_Name;
};
} // namespace AMP
#endif
