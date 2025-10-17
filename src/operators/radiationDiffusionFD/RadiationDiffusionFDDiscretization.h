#ifndef RAD_DIF_FD_DISCRETIZATION
#define RAD_DIF_FD_DISCRETIZATION

#include "AMP/IO/AsciiWriter.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/geometry/shapes/Box.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <optional>
#include <variant>


/** The classes in this file are (or are associated with) spatial finite-discretizations of
 * the radiation-diffusion problem:
 *      u'(t) - D(u) - R(u)  = s(t), u(0) = u_0
 * over the spatial domain Omega subset R^d for d in {1,2,3}. Where:
 *      1. u = [u0, u1] = [E, T] is a block vector, holding E and T.
 *      2. D(u) = [grad dot ( D0 * grad u0 ), grad dot ( D1 * grad u1 )] is a (possibly) nonlinear
 * diffusion operator
 *      3. R(u) is a (possibly) nonlinear reaction operator
 *
 * In more detail, the diffusion operator is
 *      D(u) = [grad dot (k11*D_E grad E), grad dot (k21*D_T grad T)],
 * where diffusive fluxes depend on the (currently static) "IsNonlinear" bool as:
 *      1. if false:
 *          D_E = D_T = 1.0
 *      2. if true:
 *          D_E = 1/(3*sigma), D_T = T^2.5, and sigma = (zatom/T)^3
 *
 * The reaction term is dependent on the "IsNonlinear" bool as:
 *      1. if false:
 *          R(u) = [k12*(T - E), -k22*(T - E)]
 *      2. if true:
 *          R(u) = [k12*simga*(T^4 - E), -k22*simga*(T^4 - E)]
 *
 * On boundary k in {1,...,6} the spatial boundary conditions are as follows:
 *      1. Robin on energy:
 *          ak*E + bk*hat{nk} dot k11*D_E * grad(E) = rk on boundary k
 *      2. ''pseudo'' Neumann on temperature (not genuine Neumann unless nk=0):
 *          hat{nk} dot grad(T) = nk on boundary k
 * where:
 *      hat{nk} is the outward facing normal to the boundary
 *      ak, bk are user-prescribed constants
 *      rk, nk are user-prescribed constants or functions (that can vary on the boundary)
 */


namespace AMP::Operator {

// Foward declaration of classes declared in this file
class RadDifCoefficients;      // Static class defining PDE coefficients
class FDMeshOps;               // Static class for creating mesh-related data and FD coefficients
class FDMeshGlobalIndexingOps; // Class for index conversions on global DOFs
class FDBoundaryUtils;         // Static class for boundary-related utilities
//
class RadDifOp;               // The main, nonlinear operator
class RadDifOpPJac;           // Its linearization
class RadDifOpPJacParameters; // Operator parameters for linearization
struct RadDifOpPJacData;      // Data structure for storing the linearization data

/** Static class defining coefficients in a radiation-diffusion PDE */
class RadDifCoefficients
{

public:
    // Hack. This should really be templated against. Realistically the operator RadDifOp, and its linearization, should be templated against a coefficient class, and a non-templated base class added. The realistic use case is for the nonlinear coefficients, but the linear coefficients can be useful for testing, debugging, etc. 
    //! Flag indicating whether nonlinear or linear PDE coefficients are used
    static constexpr bool IsNonlinear = true; 

    //! Prevent instantiation
    RadDifCoefficients() = delete;

    /** Energy diffusion coefficient k11*D_E, given constant k11, temperature T, and atomic number
     * zatom:
     * nonlinear: D_E = 1/(3*sigma), sigma=(zatom/T)^3
     * linear: D_E = 1.0
     */
    static double diffusionE( double k11, double T, double zatom );

    /** Temperature diffusion coefficient k21*D_T, given constant k21, and temperature T:
     * nonlinear: D_T = T^2.5
     * linear: D_T = 1.0
     */
    static double diffusionT( double k21, double T );

    //! Compute reaction coefficients REE, RET, RTE, REE
    static void reaction( double k12,
                                      double k22,
                                      double T,
                                      double zatom,
                                      double &REE,
                                      double &RET,
                                      double &RTE,
                                      double &RTT );
};


/** Static class providing mesh-related operations and utilities that are required for
 * finite-difference discretizations.
 */
class FDMeshOps
{

private:
    //! Indices used for referencing WEST, ORIGIN, and EAST entries in Loc3 data structures
    static constexpr size_t WEST   = 0;
    static constexpr size_t ORIGIN = 1;
    static constexpr size_t EAST   = 2;

public:
    // Hack. This should really be templated against. That said, for applications flux limiting is usually required.
    //! Flag indicating whether energy diffusion coefficient is limited
    static constexpr bool IsFluxLimited = true; 

    //! Prevent instantiation
    FDMeshOps() = delete;

    //! Create mesh-related data given the mesh
    static void
    createMeshData( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                    std::shared_ptr<AMP::Mesh::BoxMesh> &d_BoxMesh,
                    size_t &d_dim,
                    AMP::Mesh::GeomType &d_CellCenteredGeom,
                    std::shared_ptr<AMP::Discretization::DOFManager> &d_scalarDOFMan,
                    std::shared_ptr<AMP::Discretization::multiDOFManager> &d_multiDOFMan,
                    std::shared_ptr<AMP::Mesh::BoxMesh::Box> &d_globalBox,
                    std::shared_ptr<AMP::Mesh::BoxMesh::Box> &d_localBox,
                    std::shared_ptr<AMP::ArraySize> &d_localArraySize,
                    std::vector<double> &d_h,
                    std::vector<double> &d_rh2 );

    /** Compute diffusion coefficients at two cell faces using 3-point stencil data in ELoc3 and
     * TLoc3
     * @param[in] computeE flag indicating whether to compute E diffusion coefficients
     * @param[in] computeT flag indicating whether to compute T diffusion coefficients
     * @param[out] Dr_WO WEST coefficient in for energy
     * @param[out] Dr_OE EAST coefficient in for energy
     * @param[out] DT_WO WEST coefficient in for temperature
     * @param[out] DT_OE EAST coefficient in for temperature
     */
    template<bool computeE, bool computeT>
    static void FaceDiffusionCoefficients( std::array<double, 3> &ELoc3,
                                           std::array<double, 3> &TLoc3,
                                           double k11,
                                           double k21,
                                           double zatom,
                                           double h,
                                           double *Dr_WO,
                                           double *Dr_OE,
                                           double *DT_WO,
                                           double *DT_OE );

    // Helper function
private:
    //! Create DOFManagers given the geometry and mesh.
    static void
    createDOFManagers( const AMP::Mesh::GeomType &Geom,
                       std::shared_ptr<AMP::Mesh::BoxMesh> &mesh,
                       std::shared_ptr<AMP::Discretization::DOFManager> &scalarDOFMan,
                       std::shared_ptr<AMP::Discretization::multiDOFManager> &multiDOFMan );
};


/** Abstract class providing mesh-indexing-related operations given some mesh and geometry. In
 * particular, this class provides grid index conversions for global indices, but for interior
 * indices these utilites are slower than those based on ArraySize.
 */
class FDMeshGlobalIndexingOps
{

    // Data
private:
    //! Indices used for referencing WEST, ORIGIN, and EAST entries in Loc3 data structures
    static constexpr size_t WEST   = 0;
    static constexpr size_t ORIGIN = 1;
    static constexpr size_t EAST   = 2;

    //! Mesh; keep a pointer to save having to downcast repeatedly
    std::shared_ptr<AMP::Mesh::BoxMesh> d_BoxMesh;
    //! Geometry type
    AMP::Mesh::GeomType d_geom;
    //! DOFManager for E and T individually
    std::shared_ptr<AMP::Discretization::DOFManager> d_scalarDOFMan;
    //! MultiDOFManager for managing [E,T] multivectors
    std::shared_ptr<AMP::Discretization::multiDOFManager> d_multiDOFMan;


public:
    FDMeshGlobalIndexingOps( std::shared_ptr<AMP::Mesh::BoxMesh> BoxMesh,
                             AMP::Mesh::GeomType &geom,
                             std::shared_ptr<AMP::Discretization::DOFManager> scalarDOFMan,
                             std::shared_ptr<AMP::Discretization::multiDOFManager> multiDOFMan );

    //! Map from grid index to a the corresponding DOF
    size_t gridIndsToScalarDOF( const std::array<size_t, 3> &ijk ) const;

    //! Map from scalar DOF to grid indices i, j, k
    std::array<size_t, 3> scalarDOFToGridInds( size_t dof ) const;

    //! Map from grid index to a MeshElement
    AMP::Mesh::MeshElement gridIndsToMeshElement( const std::array<size_t, 3> &ijk ) const;
};


/** Static class bundling together boundary-related utility data structures and functionality */
class FDBoundaryUtils
{

private:
    //! Keys used to access db constants
    static constexpr std::string_view a_keys[] = { "a1", "a2", "a3", "a4", "a5", "a6" };
    static constexpr std::string_view b_keys[] = { "b1", "b2", "b3", "b4", "b5", "b6" };
    static constexpr std::string_view r_keys[] = { "r1", "r2", "r3", "r4", "r5", "r6" };
    static constexpr std::string_view n_keys[] = { "n1", "n2", "n3", "n4", "n5", "n6" };

    //
public:
    //! Prevent instantiation
    FDBoundaryUtils() = delete;

    //! Defines a boundary in a given dimension (WEST is the first boundary, and EAST the second)
    enum class BoundarySide { WEST, EAST };

    /** Return the boundaryID in {1,...,6} corresponding to a dim in {0,1,2}, given the
     * corresponding side.
     */
    static size_t getBoundaryIDFromDim( size_t dim, BoundarySide side );

    /** Assuming a boundaryID in {1,...,6}, get the dimension of the boundary, i.e., boundaries 1
     * and 2 live in the first dimension, 3 and 4 the second, and 5 and 6 the third.
     */
    static size_t getDimFromBoundaryID( size_t boundaryID );

    //! Get the constants ak, bk, rk, nk from the database for the given boundaryID. Note that rk and nk need not exist in the db, and are returned with a default of 0
    static void getBCConstantsFromDB( const AMP::Database &db,
                                            size_t boundaryID,
                                            double &ak,
                                            double &bk, 
                                            double &rk, 
                                            double &nk );

    /** On boundary k we have the two equations:
     *     ak*E + bk * hat{nk} dot k11*D_E(T) grad E = rk,
     *                 hat{nk} dot            grad T = nK,
     * where ak, bk, rk, and nk are all known constants and hat{nk} is the outward-facing normal
     * vector at the boundary.
     * The discretization of these conditions involves one ghost point for E and T (Eg and Tg), and
     * one interior point (Eint and Tint). Here we solve for the ghost points and return them. Note
     * that this system, although nonlinear, can be solved directly by forward substitution.
     * @param[in] cHandle is a function taking in T and returning k11*D_E(T)
     * @param[out] Eg ghost-point value for E that satisfies the discretized BCs
     * @param[out] Tg ghost-point value for T that satisfies the discretized BCs
     */
    static void ghostValuesSolve( double a,
                                  double b,
                                  const std::function<double( double T )> &cHandle,
                                  double r,
                                  double n,
                                  double h,
                                  double Eint,
                                  double Tint,
                                  double &Eg,
                                  double &Tg );

    /** On boundary k we have the equation:
     *      hat{nk} dot grad T = nk,
     * where nk is a known constant and hat{nk} is the outward-facing normal vector at the
     * boundary. This BC is discretized as
     *      sign(hat{nk}) * [Tg_k - Tint_k]/h = nk.
     * Here we solve for the ghost point Tg_k and return it
     * @param[out] Tg ghost-point value that satisfies the discretized BC
     */
    static double ghostValueSolveT( double n, double h, double Tint );

    /** On boundary k we have the equation:
     *      ak*E + bk * hat{nk} dot ck grad E = rk,
     * where ak, bk, rk, nk, and ck=+k11*D_E(T) are all known constants, and hat{nk} is the
     * outward-facing normal vector at the boundary. This BC is discretized as
     *      ak*0.5*[Eg_k + Eint_k] + bk*ck*sign(hat{nk}) *[Eg_k - Eint_k]/h = rk.
     * Here we solve for the ghost point Eg_k and return it.
     * @param[out] Eg ghost-point value that satisfies the discretized BC
     * @note ck=+k11*D_E(T) and not -k11*D_E(T)
     */
    static double ghostValueSolveE( double a, double b, double r, double c, double h, double Eint );
};


/** Finite-difference discretization of the spatial operator in the above radiation-diffusion
 * equation. This discretization is based on that described in
 *      "Dynamic implicit 3D adaptive mesh refinement for non-equilibrium radiation diffusion"
 * by B.Philip, Z.Wangb, M.A.Berrill, M.Birkeb, M.Pernice in Journal of Computational Physics 262
 * (2014) 17–37. The primary difference is that this implementation does not use adaptive mesh
 * refiment, and instead assumes the mesh spacing in each dimension is constant.
 *
 * The database in the incoming OperatorParameters must contain the following parameters:
 *  1. ak, bk (for all boundaries k)
 * doubles. Specify constants in Robin BCs on energy.
 * Optionally, the RHS boundary condition values of rk and nk can be provided. However, the user
 * can also specify these RHS values as boundary- and spatially-dependent functions via calls to
 * 'setBoundaryFunctionE' and 'setBoundaryFunctionT', respectively, in which case whatever values
 * of rk and nk exist in the database will be ignored.
 *
 *  2. zatom:
 * double. Atomic number constant in the nonlinear problem. Default value is 1.0.
 *
 *  3. k11, k12, k21, k22
 * doubles. Scaling constants in PDE
 *
 * Mesh: The discretization expects a non-periodic AMP::Mesh::BoxMesh generated from the "cube"
 * generator. The boundaryIDs on the mesh must be:
 *      1,2 for xmin,xman,
 *      3,4 for ymin,ymax,
 *      5,6 for zmin,zmax.
 * The mesh range and number of points in each dimension need not be the same.
 *
 * NOTES:
 *      * There is no mass matrix: After integrating the PDEs over a spatial volume, the discrete
 * equations are re-scaled such that the mesh volume appearing in front of the time derivative is 1.
 *      * Each dimension [xmin, xmax] is divided into nx cells with centers xi = xmin + (i+1/2)*hx
 * with hx = (xmax-xmin)/nx, for i = 0,...,nx-1. The computational unknowns in each dimension that
 * we discretize are the point values of E and T at these nx cell centers. The placement of DOFs at
 * cell centers is handled internally by the class.
 *      * Boundary conditions are implemented by placing one ghost cell at each end of the domain,
 * with corresponding unknowns at the centers of those ghost cells. Where ghost values are required
 * (i.e., when evaluating the 3-point stencil of a boundary-adjacent DOF), the ghost value is
 * evaluated in terms the boundary-adjacent DOF using the discretized boundary condition. That is,
 * ghost points are not active DOFs.
 */
class RadDifOp : public AMP::Operator::Operator
{

    // Data
public:
    //! Parameters required by the discretization
    std::shared_ptr<AMP::Database> d_db = nullptr;


    // Methods
public:
    //! Constructor
    RadDifOp( std::shared_ptr<const AMP::Operator::OperatorParameters> params );
    //! Destructor
    virtual ~RadDifOp(){};

    std::string type() const override { return "RadDifOp"; };

    //! Compute LET = L(ET)
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET,
                std::shared_ptr<AMP::LinearAlgebra::Vector> LET ) override;

    //! E and T must be positive
    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) override;

    //! Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override;

    //! Vector of hx, hy, hz
    std::vector<double> getMeshSize() const;

    //! Geometry used in the mesh
    AMP::Mesh::GeomType getGeomType() const;

    //! DOFManager used for each of E and T
    std::shared_ptr<const AMP::Discretization::DOFManager> getScalarDOFManager() const;

    //! Set the Robin return function for the energy.
    void setBoundaryFunctionE(
        const std::function<double( size_t boundaryID, const AMP::Mesh::Point &boundaryPoint )>
            &fn_ );

    //! Set the pseudo-Neumann return function for the temperature.
    void setBoundaryFunctionT(
        const std::function<double( size_t boundaryID, const AMP::Mesh::Point &boundaryPoint )>
            &fn_ );

    // Data
private:
    //! Constant scaling factors in the PDE
    const double d_k11;
    const double d_k12;
    const double d_k21;
    const double d_k22;

    //! Constants in boundary conditions from incoming db. The constant for a given boundaryID is in index boundaryID-1
    std::array<double, 6> d_ak;
    std::array<double, 6> d_bk;
    std::array<double, 6> d_rk;
    std::array<double, 6> d_nk;

    //! Mesh-related data
private:
    //! MultiDOFManager for managing [E,T] multivectors
    std::shared_ptr<AMP::Discretization::multiDOFManager> d_multiDOFMan;
    //! DOFManager for E and T individually
    std::shared_ptr<AMP::Discretization::DOFManager> d_scalarDOFMan;
    //! Mesh; keep a pointer to save having to downcast repeatedly
    std::shared_ptr<AMP::Mesh::BoxMesh> d_BoxMesh;
    //! Global grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_globalBox = nullptr;
    //! Local grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_localBox = nullptr;
    //! Local array size
    std::shared_ptr<AMP::ArraySize> d_localArraySize = nullptr;
    //! Placeholder for geometry that results in cell-centered data
    AMP::Mesh::GeomType CellCenteredGeom;
    //! Problem dimension
    size_t d_dim = -1;
    //! Mesh sizes, hx, hy, hz. We compute these based on the incoming mesh
    std::vector<double> d_h;
    //! Reciprocal squares of mesh sizes
    std::vector<double> d_rh2;

    //! Indices used for referencing WEST, ORIGIN, and EAST entries in Loc3 data structures
    static constexpr size_t WEST   = 0;
    static constexpr size_t ORIGIN = 1;
    static constexpr size_t EAST   = 2;

    //! Mesh-indexing functions
    std::shared_ptr<FDMeshGlobalIndexingOps> d_meshIndexingOps = nullptr;


private:
    //! Apply operator over DOFs living on the interior of process
    void applyInterior( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                        std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> LE_vec,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> LT_vec );

    //! Apply operator over DOFs living on the boundary of process
    void applyBoundary( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                        std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> LE_vec,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> LT_vec );


    /** This is a wrapper around FDBoundaryUtils::ghostValuesSolve to pass our specific constants
     * and boundary-function evaluations.
     * @param[out] Eg value of E at the ghost point
     * @param[out] Tg value of T at the ghost point
     */
    void ghostValuesSolveWrapper( size_t boundaryID,
                                  const AMP::Mesh::Point &boundaryPoint,
                                  double Eint,
                                  double Tint,
                                  double &Eg,
                                  double &Tg );

    /** Get nearest neighbor data, i.e., WEST and EAST values. This is valid in the case that
     * ORIGIN is a boundary point, either process (hence requiring ghost data) or physical (hence
     * requiring a ghost-point solve).
     * the latter case.
     * @param[in] E_vec vector of all E values
     * @param[in] T_vec vector of all T values
     * @param[in] ijk grid indices of DOF for which 3-point stencil values are to be unpacked
     * @param[in] dim dimension in which the 3-point extends
     * @param[out] ELoc3 E values in the 3-point stencil (WEST, ORIGIN, UPPER)
     * @param[out] TLoc3 T values in the 3-point stencil (WEST, ORIGIN, UPPER)
     *
     * @note ijk is modified inside the function, but upon conclusion of the function is in its
     * original state
     */
    void getNNDataBoundary( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                            std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                            std::array<size_t, 3> &ijk,
                            size_t dim,
                            std::array<double, 3> &ELoc3,
                            std::array<double, 3> &TLoc3 );


    /** Prototype of function returning value of Robin BC of E on given boundary at given node. The
     * user can specify any function with this signature via 'setBoundaryFunctionE'
     * @param[in] boundaryID ID of the boundary
     * @param[in] boundaryPoint the point in space where the function is to be evaluated (this will
     * be a point on the corresponding boundary)
     */
    std::function<double( size_t boundaryID, const AMP::Mesh::Point &boundaryPoint )>
        d_robinFunctionE;

    /** Prototype of function returning value of pseudo-Neumann BC of T on given boundary at given
     * node. The user can specify any function with this signature via 'setBoundaryFunctionT'
     * @param[in] boundaryID ID of the boundary
     * @param[in] boundaryPoint the point in space where the function is to be evaluated (this will
     * be a point on the corresponding boundary)
     */
    std::function<double( size_t boundaryID, const AMP::Mesh::Point &boundaryPoint )>
        d_pseudoNeumannFunctionT;

    //
protected:
    /** Returns a parameter object that can be used to reset the corresponding
     * RadDifOpPJac operator. Note that in the base class's getParameters get's redirected to this
     * function.
     * @param[in] u_in is the current nonlinear iterate.
     */
    std::shared_ptr<AMP::Operator::OperatorParameters>
    getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) override;
};
// End of RadDifOp


/** Picard linearization of a RadDifOp.
 * Specifically, the spatial operators in the radiation-diffusion equation can be written as
 *          L(u) = -D(u) -R(u) <==> hat{L}(u)*u = -hat{D}(u)*u -hat{R}(u)*u
 * where hat{L}(u), hat{D}(u) and hat{R}(u) are block 2x2 matrices dependent upon the state u. This
 * class implements the LinearOperator hat{L}(u), which is a Picard linearization of L(u).
 */
class RadDifOpPJac : public AMP::Operator::LinearOperator
{

    //
public:
    std::shared_ptr<AMP::Database> d_db = nullptr;
    //! Representation of this operator as a block 2x2 matrix
    std::shared_ptr<RadDifOpPJacData> d_data = nullptr;
    //! The vector the above operator is linearized about
    AMP::LinearAlgebra::Vector::shared_ptr d_frozenVec = nullptr;

    //! Constructor
    RadDifOpPJac( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ );

    //! Destructor
    virtual ~RadDifOpPJac(){};

    //! Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override;

    //! Used by OperatorFactory to create a RadDifOpPJac
    static std::unique_ptr<AMP::Operator::Operator>
    create( std::shared_ptr<AMP::Operator::OperatorParameters> params )
    {
        return std::make_unique<RadDifOpPJac>( params );
    };

    /** Reset the operator based on the incoming parameters.
     * Primarily, this updates frozen value of current solution, d_frozenSolution. It also resets
     * the associated linearized data. Note that it does not update any mesh-related data since
     * neither does the base classes reset(). It also doesn't update any boundary-related data
     */
    void reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params ) override;

    std::string type() const override { return "RadDifOpPJac"; };

    //! Compute LET = L(ET)
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET,
                std::shared_ptr<AMP::LinearAlgebra::Vector> LET ) override;

    /** Allows an apply from Jacobian data that's been modified by an outside class (this is an
     * acknowledgement that the caller of the apply deeply understands what they're doing)
     */
    void applyWithOverwrittenDataIsValid() { d_applyWithOverwrittenDataIsValid = true; };


    // Data
private:
    //! Constant scaling factors in the PDE
    const double d_k11;
    const double d_k12;
    const double d_k21;
    const double d_k22;

    //! Constants in boundary conditions from incoming db. The constant for a given boundaryID is in index boundaryID-1
    std::array<double, 6> d_ak;
    std::array<double, 6> d_bk;
    std::array<double, 6> d_rk;
    std::array<double, 6> d_nk;

    //! Flag indicating whether apply with overwritten Jacobian data is valid. This is reset to
    //! false at the end of every apply call, and can be set t true by the public member function
    bool d_applyWithOverwrittenDataIsValid = false;


    //! Mesh-related data
private:
    //! MultiDOFManager for managing [E,T] multivectors
    std::shared_ptr<AMP::Discretization::multiDOFManager> d_multiDOFMan;
    //! DOFManager for E and T individually
    std::shared_ptr<AMP::Discretization::DOFManager> d_scalarDOFMan;
    //! Mesh; keep a pointer to save having to downcast repeatedly
    std::shared_ptr<AMP::Mesh::BoxMesh> d_BoxMesh;
    //! Global grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_globalBox = nullptr;
    //! Local grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_localBox = nullptr;
    //! Local array size
    std::shared_ptr<AMP::ArraySize> d_localArraySize = nullptr;
    //! Placeholder for geometry that results in cell-centered data
    AMP::Mesh::GeomType CellCenteredGeom;
    //! Problem dimension
    size_t d_dim = -1;
    //! Mesh sizes, hx, hy, hz. We compute these based on the incoming mesh
    std::vector<double> d_h;
    //! Reciprocal squares of mesh sizes
    std::vector<double> d_rh2;

    //! Placeholder for grid indices. Size 5 is because ArraySize deals with 5 grid indcies
    std::array<size_t, 5> d_ijk;
    //! Placeholder arrays for values used in 3-point stencils
    std::array<double, 3> d_ELoc3;
    std::array<double, 3> d_TLoc3;
    //! Placeholder array for dofs we connect to in 3-point stencil
    std::array<size_t, 3> d_dofsLoc3;

    //! Indices used for referencing WEST, ORIGIN, and EAST entries in Loc3 data structures
    static constexpr size_t WEST   = 0;
    static constexpr size_t ORIGIN = 1;
    static constexpr size_t EAST   = 2;

    //! Mesh indexing functions
    std::shared_ptr<FDMeshGlobalIndexingOps> d_meshIndexingOps = nullptr;
    //
private:
    //! Apply action of the operator utilizing its representation in d_data
    void applyFromData( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> LET_ );

    //! Set our d_data member
    void setData();

    /** Sets the reaction-related vectors in our d_data member.
     * This code is based on stripping out the reaction component of the apply of the nonlinear
     * operator.
     * @param[in] T_vec T component of the frozen vector d_frozenVec
     */
    void setDataReaction( std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec );

    /** Fill the given input diffusion matrix with CSR data
     * @param[in] component 0 (for energy) or 1 (for temperature)
     */
    template<size_t Component>
    void fillDiffusionMatrixWithData( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix );

    /** Get CSR data for a row of the Picard-linearized diffusion matrix dE or dT.
     * @param[in] component 0 (energy) or 1 (temperature) to get CSR data for
     * @param[in] E_vec E component of the frozen vector d_frozenVec
     * @param[in] T_vec T component of the frozen vector d_frozenVec
     * @param[in] E_rawData local raw data array for E
     * @param[in] T_rawData local raw data array for T
     * @param[in] row the row to retrieve (a scalar index)
     * @param[out] cols the column indices for the non-zeros in the given row, with the diagonal
     * entry first
     * @param[out] data the data for the non-zeros in the given row
     *
     * @note this function implicity assumes that the stencil does not touch both boundaries at
     * once (corresponding to the number of interior DOFs in the given dimension being larger than
     * one)
     */
    template<size_t Component>
    void getCSRDataDiffusionMatrix( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                                    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                                    const double * E_rawData,
                                    const double * T_rawData,
                                    size_t row,
                                    std::vector<size_t> &cols,
                                    std::vector<double> &data );

    /** Get cols and data for given row, when the row lives on the interior of a process
     * @note ijkLocal is modified internally, but returned in its original state
     */
    template<size_t Component>
    void getCSRDataDiffusionMatrixInterior( const double * E_rawData,
                                            const double * T_rawData,
                                            size_t rowLocal,
                                            std::array<size_t, 5> &ijkLocal,
                                            std::vector<size_t> &colsLocal,
                                            std::vector<double> &data );

    //! Get cols and data for given row, when the row lives on a process boundary
    template<size_t Component>
    void getCSRDataDiffusionMatrixBoundary( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                                            std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                                            size_t row,
                                            std::vector<size_t> &cols,
                                            std::vector<double> &data );


    /** We have ghost values that satisfy
     * Eg = alpha_E*Eint + beta_E
     * Tg = alpha_T*Tint + beta_T
     * Here we return the coefficient alpha_
     *
     * @param[in] component 0 (for energy), or 1 (for temperature)
     * @param[in] boundaryID the boundary
     * @param[in] ck = +k11*D_E(T), but is ignored if component == 1.
     */
    double PicardCorrectionCoefficient( size_t component, size_t boundaryID, double ck ) const;

    /** This is a wrapper around FDBoundaryUtils::ghostValuesSolve to pass our specific constants
     * and boundary-function evaluations.
     * @param[out] Eg value of E at the ghost point
     * @param[out] Tg value of T at the ghost point
     */
    void ghostValuesSolveWrapper( size_t boundaryID,
                                  const AMP::Mesh::Point &boundaryPoint,
                                  double Eint,
                                  double Tint,
                                  double &Eg,
                                  double &Tg );

    /** Closely related to RadDifOp::getNNDataBoundary, except there are two additional outputs:
     * @param[out] d_dofsLoc3 indices of the dofs in the 3-point stencil
     * @param[out] boundaryIntersection flag indicating if the stencil touches a physical boundary
     * (and which one if it does)
     *
     * @note if the stencil touches a physical boundary then the corresponding value in dofs is
     * meaningless
     * @note this function implicity assumes that the stencil does not touch both boundaries at
     * once (corresponding to the number of interior DOFs in the given dimension being larger than
     * one)
     */
    void getNNDataBoundary(
        std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
        std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
        std::array<size_t, 3> &ijk, // is modified locally, but returned in same state
        size_t dim,
        std::array<double, 3> &ELoc3,
        std::array<double, 3> &TLoc3,
        std::array<size_t, 3> &dofsLoc3,
        std::optional<FDBoundaryUtils::BoundarySide> &boundaryIntersection );

    /** Prototype of function returning value of Robin BC of E on given boundary at given node. The
     * user can specify any function with this signature via 'setBoundaryFunctionE'
     * @param[in] boundaryID ID of the boundary
     * @param[in] boundaryPoint the point in space where the function is to be evaluated (this will
     * be a point on the corresponding boundary)
     */
    std::function<double( size_t boundaryID, const AMP::Mesh::Point &boundaryPoint )>
        d_robinFunctionE;

    /** Prototype of function returning value of pseudo-Neumann BC of T on given boundary at given
     * node. The user can specify any function with this signature via 'setBoundaryFunctionT'
     * @param[in] boundaryID ID of the boundary
     * @param[in] boundaryPoint the point in space where the function is to be evaluated (this will
     * be a point on the corresponding boundary)
     */
    std::function<double( size_t boundaryID, const AMP::Mesh::Point &boundaryPoint )>
        d_pseudoNeumannFunctionT;
};
// End of RadDifOpPJac


/** Data structure for storing the 2x2 block matrix hat{L} associated with the RadDifOpPJac
 * Specifically, this Picard Linearization is a LinearOperator with the following block structure:
 * [ d_E 0   ]   [ diag(r_EE) diag(r_ET) ]
 * [ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
 * where the first block matrix contains diffusion terms and the second contains the reaction terms.
 */
struct RadDifOpPJacData {
    //! The apply of RadDifOpPJac uses our private data members
    friend class RadDifOpPJac;

public:
    //! Getter routine; any external updates to the private data members below done via this
    std::tuple<
    std::shared_ptr<AMP::LinearAlgebra::Matrix>,std::shared_ptr<AMP::LinearAlgebra::Matrix>,
    std::shared_ptr<AMP::LinearAlgebra::Vector>,std::shared_ptr<AMP::LinearAlgebra::Vector>,std::shared_ptr<AMP::LinearAlgebra::Vector>,std::shared_ptr<AMP::LinearAlgebra::Vector>
    > get();

private:
    //! Flag indicating whether our data has been accessed, and hence possibly modified, by a
    //! non-friend class (e.g., a BDF wrapper of a RadDifOpPJac). This is set to true any time a
    //! getter is called.
    bool d_dataMaybeOverwritten = false;

    //! Members used to store matrix components
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_E  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_T  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_EE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_ET = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TT = nullptr;
};


/** OperatorParameters for creating the linearized operator RadDifOpPJac. This is just regular
 * OperatorParameters appended with:
 * 1. a vector containg that the operator is to be linearized about
 * 2. std::functions that return boundary values of E and T. These values are optional; if
 * they are set the linearized operator will use them, otherwise it will default to using the
 * constant boudnary rk and nk values in the incoming database.
 */
class RadDifOpPJacParameters : public AMP::Operator::OperatorParameters
{
public:
    // Constructor
    explicit RadDifOpPJacParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db ){};
    virtual ~RadDifOpPJacParameters(){};

    AMP::LinearAlgebra::Vector::shared_ptr d_frozenSolution = nullptr;

    std::function<double( size_t boundaryID, const AMP::Mesh::Point &boundaryPoint )>
        d_robinFunctionE;
    std::function<double( size_t boundaryID, const AMP::Mesh::Point &boundaryPoint )>
        d_pseudoNeumannFunctionT;
};

} // namespace AMP::Operator


#endif