#ifndef RAD_DIF_FD_DISCRETIZATION
#define RAD_DIF_FD_DISCRETIZATION

#include <optional>

#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/IO/AsciiWriter.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/geometry/shapes/Box.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/LinearOperator.h"

namespace AMP::Operator {

// Classes declared here
class RadDifOp;
class RadDifOpPJac;
struct RadDifOpPJacData;
class RadDifOpPJacParameters;

// Friend classes
class BERadDifOp;
class BERadDifOpPJac;

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
 * where diffusive fluxes depend on the "model" parameter as:
 *      1. if model == "linear": 
 *          D_E = D_T = 1.0
 *      2. if model == "nonlinear": 
 *          D_E = 1/(3*sigma), D_T = T^2.5, and sigma = (zatom/T)^3
 *      
 * The reaction term is dependent on the "model" parameter as:
 *      1. if model == "linear":
 *          R(u) = [k12*(T - E), -k22*(T - E)]
 *      2. if model == "nonlinear":
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


/** Finite-difference discretization of the spatial operator in the above radiation-diffusion 
 * equation. This discretization is based on that described in 
 *      "Dynamic implicit 3D adaptive mesh refinement for non-equilibrium radiation diffusion"
 * by B.Philipa, Z.Wangb, M.A.Berrilla, M.Birkeb, M.Pernice in Journal of Computational Physics 262
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
 *  4. model:
 * must be either 'linear' or 'nonlinear'. Specifies which PDE is discretized.
 * 
 *  5. fluxLimited:
 * bool. Flag indicating whether flux limiting is for diffusion of energy
 * 
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
class RadDifOp : public AMP::Operator::Operator {

//
private:
    //! Constant scaling factors in the PDE
    const double d_k11;
    const double d_k12;
    const double d_k21;
    const double d_k22;

    //! Problem dimension
    size_t d_dim  = 0;
    //! Flag whether we consider the linear or nonlinear PDE
    bool d_nonlinearModel;
    //! Flag whether flux limiting is used in the energy equation
    bool d_fluxLimited;

    //! Mesh sizes, hx, hy, hz. We compute these based on the incoming mesh
    std::vector<double> d_h;
    //! Global grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_globalBox = nullptr;
    //! Local grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_localBox = nullptr;
    
    //! Convenience member. Geometry type that results in cell centered data. In 1D: Line, 2D: Edge, 3D Cell
    AMP::Mesh::GeomType CellCenteredGeom;
    
//
public: 

    //! I + gamma*L, where L is a RadDifOp
    friend class BERadDifOp; 
    //! hat{L}, where hat{L} is a Picard linearization of L
    friend class RadDifOpPJac; 

    //! MultiDOFManager for managing [E,T] multivectors
    std::shared_ptr<AMP::Discretization::multiDOFManager> d_multiDOFMan;
    //! DOFManager for E and T individually
    std::shared_ptr<AMP::Discretization::DOFManager>      d_scalarDOFMan;
    //! Parameters required by the discretization
    std::shared_ptr<AMP::Database>                        d_db;
    //! Mesh; keep a pointer to save having to downcast repeatedly
    std::shared_ptr<AMP::Mesh::BoxMesh>                   d_BoxMesh;
    

    //! Constructor
    RadDifOp(std::shared_ptr<const AMP::Operator::OperatorParameters> params);
    //! Destructor
    virtual ~RadDifOp() {};
    
    std::string type() const override { return "RadDifOp"; };
    
    //! Compute LET = L(ET)
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> LET) override;

    //! E and T must be positive
    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) override;

    //! Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const; 

    //! Vector of hx, hy, hz
    std::vector<double> getMeshSize() const;

    //! Populate the given multivector a function of the given type
    void fillMultiVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> vec_, std::function<double( size_t component, AMP::Mesh::Point &point )> fun ) const;

    //! Set the Robin return function for the energy. If the user does not use this function then the Robin values rk from the input database will be used.
    void setBoundaryFunctionE( std::function<double( size_t boundaryID, AMP::Mesh::Point &boundaryPoint )> fn_ ); 
    
    //! Set the pseudo-Neumann return function for the temperature. If the user does not use this function then the pseudo-Neumann values nk from the input database will be used.
    void setBoundaryFunctionT( std::function<double( size_t boundaryID, AMP::Mesh::Point &boundaryPoint )> fn_ ); 


// Boundary-related data and routines
private:

    /** Prototype of function returning value of Robin BC of E on given boundary at given node. The
     * user can specify any function with this signature via 'setBoundaryFunctionE'
     * @param[in] boundaryID ID of the boundary
     * @param[in] boundaryPoint the point in space where the function is to be evaluated (this will 
     * be a point on the corresponding boundary) 
     */
    std::function<double( size_t boundaryID, AMP::Mesh::Point &boundaryPoint )> d_robinFunctionE;

    /** Prototype of function returning value of pseudo-Neumann BC of T on given boundary at given
     * node. The user can specify any function with this signature via 'setBoundaryFunctionT'
     * @param[in] boundaryID ID of the boundary
     * @param[in] boundaryPoint the point in space where the function is to be evaluated (this will 
     * be a point on the corresponding boundary) 
     */
    std::function<double( size_t boundaryID, AMP::Mesh::Point &boundaryPoint )> d_pseudoNeumannFunctionT;


    //! Defines a boundary in a given dimension (WEST is the first boundary, and EAST the second) 
    enum class BoundarySide { WEST, EAST };

    //! Ghost values for E and T; set via "setGhostData". I.e., values of E and T at the centroid of the ghost cell that neighbors the interior cell considered with "setGhostData"
    std::array<double, 2> d_ghostData;

    /** Set values in the member ghost values array, d_ghostData. 
     * @param[in] boundaryID ID of the boundary being considered
     * @param[in] boundaryPoint explicit point on the boundary being considered
     * @param[in] Eint value of E at centroid immediately interior to the boundary
     * @param[in] Tint value of T at centroid immediately interior to the boundary
     * @param[out] d_ghostData array that computed ghost values are stored in  
     */
    void setGhostData( size_t boundaryID, AMP::Mesh::Point &boundaryPoint, double Eint, double Tint );    
    
    /** Return the boundaryID in {1,...,6} corresponding to a dim in {0,1,2}, given the 
     * corresponding side.
     */
    size_t getBoundaryIDFromDim(size_t dim, BoundarySide side) const;

    /** Assuming a boundaryID in {1,...,6}, get the dimension of the boundary, i.e., boundaries 1
     * and 2 live in the first dimension, 3 and 4 the second, and 5 and 6 the third. 
     */
    size_t getDimFromBoundaryID(size_t boundaryID) const;


    //! Get the Robin constants ak and bk from the database for the given boundaryID
    void getLHSRobinConstantsFromDB(size_t boundaryID, double &ak, double &bk) const;

    //! Return database constant rk for boundary k
    double getBoundaryFunctionValueFromDBE( size_t boundaryID ) const; 

    //! Return database constant nk for boundary k
    double getBoundaryFunctionValueFromDBT( size_t boundaryID ) const; 
    
    /** On boundary k we have the two equations:
     *     ak*E + bk * hat{nk} dot k11*D_E(T) grad E = rk,
     *                 hat{nk} dot            grad T = nK,
     * where ak, bk, rk, and nk are all known constants and hat{nk} is the outward-facing normal
     * vector at the boundary.
     * The discretization of these conditions involves one ghost point for E and T (Eg and Tg), and
     * one interior point (Eint and Tint). Here we solve for the ghost points and return them. Note
     * that this system, although nonlinear, can be solved directly by forward substitution. 
     * @param[out] Eg ghost-point value for E that satisfies the discretized BCs
     * @param[out] Tg ghost-point value for T that satisfies the discretized BCs
     */
    void ghostValuesSolve( double a, double b, double r, double n, double h, double Eint, double Tint, double &Eg, double &Tg ) const; 

    /** On boundary k we have the equation:
     *      hat{nk} dot grad T = nk,
     * where nk is a known constant and hat{nk} is the outward-facing normal vector at the
     * boundary. This BC is discretized as
     *      sign(hat{nk}) * [Tg_k - Tint_k]/h = nk.
     * Here we solve for the ghost point Tg_k and return it
     * @param[out] Tg ghost-point value that satisfies the discretized BC
     */
    double ghostValueSolveT( double n, double h, double Tint ) const;
    
    /** On boundary k we have the equation:
     *      ak*E + bk * hat{nk} dot ck grad E = rk,
     * where ak, bk, rk, nk, and ck=+k11*D_E(T) are all known constants, and hat{nk} is the 
     * outward-facing normal vector at the boundary. This BC is discretized as
     *      ak*0.5*[Eg_k + Eint_k] + bk*ck*sign(hat{nk}) *[Eg_k - Eint_k]/h = rk.
     * Here we solve for the ghost point Eg_k and return it.
     * @param[out] Eg ghost-point value that satisfies the discretized BC
     * @note ck=+k11*D_E(T) and not -k11*D_E(T)
     */
    double ghostValueSolveE( double a, double b, double r, double c, double h, double Eint ) const;

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


private: 
    //! Indices used for referencing WEST, ORIGIN, and EAST entries in Loc3 data structures
    static constexpr size_t WEST   = 0;
    static constexpr size_t ORIGIN = 1;
    static constexpr size_t EAST   = 2;

    //! Placeholder arrays for values used in 3-point stencils. Set by setLoc3Data
    std::array<double, 3> d_ELoc3;
    std::array<double, 3> d_TLoc3;
    //! Placeholder array for dofs we connect to in 3-point stencil. Set by setLoc3Data
    std::array<size_t, 3> d_dofsLoc3;

    //! Create and set member DOFManagers based on the mesh
    void setDOFManagers();

    /** Given 3-point stencils compute FD diffusion coefficients 
     * @param[in] computeE flag indicating whether to compute E diffusion coefficients
     * @param[in] computeT flag indicating whether to compute T diffusion coefficients 
     * @param[out] Dr_WO WEST coefficient in for energy
     * @param[out] Dr_OE EAST coefficient in for energy
     * @param[out] DT_WO WEST coefficient in for temperature
     * @param[out] DT_OE EAST coefficient in for temperature
    */
    void getLocalFDDiffusionCoefficients( const std::array<double,3> &ELoc3,
                                            const std::array<double,3> &TLoc3,
                                            double zatom,
                                            double h,
                                            bool computeE,
                                            double &Dr_WO, 
                                            double &Dr_OE,
                                            bool computeT,
                                            double &DT_WO, 
                                            double &DT_OE) const;

    //! Energy diffusion coefficient D_E given temperature T
    double diffusionCoefficientE( double T, double zatom ) const;

    //! Temperature diffusion coefficient D_E given temperature T
    double diffusionCoefficientT( double T ) const;

    //! Compute quasi-linear reaction coefficients REE, RET, RTE, REE
    void getSemiLinearReactionCoefficients( double T, double zatom, double &REE, double &RET, double &RTE, double &RTT ) const;

    //! Scale semi-linear reaction coefficients by constants k_ij in PDE
    void scaleReactionCoefficientsBy_kij( double &REE, double &RET, double &RTE, double &RTT ) const;
    //! Scale D_E by k11
    void scaleDiffusionCoefficientEBy_kij( double &D_E ) const;
    //! Scale D_T by k21
    void scaleDiffusionCoefficientTBy_kij( double &D_T ) const;

    /** Pack local 3-point stencil data into the arrays d_ELoc3 and d_TLoc3 for the given 
     * dimension. This involves a ghost-point solve if the stencil extends to a ghost point.
     * @param[in] E_vec vector of all (local) E values
     * @param[in] T_vec vector of all (local) T values
     * @param[in] ijk grid indices of DOF for which 3-point stencil values are to be unpacked
     * @param[in] dim dimension in which the 3-point extends
     * @param[out] d_ELoc3 E values in the 3-point stencil (WEST, ORIGIN, UPPER)
     * @param[out] d_TLoc3 T values in the 3-point stencil (WEST, ORIGIN, UPPER)
     * 
     * @note ijk is modified inside the function, but upon conclusion of the function is in its 
     * original state 
     */
    void setLoc3Data( 
        std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, 
        std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,  
        std::array<int, 3> &ijk, 
        int dim);

    /** Overloaded version of the above with two additional output parameters
     * @param[out] d_dofsLoc3 indices of the dofs in the 3-point stencil
     * @param[out] boundaryIntersection flag indicating if the stencil touches a boundary (and in 
     * which one if it does) 
     * 
     * @note if the stencil touches the boundary then the corresponding value in dofs is meaningless
     * @note this function implicity assumes that the stencil does not touch both boundaries at 
     * once (corresponding to the number of interior DOFs in the given dimension being larger than 
     * one) 
     */
    void setLoc3Data( 
        std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, 
        std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,  
        std::array<int, 3> &ijk, // is modified locally, but returned in same state
        int dim,
        std::optional<BoundarySide> &boundaryIntersection);

    
    //! Map from grid index to a the corresponding DOF
    size_t gridIndsToScalarDOF( int i, int j = 0, int k = 0 ) const; 

    //! Map from grid index to a the corresponding DOF
    size_t gridIndsToScalarDOF( std::array<int,3> ijk ) const;

    //! Map from grid index to a MeshElement
    AMP::Mesh::MeshElement gridIndsToMeshElement( int i, int j = 0, int k = 0 ) const; 

    //! Map from grid index to a MeshElement
    AMP::Mesh::MeshElement gridIndsToMeshElement( std::array<int,3> ijk ) const; 

    //! Map from scalar DOF to grid indices i, j, k
    std::array<int,3> scalarDOFToGridInds( size_t dof ) const;

    /** Gets a global node box (by converting global element box)
     * Modified from src/mesh/test/test_BoxMeshIndex.cpp by removing the possibility of any of the
     * grid dimensions being periodic.
     */
    AMP::Mesh::BoxMesh::Box getGlobalNodeBox() const;

    /** Gets a local node box (by converting local element box)
     * Modified from src/mesh/test/test_BoxMeshIndex.cpp by removing the possibility of any of the
     * grid dimensions being periodic.
     */
    AMP::Mesh::BoxMesh::Box getLocalNodeBox() const; 


    /** Helper function that first checks the incoming mesh satisfies requirements from the 
     * operator and then sets mesh-related member data 
     */
    void setAndCheckMeshData();

    // oktodo: delete
    void printMeshNodes();

//
protected:
    
    /** Returns a parameter object that can be used to reset the corresponding
     * RadDifOpPJac operator. Note that in the base class's getParameters get's redirected to this
     * function. 
     * @param[in] u_in is the current nonlinear iterate. 
     */
    std::shared_ptr<AMP::Operator::OperatorParameters> getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) override;

};
// End of RadDifOp



/** Picard linearization of a RadDifOp. 
 * Specifically, the spatial operators in the radiation-diffusion equation can be written as
 *          L(u) = -D(u) -R(u) <==> hat{L}(u)*u = -hat{D}(u)*u -hat{R}(u)*u
 * where hat{L}(u), hat{D}(u) and hat{R}(u) are block 2x2 matrices dependent upon the state u. This 
 * class implements the LinearOperator hat{L}(u), which is a Picard linearization of L(u).   
 */
class RadDifOpPJac : public AMP::Operator::LinearOperator {

private:
    // Indices used for referencing WEST, ORIGIN, and EAST entries in 3-point stencils
    static constexpr size_t WEST   = 0;
    static constexpr size_t ORIGIN = 1;
    static constexpr size_t EAST   = 2;

//
public:
    std::shared_ptr<AMP::Database>          d_db        = nullptr;
    //! Representation of this operator as a block 2x2 matrix
    std::shared_ptr<RadDifOpPJacData>       d_data      = nullptr;
    //! Raw pointer to the underlying nonlinear operator
    RadDifOp *                              d_RadDifOp  = nullptr;
    //! The vector the above operator is linearized about
    AMP::LinearAlgebra::Vector::shared_ptr  d_frozenVec = nullptr;

    //! Constructor
    RadDifOpPJac(std::shared_ptr<const AMP::Operator::OperatorParameters> params_);

    //! Destructor
    virtual ~RadDifOpPJac() {};

    //! Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override;

    //! Used by OperatorFactory to create a RadDifOpPJac
    static std::unique_ptr<AMP::Operator::Operator> create( std::shared_ptr<AMP::Operator::OperatorParameters> params ) {  
        return std::make_unique<RadDifOpPJac>( params ); };

    //! Reset the operator based on the incoming parameters. 
    void reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params ) override;

    std::string type() const override { return "RadDifOpPJac"; };

    //! Compute LET = L(ET)
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> LET );

    //! Allows an apply from Jacobian data that's been modified by an outside class (this is an acknowledgement that the caller of the apply deeply understands what they're doing)
    void applyWithOverwrittenDataIsValid() { d_applyWithOverwrittenDataIsValid = true; };

//
private:

    //! Flag indicating whether apply with overwritten Jacobian data is valid. This is reset to false at the end of every apply call, and can be set t true by the public member function
    bool d_applyWithOverwrittenDataIsValid = false;

    //! Apply action of the operator utilizing its representation in d_data
    void applyFromData( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_, std::shared_ptr<AMP::LinearAlgebra::Vector> LET_  );

    //! Set our d_data member
    void setData( );

    /** Sets the reaction-related vectors in our d_data member. 
     * This code is based on stripping out the reaction component of the apply of the nonlinear 
     * operator.
     * @param[in] T_vec T component of the frozen vector d_frozenVec
     */
    void setDataReaction( std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec );

    /** Get CSR data for a row of the Picard-linearized diffusion matrix dE or dT.
     * @param[in] component 0 (energy) or 1 (temperature) to get CSR data for  
     * @param[in] E_vec E component of the frozen vector d_frozenVec
     * @param[in] T_vec T component of the frozen vector d_frozenVec
     * @param[in] row the row to retrieve (a scalar index)
     * @param[out] cols the column indices for the non-zeros in the given row, with the diagonal 
     * entry first
     * @param[out] data the data for the non-zeros in the given row
     * 
     * @note this function implicity assumes that the stencil does not touch both boundaries at 
     * once (corresponding to the number of interior DOFs in the given dimension being larger than 
     * one)
     */
    void getCSRDataDiffusionMatrix( 
                                size_t component,
                                std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                                std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                                size_t row,
                                std::vector<size_t> &cols,
                                std::vector<double> &data );

    /** Fill the given input diffusion matrix with CSR data
     * @param[in] component 0 (for energy) or 1 (for temperature) 
     */
    void fillDiffusionMatrixWithData(size_t component, std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix);
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
    //! Getter routines; any external updates to the private data members below are done via these
    std::shared_ptr<AMP::LinearAlgebra::Matrix> get_d_E();
    std::shared_ptr<AMP::LinearAlgebra::Matrix> get_d_T();
    std::shared_ptr<AMP::LinearAlgebra::Vector> get_r_EE();
    std::shared_ptr<AMP::LinearAlgebra::Vector> get_r_ET();
    std::shared_ptr<AMP::LinearAlgebra::Vector> get_r_TE();
    std::shared_ptr<AMP::LinearAlgebra::Vector> get_r_TT();

private:
    //! Flag indicating whether our data has been accessed, and hence possibly modified, by a non-friend class (e.g., a BE wrapper of a RadDifOpPJac). This is set to true any time a getter is called.
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
 * 2. a raw pointer to the outer nonlinear operator, since this has functionality that we require
 * access to (e.g., getting Robin BC data). We access all problem parameters of the outer nonlinear 
 * problem via this pointer rather than duplicating them when creating the linearized operator 
 */
class RadDifOpPJacParameters : public AMP::Operator::OperatorParameters
{
public:
    // Constructor
    explicit RadDifOpPJacParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db ) { };
    virtual ~RadDifOpPJacParameters() {};

    AMP::LinearAlgebra::Vector::shared_ptr d_frozenSolution = nullptr;
    RadDifOp *                                   d_RadDifOp = nullptr; 
};

} // namespace AMP::Operator


#endif