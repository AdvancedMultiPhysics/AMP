#ifndef RAD_DIF_FD_DISCRETIZATION
#define RAD_DIF_FD_DISCRETIZATION

#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/IO/AsciiWriter.h"

#include "AMP/vectors/CommunicationList.h"
#include "AMP/matrices/petsc/NativePetscMatrix.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/data/VectorDataNull.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"
#include "AMP/vectors/VectorBuilder.h"

#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/geometry/Geometry.h"
#include "AMP/geometry/shapes/Box.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixBuilder.h"

#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/petsc/PetscMatrixShellOperator.h"
#include "AMP/operators/OperatorFactory.h"

#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"

#include "RDUtils.h"

#include <iostream>
#include <iomanip>
#include <optional>

/** The classes in this file are (or are associated with) spatial finite-discretizations of  
 * radiation-diffusion problem:
 *      u'(t) - L(u) - R(u)  = s(t), u(0) = u_0
 * over the spatial domain [0,1]^d, for d = 1 or d = 2. Where:
 * 1. L(u) = [grad dot ( D0 * grad u0 ), grad dot ( D1 * grad u1 )] is a (possibly) nonlinear
 * diffusion operator
 * 2. R(u) is a (possibly) nonlinear reaction operator
 *
 * The vector u = [u0, u1] = [E, T] is a block vector, holding E and T.
 * 
 * In more detail, the general PDE is:
 *  * L(u) = [grad dot (k11*D_E grad E), grad dot (k21*D_T grad T)], where:
 *      * if model == "linear": 
 *          D_E = D_T = 1.0
 *      * if model == "nonlinear": 
 *          D_E = 1/(3*sigma), D_T = T^2.5, and sigma = (z/T)^3
 *      
 *  * if model == "linear":
 *      R(u) = [k12*(T - E), -k22*(T - E)]
 *  * if model == "nonlinear":
 *      R(u) = [k12*simga*(T^4 - E), -k22*simga*(T^4 - E)]
 */


class RadDifOp;
class RadDifOpPJac;
class RadDifOpPJacParameters;

// Friend classes
class BERadDifOp;
class BERadDifOpPJac;



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
    //! Flag indicating whether our data has been accessed, and hence possibly modified, by a non-friend class (i.e., a BE wrapper of a RadDifOpPJac). This is set to true any time a getter is called.
    bool d_dataMaybeOverwritten = false; 

    //! Members used to store matrix components
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_E  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_T  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_EE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_ET = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TT = nullptr;
};



/** OperatorParameters for creating the linearized operator RadDifOpPJac. This is just regular OperatorParameters appended with:
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

/** Picard linearization of a RadDifOp. */
class RadDifOpPJac : public AMP::Operator::LinearOperator {

private:
    // Indices used for referencing WEST, ORIGIN, and EAST entries in 3-point stencils
    static constexpr size_t W = 0;
    static constexpr size_t O = 1;
    static constexpr size_t E = 2;

//
public:
    std::shared_ptr<AMP::Database>          d_db;
    //! Representation of this operator as a block 2x2 matrix
    std::shared_ptr<RadDifOpPJacData>       d_data      = nullptr;
    //! The underlying nonlinear operator
    RadDifOp *                              d_RadDifOp  = nullptr;
    //! The vector the above operator is linearized about
    AMP::LinearAlgebra::Vector::shared_ptr  d_frozenVec;


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


    // void setData1D( );
    // void setData2D( );
};
// End of RadDifOpPJac





/** Finite-difference discretization of a radiation-diffusion operator */
class RadDifOp : public AMP::Operator::Operator {

//
private:
    // Indices used for referencing WEST, ORIGIN, and EAST entries in 3-point stencils
    static constexpr size_t W = 0;
    static constexpr size_t O = 1;
    static constexpr size_t E = 2;

    // Constant scaling factors in the PDE
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

    // Mesh sizes, hx, hy, hz. We compute these based on the incoming mesh
    std::vector<double> d_h;
    // Global grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_globalBox = nullptr;
    //! Convenience member
    static constexpr auto VertexGeom = AMP::Mesh::GeomType::Vertex;

//
public: 
    #if 1
    std::shared_ptr<AMP::Discretization::DOFManager>      d_nodalDOFMan;
    #endif

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

    // Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const; 

    // Vector of hx, hy, hz
    std::vector<double> getMeshSize() const;

    //! Populate the given multivector a function of the given type
    void fillMultiVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> vec_, std::function<double( int, AMP::Mesh::MeshElement & )> fun );

    //! Set the Robin return function for the energy. If the user does not use call this function then the Robin values rk from the input database will be used.
    void setRobinFunctionE( std::function<double(int, double, double, AMP::Mesh::MeshElement &)> fn_ ); 
    
    //! Set the pseudo-Neumann return function for the temperature. If the user does not use call this function then the pseudo-Neumann values nk from the input database will be used.
    void setPseudoNeumannFunctionT( std::function<double(int, AMP::Mesh::MeshElement &)> fn_ ); 


    
    #if 0
    std::shared_ptr<AMP::LinearAlgebra::Vector> createNodalInputVector() const {
        auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "" );
        return AMP::LinearAlgebra::createVector<double>( d_nodalDOFMan, var );
    };
    #endif


// Boundary-related data and routines
private:

    //! Defines a boundary in a given dimension (WEST is the first boundary, and EAST the second) 
    enum class BoundarySide { WEST, EAST };

    /** Return the boundaryID in {1,...,6} corresponding to a dim in {0,1,2}, given the 
     * corresponding side.
     */
    size_t getBoundaryIDFromDim(size_t dim, BoundarySide side) const;

    /** Assuming a boundaryID in {1,...,6}, get the dimension of the boundary, i.e., boundaries 1
     * and 2 live in the first dimension, 3 and 4 the second, and 5 and 6 the third. 
     */
    size_t getDimFromBoundaryID(size_t boundaryID) const;

    //! Ghost values for E and T; set via ''setGhostData''
    std::array<double, 2> d_ghostData;

    /** Set values in the member ghost values array, d_ghostData, corresponding to the given
     * boundary at the given node given the interior value Eint and Tint. 
     * This routine assumes Robin on E and pseudo-Neumann on T 
     */
    void setGhostData( size_t boundaryID, AMP::Mesh::MeshElement &node, double Eint, double Tint );    
    

    /** Prototype of function returning value of Robin BC of E on given boundary at given node. The
     * user can specify any function with this signature.
     */
    std::function<double( size_t boundaryID, double a, double b, AMP::Mesh::MeshElement & node )> d_robinFunctionE;

    /** Prototype of function returning value of pseudo-Neumann BC of T on given boundary at given
     * node. The user can specify any function with this signature
     */
    std::function<double( size_t boundaryID, AMP::Mesh::MeshElement & node )> d_pseudoNeumannFunctionT;


    //! Return database constant rk for boundary k
    double robinFunctionEFromDB( size_t boundaryID ); 

    //! Return database constant nk for boundary k
    double pseudoNeumannFunctionTFromDB( size_t boundaryID ); 
    
    
    /** Suppose on boundary k we have the two equations:
     *     ak*E + bk * hat{nk} dot k11*D_E(T) grad E = rk
     *                 hat{nk} dot            grad T = nK,
     * where ak, bk, rk, and nk are all known constants qne hat{nk} is the outward-facing normal
     * vector at the boundary.
     * The discretization of these conditions involves one ghost point for E and T (Eg and Tg), and
     * one interior point (Eint and Tint). Here we solve for the ghost points and return them. Note
     * that this system, although nonlinear, can be solved by forward substitution. 
     * NOTE: that the actual boundary does not matter here once the constants a, b, r, and n have
     * been specified.
     */
    void ghostValuesSolve( double a, double b, double r, double n, double h, double Eint, double Tint, double &Eg, double &Tg ); 


    /** Suppose on boundary k we have the equation:
     *      hat{nk} dot grad T = nk,
     * where nk is a known constant and hat{nk} is the outward-facing normal vector at the
     * boundary. This BC is discretized as
     *      sign(hat{nk}) * [Tg_k - Tint_k]/h = nk
     * Here we solve for the ghost point Tg_k and return it
     * 
     * @param[out] Tg ghost-point value that satisfies the discretized BC
     */
    double ghostValueSolvePseudoNeumannT( double n, double h, double Tint );
    
    /** Suppose on boundary k we have the equation:
     *      ak*E + bk * hat{nk} dot ck grad E = rk
     * where ak, bk, rk, nk, and ck are all known constants, and hat{n}_k is the outward-facing 
     * normal vector at the boundary. This BC is discretized as
     *      ak*0.5*[Eg_k + Eint_k] + bk*ck*sign(hat{nk}) *[Eg_k - Eint_k]/h = rk
     * Here we solve for the ghost point Eg_k and return it.
     * 
     * @param[out] Eg ghost-point value that satisfies the discretized BC
     */
    double ghostValueSolveRobinE( double a, double b, double r, double c, double h, double Eint );


    double PicardCorrectionCoefficient( size_t component, double ck, size_t boundaryID ) const;

    /** In the Robin BC for energy we get Eghost = alpha*Eint + beta, this function returns the
     * coefficient alpha, which is the Picard linearization of this equation w.r.t Eint, where ck
     * is the energy flux in the BC.
     */
    double PicardCorrectionRobinE( double ck, size_t boundary ) const;

    
    
    #if 0
    void apply1D( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> rET);

    void apply2D( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> rET);

    // Values in the 3-point stencil
    std::vector<double> unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, int i );
    // Values in the 5-point stencil
    std::vector<double> unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, int i, int j );

        // Overloaded version of above also returning corresponding DOFs
    std::vector<double> unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, int i, std::vector<size_t> &dofs, bool &onBoundary );

    
    // Overloaded version of above also returning corresponding DOFs
    std::vector<double> unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, int i, int j, std::vector<size_t> &dofs, bool &onBoundary );
    #endif

private:    
    // Set d_multiDOFManagers after creating it from this Operators's mesh
    void setDOFManagers();

    void getLocalFDDiffusionCoefficients(
    const std::array<double,3> &ELoc3,
    const std::array<double,3> &TLoc3,
    double z,
    double h,
    bool computeE,
    double &Dr_WO, 
    double &Dr_OE,
    bool computeT,
    double &DT_WO, 
    double &DT_OE) const;

    //! Energy diffusion coefficient D_E given temperature T
    double diffusionCoefficientE( double T, double z ) const;

    //! Temperature diffusion coefficient D_E given temperature T
    double diffusionCoefficientT( double T ) const;

    //! Compute quasi-linear reaction coefficients REE, RET, RTE, REE
    void getSemiLinearReactionCoefficients( double T, double z, double &REE, double &RET, double &RTE, double &RTT ) const;

    //! Scale semi-linear reaction coefficients by constants k_ij in PDE
    void scaleReactionCoefficientsBy_kij( double &REE, double &RET, double &RTE, double &RTT ) const;
    //! Scale D_E by k11
    void scaleDiffusionCoefficientEBy_kij( double &D_E ) const;
    //! Scale D_T by k21
    void scaleDiffusionCoefficientTBy_kij( double &D_T ) const;


    void unpackLocalStencilData( 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,  
    std::array<int, 3> &ijk, // is modified locally, but returned in same state
    int dim,
    std::array<double, 3> &ELoc3, 
    std::array<double, 3> &TLoc3);

    void unpackLocalStencilData( 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,  
    std::array<int, 3> &ijk, // is modified locally, but returned in same state
    int dim,
    std::array<double, 3> &ELoc3, 
    std::array<double, 3> &TLoc3, 
    std::array<size_t, 3> &dofs,
    std::optional<BoundarySide> &boundaryIntersection);

    
    // Map from grid index i, or i,j, or i,j,k to a MeshElementIndex to a MeshElementId and then to the corresponding DOF
    size_t gridIndsToScalarDOF( int i, int j = 0, int k = 0 ); 

    size_t gridIndsToScalarDOF( std::array<int,3> ijk ) {
        return gridIndsToScalarDOF( ijk[0], ijk[1], ijk[2] );
    }

    // Map from grid index to a MeshElement
    AMP::Mesh::MeshElement gridIndsToMeshElement( int i, int j = 0, int k = 0 ); 

    AMP::Mesh::MeshElement gridIndsToMeshElement( std::array<int,3> ijk ) {
        return gridIndsToMeshElement( ijk[0], ijk[1], ijk[2] );
    } 

    //! Map from scalar DOF to grid indices i, j, k
    std::array<int,3> scalarDOFToGridInds( size_t dof ) const;


    // Convert a global element box to a global node box.
    // Modified from src/mesh/test/test_BoxMeshIndex.cpp by removing the possibility of any of the
    // grid dimensions being periodic.
    AMP::Mesh::BoxMesh::Box getGlobalNodeBox() const;


protected:
    
/* virtual function returning a parameter object that can be used to reset the corresponding
    RadDifOpPJac operator. Note that in the base class's getParameters get's redirected to this function. 
    u_in is the current nonlinear iterate. */
    std::shared_ptr<AMP::Operator::OperatorParameters> getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) override;

};
// End of RadDifOp



#endif