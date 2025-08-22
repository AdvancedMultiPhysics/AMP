/* The classes here wrap RadiationDiffusionFDDiscretization-based operators as backward Euler (BE) operators */
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


class RadDifOp;
class RadDifOpPJac;
class RadDifOpPJacParameters;

// Friend classes
class BERadDifOp;
class BERadDifOpPJac;


/* The Picard Linearization is a LinearOperator with the following structure: 
[ d_E 0   ]   [ diag(r_EE) diag(r_ET) ]
[ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]

where the first matrix is the diffusion terms, and the second the reaction terms.
*/


/* Data required to store and apply the Picard linearization of a RadDifOp. 
The data is stored as two matrices and 4 vectors. */
struct RadDifOpPJacData {
    // Flag indicating whether the data here has been overwitten to a BEOper
    bool overwrittenByBEOper                         = false; 

    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_E  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_T  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_EE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_ET = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TT = nullptr;
};



/* OperatorParameters for a RadDifOpPJac. Just regular OperatorParameters with:
    1. a vector representing the current outer approximate solution
    2. a raw pointer to the outer nonlinear Operator since this has functionality that we require access to (e.g., getting Robin BC data). We access all problem parameters that pass over into the linearized problem through this pointer rather than duplicating them when creating the linearized operator */
class RadDifOpPJacParameters : public AMP::Operator::OperatorParameters
{
public:
    // Constructor
    explicit RadDifOpPJacParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db ) { };
    virtual ~RadDifOpPJacParameters() {};

    AMP::LinearAlgebra::Vector::shared_ptr d_frozenSolution = nullptr;
    RadDifOp *                                   d_RadDifOp = nullptr; // This should really be a reference to a const
};

/* Picard linearization of a RadDifOp. */
class RadDifOpPJac : public AMP::Operator::LinearOperator {

public:

    std::shared_ptr<AMP::Database>          d_db;
    AMP::LinearAlgebra::Vector::shared_ptr  d_frozenVec;
    std::shared_ptr<RadDifOpPJacData>       d_data      = nullptr;
    RadDifOp *                              d_RadDifOp  = nullptr;
    // Flag an outer BEOper has to set to true every time it calls an apply...
    bool applyWithOverwrittenBEOperDataIsValid          = false;

    // Constructor
    RadDifOpPJac(std::shared_ptr<const AMP::Operator::OperatorParameters> params_);

    virtual ~RadDifOpPJac() {};

    // Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override;

    // Used by OperatorFactory to create a RadDifOpPJac
    static std::unique_ptr<AMP::Operator::Operator> create( std::shared_ptr<AMP::Operator::OperatorParameters> params ) {  
        return std::make_unique<RadDifOpPJac>( params ); };

    /* Virtual function. Resets this operator using a RadDifOpJacParams; e.g., it updates the frozen solution vector, d_data etc. */
    void reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params ) override;

    // Pure virtual function
    std::string type() const { return "RadDifOpPJac"; };

    // Pure virtual function
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> rET );


private:

    // New data is created only if the existing one is null (as after construction or a reset)
    void setData( );

    void setData1D( );
    void setData2D( );

    void applyFromData( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_, std::shared_ptr<AMP::LinearAlgebra::Vector> LET_  );

};
// End of RadDifOpPJac




/* ----------------------------------------------------------------------------------------
    Implementation of a finite-difference discretization of a radiation-diffusion operator 
----------------------------------------------------------------------------------------- */
class RadDifOp : public AMP::Operator::Operator {

public: 
    // I + gamma*L, where L is a RadDifOp
    friend class BERadDifOp; 

    // hat{L}, where hat{L} is a linearization of L
    friend class RadDifOpPJac; 

    // MultiDOFManager for collectively managing E and T, and scalar for each separately
    std::shared_ptr<AMP::Discretization::multiDOFManager> d_multiDOFMan;
    std::shared_ptr<AMP::Discretization::DOFManager>      d_scalarDOFMan;
    std::shared_ptr<AMP::Discretization::DOFManager>      d_nodalDOFMan;
    // Parameters required by the discretization
    std::shared_ptr<AMP::Database>                        d_db;
    // Vertex-based geomety
    AMP::Mesh::GeomType                                   d_geomType = AMP::Mesh::GeomType::Vertex;
    std::shared_ptr<AMP::Mesh::BoxMesh>                   d_BoxMesh;

    // Constructor call's base class's constructor
    RadDifOp(std::shared_ptr<const AMP::Operator::OperatorParameters> params);

    virtual ~RadDifOp() {};
    
    // Pure virtual function
    std::string type() const { return "RadDifOp"; };
    
    // Pure virtual function
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> LET) override;

    // E and T must be non-negative
    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) override;

    // Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const; 


    // std::shared_ptr<AMP::LinearAlgebra::Vector> createNodalInputVector() const {
    //     auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "" );
    //     return AMP::LinearAlgebra::createVector<double>( d_nodalDOFMan, var );
    // };


    void fillMultiVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> vec_, std::function<double( int, AMP::Mesh::MeshElement & )> fun );

    // The user can specify any Robin return function for E with this signature; if they do then this will overwrite the default.
    void setRobinFunctionE( std::function<double(int, double, double, AMP::Mesh::MeshElement &)> fn_ ); 
    
    // The user can specify any pseudo Neumann return function for T with this signature; if they do then this will overwrite the default.
    void setPseudoNeumannFunctionT( std::function<double(int, AMP::Mesh::MeshElement &)> fn_ ); 



private:
    // Set d_multiDOFManagers after creating it from this Operators's mesh
    void setDOFManagers();

    void apply1D( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> rET);

    void apply2D( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> rET);

    // Prototype of function returning value of Robin BC of E on given boundary at given node. The user can specify any function with this signature
    std::function<double( int boundary, double a, double b, AMP::Mesh::MeshElement & node )> d_robinFunctionE;

    // Prototype of function returning value of pseudo-Neumann BC of T on given boundary at given node. The user can specify any function with this signature
    std::function<double( int boundary, AMP::Mesh::MeshElement & node )> d_pseudoNeumannFunctionT;

    // Default function for returning Robin values; this can be overridden by the user
    double robinFunctionEDefault( int boundary ); 
    
    // Default function for returning pseudo-Neumann values; this can be overridden by the user
    double pseudoNeumannFunctionTDefault( int boundary ); 
    
    std::vector<double> getGhostValues( int boundary, AMP::Mesh::MeshElement &node, double Eint, double Tint );    
    std::vector<double> ghostValuesSolve( double a, double b, double r, double n, double Eint, double Tint ); 
    double energyDiffusionCoefficientAtMidPoint( double T_left, double T_right);
    double ghostValuePseudoNeumannTSolve( double n, double Tint );
    double ghostValueRobinESolve( double a, double b, double r, double c, double Eint );
    double ghostValueRobinEPicardCoefficient( double ck, size_t boundary ) const;

    // Values in the 3-point stencil
    std::vector<double> unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, AMP::Mesh::BoxMesh::Box &globalBox, int i );
    // Overloaded version of above also returning corresponding DOFs
    std::vector<double> unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, AMP::Mesh::BoxMesh::Box &globalBox, int i, std::vector<size_t> &dofs, bool &onBoundary );

    // Values in the 5-point stencil
    std::vector<double> unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, AMP::Mesh::BoxMesh::Box &globalBox, int i, int j );
    // Overloaded version of above also returning corresponding DOFs
    std::vector<double> unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, AMP::Mesh::BoxMesh::Box &globalBox, int i, int j, std::vector<size_t> &dofs, bool &onBoundary );

    // Map from grid index i, or i,j, or i,j,k to a MeshElementIndex to a MeshElementId and then to the corresponding DOF
    size_t gridIndsToScalarDOF( int i, int j = 0, int k = 0 ); 

    // Map from grid index to a MeshElement
    AMP::Mesh::MeshElement gridIndsToMeshElement( int i, int j = 0, int k = 0 ); 

protected:
    
/* virtual function returning a parameter object that can be used to reset the corresponding
    RadDifOpPJac operator. Note that in the base class's getParameters get's redirected to this function. 
    u_in is the current nonlinear iterate. */
    std::shared_ptr<AMP::Operator::OperatorParameters> getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) override;

};
// End of RadDifOp



#endif