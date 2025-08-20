// Class definitions for discretizations, and implementation of some basic functions
#ifndef RAD_DIF_DISCRETIZATION
#define RAD_DIF_DISCRETIZATION

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

#include "utils.hpp"

#include <iostream>
#include <iomanip>


class RadDifOp;
class RadDifOpJac;
class RadDifOpJacParameters;

/* The Picard Linearization is a LinearOperator with the following structure: 
[ d_E 0   ]   [ diag(r_EE) diag(r_ET) ]
[ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]

where the first matrix is the diffusion terms, and the second the reaction terms.
*/


/* Data required to store and apply the Picard linearization of a RadDifOp. 
The data is stored as two matrices and 4 vectors. */
struct RadDifOpJacData {
    // Flag indicating whether the data here has been overwitten to a BEOper
    bool overwrittenByBEOper                         = false; 

    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_E  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_T  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_EE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_ET = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TT = nullptr;
};

/* Overwrites the above data structure to create the BEOperator 
([I 0]         [ d_E 0   ])         [ diag(r_EE) diag(r_ET) ]
([0 I] + gamma*[ 0   d_T ]) + gamma*[ diag(r_TE) diag(r_TT) ]
==
[ d_E_BE  0    ]   [ diag(r_EE_BE) diag(r_ET_BE) ]
[ 0      d_T_BE] + [ diag(r_TE_BE) diag(r_TT_BE) ]
*/
struct BERadDifOpJacData {
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_E_BE  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_T_BE  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_EE_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_ET_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TE_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TT_BE = nullptr;

    BERadDifOpJacData () {};
    BERadDifOpJacData( std::shared_ptr<RadDifOpJacData> data, double gamma ) {
        AMP_INSIST( data, "Non-null data required" );

        data->overwrittenByBEOper = true;

        // Unpack vectors
        r_EE_BE = data->r_EE;
        r_ET_BE = data->r_ET;
        r_TE_BE = data->r_TE;
        r_TT_BE = data->r_TT;
        // Scale them by gamma
        r_EE_BE->scale( gamma );
        r_ET_BE->scale( gamma );
        r_TE_BE->scale( gamma );
        r_TT_BE->scale( gamma );
        // Make everything consistent
        r_EE_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        r_ET_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        r_TE_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        r_TT_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        // Unpack matrices
        d_E_BE  = data->d_E;
        d_T_BE  = data->d_T;
        // Scale by gamma
        d_E_BE->scale( gamma );
        d_T_BE->scale( gamma );
        // Add identity perturbations 
        auto DOFMan_E = d_E_BE->getRightDOFManager(  );
        for ( auto dof = DOFMan_E->beginDOF(); dof != DOFMan_E->endDOF(); dof++ ) {
            d_E_BE->addValueByGlobalID(dof, dof, 1.0);
        }
        auto DOFMan_T = d_T_BE->getRightDOFManager(  );
        for ( auto dof = DOFMan_T->beginDOF(); dof != DOFMan_T->endDOF(); dof++ ) {
            d_T_BE->addValueByGlobalID(dof, dof, 1.0);
        }
        // Make everything consistent
        d_E_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_T_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    };
};



/*
Maybe it's just easiest to create a map from a nodal index to a variable index and vice versa

Say we have 2*n values on process (n for T) and (n for E), indices a,a+1, ..., a+2*n-1, where a is an even integer

Then variable ordering is
[a,a+1,a+2,a+3,   ...,a+(n-1)],  [a+n,a+1+n,...a+n-1+(n-1),a+n+(n-1)]
And the nodal ordering is
 a,a+n,a+1,a+1+n, ...,                      a+(n-1),a+n+(n-1) 


// This is only going to work if I know a... or for local DOFs where a=0
but that's an issue because sometimes I have connections to non-local DOFs...

For a given index, I need to know the first index on that process and how many dofs are on that process...


# Consider the following variably-ordered indices mapped to nodally-ordered indices
# Variable ordering
P0: [0 1 2 3 4] [5 6 7 8 9 10]
P1: [11 12 13 14] [16 17 18 19]

# Nodal ordering
P0: [0 5 1 6 2 7 3 8 4 9 5 10]
P1: [11 16 12 17 13 18 14 19]

So say I'm given DOFs [0 16] then I have an issue because they're on different proccesses with different starting points and different numbers of DOFs on each...

Just assume serial for the moment.... 
*/

// 0->0, 1->n, 2->1, 3->n+1 ...
// So even nodal indices get divided by 2
// and odd nodal indices get subtract 1, divide by 2 and add n  
inline void nodalOrderingToVariableOrdering( size_t n, const std::vector<size_t> &ndl, std::vector<size_t> &var ) {
    var.resize( ndl.size() );
    for ( auto i = 0; i < ndl.size(); i++ ) {
        auto dof = ndl[i];
        if ( dof % 2 == 0 ) {
            var[i] = dof/2;
        } else {
            var[i] = (dof-1)/2 + n;
        }
    }
}

// 0->0, 1->2, 2->4, ..., n->1, 1+n->3, 2+n->5
// So variable indices  <n get doubled
// and variable indices >= subtract n, double and add 1
inline void variableOrderingToNodalOrdering( size_t n, const std::vector<size_t> &var, std::vector<size_t> &ndl ) {
    ndl.resize( var.size() );
    for ( auto i = 0; i < var.size(); i++ ) {
        auto dof = var[i];
        if ( dof < n ) {
            ndl[i] = dof * 2;
        } else {
            ndl[i] = (dof-n) * 2 + 1;
        }
    }
}





/* OperatorParameters for a RadDifOpJac. Just regular OperatorParameters with:
    1. a vector representing the current outer approximate solution
    2. a raw pointer to the outer nonlinear Operator since this has functionality that we require access to (e.g., getting Robin BC data). We access all problem parameters that pass over into the linearized problem through this pointer rather than duplicating them when creating the linearized operator */
class RadDifOpJacParameters : public AMP::Operator::OperatorParameters
{
public:
    // Constructor
    explicit RadDifOpJacParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db ) { };
    virtual ~RadDifOpJacParameters() {};

    AMP::LinearAlgebra::Vector::shared_ptr d_frozenSolution = nullptr;
    RadDifOp *                                   d_RadDifOp = nullptr; // This should really be a reference to a const
};

/* Picard linearization of a RadDifOp. */
class RadDifOpJac : public AMP::Operator::LinearOperator {

public:

    std::shared_ptr<AMP::Database>          d_db;
    AMP::LinearAlgebra::Vector::shared_ptr  d_frozenVec;
    std::shared_ptr<RadDifOpJacData>        d_data      = nullptr;
    RadDifOp *                              d_RadDifOp  = nullptr;
    // Flag an outer BEOper has to set to true every time it calls an apply...
    bool applyWithOverwrittenBEOperDataIsValid          = false;

    // Constructor
    RadDifOpJac(std::shared_ptr<const AMP::Operator::OperatorParameters> params_) : 
            AMP::Operator::LinearOperator( params_ ) {

        if ( d_iDebugPrintInfoLevel > 0 )
            AMP::pout << "RadDifOpJac::RadDifOpJac() " << std::endl;

        auto params = std::dynamic_pointer_cast<const RadDifOpJacParameters>( params_ );
        AMP_INSIST( params, "params must be of type RadDifOpJacParameters" );

        // Unpack parameters
        d_frozenVec = params->d_frozenSolution;
        d_RadDifOp  = params->d_RadDifOp;

        setData( );
    };

    // Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override;

    // Used by OperatorFactory to create a RadDifOpJac
    static std::unique_ptr<AMP::Operator::Operator> create( std::shared_ptr<AMP::Operator::OperatorParameters> params ) {  
        return std::make_unique<RadDifOpJac>( params ); };

    /* Virtual function. Resets this operator using a RadDifOpJacParams; e.g., it updates the frozen solution vector, d_data etc. */
    void reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params ) override;

    // Pure virtual function
    std::string type() const { return "RadDifOpJac"; };

    // Pure virtual function
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> rET ) {

        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "RadDifOpJac::apply() " << std::endl;

        // If the data has been overwritten by a BEOper, then this apply will be an apply of that operator. That's fine, but so as to not cause any confusion about the state of the data the BEOper must acknowledge before every apply that's indeed what they're trying to do.
        if ( d_data->overwrittenByBEOper ) {
            AMP_INSIST( applyWithOverwrittenBEOperDataIsValid, "This apply is invalid because the data has been mutated by a BEOper; you must first set the flag 'applyWithOverwrittenBEOperDataIsValid' to true if you really want to do an apply" );
        }

        applyFromData( ET, rET );

        // Reset flag
        applyWithOverwrittenBEOperDataIsValid = false;
    };


private:

    // New data is created only if the existing one is null (as after construction or a reset)
    void setData( ) {
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpJac::setData() " << std::endl;    

        auto meshDim = this->getMesh()->getDim();
        if ( meshDim == 1 ) {
            setData1D( );
        } else if ( meshDim == 2 ) {
            setData2D( );
        } else {
            AMP_ERROR( "Invalid dimension" );
        }
    }

    void setData1D( );
    void setData2D( );

    void applyFromData( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_, std::shared_ptr<AMP::LinearAlgebra::Vector> LET_  );

};
// End of RadDifOpJac




/* Abstract class implementing a radiation-diffusion operator */
class RadDifOp : public AMP::Operator::Operator {

public: 

    // I + gamma*L, where L is a RadDifOp
    friend class BERadDifOp; 

    // hat{L}, where hat{L} is a linearization of L
    friend class RadDifOpJac; 

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
    RadDifOp(std::shared_ptr<const AMP::Operator::OperatorParameters> params) : 
            AMP::Operator::Operator( params ) {

        if ( d_iDebugPrintInfoLevel > 0 )
            AMP::pout << "RadDifOp::RadDifOp() " << std::endl; 

        // Keep a pointer to my BoxMesh to save having to do this downcast repeatedly 
        d_BoxMesh = std::dynamic_pointer_cast<AMP::Mesh::BoxMesh>( this->getMesh() );
        AMP_INSIST( d_BoxMesh, "Mesh must be a AMP::Mesh::BoxMesh" );

        // Set PDE parameters
        d_db = params->d_db;
        AMP_INSIST(  d_db, "Requires non-null db" );

        // Set DOFManagers
        this->setDOFManagers();
        AMP_INSIST(  d_multiDOFMan, "Requires non-null multiDOF" );

        AMP_INSIST( d_db->getDatabase( "PDE" ),  "PDE_db is null" );
        AMP_INSIST( d_db->getDatabase( "mesh" ), "mesh_db is null" );

        auto model = d_db->getDatabase( "PDE" )->getWithDefault<std::string>( "model", "" );
        AMP_INSIST( model == "linear" || model == "nonlinear", "model must be 'linear' or 'nonlinear'" );

        // Specify default Robin return function for E
        std::function<double( int, double, double, AMP::Mesh::MeshElement & )> wrapperE = [&]( int boundary, double, double, AMP::Mesh::MeshElement & ) { return robinFunctionEDefault( boundary ); };
        this->setRobinFunctionE( wrapperE );
        // Specify default Neumann return function for T
        std::function<double( int, AMP::Mesh::MeshElement & )> wrapperT = [&]( int boundary,  AMP::Mesh::MeshElement & ) { return pseudoNeumannFunctionTDefault( boundary ); };
        this->setPseudoNeumannFunctionT( wrapperT );
    };
    
    
    // Pure virtual function
    std::string type() const { return "RadDifOp"; };
    
    // Pure virtual function
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> LET) override {

        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "RadDifOp::apply() " << std::endl;

        auto meshDim = d_BoxMesh->getDim();
        if ( meshDim == 1 ) {
            apply1D(ET, LET);
        } else {
            apply2D(ET, LET);
        }
    };

    // E and T must be non-negative
    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) override;

    // Create a multiVector of E and T over the mesh.
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const {
        auto ET_var = std::make_shared<AMP::LinearAlgebra::Variable>( "ET" );
        // auto E_var = std::make_shared<AMP::LinearAlgebra::Variable>( "E" );
        // auto T_var = std::make_shared<AMP::LinearAlgebra::Variable>( "T" );
        // std::vector<std::shared_ptr<AMP::LinearAlgebra::Variable>> vec = { E_var, T_var };
        // auto ET_var = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "ET", vec );
        auto ET_vec = AMP::LinearAlgebra::createVector<double>( this->d_multiDOFMan, ET_var );
        return ET_vec;
    };

    std::shared_ptr<AMP::LinearAlgebra::Vector> createNodalInputVector() const {
        auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "" );
        return AMP::LinearAlgebra::createVector<double>( d_nodalDOFMan, var );
    };



    void fillMultiVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> vec_, std::function<double( int, AMP::Mesh::MeshElement & )> fun );

    // The user can specify any Robin return function for E with this signature; if they do then this will overwrite the default.
    void setRobinFunctionE( std::function<double(int, double, double, AMP::Mesh::MeshElement &)> fn_ ) { d_robinFunctionE = fn_; };
    // The user can specify any pseudo Neumann return function for T with this signature; if they do then this will overwrite the default.
    void setPseudoNeumannFunctionT( std::function<double(int, AMP::Mesh::MeshElement &)> fn_ ) { d_pseudoNeumannFunctionT = fn_; };


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
    size_t gridIndsToScalarDOF( int i, int j = 0, int k = 0 ) {
        AMP::Mesh::BoxMesh::MeshElementIndex ind(
                        AMP::Mesh::GeomType::Vertex, 0, i, j, k );
        AMP::Mesh::MeshElementID id = d_BoxMesh->convert( ind );
        std::vector<size_t> dof;
        d_scalarDOFMan->getDOFs(id, dof);
        return dof[0];
    };

    // Map from grid index to a MeshElement
    AMP::Mesh::MeshElement gridIndsToMeshElement( int i, int j = 0, int k = 0 ) {
        AMP::Mesh::BoxMesh::MeshElementIndex ind(
                        AMP::Mesh::GeomType::Vertex, 0, i, j, k );
        return d_BoxMesh->getElement( ind );
    };

protected:
    
/* virtual function returning a parameter object that can be used to reset the corresponding
    RadDifOpJac operator. Note that in the base class's getParameters get's redirected to this function. 
    u_in is the current nonlinear iterate. */
    std::shared_ptr<AMP::Operator::OperatorParameters> getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) override {

        // Create a copy of d_db using Database copy constructor
        auto db = std::make_shared<AMP::Database>( *d_db );
        //auto db = std::make_shared<AMP::Database>( "JacobianParametersDB" );
        // OperatorParameters database must contain the "name" of the Jacobian operator that will be created from this
        db->putScalar( "name", "RadDifOpJac");
        //db->putScalar<int>( "print_info_level", d_db->getScalar<int>( "print_info_level" ) );
        // Create derived OperatorParameters for Jacobian
        auto jacOpParams    = std::make_shared<RadDifOpJacParameters>( db );
        // Set its mesh
        jacOpParams->d_Mesh = this->getMesh();

        jacOpParams->d_frozenSolution = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( u_in );
        jacOpParams->d_RadDifOp = this;

        return jacOpParams;
    }
};





/* ------------------------------------------------
    Class implementing a backward Euler operator 
------------------------------------------------- */
/* Implements the Operator I + gamma*L where L is a RadDifOp. This operator arises from the BDF discretization of the ODEs u'(t) + L*u = s(t). 

    The parsed OperatorParameters are used to create operator L. */ 
class BERadDifOp : public AMP::Operator::Operator {

public:
    double                    d_gamma    = -1.0;
    std::shared_ptr<RadDifOp> d_RadDifOp = nullptr;
    
    // Constructor call's base class's constructor
    BERadDifOp( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ ) : 
            AMP::Operator::Operator( params_ ) 
    {
        if ( d_iDebugPrintInfoLevel > 0 )
            AMP::pout << "BERadDifOp::BERadDifOp() " << std::endl;   

        // Create my RadDifOp
        d_RadDifOp = std::make_shared<RadDifOp>( params_ );
    }

    // Compute r <- (I + gamma*L)*u
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override
    {
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOp::apply() " << std::endl;  
        AMP_INSIST( d_RadDifOp, "RadDifOp not set!" );
        d_RadDifOp->apply( u_in, r );
        r->axpby (1.0, d_gamma, *u_in); // r <- 1.0*u + d_gamma * r
    }

    // Used to register this operator in a factory
    std::string type() const override {
        return "BERadDifOp";
    }

    // E and T must be non-negative
    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) override {
        return d_RadDifOp->isValidVector( ET );
    }

    // Set the time-step size in the operator
    void setGamma( AMP::Scalar gamma_ ) { 
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOp::setGamma() " << std::endl;  
        d_gamma = double( gamma_ );
    }

protected:

    /** virtual function returning a parameter object that can be used to reset the corresponding
    RadDifJacobianOp operator. Note that in the base class's getParameters get's redirected to this function. 
    u_in is the current nonlinear iterate. 
    The returned parameter object is used to create the Jacobian of this operator where required.
    */
    std::shared_ptr<AMP::Operator::OperatorParameters> getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) override {

        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOp::getJacobianParameters() " << std::endl;  

        // Get RadDifOp params
        auto jacOpParams = d_RadDifOp->getJacobianParameters( u_in );

        // Update the name from "RadDifOpJac"
        jacOpParams->d_db->setDefaultAddKeyBehavior( AMP::Database::Check::Overwrite, true );
        jacOpParams->d_db->putScalar<std::string>( "name", "BERadDifOpJac" );

        // Specify time-step size
        jacOpParams->d_db->putScalar<double>( "gamma", d_gamma );

        return jacOpParams;
    }

};

/* ------------------------------------------------
    Class implementing a backward Euler operator 
------------------------------------------------- */
/* Implements the Operator I + gamma*hat{L} where hat{L} is a RadDifOpJac. */
class BERadDifOpJac : public AMP::Operator::LinearOperator {

public:

    double                             d_gamma;
    std::shared_ptr<RadDifOpJac>       d_RadDifOpJac;
    std::shared_ptr<BERadDifOpJacData> d_data = nullptr;

    // Monolithic Jacobian in nodal ordering
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_JNodal;

    BERadDifOpJac( std::shared_ptr<AMP::Operator::OperatorParameters> params ) : AMP::Operator::LinearOperator( params ) {

        if ( d_iDebugPrintInfoLevel > 0 )
            AMP::pout << "BERadDifOpJac::BERadDifOpJac() " << std::endl;

        // Create my RadDifOpJac
        d_RadDifOpJac = std::make_shared<RadDifOpJac>( params );

        // Set the time step-size
        d_gamma = params->d_db->getScalar<double>( "gamma" );   

        setData();

        // 
        setLocalPermutationArrays();
    }

    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override { return d_RadDifOpJac->createInputVector( ); };

    // Used by OperatorFactory to create a BERadDifOpJac
    static std::unique_ptr<AMP::Operator::Operator> create( std::shared_ptr<AMP::Operator::OperatorParameters> params ) {  
        return std::make_unique<BERadDifOpJac>( params ); };

    // Compute r <- (I + gamma*hat{L})*u
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override
    {
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpJac::apply() " << std::endl;  
        AMP_INSIST( d_RadDifOpJac, "RadDifOpJac not set!" );

        d_RadDifOpJac->applyWithOverwrittenBEOperDataIsValid = true;
        d_RadDifOpJac->apply( u_in, r );

        // d_RadDifOpJac->apply( u_in, r );
        // r->axpby (1.0, d_gamma, *u_in); // r <- 1.0*u + d_gamma * r



        #if 0
        /* Test apply to check nodal ordering...
            Seems to be working in 1D and 2D....
        */
        auto r_test = d_RadDifOpJac->createInputVector();
        applyNodalToVariableVectors( u_in, r_test );
        std::cout << "+++Apply comparison: Variable vs nodal ordering\n";
        auto dof = d_RadDifOpJac->d_RadDifOp->d_multiDOFMan; 
        for ( auto i = dof->beginDOF(); i != dof->endDOF(); i++ ) {
            auto ri = r->getValueByGlobalID( i );
            auto riTemp = r_test->getValueByGlobalID( i );
            std::cout << "dof=" << i << ": dr=" << 1e-16+std::fabs(ri - riTemp)/std::fabs( ri ) << ",\tr=" << ri << ",rt=" << riTemp << "\n";
        }
        //AMP_ERROR( "Halt...." );
        #endif
    }

    std::shared_ptr<AMP::LinearAlgebra::Vector> createNodalInputVector() {
        return d_RadDifOpJac->d_RadDifOp->createNodalInputVector();
    }

    // Compute r <- (I + gamma*hat{L})*u
    // Serial only...
    void applyNodalToVariableVectors( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r )
    {
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpJac::apply() " << std::endl;  
        AMP_INSIST( d_RadDifOpJac, "RadDifOpJac not set!" );

        // Create vectors for nodal ordering
        auto u_inNodal = createNodalInputVector();
        auto rNodal    = createNodalInputVector(); 

        // Get u_in in nodal ordering
        createNodalOrderedCopy( u_in, u_inNodal );
        u_inNodal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        // Do the apply
        d_JNodal->mult( u_inNodal, rNodal );

        // Permute r from nodal ordering to variable ordering
        // Get u_in in nodal ordering
        createVariableOrderedCopy( rNodal, r );
    }

    void applyNodal( AMP::LinearAlgebra::Vector::const_shared_ptr u_in_ndl,
                AMP::LinearAlgebra::Vector::shared_ptr r_ndl )
    {
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpJac::apply() " << std::endl;  
        AMP_INSIST( d_RadDifOpJac, "RadDifOpJac not set!" );

        // Do the apply
        d_JNodal->mult( u_in_ndl, r_ndl );
    }

    void residualNodal( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                         AMP::LinearAlgebra::Vector::const_shared_ptr u,
                         AMP::LinearAlgebra::Vector::shared_ptr r )
    {
        AMP_INSIST( u, "NULL Solution Vector" );
        AMP_INSIST( r, "NULL Residual Vector" );
        AMP_ASSERT( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

        applyNodal( u, r );

        auto rInternal = subsetOutputVector( r );
        AMP_INSIST( ( rInternal ), "rInternal is NULL" );

        // the rhs can be NULL
        if ( f ) {
            auto fInternal = subsetOutputVector( f );
            rInternal->subtract( *fInternal, *rInternal );
        } else {
            rInternal->scale( -1.0 );
        }

        rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
    

    // Virtual function. This re-initializes/updates the operator. 
    void reset(std::shared_ptr<const AMP::Operator::OperatorParameters> params) override {

        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpJac::reset()  " << std::endl;

        // Calls with empty parameters are to be ignored
        if ( !params ) { 
            if ( d_iDebugPrintInfoLevel > 1 )
                AMP::pout << "  Called with empty parameters...  not resetting anything" << std::endl;
            return; 
        }

        AMP::Operator::LinearOperator::reset( params );

        // Reset my RadDifOpJac
        d_RadDifOpJac->reset( params );

        // Reset time step-size
        d_gamma = params->d_db->getScalar<double>( "gamma" );  

        // Reset my data
        setData( );

        // Reset my monolithic Jacobian
        d_JNodal = createMonolithicJac( );
    }  

    // Copy contents of variable-ordered vector into nodal-ordered vector 
    void createNodalOrderedCopy( std::shared_ptr<const AMP::LinearAlgebra::Vector> inVar, std::shared_ptr<AMP::LinearAlgebra::Vector> outNdl ) {
        for ( size_t i = 0; i < d_ndlInds.size(); i++ ) {
            outNdl->setValueByGlobalID( d_ndlInds[i], inVar->getValueByGlobalID( d_varInds[i] ) );
        }
    }
    // Copy contents of nodal-ordered vector into variable-ordered vector 
    void createVariableOrderedCopy( std::shared_ptr<const AMP::LinearAlgebra::Vector> inNdl, std::shared_ptr<AMP::LinearAlgebra::Vector> outVar ) {
        for ( size_t i = 0; i < d_ndlInds.size(); i++ ) {
            outVar->setValueByGlobalID( d_varInds[i], inNdl->getValueByGlobalID( d_ndlInds[i] ) );
        }
    }
    
private:
    // Create new d_data based on my RadDifOpJac's d_data
    void setData() {
        d_data = std::make_shared<BERadDifOpJacData>( d_RadDifOpJac->d_data, d_gamma );
    }

    void monolithicNodalJacGetRow( size_t row, std::vector<size_t> &cols, std::vector<double> &data ) const;

    // Helper function to fill matrix with CSR data
    void fillMatrixWithCSRData( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix ) const {

        // Place-holders for CSR data in each row
        std::vector<size_t> cols;
        std::vector<double> data;
        // Create wrapper around CSR data function that sets cols and data
        std::function<void( size_t dof )> setColsAndData = [&]( size_t dof ) { monolithicNodalJacGetRow( dof, cols, data ); }; 

        // Iterate through local rows in matrix
        auto multiDOF = d_RadDifOpJac->d_RadDifOp->d_multiDOFMan;
        size_t nrows = 1;
        for ( size_t dof = multiDOF->beginDOF(); dof != multiDOF->endDOF(); dof++ ) {
            setColsAndData( dof );
            matrix->setValuesByGlobalID<double>( nrows, cols.size(), &dof, cols.data(), data.data() );
        }
    }

    // Arrays that allow permutation between nodal and variable ordering
    std::vector<size_t> d_ndlInds;
    std::vector<size_t> d_varInds;
    // Set the above arrays
    void setLocalPermutationArrays() {
        auto N = d_RadDifOpJac->d_RadDifOp->d_multiDOFMan->numLocalDOF();
        d_ndlInds.resize( N );
        d_varInds.resize( N );
        for ( size_t i = 0; i < N; i++ ) {
            d_varInds[i] = i;
        } 
        // Get corresponding nodal ordering
        variableOrderingToNodalOrdering( N/2, d_varInds, d_ndlInds );
    }



    // Duplicated data from the block 2x2 matrix...
    std::shared_ptr<AMP::LinearAlgebra::Matrix> A;
    std::shared_ptr<AMP::LinearAlgebra::Vector> B;
    std::shared_ptr<AMP::LinearAlgebra::Vector> C;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> D;
    // [ d_E_BE  0    ]   [ diag(r_EE_BE) diag(r_ET_BE) ]
    // [ 0      d_T_BE] + [ diag(r_TE_BE) diag(r_TT_BE) ]
    // */
    // struct BERadDifOpJacData {
    //     std::shared_ptr<AMP::LinearAlgebra::Matrix> d_E_BE  = nullptr;
    //     std::shared_ptr<AMP::LinearAlgebra::Matrix> d_T_BE  = nullptr;
    //     std::shared_ptr<AMP::LinearAlgebra::Vector> r_EE_BE = nullptr;
    //     std::shared_ptr<AMP::LinearAlgebra::Vector> r_ET_BE = nullptr;
    //     std::shared_ptr<AMP::LinearAlgebra::Vector> r_TE_BE = nullptr;
    //     std::shared_ptr<AMP::LinearAlgebra::Vector> r_TT_BE = nullptr;

    std::shared_ptr<AMP::LinearAlgebra::Matrix> createMonolithicJac() {

        AMP_INSIST( this->getMesh()->getComm().getSize() == 1, "This function implemented serially only" );

        // I'm duplicating the BE data here because otherwise what I do is going to mess up the apply since I need to consolidate stuff on the diagonal...
        // Allocate space for matrices and vectors
        A = d_data->d_E_BE->clone();
        B = d_data->r_ET_BE->clone();
        C = d_data->r_TE_BE->clone();
        D = d_data->d_T_BE->clone();
        // Copy their values
        A->copy( d_data->d_E_BE );
        B->copyVector( d_data->r_ET_BE );
        C->copyVector( d_data->r_TE_BE );
        D->copy( d_data->d_T_BE );
        // Add extra info into diagonal of A and D
        auto DOFMan_E = d_data->d_E_BE->getRightDOFManager(  );
        for ( auto dof = DOFMan_E->beginDOF(); dof != DOFMan_E->endDOF(); dof++ ) {
            A->addValueByGlobalID( dof, dof, d_data->r_EE_BE->getValueByGlobalID( dof ) );
            D->addValueByGlobalID( dof, dof, d_data->r_TT_BE->getValueByGlobalID( dof ) );
        }

        // Place-holders for CSR data in each row
        std::vector<size_t> cols;
        std::vector<double> data;

        // This is a test (for small n) to check the index conversion all works properly inside the get row function. Looks OK for 1D with e.g., n = 8.
        # if 0
        auto multiDOF = d_RadDifOpJac->d_RadDifOp->d_multiDOFMan;
        for ( auto dof = multiDOF->beginDOF(); dof != multiDOF->endDOF(); dof++ ) {
            std::cout << "Nodal dof=" << dof << "\n";
            monolithicNodalJacGetRow( dof, cols, data );
            std::cout << "\tNodal ordering:\n";
            for ( auto i = 0; i < cols.size(); i++ ) {
                std::cout << "\t\tcol=" << cols[i] << ", data=" << data[i] << "\n";
            }
        }
        AMP_ERROR( "Halt...." );   
        #endif 

        auto inVec = d_RadDifOpJac->d_RadDifOp->createNodalInputVector();
        auto outVec = d_RadDifOpJac->d_RadDifOp->createNodalInputVector();
        
        // I don't know why, but the below is not working... 
        // auto inVec  = d_RadDifOpJac->d_RadDifOp->createInputVector( );
        // auto outVec = d_RadDifOpJac->d_RadDifOp->createInputVector( );

        // Create wrapper around CSR data function that sets cols and data
        std::function<void( size_t dof )> setColsAndData = [&]( size_t dof ) { monolithicNodalJacGetRow( dof, cols, data ); }; 

        // Create Lambda to return col inds from a given row ind
        auto getColumnIDs = [&]( size_t row ) { 
            setColsAndData( row );
            // std::cout << "getColumnIDs: row=" << row << "\n";
            // for ( auto i = 0; i < cols.size(); i++ ) {
            //     std::cout << "\t\tcol=" << cols[i] << ", data=" << data[i] << "\n";
            // }
            return cols;
        };

        // Create CSR matrix
        std::shared_ptr<AMP::LinearAlgebra::Matrix> A = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix", getColumnIDs );
        fillMatrixWithCSRData( A );
        
        // Finalize A
        A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        AMP::IO::AsciiWriter matWriter;
        matWriter.registerMatrix( A );
        matWriter.writeFile( "ANodalOut", 0 );

        //AMP_ERROR( "Halt...." );   

        return A;

    }
    
};



#endif