#ifndef RAD_DIF_FD_BE_WRAPPERS
#define RAD_DIF_FD_BE_WRAPPERS

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

#include "RadiationDiffusionFDDiscretization.h"

/** The classes in this file either are (or are associated with) wrapping spatial radiation 
 * diffusion operators as backward Euler operators, i.e., multiplying them by some time-step size 
 * and adding an identity. These operators here are closely related to 
 * AMP::TimeIntegrator::TimeOperator  
 */


/** Data structure for storing the 2x2 block matrix associated with the BERadDifOpPJac, I + 
 * gamma*hat{L}, where hat{L} is a RadDifOpPJac.
 * 
 * Specifically, the constructor here takes in the data structure used to store the 2x2 block matrix associated with hat{L}, i.e., 
 *  [ d_E 0   ] + [ diag(r_EE) diag(r_ET) ]
 *  [ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
 * and overwrites it to create
 *  ([I 0]         [ d_E 0   ])         [ diag(r_EE) diag(r_ET) ]
 *  ([0 I] + gamma*[ 0   d_T ]) + gamma*[ diag(r_TE) diag(r_TT) ]
 *      ==
 *  [ d_E_BE  0    ]   [ diag(r_EE_BE) diag(r_ET_BE) ]
 *  [ 0      d_T_BE] + [ diag(r_TE_BE) diag(r_TT_BE) ]
 * 
 * Note: We store the data in the above format for two reasons:
 * 1. It allows an operator-split preconditioner to be build, wherein the diffusion blocks must
 * contain the identity perturbation since AMG is applied to them, and it's easy enough to add an
 * identity perturbation on the fly to the reaction blocks when decoupled 2x2 solves are done there
 * 2. The modification that we make to the data means that our underlying RadDifOpPJac's apply will 
 * actually be an apply of a BERadDifOpPJac (we would have to write another apply routine if we 
 * also added an identity perturbation into the reaction block).
 */
struct BERadDifOpPJacData {
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_E_BE  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_T_BE  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_EE_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_ET_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TE_BE = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_TT_BE = nullptr;

    BERadDifOpPJacData( ) { };
    BERadDifOpPJacData( std::shared_ptr<RadDifOpPJacData> data, double gamma );
};



/** ----------------------------------------------------------------
 *      Class wrapping a RadDifOp as a backward Euler operator 
----------------------------------------------------------------- */
/** Implements the Operator I + gamma*L where L is a RadDifOp. 
 * This operator arises from the BDF discretization of the ODEs 
 *      u'(t) + L*u = s(t). 
 * The incoming OperatorParameters are used to create operator L. 
 */ 
class BERadDifOp : public AMP::Operator::Operator {

public:
    //! Time-step size (up to scaling by BDF constants)
    double                    d_gamma    = -1.0;
    //! The underlying radiation diffusion spatial operator
    std::shared_ptr<RadDifOp> d_RadDifOp = nullptr;
    

    //! Constructor
    BERadDifOp( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ );

    //! Destructor
    virtual ~BERadDifOp() {};

    //! Compute r <- (I + gamma*L)*u
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    //! Used to register this operator in a factory
    std::string type() const override { return "BERadDifOp"; }

    //! Returns RadDifOp's isValidVector
    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) override {
        return d_RadDifOp->isValidVector( ET );
    }

    //! Set the scaled time-step size of the operator
    void setGamma( AMP::Scalar gamma_ ); 


protected:

    /** \brief Returns a parameter object that may be used to reset the associated RadDifPJacOp
     * operator. 
     * \param[in] u_in The current nonlinear iterate. 
     * \details Note that the base class's getParameters() get's redirected to this function. 
     */
    std::shared_ptr<AMP::Operator::OperatorParameters> getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) override;

};


/** ----------------------------------------------------------------
 *      Class wrapping a RadDifOp as a backward Euler operator 
----------------------------------------------------------------- */
/** Implements the Operator I + gamma*hat{L} where hat{L} is a 
 * RadDifOpPJac, i.e., a Picard linearization of a radiation diffusion 
 * operator, RadDifOp  
 */
class BERadDifOpPJac : public AMP::Operator::LinearOperator {

public:

    //! Time-step size (up to scaling by BDF constants)
    double                              d_gamma    = -1.0;
    //! The underlying linearized radiation diffusion spatial operator
    std::shared_ptr<RadDifOpPJac>       d_RadDifOpPJac;
    //! Data structure for storing block 2x2 matrix
    std::shared_ptr<BERadDifOpPJacData> d_data = nullptr;


    //! Constructor
    BERadDifOpPJac( std::shared_ptr<AMP::Operator::OperatorParameters> params );

    //! Destructor
    virtual ~BERadDifOpPJac() {};

    //! Used by OperatorFactory to create a BERadDifOpPJac
    static std::unique_ptr<AMP::Operator::Operator> create( std::shared_ptr<AMP::Operator::OperatorParameters> params ) {  
        return std::make_unique<BERadDifOpPJac>( params ); };

    //! Compute r <- (I + gamma*hat{L})*u
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;
    
    //! Reset the operator based on the incoming parameters 
    void reset(std::shared_ptr<const AMP::Operator::OperatorParameters> params) override;

    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override; 

    
private:
    //! Create new d_data based on my RadDifOpPJac's d_data
    void setData() {
        d_data = std::make_shared<BERadDifOpPJacData>( d_RadDifOpPJac->d_data, d_gamma );
    }

    #if 0
    // Monolithic Jacobian in nodal ordering
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_JNodal;

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

    std::shared_ptr<AMP::LinearAlgebra::Vector> createNodalInputVector() {
        return d_RadDifOpPJac->d_RadDifOp->createNodalInputVector();
    }

    // Compute r <- (I + gamma*hat{L})*u
    // Serial only...
    void applyNodalToVariableVectors( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r )
    {
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpPJac::apply() " << std::endl;  
        AMP_INSIST( d_RadDifOpPJac, "RadDifOpPJac not set!" );

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
            AMP::pout << "BERadDifOpPJac::apply() " << std::endl;  
        AMP_INSIST( d_RadDifOpPJac, "RadDifOpPJac not set!" );

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
    #endif

    
    #if 0
    // Helper function to fill matrix with CSR data
    void fillMatrixWithCSRData( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix ) const {

        // Place-holders for CSR data in each row
        std::vector<size_t> cols;
        std::vector<double> data;
        // Create wrapper around CSR data function that sets cols and data
        std::function<void( size_t dof )> setColsAndData = [&]( size_t dof ) { monolithicNodalJacGetRow( dof, cols, data ); }; 

        // Iterate through local rows in matrix
        auto multiDOF = d_RadDifOpPJac->d_RadDifOp->d_multiDOFMan;
        size_t nrows = 1;
        for ( size_t dof = multiDOF->beginDOF(); dof != multiDOF->endDOF(); dof++ ) {
            setColsAndData( dof );
            matrix->setValuesByGlobalID<double>( nrows, cols.size(), &dof, cols.data(), data.data() );
        }
    }

    void monolithicNodalJacGetRow( size_t row, std::vector<size_t> &cols, std::vector<double> &data ) const;


    // Arrays that allow permutation between nodal and variable ordering
    std::vector<size_t> d_ndlInds;
    std::vector<size_t> d_varInds;
    // Set the above arrays
    void setLocalPermutationArrays() {
        auto N = d_RadDifOpPJac->d_RadDifOp->d_multiDOFMan->numLocalDOF();
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
    // struct BERadDifOpPJacData {
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
        auto multiDOF = d_RadDifOpPJac->d_RadDifOp->d_multiDOFMan;
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

        auto inVec = d_RadDifOpPJac->d_RadDifOp->createNodalInputVector();
        auto outVec = d_RadDifOpPJac->d_RadDifOp->createNodalInputVector();
        
        // I don't know why, but the below is not working... 
        // auto inVec  = d_RadDifOpPJac->d_RadDifOp->createInputVector( );
        // auto outVec = d_RadDifOpPJac->d_RadDifOp->createInputVector( );

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
    #endif
    
};


#if 0

    // MultiDofManager does have a getGlocalDof which converts DOFs from a sub-manager to their global values.---there is some "local" language here that's a big confusing, but I think that's probably the function I want. I.e, we use this to re-map column indices. Potentially the "getSubDOF" does the reverse of this, giving us a local DOF based on a global one. This is how we could map from the global row into the local rows of the matrix. 

    // std::vector<size_t> multiDOFManager::getGlobalDOF( const int manager,
    //                                                const std::vector<   size_t> &dofs )
    // std::vector<size_t> multiDOFManager::getSubDOF( const int manager,
    //                                                 const std::vector<size_t> &dofs )



/* Build a monolithic matrix representing the BERadDifOpJac. I.e., we build the matrix:
[A B]
[C D]  

But we need to do it in nodal ordering, where as the simplest thing is to get it in variable ordering.

[(E0,T0), (E1,T1), ..., (Ek, Tk)]

Given a row index in the nodal matrix, return the corresponding cols and data values. 
*/
void BERadDifOpJac::monolithicNodalJacGetRow( size_t rowNodal, std::vector<size_t> &cols, std::vector<double> &data ) const {

    // TODO: Need to sort out what I'm doing here in terms of resizing, pre-allocation, etc.
    // it's a bit hard because I have to extract CSR data from a matrix.
    std::vector<size_t> aux_dof1(1, 0);
    std::vector<size_t> aux_dof2(1, 0);
    std::vector<size_t> aux_cols;
    std::vector<double> aux_vals;
    cols = {};
    data = {};

    // Unpack DOF manager
    auto multiDOF = d_RadDifOpPJac->d_RadDifOp->d_multiDOFMan;

    // Number of local DOFs in each variable 
    auto nLocalDOFs = multiDOF->numLocalDOF() / 2;

    // We're given a nodal row index, we need to convert it to a variable row index so we can extract the correct variable rows of the submatrices (these live in variable index space)
    aux_dof1[0] = rowNodal;
    nodalOrderingToVariableOrdering( nLocalDOFs, aux_dof1, aux_dof2 );
    size_t rowIndVar = aux_dof2[0];

    // E variables live in even nodal rows
    // Strip out corresponding variable row of [A, B]. B is diagonal.
    if ( rowNodal % 2 == 0 ) {

        // Convert the global variable row index into an E variable index space
        // This is the row in [A, B] we need to extract
        aux_dof1[0] = rowIndVar;
        std::vector<size_t> rowIndEVar = multiDOF->getSubDOF( 0, aux_dof1 );
        
        // Get row of A in aux_cols and data.
        A->getRowByGlobalID( rowIndEVar[0], aux_cols, data );

        // Map cols of A from EVar index space into global variable index space
        cols = multiDOF->getGlobalDOF( 0, aux_cols );

        // Now concatenate single entry from diag matrix B
        // Get entry of B in current E var row
        data.push_back( B->getValueByGlobalID( rowIndEVar[0] ) ); 
        // Get index of T connection in global index space
        cols.push_back( multiDOF->getGlobalDOF( 1, rowIndEVar )[0] );

    // T variables live in odd nodal rows
    // Strip out corresponding variable row of [C, D]. C is diagonal
    } else {
        
        // Convert the global variable row index into a T variable index space
        // This is the row in [C, D] we need to extract
        aux_dof1[0] = rowIndVar;
        std::vector<size_t> rowIndTVar = multiDOF->getSubDOF( 1, aux_dof1 );
        
        // Get row of D in aux_cols and data.
        D->getRowByGlobalID( rowIndTVar[0], aux_cols, data );

        // Map cols of D from TVar index space into global variable index space
        cols = multiDOF->getGlobalDOF( 1, aux_cols );

        // Now concatenate single entry from diag matrix C
        // Get entry of C in current T var row
        data.push_back( C->getValueByGlobalID( rowIndTVar[0] ) ); 
        // Get index of E connection in global index space
        cols.push_back( multiDOF->getGlobalDOF( 0, rowIndTVar )[0] );

    }

    #if 0
    std::cout << "\tVariable ordering:\n"; 
    for ( auto i = 0; i < cols.size(); i++ ) {
        std::cout << "\t\tcol=" << cols[i] << ", data=" << data[i] << "\n";
    }
    #endif

    // Convert cols from variable ordering into nodal ordering.
    std::vector<size_t> cols_nodal;
    variableOrderingToNodalOrdering( nLocalDOFs, cols, cols_nodal );
    cols = cols_nodal;
}
#endif



#endif