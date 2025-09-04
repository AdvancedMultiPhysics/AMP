#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"

namespace AMP::Operator {

/** -------------------------------------------------------- *
 *  ----------- Implementation of RadDifOpPJacData --------- *
 *  -------------------------------------------------------- */
std::shared_ptr<AMP::LinearAlgebra::Matrix> RadDifOpPJacData::get_d_E() {
    d_dataMaybeOverwritten = true;
    return d_E;
}
std::shared_ptr<AMP::LinearAlgebra::Matrix> RadDifOpPJacData::get_d_T() {
    d_dataMaybeOverwritten = true;
    return d_T;
}
std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOpPJacData::get_r_EE() {
    d_dataMaybeOverwritten = true;
    return r_EE;
}
std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOpPJacData::get_r_ET() {
    d_dataMaybeOverwritten = true;
    return r_ET;
}
std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOpPJacData::get_r_TE() {
    d_dataMaybeOverwritten = true;
    return r_TE;
}
std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOpPJacData::get_r_TT() {
    d_dataMaybeOverwritten = true;
    return r_TT;
}


/** -------------------------------------------------------- *
 *  ------------ Implementation of RadDifOpPJac ------------ *
 *  -------------------------------------------------------- */
RadDifOpPJac::RadDifOpPJac(std::shared_ptr<const AMP::Operator::OperatorParameters> params_) : 
        AMP::Operator::LinearOperator( params_ ) {

    if ( d_iDebugPrintInfoLevel > 0 )
        AMP::pout << "RadDifOpPJac::RadDifOpPJac() " << std::endl;

    auto params = std::dynamic_pointer_cast<const RadDifOpPJacParameters>( params_ );
    AMP_INSIST( params, "params must be of type RadDifOpPJacParameters" );

    // Unpack parameters
    d_frozenVec = params->d_frozenSolution;
    d_RadDifOp  = params->d_RadDifOp;

    setData( );
};

void RadDifOpPJac::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> rET ) {

    PROFILE( "RadDifOpPJac::apply" );

    if ( d_iDebugPrintInfoLevel > 1 )
        AMP::pout << "RadDifOpPJac::apply() " << std::endl;

    // If the data has been overwritten by a BEOper, then this apply will be an apply of that operator. That's fine, but so as to not cause any confusion about the state of the data the BEOper must acknowledge before every apply that's indeed what they're trying to do.
    if ( d_data->d_dataMaybeOverwritten ) {
        AMP_INSIST( d_applyWithOverwrittenDataIsValid, "This apply is invalid because the data has been mutated by a BEOper; you must first set the flag 'applyWithOverwrittenBEOperDataIsValid' to true if you really want to do an apply" );
    }

    applyFromData( ET, rET );

    // Reset flag
    d_applyWithOverwrittenDataIsValid = false;
};


std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOpPJac::createInputVector() const {
    return d_RadDifOp->createInputVector();
};


/** Picard linearization of a RadDifOp. This LinearOperator has the following structure: 
 * [ d_E 0   ]   [ diag(r_EE) diag(r_ET) ]
 * [ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
 */
void RadDifOpPJac::applyFromData( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_, std::shared_ptr<AMP::LinearAlgebra::Vector> LET_  ) {

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BERadDifOpJac::applyFromData() " << std::endl;
    }

    // Check that if Jacobian data has been modified that the caller of this function really intends to do the apply with modified data.
    if ( d_data->d_dataMaybeOverwritten ) {
        AMP_INSIST( d_applyWithOverwrittenDataIsValid, "Jacobian data may have been modified. If you understand what that means for an apply with it, you must first call RadDifOpPJac::applyWithOverwrittenDataIsValid()" );
    }

    // Downcast input Vectors to MultiVectors 
    auto ET  = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( ET_ );
    auto LET = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( LET_ );
    AMP_INSIST( ET,  "ET downcast to MultiVector unsuccessful" );
    AMP_INSIST( LET, "LET downcast to MultiVector unsuccessful" );

    // Unpack scalar vectors from multivectors
    auto E  = ET->getVector(0);
    auto T  = ET->getVector(1);
    auto LE = LET->getVector(0);
    auto LT = LET->getVector(1);

    auto temp_ = this->createInputVector();
    auto temp  = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( temp_ );
    auto temp1 = temp->getVector(0);
    auto temp2 = temp->getVector(1);

    // LE = d_E * E + r_EE * E + r_ET * T
    // LT = d_T * T + r_TE * E + r_TT * T
    auto d_E  = d_data->d_E;
    auto d_T  = d_data->d_T;
    auto r_EE = d_data->r_EE;
    auto r_ET = d_data->r_ET;
    auto r_TE = d_data->r_TE;
    auto r_TT = d_data->r_TT;

    // add(x, y) = \(\mathit{this}_i = x_i + y_i\).
    // axpby(alpha, beta, x) = \(\mathit{this}_i = \alpha x_i + \beta \mathit{this}_i \). 
    // multiply(x, y) = \(\mathit{this}_i = x_i y_i\).

    // First component
    temp1->multiply( *r_EE, *E );
    temp2->multiply( *r_ET, *T );
    LE->add( *temp1, *temp2 );
    d_E->mult( E, temp1 );
    LE->axpby( 1.0, 1.0, *temp1 );

    // Second component
    temp1->multiply( *r_TE, *E );
    temp2->multiply( *r_TT, *T );
    LT->add( *temp1, *temp2 );
    d_T->mult( T, temp1 );
    LT->axpby( 1.0, 1.0, *temp1 );

    // Reset flag indicating apply with overwritten Jacobian data is invalid
    d_applyWithOverwrittenDataIsValid = false;
}


/** Populate our Jacobian data, d_data 
 * 
 * Data required to store the Picard linearization of a RadDifOp. The data is stored as two
 * matrices and 4 vectors.
 * The underlying LinearOperator has the following structure: 
 *      [ d_E 0   ]   [ diag(r_EE) diag(r_ET) ]
 *      [ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
 * 
 * Some optimizations that can be done to this function in the future regarding the diffusion 
 * matrices:
 * 1. The diffusion matrices are built even if they're multipled by a zero coefficient (k11 in the 
 * case of d_E and k21 in the case of d_T). Obviously there's no need to do this, but then one 
 * needs to take care: 
 *      i. when building the BERadDifOpPJacData because this just adds an identity perturbation to 
 * these matrices (and hence assumes they exist)
 *      ii. that the matrix doesn't exist during the apply (which has implications for the 
 * BERadDifOpPJacData since this uses the apply)
 *      iii. in the operator-split preconditioner since this assumes the matrices exist
 * 2. The sparsity pattern of both matrices is the same, and is independent of the linearization 
 * point. So, the sparsity pattern should be computed once, and then the values in the matrices 
 * reset when ever the linearization point changes (currently fresh matrices are built every time 
 * this function is called).
*/
void RadDifOpPJac::setData() {
    
    PROFILE( "RadDifOpPJac::setData" );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BERadDifOpJac::setData() " << std::endl; 
    }

    // --- Unpack frozen vector ---
    auto ET_vec = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( this->d_frozenVec );
    AMP_INSIST( ET_vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto E_vec = ET_vec->getVector(0);
    auto T_vec = ET_vec->getVector(1);

    // Initialize+allocate d_data if it's not been done already
    // We don't allocate matrices here since they're going to be created below. 
    if ( !d_data ) {
        d_data       = std::make_shared<RadDifOpPJacData>();
        d_data->r_EE = E_vec->clone();
        d_data->r_ET = E_vec->clone();
        d_data->r_TE = E_vec->clone();
        d_data->r_TT = E_vec->clone();
    // I believe the following should free the matrices d_E and d_T were pointing to
    } else {
        d_data->d_E = nullptr;
        d_data->d_T = nullptr;
    }

    // Set the reaction data members in d_data
    setDataReaction( T_vec );

    // --- Create matrices
    // Place-holders for CSR data in each row
    std::vector<size_t> cols_dE;
    std::vector<double> data_dE;
    std::vector<size_t> cols_dT;
    std::vector<double> data_dT;
    // Create wrapper around CSR data function that sets cols and data
    std::function<void( size_t dof )> setColsAndData_dE = [&]( size_t dof ) {
        getCSRDataDiffusionMatrix( 0, E_vec, T_vec, dof, cols_dE, data_dE );
    };
    std::function<void( size_t dof )> setColsAndData_dT = [&]( size_t dof ) {
        getCSRDataDiffusionMatrix( 1, E_vec, T_vec, dof, cols_dT, data_dT );
    };

    // Create Lambda to return col inds from a given row ind
    auto getColumnIDs_dE = [&]( size_t row ) {
        setColsAndData_dE( row );
        return cols_dE;
    };
    auto getColumnIDs_dT = [&]( size_t row ) {
        setColsAndData_dT( row );
        return cols_dT;
    };

    // Create CSR matrices
    auto inVec = d_data->r_EE, outVec = d_data->r_EE;
    auto d_E_mat = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix", getColumnIDs_dE );
    auto d_T_mat = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix", getColumnIDs_dT );
    // Fill matrices with data
    fillDiffusionMatrixWithData( 0, d_E_mat );
    fillDiffusionMatrixWithData( 0, d_T_mat );
    // Finalize matrix construction
    d_E_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_T_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    d_data->d_E = d_E_mat;
    d_data->d_T = d_T_mat;
}


void RadDifOpPJac::fillDiffusionMatrixWithData(
    size_t component,
    std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix ) 
{
    PROFILE( "RadDifOpPJac::fillDiffusionMatricesWithData" );

    // --- Unpack frozen vector ---
    auto ET_vec = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( this->d_frozenVec );
    AMP_INSIST( ET_vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto E_vec = ET_vec->getVector(0);
    auto T_vec = ET_vec->getVector(1);

    // Place-holders for CSR data in each row
    std::vector<size_t> cols;
    std::vector<double> data;
    // Create wrapper around CSR data function that sets cols and data
    std::function<void( size_t dof )> setColsAndData = [&]( size_t dof ) {
        getCSRDataDiffusionMatrix( component, E_vec, T_vec, dof, cols, data );
    };

    // Iterate through local rows in matrix
    size_t nrows = 1;
    for ( size_t dof = d_RadDifOp->d_scalarDOFMan->beginDOF(); dof != d_RadDifOp->d_scalarDOFMan->endDOF(); dof++ ) {
        setColsAndData( dof );
        matrix->setValuesByGlobalID<double>( nrows, cols.size(), &dof, cols.data(), data.data() );
    }
}


void RadDifOpPJac::setDataReaction( std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec ) {
    
    PROFILE( "RadDifOpPJac::setDataReaction" );

    // Unpack z
    auto zatom = d_RadDifOp->d_db->getWithDefault<double>( "zatom", 1.0 ); 

    // --- Iterate over all local rows ---
    // Placeholder for current grid index
    std::array<int, 3> ijk;

    // Iterate over local box
    for ( auto k = d_RadDifOp->d_localBox->first[2]; k <= d_RadDifOp->d_localBox->last[2]; k++ ) {
        ijk[2] = k;
        for ( auto j = d_RadDifOp->d_localBox->first[1]; j <= d_RadDifOp->d_localBox->last[1]; j++ ) {
            ijk[1] = j;
            for ( auto i = d_RadDifOp->d_localBox->first[0]; i <= d_RadDifOp->d_localBox->last[0]; i++ ) {
                ijk[0] = i;

                // Get T at current node (reaction stencil doesn't depend on E) 
                size_t dof  = d_RadDifOp->gridIndsToScalarDOF( ijk );
                double TLoc = T_vec->getValueByGlobalID( dof ); 

                // Compute semi-linear reaction coefficients at cell centers using T
                double REE, RET, RTE, RTT;
                d_RadDifOp->getSemiLinearReactionCoefficients( TLoc, zatom, REE, RET, RTE, RTT );

                // Insert values into the vectors
                d_data->r_EE->setValueByGlobalID<double>( dof, REE );
                d_data->r_ET->setValueByGlobalID<double>( dof, RET );
                d_data->r_TE->setValueByGlobalID<double>( dof, RTE );
                d_data->r_TT->setValueByGlobalID<double>( dof, RTT );
            } // Loop over i
        } // Loop over j
    } // Loop over k
    d_data->r_EE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_data->r_ET->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_data->r_TE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_data->r_TT->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


void RadDifOpPJac::getCSRDataDiffusionMatrix( 
                                size_t component,
                                std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                                std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                                size_t row,
                                std::vector<size_t> &cols,
                                std::vector<double> &data )
{
    PROFILE( "RadDifOpPJac::getCSRDataDiffusionMatrix" );

    AMP_INSIST( component == 0 || component == 1, "Invalid component" );

    // Unpack z
    auto zatom = d_RadDifOp->d_db->getWithDefault<double>( "zatom", 1.0 ); 

    // Flag indicating which boundary the 3-point stencil intersects with (unset by default)
    std::optional<RadDifOp::BoundarySide> boundaryIntersection;
    
    // Convert dof to a grid index
    std::array<int, 3> ijk = d_RadDifOp->scalarDOFToGridInds( row );
                
    /**
     * We sum over dimensions, resulting in columns ordered as
     * O, W, E, S, N, D, U
     * But note that in boundary-adjacent rows some of these connections can disappear
     * The number of DOFs per non-boundary row is 1 + 2*dim. 
     */

    // Initial resize here (and one at the end based on how many nnz were added)
    cols.resize( 1 + 2*d_RadDifOp->d_dim );
    data.resize( 1 + 2*d_RadDifOp->d_dim );
    // Initialize ORIGIN connection to zero since it's incremented in each dimension
    data[0] = 0.0;
    // Counter for actual number of off-diag connections set
    size_t nnzOffDiag = 0;

    // Loop over each dimension, adding in its contribution to the total stencil
    for ( size_t dim = 0; dim < d_RadDifOp->d_dim; dim++ ) {

        double h   = d_RadDifOp->d_h[dim];
        double rh2 = 1.0/(h*h); // Reciprocal h squared

        // Get WEST, ORIGIN, and EAST ET data for the given dimension
        d_RadDifOp->setLoc3Data(E_vec, T_vec, ijk, dim, boundaryIntersection); 

        // Compute diffusion coefficients for E or T
        double D_WO, D_OE;
        double dummy1, dummy2; // Dummy values for coefficients we don't set
        // Get energy coefficients
        if ( component == 0 ) {
            d_RadDifOp->getLocalFDDiffusionCoefficients( zatom, h, true, D_WO, D_OE, false, dummy1, dummy2 );
        // Get temperature coefficients
        } else {
            d_RadDifOp->getLocalFDDiffusionCoefficients( zatom, h, false, dummy1, dummy2, true, D_WO, D_OE );
        }
        
        /** Recall the stencil is applied in the following fashion:
         * dif_action += 
         *  [ -D_OE*(d_ELoc3[EAST]-d_ELoc3[ORIGIN]) + D_WO*(d_ELoc3[ORIGIN]-d_ELoc3[WEST]) ]*rh2
         *  == 
         * - [D_WO*rh2]*WEST + [(D_OE+D_WO)*rh2]*ORIGIN - [D_OE*rh2]*E
         */

        //--- Pack stencil and column data
        // Set diagonal connection
        data[0] += ( D_OE + D_WO )*rh2;
        cols[0]  = d_RadDifOp->d_dofsLoc3[ORIGIN]; 

        // Set off-diagonal connections
        // Case 1: Stencil does not intersect boundary
        if ( !boundaryIntersection.has_value() ) {
            // WEST connection
            data[nnzOffDiag+1] = -D_WO*rh2;
            cols[nnzOffDiag+1] = d_RadDifOp->d_dofsLoc3[WEST];
            // EAST connection
            data[nnzOffDiag+2] = -D_OE*rh2;
            cols[nnzOffDiag+2] = d_RadDifOp->d_dofsLoc3[EAST];
            nnzOffDiag += 2;

        /** Case 2: Stencil intersects a boundary. This means that one of the stencil connections 
         * is to a ghost. However, recall that the ghost value is a function of the ORIGIN point, 
         * as well as some other stuff. That is, the ghost point values satisfy
         *      Eg = alpha_E*Eint + blah_E,
         *      Tg = alpha_T*Tint + blah_T.
         * In the spirit of Picard linearization when we linearize these equations w.r.t. Eint and 
         * Tint we ignore the "blah_E" and "blah_T" (even though these terms are actually functions 
         * of Eint and Tint). As such, the stencil connection to the ghost needs to get added into 
         * the ORIGIN connection with weight alpha.
         */
        // Case 2a: Stencil intersects WEST boundary -> WEST neighbor is a ghost
        } else if ( boundaryIntersection.value() == RadDifOp::BoundarySide::WEST ) {
            boundaryIntersection.reset(); // Reset to have no value
            // Add WEST connection into diagonal with weight alpha
            size_t boundaryID = d_RadDifOp->getBoundaryIDFromDim(dim, RadDifOp::BoundarySide::WEST);
            double alpha = d_RadDifOp->PicardCorrectionCoefficient( component, boundaryID, D_WO );
            data[0] += alpha * -D_WO*rh2;

            // Add in EAST connection
            data[nnzOffDiag+1] = -D_OE*rh2;
            cols[nnzOffDiag+1] = d_RadDifOp->d_dofsLoc3[EAST];
            nnzOffDiag += 1;

        // Case 2b: Stencil intersects EAST boundary -> EAST neighbor is a ghost
        } else {
            boundaryIntersection.reset(); // Reset to have no value
            // Add in WEST connection
            data[nnzOffDiag+1] = -D_WO*rh2;
            cols[nnzOffDiag+1] = d_RadDifOp->d_dofsLoc3[WEST];
            nnzOffDiag += 1;

            // Add EAST connection into diagonal with weight alpha
            size_t boundaryID = d_RadDifOp->getBoundaryIDFromDim(dim, RadDifOp::BoundarySide::EAST);
            double alpha = d_RadDifOp->PicardCorrectionCoefficient( component, boundaryID, D_OE );
            data[0] += alpha * -D_OE*rh2;

        }
    } // Loop over dimension

    // Do a final resize to trim excess in the event we're in a boundary-adjacent row 
    cols.resize( 1 + nnzOffDiag );
    data.resize( 1 + nnzOffDiag );
}



/** Updates frozen value of current solution. I.e., when this is called, the d_frozenSolution 
 * vector in params is the current approximation to the outer nonlinear problem.
 * Also direct d_data to null, indicating it's out of date
 */
void RadDifOpPJac::reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ ) {

    if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "RadDifOpPJac::reset() " << std::endl;
    }

    AMP_ASSERT( params_ );
    // Downcast OperatorParameters to their derived class
    auto params = std::dynamic_pointer_cast<const RadDifOpPJacParameters>( params_ );
    AMP_INSIST( ( params.get() ) != nullptr, "NULL parameters" );
    AMP_INSIST( ( ( params->d_db ).get() ) != nullptr, "NULL database" );
    AMP_INSIST( ( ( params->d_frozenSolution ).get() ) != nullptr, "NULL frozen solution" );
    AMP_INSIST( params->d_RadDifOp != nullptr, "NULL RadDifOp" );

    // Update my data based on these parameters
    d_frozenVec = params->d_frozenSolution;
    d_RadDifOp  = params->d_RadDifOp;

    // Point data to null, that way it will be reconstructed from new data during the next apply call
    d_data = nullptr;
    setData( );
}; 


/** -------------------------------------------------------- *
 *  --------------- Implementation of RadDifOp ------------- *
 *  -------------------------------------------------------- */
// Constants scaling factors k_ij in the PDE are declared const, so must be set in the initializer list. Note that initializer list will fail if these constants do not exist (or their parent database does not)
RadDifOp::RadDifOp(std::shared_ptr<const AMP::Operator::OperatorParameters> params) : 
    AMP::Operator::Operator( params ),        
    d_k11( params->d_db->getScalar<double>( "k11" ) ),
    d_k12( params->d_db->getScalar<double>( "k12" ) ),
    d_k21( params->d_db->getScalar<double>( "k21" ) ),
    d_k22( params->d_db->getScalar<double>( "k22" ) )
{
    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "RadDifOp::RadDifOp() " << std::endl; 
    }

    // Unpack parameter database
    d_db = params->d_db;

    // Some basic input checking on the incoming database
    // PDE parameters
    auto model = d_db->getWithDefault<std::string>( "model", "" );
    AMP_INSIST( model == "linear" || model == "nonlinear", "model must be 'linear' or 'nonlinear'" );
    d_nonlinearModel = ( d_db->getScalar<std::string>( "model" ) == "nonlinear" );
    d_fluxLimited = d_db->getScalar<bool>( "fluxLimited" );

    // Set mesh-related data
    setAndCheckMeshData();

    // Specify default Robin return function for E
    std::function<double( size_t, AMP::Mesh::Point & )> wrapperE = [&]( size_t boundaryID, AMP::Mesh::Point & ) { return getBoundaryFunctionValueFromDBE( boundaryID ); };
    this->setBoundaryFunctionE( wrapperE );
    
    // Specify default Neumann return function for T
    std::function<double( size_t, AMP::Mesh::Point & )> wrapperT = [&]( size_t boundaryID,  AMP::Mesh::Point & ) { return getBoundaryFunctionValueFromDBT( boundaryID ); };
    this->setBoundaryFunctionT( wrapperT );
};


void RadDifOp::setAndCheckMeshData() {

    // Keep a pointer to my BoxMesh to save having to do this downcast repeatedly
    d_BoxMesh = std::dynamic_pointer_cast<AMP::Mesh::BoxMesh>( this->getMesh() );
    AMP_INSIST( d_BoxMesh, "Mesh must be a AMP::Mesh::BoxMesh" );

    d_dim = d_BoxMesh->getDim();
    AMP_INSIST( d_dim == 1 || d_dim == 2 || d_dim == 3,
                "Invalid dimension: dim=" + std::to_string( d_dim ) +
                    std::string( " !in {1,2,3}" ) );

    // Set the geometry type that gives us cell centered data
    if ( d_dim == 1 ) {
        CellCenteredGeom = AMP::Mesh::GeomType::Edge;
    } else if ( d_dim == 2 ) {
        CellCenteredGeom = AMP::Mesh::GeomType::Face;
    } else {
        CellCenteredGeom = AMP::Mesh::GeomType::Cell;
    }

    // Set DOFManagers
    setDOFManagers();
    AMP_INSIST( d_multiDOFMan, "Requires non-null multiDOF" );

    /** Now ensure that the geometry of the Mesh is a Box, which is what happens only when a "cube"
     * generator is used, as opposed to any other generator; see the function:
     *      std::shared_ptr<AMP::Geometry::Geometry>
     *          Geometry::buildGeometry( std::shared_ptr<const AMP::Database> db )
     *  I.e., check whether down casting the geometry to a Box passes. Note that Box is a
     * dimension-templated class, so we need to resolve each case separately.
    */
    auto meshGeom = d_BoxMesh->getGeometry();
    if ( d_dim == 1 ) {
        AMP_INSIST( std::dynamic_pointer_cast<AMP::Geometry::Box<1>>( meshGeom ),
                    "Mesh must be generated with 'cube generator'!" );
    } else if ( d_dim == 2 ) {
        AMP_INSIST( std::dynamic_pointer_cast<AMP::Geometry::Box<2>>( meshGeom ),
                    "Mesh must be generated with 'cube generator'!" );
    } else if ( d_dim == 3 ) {
        AMP_INSIST( std::dynamic_pointer_cast<AMP::Geometry::Box<3>>( meshGeom ),
                    "Mesh must be generated with 'cube generator'!" );
    }

    // Ensure boundaryIDs are 1,2, 3,4, 5,6 for dims 0, 1, 2
    std::vector<int> boundaryIDs = d_BoxMesh->getBoundaryIDs();
    for ( size_t dim = 0; dim < d_dim; dim++ ) {
        AMP_INSIST(boundaryIDs[2*dim]   == int(2*dim+1), "Invalid boundaryID");
        AMP_INSIST(boundaryIDs[2*dim+1] == int(2*dim+2), "Invalid boundaryID");  
    }

    // Discretization assumes Dirichlet boundaries in all directions
    for ( auto periodic : d_BoxMesh->periodic() ) {
        AMP_INSIST( periodic == false, "Mesh cannot be periodic in any direction" );
    }

    // Compute mesh spacings in each dimension
    // [ x_min x_max y_min y_max z_min z_max ]
    auto range = d_BoxMesh->getBoundingBox();
    // Set node boxes
    d_globalBox = std::make_shared<AMP::Mesh::BoxMesh::Box>( d_BoxMesh->getGlobalBox() );
    d_localBox  = std::make_shared<AMP::Mesh::BoxMesh::Box>( d_BoxMesh->getLocalBox() );

    // There are nk cells in dimension k, nk = d_globalBox.last[k] - d_globalBox.first[k]+1, such that the mesh spacing is hk = (xkMax - xkMin)/nk
    for ( size_t k = 0; k < d_dim; k++ ) {
        auto nk    = d_globalBox->last[k] - d_globalBox->first[k] + 1;
        auto xkMin = range[2 * k];
        auto xkMax = range[2 * k + 1];
        d_h.push_back( ( xkMax - xkMin ) / nk );
    }

    //printMeshNodes();
}

// oktodo: remove the following once i've verifies accuracy
void RadDifOp::printMeshNodes() {

    AMP::pout << "h=";
    for ( auto h : d_h ) {
        AMP::pout << h << ", ";
    }

    AMP::pout << "\n\nprintMeshNodes()\n";
    std::array<int, 3> ijk;

    // Iterate over local box
    for ( auto k = d_localBox->first[2]; k <= d_localBox->last[2]; k++ ) {
        ijk[2] = k;
        std::cout << "k=" << k << "\n";
        for ( auto j = d_localBox->first[1]; j <= d_localBox->last[1]; j++ ) {
            ijk[1] = j;
            std::cout << "  j=" << j << "\n";
            for ( auto i = d_localBox->first[0]; i <= d_localBox->last[0]; i++ ) {
                ijk[0] = i;
                std::cout << "    i=" << i << "\t";

                auto node = gridIndsToMeshElement( ijk );
                //auto coord = node.coord();
                auto point = node.centroid();
                std::cout << "x=" << point[0] << ", ";
                if (d_dim >= 2)
                    std::cout << "y=" << point[1] << ", ";
                if ( d_dim >= 3 )
                    std::cout << "z=" << point[2];
                std::cout << "\n";
            }
        }
    }

    this->getMesh()->getComm().barrier();
    AMP_ERROR( "halt here pls" );
}


std::vector<double> RadDifOp::getMeshSize() const { return d_h; }

std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOp::createInputVector() const {
    auto ET_var = std::make_shared<AMP::LinearAlgebra::Variable>( "ET" );
    auto ET_vec = AMP::LinearAlgebra::createVector<double>( this->d_multiDOFMan, ET_var );
    return ET_vec;
};


void RadDifOp::setBoundaryFunctionE( std::function<double( size_t, AMP::Mesh::Point &)> fn_ ) 
{ 
    d_robinFunctionE = fn_; 
};


void RadDifOp::setBoundaryFunctionT( std::function<double( size_t, AMP::Mesh::Point &)> fn_ ) 
{ 
    d_pseudoNeumannFunctionT = fn_; 
};


AMP::Mesh::MeshElement RadDifOp::gridIndsToMeshElement( int i, int j, int k ) const {
    AMP::Mesh::BoxMesh::MeshElementIndex ind( CellCenteredGeom, 0, i, j, k );
    return d_BoxMesh->getElement( ind );
};

AMP::Mesh::MeshElement RadDifOp::gridIndsToMeshElement( std::array<int,3> ijk ) const {
    return gridIndsToMeshElement( ijk[0], ijk[1], ijk[2] );
} 

size_t RadDifOp::gridIndsToScalarDOF( int i, int j, int k ) const {
    AMP::Mesh::BoxMesh::MeshElementIndex ind( CellCenteredGeom, 0, i, j, k );
    AMP::Mesh::MeshElementID id = d_BoxMesh->convert( ind );
    std::vector<size_t> dof;
    d_scalarDOFMan->getDOFs(id, dof);
    return dof[0];
};

size_t RadDifOp::gridIndsToScalarDOF( std::array<int,3> ijk ) const {
    return gridIndsToScalarDOF( ijk[0], ijk[1], ijk[2] );
}


std::shared_ptr<AMP::Operator::OperatorParameters> RadDifOp::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) {

    // Create a copy of d_db using Database copy constructor
    auto db = std::make_shared<AMP::Database>( *d_db );
    // OperatorParameters database must contain the "name" of the Jacobian operator that will be created from this
    db->putScalar( "name", "RadDifOpPJac");
    // Create derived OperatorParameters for Jacobian
    auto jacOpParams    = std::make_shared<RadDifOpPJacParameters>( db );
    // Set its mesh
    jacOpParams->d_Mesh = this->getMesh();

    jacOpParams->d_frozenSolution = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( u_in );
    jacOpParams->d_RadDifOp = this;

    return jacOpParams;
}


void RadDifOp::fillMultiVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> vec_, std::function<double( size_t component, AMP::Mesh::Point &point )> fun ) const {

    // Unpack multiVector
    auto vec = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( vec_ );
    AMP_INSIST( vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto vec0 = vec->getVector(0);
    auto vec1 = vec->getVector(1);

    double u0, u1;
    auto it = d_BoxMesh->getIterator( CellCenteredGeom ); // Mesh iterator
    for ( auto elem = it.begin(); elem != it.end(); elem++ ) {
        auto point = elem->centroid();
        u0 = fun( 0, point );
        u1 = fun( 1, point );
        std::vector<size_t> i;
        d_scalarDOFMan->getDOFs( elem->globalID(), i );
        vec0->setValueByGlobalID( i[0], u0 );
        vec1->setValueByGlobalID( i[0], u1 );
    }
    vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


std::array<int,3> RadDifOp::scalarDOFToGridInds( size_t dof ) const
{
    // Get ElementID
    AMP::Mesh::MeshElementID id = d_scalarDOFMan->getElementID( dof );
    // Convert ElementID into a MeshElementIndex
    AMP::Mesh::BoxMesh::MeshElementIndex ind = d_BoxMesh->convert( id );
    // Get grid index along each component direction
    return { ind.index( 0 ), ind.index( 1 ), ind.index( 2 ) };
}


/* Build and set d_multiDOFMan and d_scalarDOFMan */
void RadDifOp::setDOFManagers() {

    // Number of DOFs per mesh element (make 1, even though we have two variables. We'll create separate DOF managers for them)
    int myDOFsPerElement = 1; 
    int gcw   = 1; // Ghost-cell width; stencils are 3-point
    auto mesh = this->getMesh();
    auto comm = mesh->getComm();

    // E and T use the same DOFManager under the hood
    std::shared_ptr<AMP::Discretization::DOFManager> scalarDOFManager = AMP::Discretization::boxMeshDOFManager::create(mesh, CellCenteredGeom, gcw, myDOFsPerElement);
    auto T_DOFManager = scalarDOFManager;
    auto E_DOFManager = scalarDOFManager;

    // Create a multiDOFManager that wraps both DOF managers
    std::vector<std::shared_ptr<AMP::Discretization::DOFManager>> DOFManagersVec = { E_DOFManager, T_DOFManager };
    auto multiDOFManager = std::make_shared<AMP::Discretization::multiDOFManager>(comm, DOFManagersVec, mesh);

    d_scalarDOFMan = scalarDOFManager;
    d_multiDOFMan  = multiDOFManager;
}


bool RadDifOp::isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) {
    return ( ET->min() > 0.0 ); 
}


/**  The operator is computed as
 *  diffusion + reaction == 
 *      [ d_E 0   ][E]   [ diag(r_EE) diag(r_ET) ][E]
 *      [ 0   d_T ][T] + [ diag(r_TE) diag(r_TT) ][T]
 *  where: 
 * 1. d_E*E        = -k11 * div grad( DE*E ) 
 * 2. diag(r_EE)*E = -k12 * diag(REE) * E
 * 3. diag(r_ET)*T = -k12 * diag(RET) * T
 *      
 * 4. d_T*T        = -k21 * div grad( DT*T ) 
 * 5. diag(r_TE)*E = +k22 * diag(RTE) * E
 * 6. diag(r_TT)*T = +k22 * diag(RTT) * T
 * 
 * In the nonlinear model:
 *      REE = RTE = -sigma
 *      RET = RTT = +sigma*T^3
 * In the linear model:
 *      REE = RTE = -1
 *      RET = RTT = +1
 */
void RadDifOp::apply(std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_vec_,
            std::shared_ptr<AMP::LinearAlgebra::Vector> LET_vec_) 
{
    PROFILE( "RadDifOp::apply" );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "RadDifOp::apply() " << std::endl;
    }

    // --- Unpack parameters ---
    double zatom = this->d_db->getWithDefault<double>( "zatom", 1.0 );  
    
    // --- Unpack inputs ---
    // Downcast input Vectors to MultiVectors 
    auto ET_vec  = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( ET_vec_ );
    auto LET_vec = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( LET_vec_ );
    // Check to see if this is a multivector
    AMP_INSIST( ET_vec, "ET downcast to MultiVector unsuccessful" );
    AMP_INSIST( LET_vec, "LET downcast to MultiVector unsuccessful" );
    // Unpack vectors
    auto E_vec  = ET_vec->getVector(0);
    auto T_vec  = ET_vec->getVector(1);
    auto LE_vec = LET_vec->getVector(0);
    auto LT_vec = LET_vec->getVector(1);


    // --- Iterate over all local rows ---
    // Placeholder for current grid index
    std::array<int, 3> ijk;

    // Iterate over local box
    for ( auto k = d_localBox->first[2]; k <= d_localBox->last[2]; k++ ) {
        ijk[2] = k;
        for ( auto j = d_localBox->first[1]; j <= d_localBox->last[1]; j++ ) {
            ijk[1] = j;
            for ( auto i = d_localBox->first[0]; i <= d_localBox->last[0]; i++ ) {
                ijk[0] = i;

                /** --- Compute coefficients to apply stencil in a quasi-linear fashion */
                // Action of diffusion operator: Loop over each dimension, adding in its contribution to the total
                double dif_E_action = 0.0; // d_E * E
                double dif_T_action = 0.0; // d_T * T
                for ( size_t dim = 0; dim < d_dim; dim++ ) {

                    double h   = d_h[dim];
                    double rh2 = 1.0/(h*h); // Reciprocal h squared

                    // Get WEST, ORIGIN, EAST ET data for the given dimension
                    setLoc3Data(E_vec, T_vec, ijk, dim);        
                    // Get diffusion coefficients for both E and T
                    double Dr_WO, Dr_OE, DT_WO, DT_OE;
                    getLocalFDDiffusionCoefficients(zatom, h, true, Dr_WO, Dr_OE, true, DT_WO, DT_OE);
                    
                    // Apply diffusion operators in quasi-linear fashion
                    dif_E_action += ( -Dr_OE*(d_ELoc3[EAST]-d_ELoc3[ORIGIN]) 
                                        + Dr_WO*(d_ELoc3[ORIGIN]-d_ELoc3[WEST]) )*rh2;
                    dif_T_action += ( -DT_OE*(d_TLoc3[EAST]-d_TLoc3[ORIGIN]) 
                                        + DT_WO*(d_TLoc3[ORIGIN]-d_TLoc3[WEST]) )*rh2;
                }
                // Finished looping over dimensions for diffusion discretizations
                AMP_INSIST( d_TLoc3[ORIGIN] > 1e-14, "PDE coefficients ill-defined for T <= 0" );

                // Compute semi-linear reaction coefficients at cell centers using T value set in the last iteration of the above loop
                double REE, RET, RTE, RTT;
                getSemiLinearReactionCoefficients( d_TLoc3[ORIGIN], zatom, REE, RET, RTE, RTT );
                
                // Sum diffusion and reaction terms
                double LE = dif_E_action + ( REE*d_ELoc3[ORIGIN] + RET*d_TLoc3[ORIGIN] );
                double LT = dif_T_action + ( RTE*d_ELoc3[ORIGIN] + RTT*d_TLoc3[ORIGIN] );

                // Insert values into the vectors
                size_t dof_O = gridIndsToScalarDOF( ijk );
                LE_vec->setValueByGlobalID<double>( dof_O, LE );
                LT_vec->setValueByGlobalID<double>( dof_O, LT );

            } // Loop over i
        } // Loop over j
    } // Loop over k
    LET_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


void RadDifOp::getLocalFDDiffusionCoefficients(
    double zatom,
    double h,
    bool computeE,
    double &Dr_WO, 
    double &Dr_OE,
    bool computeT,
    double &DT_WO, 
    double &DT_OE) const
{

    // Compute temp at mid points             
    double T_WO = 0.5*( d_TLoc3[WEST]   + d_TLoc3[ORIGIN] ); // T_{i-1/2}
    double T_OE = 0.5*( d_TLoc3[ORIGIN] + d_TLoc3[EAST]   ); // T_{i+1/2}

    // Get diffusion coefficients at cell faces, i.e., mid points
    // Energy
    if ( computeE ) {
        Dr_WO = diffusionCoefficientE( T_WO, zatom );
        Dr_OE = diffusionCoefficientE( T_OE, zatom );
        // Limit the energy flux if need be, eq. (17)
        if ( d_fluxLimited ) {
            double DE_WO = Dr_WO/( 1.0 + Dr_WO*( abs( d_ELoc3[ORIGIN] - d_ELoc3[WEST] )/( h*0.5*(d_ELoc3[ORIGIN] + d_ELoc3[WEST]) ) ) );
            double DE_OE = Dr_OE/( 1.0 + Dr_OE*( abs( d_ELoc3[EAST] - d_ELoc3[ORIGIN] )/( h*0.5*(d_ELoc3[EAST] + d_ELoc3[ORIGIN]) ) ) );
            Dr_WO = DE_WO;
            Dr_OE = DE_OE;

            AMP_WARNING("flux limiting is not working properly...");
        }
        // Scale coefficients by constants in PDE
        scaleDiffusionCoefficientEBy_kij( Dr_WO );
        scaleDiffusionCoefficientEBy_kij( Dr_OE );
    }
    
    // Temperature
    if ( computeT ) {
        DT_WO = diffusionCoefficientT( T_WO );
        DT_OE = diffusionCoefficientT( T_OE );
        scaleDiffusionCoefficientTBy_kij( DT_WO );
        scaleDiffusionCoefficientTBy_kij( DT_OE );
    }
}


void RadDifOp::setGhostData( size_t boundaryID, AMP::Mesh::Point &boundaryPoint, double Eint, double Tint )
{
    // Get the Robin constants for the given boundaryID
    double ak, bk; 
    getLHSRobinConstantsFromDB(boundaryID, ak, bk);
    
    // Now get the corresponding Robin value
    double rk = d_robinFunctionE( boundaryID, boundaryPoint );
    // Get Neumann value
    double nk = d_pseudoNeumannFunctionT( boundaryID, boundaryPoint );
    
    // Spatial mesh size
    double hk = d_h[getDimFromBoundaryID(boundaryID)]; 

    // Solve for ghost values given these constants, storing results in member array
    ghostValuesSolve( ak, bk, rk, nk, hk, Eint, Tint, d_ghostData[0], d_ghostData[1] );
}


size_t RadDifOp::getDimFromBoundaryID(size_t boundaryID) const {
    AMP_INSIST( boundaryID >= 1 && boundaryID <= 6, "boundaryID not recognised" );
    return (boundaryID-1)/2; // Note the integer division
}


size_t RadDifOp::getBoundaryIDFromDim(size_t dim, BoundarySide side) const {
    if ( side == BoundarySide::WEST ) {
        return 2*dim + 1;
    } else if ( side == BoundarySide::EAST ) {
        return 2*dim + 2; 
    } else {
        AMP_ERROR( "Invalid side" );
    }
}


void RadDifOp::getLHSRobinConstantsFromDB(size_t boundaryID, double &ak, double &bk) const
{
    if ( boundaryID == 1 ) {
        ak = d_db->getScalar<double>( "a1" );
        bk = d_db->getScalar<double>( "b1" );
    } else if ( boundaryID == 2 ) {
        ak = d_db->getScalar<double>( "a2" );
        bk = d_db->getScalar<double>( "b2" );
    } else if ( boundaryID == 3 ) {
        ak = d_db->getScalar<double>( "a3" );
        bk = d_db->getScalar<double>( "b3" );
    } else if ( boundaryID == 4 ) {
        ak = d_db->getScalar<double>( "a4" );
        bk = d_db->getScalar<double>( "b4" );
    } else if ( boundaryID == 5 ) {
        ak = d_db->getScalar<double>( "a5" );
        bk = d_db->getScalar<double>( "b5" );
    } else if ( boundaryID == 6 ) {
        ak = d_db->getScalar<double>( "a6" );
        bk = d_db->getScalar<double>( "b6" );
    } else {
        AMP_ERROR( "Invalid boundaryID" );
    }
}

double RadDifOp::getBoundaryFunctionValueFromDBE( size_t boundaryID ) const {
    if ( boundaryID == 1 ) {
        return d_db->getScalar<double>( "r1" );
    } else if ( boundaryID == 2 ) {
        return d_db->getScalar<double>( "r2" );
    } else if ( boundaryID == 3 ) {
        return d_db->getScalar<double>( "r3" );
    } else if ( boundaryID == 4 ) {
        return d_db->getScalar<double>( "r4" );
    } else if ( boundaryID == 5 ) {
        return d_db->getScalar<double>( "r5" );
    } else if ( boundaryID == 6 ) {
        return d_db->getScalar<double>( "r6" );
    } else { 
        AMP_ERROR( "Invalid boundaryID" );
    }
}

double RadDifOp::getBoundaryFunctionValueFromDBT( size_t boundaryID ) const {
    if ( boundaryID == 1 ) {
        return d_db->getScalar<double>( "n1" );
    } else if ( boundaryID == 2 ) {
        return d_db->getScalar<double>( "n2" );
    } else if ( boundaryID == 3 ) {
        return d_db->getScalar<double>( "n3" );
    } else if ( boundaryID == 4 ) {
        return d_db->getScalar<double>( "n4" );
    } else if ( boundaryID == 5 ) {
        return d_db->getScalar<double>( "n5" );
    } else if ( boundaryID == 6 ) {
        return d_db->getScalar<double>( "n6" );
    } else { 
        AMP_ERROR( "Invalid boundaryID" );
    }
}


void RadDifOp::ghostValuesSolve( double a, double b, double r, double n, double h, double Eint, double Tint, double &Eg, double &Tg ) const {

    // Unpack parameters
    auto zatom = d_db->getWithDefault<double>( "zatom", 1.0 );

    // Solve for Tg
    Tg = ghostValueSolveT( n, h, Tint );

    // Compute energy diffusion coefficient on the boundary, i.e., the mid-point between Tg and Tint
    double T_midpoint = 0.5*( Tg + Tint );
    double D_E = diffusionCoefficientE( T_midpoint, zatom );
    // The below solve requires the finalized flux in the form of c = k11*D_E
    scaleDiffusionCoefficientEBy_kij( D_E );  
    auto c = D_E; 

    // Solve for Eg
    Eg = ghostValueSolveE( a, b, r, c, h, Eint );
}

double RadDifOp::ghostValueSolveT( double n, double h, double Tint ) const {
    double alpha = 1.0;
    double beta  = h*n; 
    double Tg = alpha*Tint + beta;
    return Tg;
}

double RadDifOp::ghostValueSolveE( double a, double b, double r, double c, double h, double Eint ) const {
    double alpha = (2*c*b - a*h)/(2*c*b + a*h);
    double beta  = 2*h*r/(2*c*b + a*h);
    double Eg = alpha*Eint + beta;
    return Eg;
}

double RadDifOp::PicardCorrectionCoefficient( size_t component, size_t boundaryID, double ck ) const {
    // Energy
    if ( component == 0 ) {
        // Get the Robin constants for the given boundaryID
        double ak, bk; 
        getLHSRobinConstantsFromDB(boundaryID, ak, bk);
        // Spatial mesh size
        double hk = d_h[getDimFromBoundaryID(boundaryID)];

        // The value we require coincides with r=0 and Eint=1 in "ghostValueSolveE"
        return ghostValueSolveE( ak, bk, 0.0, ck, hk, 1.0 );
    
    // Temperature; correction coefficient is 1.0, because we get Tg = 1.0*Tint + blah
    } else if ( component == 1 ) {

        // The value we require coincides with any n and h in "ghostValueSolveT"
        return ghostValueSolveT( 0.0, 0.0, 1.0 );

    } else {
        AMP_ERROR( "Invalid component" );
    }
}


double RadDifOp::diffusionCoefficientE( double T, double zatom ) const {
    if ( d_nonlinearModel ) {
        double sigma = std::pow( zatom/T, 3.0 ); 
        return 1.0/(3*sigma);
    } else {
        return 1.0;
    }
}

double RadDifOp::diffusionCoefficientT( double T ) const {
    if ( d_nonlinearModel ) {
        return pow( T, 2.5 );
    } else {
        return 1.0;
    }
}

void RadDifOp::getSemiLinearReactionCoefficients( double T, double zatom, double &REE, double &RET, double &RTE, double &RTT ) const {
    if ( d_nonlinearModel ) {
        double sigma = std::pow( zatom/T, 3.0 );
        REE = RTE = -sigma;
        RET = RTT = +sigma * pow( T, 3.0 );
    } else {
        REE = RTE = -1.0;
        RET = RTT = +1.0;
    }

    // Scale coefficients by constants in PDE
    scaleReactionCoefficientsBy_kij( REE, RET, RTE, RTT );
}

void RadDifOp::scaleReactionCoefficientsBy_kij( double &REE, double &RET, double &RTE, double &RTT ) const {
    REE *= -d_k12, RET *= -d_k12; // Energy equation
    RTE *=  d_k22, RTT *=  d_k22; // Temperature equation
}

void RadDifOp::scaleDiffusionCoefficientEBy_kij( double &D_E ) const {
    D_E *= d_k11; // Energy equation
}

void RadDifOp::scaleDiffusionCoefficientTBy_kij( double &D_T ) const {
    D_T *= d_k21; // Temperature equation
}



void RadDifOp::setLoc3Data( 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,  
    std::array<int, 3> &ijk,
    int dim) 
{

    // The current DOF
    size_t dof_O = gridIndsToScalarDOF( ijk );
    d_ELoc3[ORIGIN] = E_vec->getValueByGlobalID<double>( dof_O );
    d_TLoc3[ORIGIN] = T_vec->getValueByGlobalID<double>( dof_O );

    // Get WEST neighboring value
    // At WEST boundary, so WEST neighbor is a ghost
    if ( ijk[dim] == d_globalBox->first[dim] ) {
        auto boundaryID = getBoundaryIDFromDim(dim, BoundarySide::WEST);
        // Get point on the boundary
        auto boundaryPoint = gridIndsToMeshElement( ijk ).centroid(); // Point in cell center
        boundaryPoint[dim] -= d_h[dim]/2; // The boundary is h/2 WEST of the cell center
        // Set member array holding ghost data and unpack into local variables
        setGhostData( boundaryID, boundaryPoint, d_ELoc3[1], d_TLoc3[1] );
        d_ELoc3[WEST] = d_ghostData[0]; 
        d_TLoc3[WEST] = d_ghostData[1];

    // At interior DOF; WEST neighbor is an interior DOF
    } else {
        ijk[dim] -= 1;
        size_t dof_W = gridIndsToScalarDOF( ijk );
        ijk[dim] += 1; // reset to ORIGIN
        d_ELoc3[WEST] = E_vec->getValueByGlobalID( dof_W ); 
        d_TLoc3[WEST] = T_vec->getValueByGlobalID( dof_W );
    }

    // Get EAST neighboring value
    // At EAST boundary, so EAST neighbor is a ghost
    if ( ijk[dim] == d_globalBox->last[dim] ) {
        auto boundaryID = getBoundaryIDFromDim(dim, BoundarySide::EAST);
        // Get point on the boundary
        auto boundaryPoint = gridIndsToMeshElement( ijk ).centroid(); // Point in cell center
        boundaryPoint[dim] += d_h[dim]/2; // The boundary is h/2 EAST of the cell center
        // Set member array holding ghost data and unpack into local variables
        setGhostData( boundaryID, boundaryPoint, d_ELoc3[1], d_TLoc3[1] );
        d_ELoc3[EAST] = d_ghostData[0]; 
        d_TLoc3[EAST] = d_ghostData[1];

    // At interior DOF; EAST neighbor is an interior DOF
    } else {
        ijk[dim] += 1;
        size_t dof_E = gridIndsToScalarDOF( ijk );
        ijk[dim] -= 1; // reset to ORIGIN
        d_ELoc3[EAST]   = E_vec->getValueByGlobalID( dof_E ); 
        d_TLoc3[EAST]   = T_vec->getValueByGlobalID( dof_E );
    }
}



void RadDifOp::setLoc3Data( 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,  
    std::array<int, 3> &ijk, 
    int dim,
    std::optional<BoundarySide> &boundaryIntersection) 
{

    // The current DOF
    size_t dof_O = gridIndsToScalarDOF( ijk );
    d_ELoc3[ORIGIN] = E_vec->getValueByGlobalID<double>( dof_O );
    d_TLoc3[ORIGIN] = T_vec->getValueByGlobalID<double>( dof_O );
    //
    d_dofsLoc3[ORIGIN] = dof_O;

    // Get WEST neighboring value
    // At WEST boundary, so WEST neighbor is a ghost
    if ( ijk[dim] == d_globalBox->first[dim] ) {
        auto boundaryID = getBoundaryIDFromDim(dim, BoundarySide::WEST);
        // Get point on the boundary
        auto boundaryPoint = gridIndsToMeshElement( ijk ).centroid(); // Point in cell center
        boundaryPoint[dim] -= d_h[dim]/2; // The boundary is h/2 WEST of the cell center
        // Set member array holding ghost data and unpack into local variables
        setGhostData( boundaryID, boundaryPoint, d_ELoc3[1], d_TLoc3[1] );
        d_ELoc3[WEST] = d_ghostData[0]; 
        d_TLoc3[WEST] = d_ghostData[1];
        //
        // Flag that we're on the WEST boundary
        boundaryIntersection = BoundarySide::WEST;

    // At interior DOF; WEST neighbor is an interior DOF
    } else {
        ijk[dim] -= 1;
        size_t dof_W = gridIndsToScalarDOF( ijk );
        ijk[dim] += 1; // reset to ORIGIN
        d_ELoc3[WEST] = E_vec->getValueByGlobalID( dof_W ); 
        d_TLoc3[WEST] = T_vec->getValueByGlobalID( dof_W );
        //
        d_dofsLoc3[WEST] = dof_W;
    }

    // Get EAST neighboring value
    // At EAST boundary, so EAST neighbor is a ghost
    if ( ijk[dim] == d_globalBox->last[dim] ) {
        auto boundaryID = getBoundaryIDFromDim(dim, BoundarySide::EAST);
        // Get point on the boundary
        auto boundaryPoint = gridIndsToMeshElement( ijk ).centroid(); // Point in cell center
        boundaryPoint[dim] += d_h[dim]/2; // The boundary is h/2 EAST of the cell center
        // Set member array holding ghost data and unpack into local variables
        setGhostData( boundaryID, boundaryPoint, d_ELoc3[1], d_TLoc3[1] );
        d_ELoc3[EAST] = d_ghostData[0]; 
        d_TLoc3[EAST] = d_ghostData[1];
        //
        // Flag that we're on the EAST boundary
        boundaryIntersection = BoundarySide::EAST;

    // At interior DOF; EAST neighbor is an interior DOF
    } else {
        ijk[dim] += 1;
        size_t dof_E = gridIndsToScalarDOF( ijk );
        ijk[dim] -= 1; // reset to ORIGIN
        d_ELoc3[EAST]   = E_vec->getValueByGlobalID( dof_E ); 
        d_TLoc3[EAST]   = T_vec->getValueByGlobalID( dof_E );
        //
        d_dofsLoc3[EAST] = dof_E;
    }
}

} // namespace AMP::Operator