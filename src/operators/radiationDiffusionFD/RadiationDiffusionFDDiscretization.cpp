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
*/
void RadDifOpPJac::setData() {
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BERadDifOpJac::setData() " << std::endl; 
    }

    // --- Unpack frozen vector ---
    auto ET_vec = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( this->d_frozenVec );
    AMP_INSIST( ET_vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto E_vec = ET_vec->getVector(0);
    auto T_vec = ET_vec->getVector(1);

    // Initialize+allocate d_data if it's not been done already
    // We don't allocate matrices here since they're going to be created below. Possibly there's an option to do something smarter where we just reset the values because the sparsity structure of each matrix is always identical.
    if ( !d_data ) {
        d_data       = std::make_shared<RadDifOpPJacData>();
        d_data->r_EE = E_vec->clone();
        d_data->r_ET = E_vec->clone();
        d_data->r_TE = E_vec->clone();
        d_data->r_TT = E_vec->clone();
    // I believe the following should free the matrices d_E and d_T were pointing to
    } else {
        d_data->d_E.reset();
        d_data->d_T.reset();
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
                size_t dof = d_RadDifOp->gridIndsToScalarDOF( ijk );
                auto TLoc  = T_vec->getValueByGlobalID( dof ); 

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
    AMP_INSIST( component == 0 || component == 1, "Invalid component" );

    // Unpack z
    auto zatom = d_RadDifOp->d_db->getWithDefault<double>( "zatom", 1.0 ); 

    // Placeholder arrays for values used in 3-point stencils
    std::array<double, 3> ELoc3;
    std::array<double, 3> TLoc3;
    // Placeholder array for dofs we connect to in 3-point stencil
    std::array<size_t, 3> colsLoc3;
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
    // Initialize O connection to zero since it's incremented in each dimension
    data[0] = 0.0;
    // Counter for actual number of off-diag connections set
    size_t nnzOffDiag = 0;

    // Loop over each dimension, adding in its contribution to the total stencil
    for ( size_t dim = 0; dim < d_RadDifOp->d_dim; dim++ ) {

        double h   = d_RadDifOp->d_h[dim];
        double rh2 = 1.0/(h*h); // Reciprocal h squared

        // Get WEST, ORIGIN, and EAST ET data for the given dimension
        d_RadDifOp->unpackLocalStencilData(E_vec, T_vec, ijk, dim, ELoc3, TLoc3, colsLoc3, boundaryIntersection); 

        // Compute diffusion coefficients for E or T
        double D_WO, D_OE;
        double dummy1, dummy2; // Dummy values for coefficients we don't set
        // Get energy coefficients
        if ( component == 0 ) {
            d_RadDifOp->getLocalFDDiffusionCoefficients(ELoc3, TLoc3, zatom, h, true, D_WO, D_OE, false, dummy1, dummy2);
        // Get temperature coefficients
        } else {
            d_RadDifOp->getLocalFDDiffusionCoefficients(ELoc3, TLoc3, zatom, h, false, dummy1, dummy2, true, D_WO, D_OE);
        }
        
        /** Recall the stencil is applied in the following fashion:
         * dif_action += 
         *  [ -D_OE*(ELoc3[E]-ELoc3[O]) + D_WO*(ELoc3[O]-ELoc3[W]) ]*rh2
         *  == 
         * - [D_WO*rh2]*W + [(D_OE+D_WO)*rh2]*O - [D_OE*rh2]*E
         */

        //--- Pack stencil and column data
        // Set diagonal connection
        data[0] += ( D_OE + D_WO )*rh2;
        cols[0]  = colsLoc3[O]; 

        // Set off-diagonal connections
        // Case 1: Stencil does not intersect boundary
        if ( !boundaryIntersection.has_value() ) {
            // WEST connection
            data[nnzOffDiag+1] = -D_WO*rh2;
            cols[nnzOffDiag+1] = colsLoc3[W];
            // EAST connection
            data[nnzOffDiag+2] = -D_OE*rh2;
            cols[nnzOffDiag+2] = colsLoc3[E];
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
            // Add WEST connection into diagonal with weight alpha
            size_t boundaryID = d_RadDifOp->getBoundaryIDFromDim(dim, RadDifOp::BoundarySide::WEST);
            double alpha = d_RadDifOp->PicardCorrectionCoefficient( component, boundaryID, D_WO );
            data[0] += alpha * -D_WO*rh2;

            // Add in EAST connection
            data[nnzOffDiag+1] = -D_OE*rh2;
            cols[nnzOffDiag+1] = colsLoc3[E];
            nnzOffDiag += 1;

        // Case 2b: Stencil intersects EAST boundary -> EAST neighbor is a ghost
        } else {
            // Add in WEST connection
            data[nnzOffDiag+1] = -D_WO*rh2;
            cols[nnzOffDiag+1] = colsLoc3[W];
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

    if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "RadDifOpPJac::reset() " << std::endl;

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
    // Mesh database

    // Set DOFManagers
    this->setDOFManagers();
    AMP_INSIST( d_multiDOFMan, "Requires non-null multiDOF" );

    // Keep a pointer to my BoxMesh to save having to do this downcast repeatedly
    d_BoxMesh = std::dynamic_pointer_cast<AMP::Mesh::BoxMesh>( this->getMesh() );
    AMP_INSIST( d_BoxMesh, "Mesh must be a AMP::Mesh::BoxMesh" );

    d_dim = d_BoxMesh->getDim();
    AMP_INSIST( d_dim == 1 || d_dim == 2 || d_dim == 3,
                "Invalid dimension: dim=" + std::to_string( d_dim ) +
                    std::string( " !in {1,2,3}" ) );

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
    d_globalBox = std::make_shared<AMP::Mesh::BoxMesh::Box>( getGlobalNodeBox() );
    d_localBox  = std::make_shared<AMP::Mesh::BoxMesh::Box>( getLocalNodeBox() );

    // There are nk+1 grid points in dimension k, nk = d_globalBox.last[k] - d_globalBox.first[k], such
    // that the mesh spacing is hk = (xkMax - xkMin)/nk
    for ( size_t k = 0; k < d_dim; k++ ) {
        auto nk    = d_globalBox->last[k] - d_globalBox->first[k];
        auto xkMin = range[2 * k];
        auto xkMax = range[2 * k + 1];
        d_h.push_back( ( xkMax - xkMin ) / nk );
    }


    // Specify default Robin return function for E
    std::function<double( int, double, double, AMP::Mesh::MeshElement & )> wrapperE = [&]( size_t boundaryID, double, double, AMP::Mesh::MeshElement & ) { return robinFunctionEFromDB( boundaryID ); };
    this->setRobinFunctionE( wrapperE );
    // Specify default Neumann return function for T
    std::function<double( int, AMP::Mesh::MeshElement & )> wrapperT = [&]( size_t boundaryID,  AMP::Mesh::MeshElement & ) { return pseudoNeumannFunctionTFromDB( boundaryID ); };
    this->setPseudoNeumannFunctionT( wrapperT );
};

std::vector<double> RadDifOp::getMeshSize() const { return d_h; }


AMP::Mesh::BoxMesh::Box RadDifOp::getGlobalNodeBox() const
{
    auto global = d_BoxMesh->getGlobalBox();
    for ( int d = 0; d < 3; d++ ) {
        // An empty box in dimension d has a last index of 0; we should preserve that behavior
        if ( global.last[d] > 0 ) { 
            global.last[d]++;
        }
    }
    return global;
}

AMP::Mesh::BoxMesh::Box RadDifOp::getLocalNodeBox() const {
    auto local  = d_BoxMesh->getGlobalBox();
    auto global = d_BoxMesh->getGlobalBox();
    for ( int d = 0; d < 3; d++ ) {
        if ( local.last[d] == global.last[d] ) {
            // An empty box in dimension d has a last index of 0; we should preserve that behavior
            if ( local.last[d] > 0 ) { 
                local.last[d]++;
            }
        }
    }
    return local;
}

std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOp::createInputVector() const {
    auto ET_var = std::make_shared<AMP::LinearAlgebra::Variable>( "ET" );
    auto ET_vec = AMP::LinearAlgebra::createVector<double>( this->d_multiDOFMan, ET_var );
    return ET_vec;
};

void RadDifOp::setPseudoNeumannFunctionT( std::function<double(int, AMP::Mesh::MeshElement &)> fn_ ) { d_pseudoNeumannFunctionT = fn_; };
void RadDifOp::setRobinFunctionE( std::function<double(int, double, double, AMP::Mesh::MeshElement &)> fn_ ) { d_robinFunctionE = fn_; };

AMP::Mesh::MeshElement RadDifOp::gridIndsToMeshElement( int i, int j, int k ) {
    AMP::Mesh::BoxMesh::MeshElementIndex ind( VertexGeom, 0, i, j, k );
    return d_BoxMesh->getElement( ind );
};

size_t RadDifOp::gridIndsToScalarDOF( int i, int j, int k ) {
    AMP::Mesh::BoxMesh::MeshElementIndex ind( VertexGeom, 0, i, j, k );
    AMP::Mesh::MeshElementID id = d_BoxMesh->convert( ind );
    std::vector<size_t> dof;
    d_scalarDOFMan->getDOFs(id, dof);
    return dof[0];
};

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


void RadDifOp::fillMultiVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> vec_, std::function<double( int, AMP::Mesh::MeshElement & )> fun ) {

    // Unpack multiVector
    auto vec = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( vec_ );
    AMP_INSIST( vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto vec0 = vec->getVector(0);
    auto vec1 = vec->getVector(1);

    double u0, u1;
    auto it = d_BoxMesh->getIterator( VertexGeom ); // Mesh iterator
    for ( auto elem = it.begin(); elem != it.end(); elem++ ) {
        u0 = fun( 0, *elem );
        u1 = fun( 1, *elem );
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

    // Specify mesh is to use cell-based geometry
    AMP::Mesh::GeomType myGeomType = VertexGeom;
    // Number of DOFs per mesh element (make 1, even though we have two variables. We'll create separate DOF managers for them)
    int myDOFsPerElement = 1; 
    int gcw   = 1; // Ghost-cell width. 
    auto mesh = this->getMesh();
    auto comm = mesh->getComm();

    // E and T use the same DOFManager under the hood
    std::shared_ptr<AMP::Discretization::DOFManager> scalarDOFManager = AMP::Discretization::boxMeshDOFManager::create(mesh, myGeomType, gcw, myDOFsPerElement);
    auto T_DOFManager = scalarDOFManager;
    auto E_DOFManager = scalarDOFManager;

    // Create a multiDOFManager that wraps both DOF managers
    std::vector<std::shared_ptr<AMP::Discretization::DOFManager>> DOFManagersVec = { E_DOFManager, T_DOFManager };
    auto multiDOFManager = std::make_shared<AMP::Discretization::multiDOFManager>(comm, DOFManagersVec, mesh);

    d_scalarDOFMan = scalarDOFManager;
    d_multiDOFMan  = multiDOFManager;

    #if 1
    comm.barrier();
    // This demonstrates how DOFs are organized on the mesh by multiDOFManager 
    // Iterate through the mesh, and pull out DOFs associated with each mesh element from the multiDOF
    auto iter = mesh->getIterator( VertexGeom, 0 );
    AMP::pout << "multiDOF E and T global indices" << std::endl;
    for (auto elem = iter.begin(); elem != iter.end(); elem++ ) {
        auto id = elem->globalID();
        std::vector<size_t> dofs;
        multiDOFManager->getDOFs(id, dofs);
        for (auto dof : dofs) {
            std::cout << dof << " ";
        }
        std::cout << std::endl;
    }

    // Iterate through the mesh, and pull out DOFs associated with each mesh element from their respective scalarDOFs 
    comm.barrier();
    AMP::pout << "DOF E and T global indices" << std::endl;
    for (auto elem = iter.begin(); elem != iter.end(); elem++ ) {
        auto id = elem->globalID();
        std::vector<size_t> dofs_E;
        std::vector<size_t> dofs_T;
        scalarDOFManager->getDOFs(id, dofs_E);
        scalarDOFManager->getDOFs(id, dofs_T);
        for (auto dof : dofs_E) {
            std::cout << dof << " ";
        }
        for (auto dof : dofs_T) {
            std::cout << dof << " ";
        }
        std::cout << std::endl;
    }
    #endif


    
    #if 0
    // This is how we get nodal ordering rather than variable ordering
    d_nodalDOFMan = AMP::Discretization::boxMeshDOFManager::create(mesh, myGeomType, gcw, 2);
    AMP::pout << "2 DOFs per element: DOF E and T global indices" << std::endl;
    int count = 0;
    for (auto elem = iter.begin(); elem != iter.end(); elem++ ) {
        auto id = elem->globalID();
        std::vector<size_t> dofs_ET;
        DOFMan->getDOFs(id, dofs_ET);
        for (auto dof : dofs_ET) {
            std::cout << dof << " ";
        } 
        std::cout << std::endl;
    }
    #endif
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
    // Placeholder arrays for values used in 3-point stencils
    std::array<double, 3> ELoc3;
    std::array<double, 3> TLoc3;

    // Iterate over local box
    for ( auto k = d_localBox->first[2]; k <= d_localBox->last[2]; k++ ) {
        ijk[2] = k;
        for ( auto j = d_localBox->first[1]; j <= d_localBox->last[1]; j++ ) {
            ijk[1] = j;
            for ( auto i = d_localBox->first[0]; i <= d_localBox->last[0]; i++ ) {
                ijk[0] = i;

                /** --- Compute coefficients to apply stencil in a semi-linear fashion */
                // Action of diffusion operator: Loop over each dimension, adding in its contribution to the total
                double dif_E_action = 0.0; // d_E * E
                double dif_T_action = 0.0; // d_T * T
                for ( size_t dim = 0; dim < d_dim; dim++ ) {

                    double h   = d_h[dim];
                    double rh2 = 1.0/(h*h); // Reciprocal h squared

                    // Get WEST, ORIGIN, EAST ET data for the given dimension
                    unpackLocalStencilData(E_vec, T_vec, ijk, dim, ELoc3, TLoc3);        
                    // Get diffusion coefficients for both E and T
                    double Dr_WO, Dr_OE, DT_WO, DT_OE;
                    getLocalFDDiffusionCoefficients(ELoc3, TLoc3, zatom, h, true, Dr_WO, Dr_OE, true, DT_WO, DT_OE);
                    
                    // Apply diffusion operators in quasi-linear fashion
                    dif_E_action += ( -Dr_OE*(ELoc3[E]-ELoc3[O]) + Dr_WO*(ELoc3[O]-ELoc3[W]) )*rh2;
                    dif_T_action += ( -DT_OE*(TLoc3[E]-TLoc3[O]) + DT_WO*(TLoc3[O]-TLoc3[W]) )*rh2;
                }
                // Finished looping over dimensions for diffusion discretizations
                AMP_INSIST( TLoc3[O] > 1e-14, "PDE coefficients ill-defined for T <= 0" );

                // Compute semi-linear reaction coefficients at cell centers using T value set in the last iteration of the above loop
                double REE, RET, RTE, RTT;
                getSemiLinearReactionCoefficients( TLoc3[O], zatom, REE, RET, RTE, RTT );
                
                // Sum diffusion and reaction terms
                double LE = dif_E_action + ( REE*ELoc3[O] + RET*TLoc3[O] );
                double LT = dif_T_action + ( RTE*ELoc3[O] + RTT*TLoc3[O] );

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
    const std::array<double,3> &ELoc3,
    const std::array<double,3> &TLoc3,
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
    double T_WO = 0.5*( TLoc3[W] + TLoc3[O] ); // T_{i-1/2}
    double T_OE = 0.5*( TLoc3[O] + TLoc3[E] ); // T_{i+1/2}

    // Get diffusion coefficients at cell faces, i.e., mid points
    // Energy
    if ( computeE ) {
        Dr_WO = diffusionCoefficientE( T_WO, zatom );
        Dr_OE = diffusionCoefficientE( T_OE, zatom );
        // Limit the energy flux if need be, eq. (17)
        if ( d_fluxLimited ) {
            double DE_WO = Dr_WO/( 1.0 + Dr_WO*( abs( ELoc3[O] - ELoc3[W] )/( h*0.5*(ELoc3[O] + ELoc3[W]) ) ) );
            double DE_OE = Dr_OE/( 1.0 + Dr_OE*( abs( ELoc3[E] - ELoc3[O] )/( h*0.5*(ELoc3[E] + ELoc3[O]) ) ) );
            Dr_WO = DE_WO;
            Dr_OE = DE_OE;
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


void RadDifOp::setGhostData( size_t boundaryID, AMP::Mesh::MeshElement &node, double Eint, double Tint )
{

    // Get the Robin constants for the given boundaryID
    double ak, bk; 
    getLHSRobinConstantsFromDB(boundaryID, ak, bk);
    
    // Now get the corresponding Robin value
    double rk = d_robinFunctionE( boundaryID, ak, bk, node );
    // Get Neumann value
    double nk = d_pseudoNeumannFunctionT( boundaryID, node );
    
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


void RadDifOp::getLHSRobinConstantsFromDB(size_t boundaryID, double &ak, double &bk) 
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

double RadDifOp::robinFunctionEFromDB( size_t boundaryID ){
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

double RadDifOp::pseudoNeumannFunctionTFromDB( size_t boundaryID ) {
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


void RadDifOp::ghostValuesSolve( double a, double b, double r, double n, double h, double Eint, double Tint, double &Eg, double &Tg ) {

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


double RadDifOp::ghostValueSolveT( double n, double h, double Tint ) {
    double alpha = 1.0;
    double beta  = h*n; 
    double Tg = alpha*Tint + beta;
    return Tg;
}

double RadDifOp::ghostValueSolveE( double a, double b, double r, double c, double h, double Eint ) {

    double alpha = (2*c*b - a*h)/(2*c*b + a*h);
    double beta  = 2*h*r/(2*c*b + a*h);
    double Eg = alpha*Eint + beta;
    return Eg;
    //return (2*Eint*c*b - Eint*a*h + 2*h*r)/(2*c*b + a*h);
}



double RadDifOp::PicardCorrectionCoefficient( size_t component, size_t boundaryID, double ck ) {
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




void RadDifOp::unpackLocalStencilData( 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,  
    std::array<int, 3> &ijk, // is modified locally, but returned in same state
    int dim,
    std::array<double, 3> &ELoc3, 
    std::array<double, 3> &TLoc3) 
{

    // The current DOF
    size_t dof_O = gridIndsToScalarDOF( ijk );
    AMP::pout << "++ dof O=" << dof_O << "\n";
    ELoc3[1] = E_vec->getValueByGlobalID<double>( dof_O );
    TLoc3[1] = T_vec->getValueByGlobalID<double>( dof_O );

    // Get WEST (or WEST) neighboring value
    // At WEST (or WEST) boundary, so WEST (or WEST) neighbor is a ghost
    if ( ijk[dim] == d_globalBox->first[dim] ) {
        auto node = gridIndsToMeshElement( ijk );
        // Set member array holding ghost data and unpack into local variables
        setGhostData( getBoundaryIDFromDim(dim, BoundarySide::WEST), node, ELoc3[1], TLoc3[1] );
        ELoc3[0] = d_ghostData[0]; 
        TLoc3[0] = d_ghostData[1];

    // At interior DOF; WEST (or WEST) neighbor is an interior DOF
    } else {
        ijk[dim] -= 1;
        size_t dof_W = gridIndsToScalarDOF( ijk );
        AMP::pout << "dof W=" << dof_W << "\n";
        ijk[dim] += 1; // reset to O
        ELoc3[0] = E_vec->getValueByGlobalID( dof_W ); 
        TLoc3[0] = T_vec->getValueByGlobalID( dof_W );
    }

    // Get EAST (or EAST) neighboring value
    // At EAST (or EAST) boundary, so EAST (or EAST) neighbor is a ghost
    if ( ijk[dim] == d_globalBox->last[dim] ) {
        auto node  = gridIndsToMeshElement( ijk );
        // Set member array holding ghost data and unpack into local variables
        setGhostData( getBoundaryIDFromDim(dim, BoundarySide::EAST), node, ELoc3[1], TLoc3[1] );
        ELoc3[2] = d_ghostData[0]; 
        TLoc3[2] = d_ghostData[1];

    // At interior DOF; EAST (or EAST) neighbor is an interior DOF
    } else {
        ijk[dim] += 1;
        size_t dof_E = gridIndsToScalarDOF( ijk );
        AMP::pout << "dof E=" << dof_E << "\n";
        ijk[dim] -= 1; // reset to O
        ELoc3[2]   = E_vec->getValueByGlobalID( dof_E ); 
        TLoc3[2]   = T_vec->getValueByGlobalID( dof_E );
    }
}



void RadDifOp::unpackLocalStencilData( 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, 
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,  
    std::array<int, 3> &ijk, // is modified locally, but returned in same state
    int dim,
    std::array<double, 3> &ELoc3, 
    std::array<double, 3> &TLoc3, 
    std::array<size_t, 3> &dofs,
    std::optional<BoundarySide> &boundaryIntersection) 
{

    // The current DOF
    size_t dof_O = gridIndsToScalarDOF( ijk );
    ELoc3[1] = E_vec->getValueByGlobalID<double>( dof_O );
    TLoc3[1] = T_vec->getValueByGlobalID<double>( dof_O );
    //
    dofs[1] = dof_O;

    // Get WEST (or WEST) neighboring value
    // At WEST (or WEST) boundary, so WEST (or WEST) neighbor is a ghost
    if ( ijk[dim] == d_globalBox->first[dim] ) {
        auto node = gridIndsToMeshElement( ijk );
        // Set member array holding ghost data and unpack into local variables
        setGhostData( getBoundaryIDFromDim(dim, BoundarySide::WEST), node, ELoc3[1], TLoc3[1] );
        ELoc3[0] = d_ghostData[0]; 
        TLoc3[0] = d_ghostData[1];
        //
        // Flag that we're on the WEST boundary
        boundaryIntersection = BoundarySide::WEST;

    // At interior DOF; WEST (or WEST) neighbor is an interior DOF
    } else {
        ijk[dim] -= 1;
        size_t dof_W = gridIndsToScalarDOF( ijk );
        ijk[dim] += 1; // reset to O
        ELoc3[0] = E_vec->getValueByGlobalID( dof_W ); 
        TLoc3[0] = T_vec->getValueByGlobalID( dof_W );
        //
        dofs[0] = dof_W;
    }

    // Get EAST (or EAST) neighboring value
    // At EAST (or EAST) boundary, so EAST (or EAST) neighbor is a ghost
    if ( ijk[dim] == d_globalBox->last[dim] ) {
        auto node  = gridIndsToMeshElement( ijk );
        // Set member array holding ghost data and unpack into local variables
        setGhostData( getBoundaryIDFromDim(dim, BoundarySide::EAST), node, ELoc3[1], TLoc3[1] );
        ELoc3[2] = d_ghostData[0]; 
        TLoc3[2] = d_ghostData[1];
        //
        // Flag that we're on the EAST boundary
        boundaryIntersection = BoundarySide::EAST;

    // At interior DOF; EAST (or EAST) neighbor is an interior DOF
    } else {
        ijk[dim] += 1;
        size_t dof_E = gridIndsToScalarDOF( ijk );
        ijk[dim] -= 1; // reset to O
        ELoc3[2]   = E_vec->getValueByGlobalID( dof_E ); 
        TLoc3[2]   = T_vec->getValueByGlobalID( dof_E );
        //
        dofs[2] = dof_E;
    }
}

} // namespace AMP::Operator







#if 0
void RadDifOp::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET, std::shared_ptr<AMP::LinearAlgebra::Vector> LET) {

    if ( d_iDebugPrintInfoLevel > 1 )
        AMP::pout << "RadDifOp::apply() " << std::endl;

    auto meshDim = d_BoxMesh->getDim();
    if ( meshDim == 1 ) {
        apply1D(ET, LET);
    } else {
        apply2D(ET, LET);
    }

    //applyDimAg(ET, LET);
};



void RadDifOp::apply1D(
            std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_vec_,
            std::shared_ptr<      AMP::LinearAlgebra::Vector> LET_vec_) 
{
    // --- Unpack parameters ---
    std::shared_ptr<const AMP::Database> PDE_db  = this->d_db->getDatabase( "PDE" );
    std::shared_ptr<const AMP::Database> mesh_db = this->d_db->getDatabase( "mesh" );
    double h   = mesh_db->getScalar<double>( "h" );
    double rh2 = 1.0/(h*h); // Reciprocal h squared
    double k11 = PDE_db->getScalar<double>( "k11" );
    double k12 = PDE_db->getScalar<double>( "k12" );
    double k21 = PDE_db->getScalar<double>( "k21" );
    double k22 = PDE_db->getScalar<double>( "k22" );
    double z   = PDE_db->getWithDefault<double>( "zatom", 1.0 );  

    bool nonlinearModel = ( PDE_db->getScalar<std::string>( "model" ) == "nonlinear" );
    bool fluxLimited    = PDE_db->getScalar<bool>( "fluxLimited" );


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
    // Get local grid index box w/ zero ghosts
    auto d_localBox  = getLocalNodeBox( d_BoxMesh );

    // Iterate over local box
    for ( auto i = d_localBox->first[0]; i <= d_localBox->last[0]; i++ ) {

        // Get values in the stencil
        auto   ET  = unpackLocalData( E_vec, T_vec, i );
        double E_W = ET[0], E_O = ET[1], E_E = ET[2]; 
        double T_W = ET[3], T_O = ET[4], T_E = ET[5]; 
        size_t dof_O = gridIndsToScalarDOF( i );
        AMP_INSIST( T_O > 1e-14, "PDE coefficients ill-defined for T <= 0 (T[" + std::to_string(dof_O) + std::string("]=") + std::to_string(T_O) + std::string(")") );

        /* Compute coefficients to apply stencil in a quasi-linear fashion */
        // Reaction coefficients at cell centers
        double REE, RET;
        double RTE, RTT;
        // Diffusion coefficients at cell faces
        double Dr_WO, Dr_OE; 
        double DT_WO, DT_OE; 

        // Nonlinear PDE
        if ( nonlinearModel ) {
            // --- Reaction coefficients
            double sigma = std::pow( z/T_O, 3.0 );
            REE = RTE = -sigma;
            RET = RTT =  sigma * pow( T_O, 3.0 );
            // --- Diffusion coefficients
            // Temp at mid points             
            double T_WO = 0.5*( T_W + T_O ); // T_{i-1/2}
            double T_OE = 0.5*( T_O + T_E ); // T_{i+1/2}
            // Unlimited energy flux = 1/(3*sigma). I moved the factor of 2 into Dr rather than DE (actually, I think Bobby's paper may be off somewhere by a factor of 2)
            Dr_WO = pow( T_WO, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) );
            Dr_OE = pow( T_OE, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) );
            // Limit the energy flux if need be, eq. (17)
            // DE = 1/( 3*sigma + 1/E * dE/dx ) = 1/( 1/Dr + 1/E * dE/dx ) = Dr/( 1 + Dr/E * dE/dx )
            if ( fluxLimited ) {
                double DE_WO = Dr_WO / ( 1.0 + Dr_WO*( abs( E_O - E_W )/( h*0.5*( E_O + E_W ) ) ) );
                double DE_OE = Dr_OE / ( 1.0 + Dr_OE*( abs( E_E - E_O )/( h*0.5*( E_E + E_O ) ) ) );
                Dr_WO = DE_WO;
                Dr_OE = DE_OE;
            }
            // Create face values of D_T coefficient, analogous to those for D_E
            DT_WO = pow( T_WO, 2.5 );
            DT_OE = pow( T_OE, 2.5 ); 

        // Linear PDE
        } else {
            // --- Reaction coefficients
            REE   = RTE = -1.0;
            RET   = RTT =  1.0;
            // --- Diffusion coefficients
            Dr_WO = Dr_OE = 1.0;
            DT_WO = DT_OE = 1.0;
        }    
        // Scale coefficients by constants
        // Reaction
        REE *= -k12, RET *= -k12;
        RTE *=  k22, RTT *=  k22;
        // Diffusion
        Dr_WO *= -k11, Dr_OE *= -k11;
        DT_WO *= -k21, DT_OE *= -k21;
        
        /* Apply the operators in a quasi-linear fashion */
        // Apply diffusion operators
        double dif_E_action = Dr_OE*( E_E - E_O )*rh2 - Dr_WO*( E_O - E_W )*rh2;
        double dif_T_action = DT_OE*( T_E - T_O )*rh2 - DT_WO*( T_O - T_W )*rh2;
        // Sum diffusion and reaction terms
        double LE = dif_E_action + ( REE*E_O + RET*T_O );
        double LT = dif_T_action + ( RTE*E_O + RTT*T_O );

        // Insert values into the vectors
        LE_vec->setValueByGlobalID<double>( dof_O, LE );
        LT_vec->setValueByGlobalID<double>( dof_O, LT );
    }
    LET_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


// Apply operator L to vector ET
void RadDifOp::apply2D(std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_vec_,
            std::shared_ptr<AMP::LinearAlgebra::Vector> LET_vec_) 
{
    // --- Unpack parameters ---
    std::shared_ptr<const AMP::Database> PDE_db  = this->d_db->getDatabase( "PDE" );
    std::shared_ptr<const AMP::Database> mesh_db = this->d_db->getDatabase( "mesh" );
    double h   = mesh_db->getScalar<double>( "h" );
    double rh2 = 1.0/(h*h); // Reciprocal h squared
    double k11 = PDE_db->getScalar<double>( "k11" );
    double k12 = PDE_db->getScalar<double>( "k12" );
    double k21 = PDE_db->getScalar<double>( "k21" );
    double k22 = PDE_db->getScalar<double>( "k22" );
    double z   = PDE_db->getWithDefault<double>( "zatom", 1.0 );  

    bool nonlinearModel = ( PDE_db->getScalar<std::string>( "model" ) == "nonlinear" );
    bool fluxLimited    = PDE_db->getScalar<bool>( "fluxLimited" );


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
    // Get local grid index box w/ zero ghosts
    auto d_localBox  = getLocalNodeBox( d_BoxMesh );

    // Iterate over local box
    for ( auto j = d_localBox->first[1]; j <= d_localBox->last[1]; j++ ) {
        for ( auto i = d_localBox->first[0]; i <= d_localBox->last[0]; i++ ) {

            // Get values in the stencil
            auto   ET  = unpackLocalData( E_vec, T_vec, i, j );
            double E_S = ET[0], E_W = ET[1], E_O = ET[2], E_E = ET[3], E_N = ET[4]; 
            double T_S = ET[5], T_W = ET[6], T_O = ET[7], T_E = ET[8], T_N = ET[9];
            size_t dof_O = gridIndsToScalarDOF( i, j );
            AMP_INSIST( T_O > 1e-14, "PDE coefficients ill-defined for T <= 0" );

            /* Compute coefficients to apply stencil in a quasi-linear fashion */
            // Reaction coefficients at cell centers
            double REE, RET;
            double RTE, RTT;
            // Diffusion coefficients at cell faces
            double Dr_WO, Dr_OE, Dr_SO, Dr_ON; 
            double DT_WO, DT_OE, DT_SO, DT_ON; 

            // Nonlinear PDE
            if ( nonlinearModel ) {
                // --- Reaction coefficients
                double sigma = std::pow( z/T_O, 3.0 );
                REE = RTE = -sigma;
                RET = RTT =  sigma * pow( T_O, 3.0 );
                // --- Diffusion coefficients
                // Temp at mid points             
                double T_WO = 0.5*( T_W + T_O ); // T_{i-1/2,j}
                double T_OE = 0.5*( T_O + T_E ); // T_{i+1/2,j}
                double T_SO = 0.5*( T_S + T_O ); // T_{i,j-1/2}
                double T_ON = 0.5*( T_O + T_N ); // T_{i,j+1/2}
                // Unlimited energy flux. I moved the factor of 2 into Dr rather than DE.
                Dr_WO = pow( T_WO, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) );
                Dr_OE = pow( T_OE, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) );
                Dr_SO = pow( T_SO, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) ); 
                Dr_ON = pow( T_ON, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) ); 
                // Limit the energy flux if need be, eq. (17)
                if ( fluxLimited ) {
                    double DE_WO = Dr_WO/( 1.0 + Dr_WO*( abs( E_O - E_W )/( h*0.5*(E_O + E_W) ) ) );
                    double DE_OE = Dr_OE/( 1.0 + Dr_OE*( abs( E_E - E_O )/( h*0.5*(E_E + E_O) ) ) );
                    double DE_SO = Dr_SO/( 1.0 + Dr_SO*( abs( E_O - E_S )/( h*0.5*(E_O + E_S) ) ) );
                    double DE_ON = Dr_ON/( 1.0 + Dr_ON*( abs( E_N - E_O )/( h*0.5*(E_N + E_O) ) ) );
                    Dr_WO = DE_WO;
                    Dr_OE = DE_OE;
                    Dr_SO = DE_SO;
                    Dr_ON = DE_ON;
                }
                // Create face values of D_T coefficient, analogous to those for D_E
                DT_WO = pow( T_WO, 2.5 );
                DT_OE = pow( T_OE, 2.5 ); 
                DT_SO = pow( T_SO, 2.5 );
                DT_ON = pow( T_ON, 2.5 ); 

            // Linear PDE
            } else {
                // --- Reaction coefficients
                REE   = RTE = -1.0;
                RET   = RTT =  1.0;
                // --- Diffusion coefficients
                Dr_WO = Dr_OE = Dr_SO = Dr_ON = 1.0;
                DT_WO = DT_OE = DT_SO = DT_ON = 1.0;
            }    
            // Scale coefficients by constants
            // Reaction
            REE *= -k12, RET *= -k12;
            RTE *=  k22, RTT *=  k22;
            // Diffusion
            Dr_WO *= -k11, Dr_OE *= -k11, Dr_SO *= -k11, Dr_ON *= -k11;
            DT_WO *= -k21, DT_OE *= -k21, DT_SO *= -k21, DT_ON *= -k21;
            
            /* Apply the operators in a quasi-linear fashion */
            // Diffusion operators terms
            double dif_E_action = Dr_OE*( E_E - E_O )*rh2 - Dr_WO*( E_O - E_W )*rh2 \
                    + Dr_ON*( E_N - E_O )*rh2 - Dr_SO*( E_O - E_S )*rh2;
            double dif_T_action = DT_OE*( T_E - T_O )*rh2 - DT_WO*( T_O - T_W )*rh2 \
                    + DT_ON*( T_N - T_O )*rh2 - DT_SO*( T_O - T_S )*rh2;
            // Sum diffusion and reaction terms
            double LE = dif_E_action + ( REE*E_O + RET*T_O );
            double LT = dif_T_action + ( RTE*E_O + RTT*T_O );

            // Insert values into the vectors
            LE_vec->setValueByGlobalID<double>( dof_O, LE );
            LT_vec->setValueByGlobalID<double>( dof_O, LT );
        }
    }
    LET_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}



// Get all values in the 3-point stencil centered at grid index i
std::vector<double> RadDifOp::unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, int i ) {

    double E_W, E_O, E_E; 
    double T_W, T_O, T_E;

    // The current DOF
    size_t dof_O = gridIndsToScalarDOF( i );
    E_O = E_vec->getValueByGlobalID<double>( dof_O );
    T_O = T_vec->getValueByGlobalID<double>( dof_O );

    // At WEST boundary (1), so WEST neighbor is a ghost
    if ( i == d_globalBox->first[0] ) {
        auto node = gridIndsToMeshElement( i );
        auto ETg  = getGhostValues( 1, node, E_O, T_O );
        E_W       = ETg[0]; 
        T_W       = ETg[1];

    // At interior DOF; WEST neighbor is an interior DOF
    } else {
        size_t dof_W = gridIndsToScalarDOF( i-1 );
        E_W          = E_vec->getValueByGlobalID( dof_W ); 
        T_W          = T_vec->getValueByGlobalID( dof_W );
    }

    // At EAST boundary (2), so EAST neighbor is a ghost
    if ( i == d_globalBox->last[0] ) {
        auto node = gridIndsToMeshElement( i );
        auto ETg  = getGhostValues( 2, node, E_O, T_O );
        E_E       = ETg[0]; 
        T_E       = ETg[1];

    // At interior DOF; EAST neighbor is an interior DOF
    } else {
        size_t dof_E = gridIndsToScalarDOF( i+1 );
        E_E          = E_vec->getValueByGlobalID( dof_E ); 
        T_E          = T_vec->getValueByGlobalID( dof_E );
    }

    std::vector<double> ET = { E_W, E_O, E_E, //
                               T_W, T_O, T_E };
    return ET;
}

// Get all values in the 5-point stencil centered at grid index i,j
std::vector<double> RadDifOp::unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, int i, int j ) {

    double E_S, E_W, E_O, E_E, E_N; 
    double T_S, T_W, T_O, T_E, T_N;

    // The current DOF
    size_t dof_O = gridIndsToScalarDOF( i, j );
    E_O = E_vec->getValueByGlobalID<double>( dof_O );
    T_O = T_vec->getValueByGlobalID<double>( dof_O );

    // Get WEST neighboring value
    // At WEST boundary (1), so WEST neighbor is a ghost
    if ( i == d_globalBox->first[0] ) {
        auto node = gridIndsToMeshElement( i, j );
        auto ETg  = getGhostValues( 1, node, E_O, T_O );
        E_W       = ETg[0];
        T_W       = ETg[1];

    // At interior DOF; WEST neighbor is an interior DOF
    } else {
        size_t dof_W = gridIndsToScalarDOF( i-1, j );
        E_W          = E_vec->getValueByGlobalID( dof_W ); 
        T_W          = T_vec->getValueByGlobalID( dof_W );
    }

    // Get EAST neighboring value
    // At EAST boundary (2), so EAST neighbor is a ghost
    if ( i == d_globalBox->last[0] ) {
        auto node = gridIndsToMeshElement( i, j );
        auto ETg  = getGhostValues( 2, node, E_O, T_O );
        E_E       = ETg[0]; 
        T_E       = ETg[1];

    // At interior DOF; EAST neighbor is an interior DOF
    } else {
        size_t dof_E = gridIndsToScalarDOF( i+1, j );
        E_E          = E_vec->getValueByGlobalID( dof_E ); 
        T_E          = T_vec->getValueByGlobalID( dof_E );
    }

    // Get SOUTH neighboring value
    // At SOUTH boundary (3), so SOUTH neighbor is a ghost
    if ( j == d_globalBox->first[1] ) {
        auto node = gridIndsToMeshElement( i, j );
        auto ETg  = getGhostValues( 3, node, E_O, T_O );
        E_S       = ETg[0]; 
        T_S       = ETg[1];

    // At interior DOF; SOUTH neighbor is an interior DOF
    } else {
        size_t dof_S = gridIndsToScalarDOF( i, j-1 );
        E_S          = E_vec->getValueByGlobalID( dof_S ); 
        T_S          = T_vec->getValueByGlobalID( dof_S );
    }

    // Get NORTH neighboring value
    // At NORTH boundary (4), so NORTH neighbor is a ghost
    if ( j == d_globalBox->last[1] ) {
        auto node = gridIndsToMeshElement( i, j );
        auto ETg  = getGhostValues( 4, node, E_O, T_O );
        E_N       = ETg[0]; 
        T_N       = ETg[1];

    // At interior DOF; NORTH neighbor is an interior DOF
    } else {
        size_t dof_N = gridIndsToScalarDOF( i, j+1 );
        E_N          = E_vec->getValueByGlobalID( dof_N ); 
        T_N          = T_vec->getValueByGlobalID( dof_N );
    }

    std::vector<double> ET = { E_S, E_W, E_O, E_E, E_N, //
                               T_S, T_W, T_O, T_E, T_N };
    return ET;
}


/* Get all values in the 3-point stencil centered at grid index i
 dofs will be returned with W, O, E
 onBoundary is a flag indicating whether the DOF is on a boundary. 
 In the case that it is on a boundary, values in dofs corresponding to the boundaries are not meaningful
 */
std::vector<double> RadDifOp::unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, int i, std::vector<size_t> &dofs, bool &onBoundary ) {

    onBoundary = false;
    double E_W, E_O, E_E; 
    double T_W, T_O, T_E;

    // The current DOF
    dofs[1] = gridIndsToScalarDOF( i );
    E_O = E_vec->getValueByGlobalID<double>( dofs[1] );
    T_O = T_vec->getValueByGlobalID<double>( dofs[1] );

    // At WEST boundary (1), so WEST neighbor is a ghost
    if ( i == d_globalBox->first[0] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i );
        setGhostData( 1, node, E_O, T_O );
        E_W        = d_ghostData[0]; 
        T_W        = d_ghostData[1];
        

    // At interior DOF; WEST neighbor is an interior DOF
    } else {
        dofs[0] = gridIndsToScalarDOF( i-1 );
        E_W     = E_vec->getValueByGlobalID( dofs[0] ); 
        T_W     = T_vec->getValueByGlobalID( dofs[0] );
    }

    // At EAST boundary (2), so EAST neighbor is a ghost
    if ( i == d_globalBox->last[0] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i );
        setGhostData( 2, node, E_O, T_O );
        E_E        = d_ghostData[0]; 
        T_E        = d_ghostData[1];

    // At interior DOF; EAST neighbor is an interior DOF
    } else {
        dofs[2] = gridIndsToScalarDOF( i+1 );
        E_E     = E_vec->getValueByGlobalID( dofs[2] ); 
        T_E     = T_vec->getValueByGlobalID( dofs[2] );
    }

    std::vector<double> ET = { E_W, E_O, E_E, //
                               T_W, T_O, T_E };
    return ET;
}





// Get all values in the 5-point stencil centered at grid index i,j
// S, W, O, E, N
// 0, 1, 2, 3, 4
std::vector<double> RadDifOp::unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, int i, int j, std::vector<size_t> &dofs, bool &onBoundary ) {

    onBoundary = false;
    double E_S, E_W, E_O, E_E, E_N; 
    double T_S, T_W, T_O, T_E, T_N;

    // The current DOF
    dofs[2] = gridIndsToScalarDOF( i, j );
    E_O = E_vec->getValueByGlobalID<double>( dofs[2] );
    T_O = T_vec->getValueByGlobalID<double>( dofs[2] );

    // Get WEST neighboring value
    // At WEST boundary (1), so WEST neighbor is a ghost
    if ( i == d_globalBox->first[0] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i, j );
        setGhostData( 1, node, E_O, T_O );
        E_W        = d_ghostData[0];
        T_W        = d_ghostData[1];

    // At interior DOF; WEST neighbor is an interior DOF
    } else {
        dofs[1] = gridIndsToScalarDOF( i-1, j );
        E_W     = E_vec->getValueByGlobalID( dofs[1] ); 
        T_W     = T_vec->getValueByGlobalID( dofs[1] );
    }

    // Get EAST neighboring value
    // At EAST boundary (2), so EAST neighbor is a ghost
    if ( i == d_globalBox->last[0] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i, j );
        setGhostData( 2, node, E_O, T_O );
        E_E        = d_ghostData[0]; 
        T_E        = d_ghostData[1];

    // At interior DOF; EAST neighbor is an interior DOF
    } else {
        dofs[3] = gridIndsToScalarDOF( i+1, j );
        E_E     = E_vec->getValueByGlobalID( dofs[3] ); 
        T_E     = T_vec->getValueByGlobalID( dofs[3] );
    }

    // Get SOUTH neighboring value
    // At SOUTH boundary (3), so SOUTH neighbor is a ghost
    if ( j == d_globalBox->first[1] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i, j );
        setGhostData( 3, node, E_O, T_O );
        E_S        = d_ghostData[0]; 
        T_S        = d_ghostData[1];

    // At interior DOF; SOUTH neighbor is an interior DOF
    } else {
        dofs[0] = gridIndsToScalarDOF( i, j-1 );
        E_S     = E_vec->getValueByGlobalID( dofs[0] ); 
        T_S     = T_vec->getValueByGlobalID( dofs[0] );
    }

    // Get NORTH neighboring value
    // At NORTH boundary (4), so NORTH neighbor is a ghost
    if ( j == d_globalBox->last[1] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i, j );
        setGhostData( 4, node, E_O, T_O );
        E_N        = d_ghostData[0]; 
        T_N        = d_ghostData[1];

    // At interior DOF; NORTH neighbor is an interior DOF
    } else {
        dofs[4] = gridIndsToScalarDOF( i, j+1 );
        E_N     = E_vec->getValueByGlobalID( dofs[4] ); 
        T_N     = T_vec->getValueByGlobalID( dofs[4] );
    }

    std::vector<double> ET = { E_S, E_W, E_O, E_E, E_N, //
                               T_S, T_W, T_O, T_E, T_N };
    return ET;
}



#if 0
void RadDifOpPJac::setData( ) {
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


void RadDifOpPJac::setData1D(  ) 
{
    // --- Unpack parameters ---
    std::shared_ptr<const AMP::Database> PDE_db  = d_RadDifOp->d_db->getDatabase( "PDE" );
    std::shared_ptr<const AMP::Database> mesh_db = d_RadDifOp->d_db->getDatabase( "mesh" );
    double h   = mesh_db->getScalar<double>( "h" );
    double rh2 = 1.0/(h*h); // Reciprocal h squared
    double k11 = PDE_db->getScalar<double>( "k11" );
    double k12 = PDE_db->getScalar<double>( "k12" );
    double k21 = PDE_db->getScalar<double>( "k21" );
    double k22 = PDE_db->getScalar<double>( "k22" );
    double z   = PDE_db->getWithDefault<double>( "zatom", 1.0 );  

    bool nonlinearModel = ( PDE_db->getScalar<std::string>( "model" ) == "nonlinear" );
    bool fluxLimited    = PDE_db->getScalar<bool>( "fluxLimited" );

    // --- Unpack frozen vector ---
    auto ET_vec = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( this->d_frozenVec );
    AMP_INSIST( ET_vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto E_vec = ET_vec->getVector(0);
    auto T_vec = ET_vec->getVector(1);

    // Initialize+allocate d_data if it's not been done already
    // We don't allocate matrices here since they're going to be created below... Possibly there's an option to do something smarter where we just reset the values because the sparsity structure is always identical, including for both matrices...
    if ( !d_data ) {
        d_data       = std::make_shared<RadDifOpPJacData>();
        d_data->r_EE = E_vec->clone( );
        d_data->r_ET = E_vec->clone( );
        d_data->r_TE = E_vec->clone( );
        d_data->r_TT = E_vec->clone( );
    // I believe the following should free the matrices d_E and d_T were pointing to
    } else {
        d_data->d_E.reset();
        d_data->d_T.reset();
    }
    // Unpack data structures we're going to fill
    auto r_EE_vec = d_data->r_EE;
    auto r_ET_vec = d_data->r_ET;
    auto r_TE_vec = d_data->r_TE;
    auto r_TT_vec = d_data->r_TT;


    ///////////////////////////////////////////
    AMP_INSIST( abs(k21) > 1e-14, "I shouldn't be constructing the d_T matrix if this coefficinet is zero...." );
    AMP_INSIST( abs(k11) > 1e-14, "I shouldn't be constructing the d_E matrix if this coefficinet is zero...." );
    
    // Create a map from the DOF to a pair a vectors
    // Map from a DOF to vector of col inds and associated data
    std::map<size_t, colsDataPair> localCSRData_d_E;
    std::map<size_t, colsDataPair> localCSRData_d_T;

    // Data structures we'll temporarily store information in for each row we iterate over
    bool onBoundary;             // Is the current DOF on the boundary?
    std::vector<size_t> dofs(3); // DOFs the current DOF connects to

    // --- Iterate over all local rows ---
    // Get local grid index box w/ zero ghosts
    auto d_localBox  = getLocalNodeBox( d_RadDifOp->d_BoxMesh );

    // Iterate over local box
    for ( auto i = d_localBox->first[0]; i <= d_localBox->last[0]; i++ ) {

        // Get values in the stencil
        auto   ET  = d_RadDifOp->unpackLocalData( E_vec, T_vec, i, dofs, onBoundary );
        double E_W = ET[0], E_O = ET[1], E_E = ET[2]; 
        double T_W = ET[3], T_O = ET[4], T_E = ET[5]; 
        size_t dof_O = dofs[1];

        /* Compute coefficients to apply stencil in a quasi-linear fashion */
        // Reaction coefficients at cell centers
        double REE, RET;
        double RTE, RTT;
        // Diffusion coefficients at cell faces
        double Dr_WO, Dr_OE; 
        double DT_WO, DT_OE; 

        // Nonlinear PDE
        if ( nonlinearModel ) {
            // --- Reaction coefficients
            double sigma = std::pow( z/T_O, 3.0 );
            REE = RTE = -sigma;
            RET = RTT =  sigma * pow( T_O, 3.0 );
            // --- Diffusion coefficients
            // Temp at mid points             
            double T_WO = 0.5*( T_W + T_O ); // T_{i-1/2}
            double T_OE = 0.5*( T_O + T_E ); // T_{i+1/2}
            // Unlimited energy flux. I moved the factor of 2 into Dr rather than DE.
            Dr_WO = pow( T_WO, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) );
            Dr_OE = pow( T_OE, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) );
            // Limit the energy flux if need be, eq. (17)
            if ( fluxLimited ) {
                double DE_WO = Dr_WO / ( 1.0 + Dr_WO*( abs( E_O - E_W )/( h*0.5*( E_O + E_W ) ) ) );
                double DE_OE = Dr_OE / ( 1.0 + Dr_OE*( abs( E_E - E_O )/( h*0.5*( E_E + E_O ) ) ) );
                Dr_WO = DE_WO;
                Dr_OE = DE_OE;
            }
            // Create face values of D_T coefficient, analogous to those for D_E
            DT_WO = pow( T_WO, 2.5 );
            DT_OE = pow( T_OE, 2.5 ); 

        // Linear PDE
        } else {
            // --- Reaction coefficients
            REE   = RTE = -1.0;
            RET   = RTT =  1.0;
            // --- Diffusion coefficients
            Dr_WO = Dr_OE = 1.0;
            DT_WO = DT_OE = 1.0;
        }    
        // Scale coefficients by constants
        // Reaction
        REE *= -k12, RET *= -k12;
        RTE *=  k22, RTT *=  k22;
        // Diffusion
        Dr_WO *= -k11, Dr_OE *= -k11;
        DT_WO *= -k21, DT_OE *= -k21;
        
        /* ------------------------------------------------------------------------
        The nonlinear discretization is then as follows

        // Apply diffusion operators
        double dif_E_action = Dr_OE*( E_E - E_O )*rh2 - Dr_WO*( E_O - E_W )*rh2;
        double dif_T_action = DT_OE*( T_E - T_O )*rh2 - DT_WO*( T_O - T_W )*rh2;
        // Sum diffusion and reaction terms
        double LE = dif_E_action + ( REE*E_O + RET*T_O );
        double LT = dif_T_action + ( RTE*E_O + RTT*T_O );
        */

        /* --- Reaction terms --- */
        r_EE_vec->setValueByGlobalID<double>( dof_O, REE );
        r_ET_vec->setValueByGlobalID<double>( dof_O, RET );
        r_TE_vec->setValueByGlobalID<double>( dof_O, RTE );
        r_TT_vec->setValueByGlobalID<double>( dof_O, RTT );

        /* --- Diffusion terms --- */
        // Get 3-point stencils for interior DOFs: W, O, E
        // Energy
        std::vector<double> stnE = { rh2*Dr_WO, -rh2*(Dr_WO + Dr_OE), rh2*Dr_OE };
        // Temperature
        std::vector<double> stnT = { rh2*DT_WO, -rh2*(DT_WO + DT_OE), rh2*DT_OE };

        // The current DOF is on a boundary. The ghost connection in the stencil gets added into the into the diagonal. In the temperature operator it's with coefficient 1, and in the energy operator it's with coefficient alpha_k 
        if ( onBoundary ) {
            // On WEST boundary; DOF to our WEST is a ghost. 
            if ( i == d_RadDifOp->d_globalBox->first[0] ) {
                // Compute energy diffusion coefficient on boundary
                double Dr_WO  = d_RadDifOp->diffusionCoefficientE( 0.5*(T_W+T_O), z ); // T_{i-1/2}
                double c1     = k11 * Dr_WO;
                // Get Picard coefficient
                double alpha1 = d_RadDifOp->PicardCorrectionRobinE( c1, 1 );

                std::vector<size_t> cols  = {                  dofs[1], dofs[2] };
                std::vector<double> valsE = { alpha1*stnE[0] + stnE[1], stnE[2] };
                std::vector<double> valsT = {        stnT[0] + stnT[1], stnT[2] };
                localCSRData_d_E[dof_O] = { cols, valsE };
                localCSRData_d_T[dof_O] = { cols, valsT };
                continue;
            }

            // On EAST boundary; DOF to our EAST is ghost.
            if ( i == d_RadDifOp->d_globalBox->last[0] ) {
                // Compute energy diffusion coefficient on boundary
                double Dr_OE  = d_RadDifOp->diffusionCoefficientE( 0.5*(T_O+T_E), z ); // T_{i+1/2}
                double c2     = k11 * Dr_OE;
                // Get Picard coefficient
                double alpha2 = d_RadDifOp->PicardCorrectionRobinE( c2, 2 );

                std::vector<size_t> cols  = { dofs[0], dofs[1]                  };
                std::vector<double> valsE = { stnE[0], stnE[1] + alpha2*stnE[2] };
                std::vector<double> valsT = { stnT[0], stnT[1] +        stnT[2] };
                localCSRData_d_E[dof_O]   = { cols, valsE };
                localCSRData_d_T[dof_O]   = { cols, valsT };
                continue;
            }
        }

        // At an interior DOF, both neighbors are active DOFs
        std::vector<size_t> cols = dofs;
        localCSRData_d_E[dof_O] = { cols, stnE };
        localCSRData_d_T[dof_O] = { cols, stnT };
    }

    // Finalize vector construction
    r_EE_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_ET_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_TE_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_TT_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // --- Create matrices
    // Create Lambdas to return col inds from a given row ind
    auto getColumnIDs_d_E = [&](int row) { return localCSRData_d_E[row].cols; };
    auto getColumnIDs_d_T = [&](int row) { return localCSRData_d_T[row].cols; };
    // Create CSR matrices
    auto inVec = r_EE_vec, outVec = r_EE_vec;
    auto d_E_mat = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix", getColumnIDs_d_E );
    auto d_T_mat = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix", getColumnIDs_d_T );
    // Fill matrices with data
    fillMatWithLocalCSRData( d_E_mat, d_RadDifOp->d_scalarDOFMan, localCSRData_d_E );
    fillMatWithLocalCSRData( d_T_mat, d_RadDifOp->d_scalarDOFMan, localCSRData_d_T );
    // Finalize matrix construction
    d_E_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_T_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    d_data->d_E = d_E_mat;
    d_data->d_T = d_T_mat;
}


void RadDifOpPJac::setData2D(  ) 
{
    // --- Unpack parameters ---
    std::shared_ptr<const AMP::Database> PDE_db  = d_RadDifOp->d_db->getDatabase( "PDE" );
    std::shared_ptr<const AMP::Database> mesh_db = d_RadDifOp->d_db->getDatabase( "mesh" );
    double h   = mesh_db->getScalar<double>( "h" );
    double rh2 = 1.0/(h*h); // Reciprocal h squared
    double k11 = PDE_db->getScalar<double>( "k11" );
    double k12 = PDE_db->getScalar<double>( "k12" );
    double k21 = PDE_db->getScalar<double>( "k21" );
    double k22 = PDE_db->getScalar<double>( "k22" );
    double z   = PDE_db->getWithDefault<double>( "zatom", 1.0 );  

    bool fluxLimited    = PDE_db->getScalar<bool>( "fluxLimited" );

    // --- Unpack frozen vector ---
    auto ET_vec = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( this->d_frozenVec );
    AMP_INSIST( ET_vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto E_vec = ET_vec->getVector(0);
    auto T_vec = ET_vec->getVector(1);

    // Initialize+allocate d_data if it's not been done already
    // We don't allocate matrices here since they're going to be created below... Possibly there's an option to do something smarter where we just reset the values because the sparsity structure is always identical, including for both matrices...
    if ( !d_data ) {
        d_data       = std::make_shared<RadDifOpPJacData>();
        d_data->r_EE = E_vec->clone( );
        d_data->r_ET = E_vec->clone( );
        d_data->r_TE = E_vec->clone( );
        d_data->r_TT = E_vec->clone( );
    // I believe the following should free the matrices d_E and d_T were pointing to
    } else {
        d_data->d_E.reset();
        d_data->d_T.reset();
    }
    // Unpack data structures we're going to fill
    auto r_EE_vec = d_data->r_EE;
    auto r_ET_vec = d_data->r_ET;
    auto r_TE_vec = d_data->r_TE;
    auto r_TT_vec = d_data->r_TT;


    ///////////////////////////////////////////
    AMP_INSIST( abs(k21) > 1e-14, "I shouldn't be constructing the d_T matrix if this coefficinet is zero...." );
    AMP_INSIST( abs(k11) > 1e-14, "I shouldn't be constructing the d_E matrix if this coefficinet is zero...." );
    
    // Create a map from the DOF to a pair a vectors
    // Map from a DOF to vector of col inds and associated data
    std::map<size_t, colsDataPair> localCSRData_d_E;
    std::map<size_t, colsDataPair> localCSRData_d_T;

    // Data structures we'll temporarily store information in for each row we iterate over
    bool onBoundary;             // Is the current DOF on the boundary?
    std::vector<size_t> dofs(5); // DOFs the current DOF connects to

    // --- Iterate over all local rows ---
    // Get local grid index box w/ zero ghosts
    auto d_localBox  = getLocalNodeBox( d_RadDifOp->d_BoxMesh );

    // Iterate over local box
    for ( auto j = d_localBox->first[1]; j <= d_localBox->last[1]; j++ ) {
        for ( auto i = d_localBox->first[0]; i <= d_localBox->last[0]; i++ ) {

            // Get values in the stencil; S, W, O, E, N
            auto   ET  = d_RadDifOp->unpackLocalData( E_vec, T_vec, i, j, dofs, onBoundary );
            double E_S = ET[0], E_W = ET[1], E_O = ET[2], E_E = ET[3], E_N = ET[4]; 
            double T_S = ET[5], T_W = ET[6], T_O = ET[7], T_E = ET[8], T_N = ET[9];
            size_t dof_O = dofs[2];

            /* Compute coefficients to apply stencil in a quasi-linear fashion */
            // Reaction coefficients at cell centers
            double REE, RET;
            double RTE, RTT;
            // Diffusion coefficients at cell faces
            double Dr_WO, Dr_OE, Dr_SO, Dr_ON; 
            double DT_WO, DT_OE, DT_SO, DT_ON; 

            // Nonlinear PDE
            if ( nonlinearModel ) {
                // --- Reaction coefficients
                double sigma = std::pow( z/T_O, 3.0 );
                REE = RTE = -sigma;
                RET = RTT =  sigma * pow( T_O, 3.0 );
                // --- Diffusion coefficients
                // Temp at mid points             
                double T_WO = 0.5*( T_W + T_O ); // T_{i-1/2,j}
                double T_OE = 0.5*( T_O + T_E ); // T_{i+1/2,j}
                double T_SO = 0.5*( T_S + T_O ); // T_{i,j-1/2}
                double T_ON = 0.5*( T_O + T_N ); // T_{i,j+1/2}
                // Unlimited energy flux. I moved the factor of 2 into Dr rather than DE.
                Dr_WO = pow( T_WO, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) );
                Dr_OE = pow( T_OE, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) );
                Dr_SO = pow( T_SO, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) ); 
                Dr_ON = pow( T_ON, 3.0 ) / ( 3.0*0.5*( z*z*z + z*z*z ) ); 
                // Limit the energy flux if need be, eq. (17)
                if ( fluxLimited ) {
                    double DE_WO = Dr_WO/( 1.0 + Dr_WO*( abs( E_O - E_W )/( h*0.5*(E_O + E_W) ) ) );
                    double DE_OE = Dr_OE/( 1.0 + Dr_OE*( abs( E_E - E_O )/( h*0.5*(E_E + E_O) ) ) );
                    double DE_SO = Dr_SO/( 1.0 + Dr_SO*( abs( E_O - E_S )/( h*0.5*(E_O + E_S) ) ) );
                    double DE_ON = Dr_ON/( 1.0 + Dr_ON*( abs( E_N - E_O )/( h*0.5*(E_N + E_O) ) ) );
                    Dr_WO = DE_WO;
                    Dr_OE = DE_OE;
                    Dr_SO = DE_SO;
                    Dr_ON = DE_ON;
                }
                // Create face values of D_T coefficient, analogous to those for D_E
                DT_WO = pow( T_WO, 2.5 );
                DT_OE = pow( T_OE, 2.5 ); 
                DT_SO = pow( T_SO, 2.5 );
                DT_ON = pow( T_ON, 2.5 ); 

            // Linear PDE
            } else {
                // --- Reaction coefficients
                REE   = RTE = -1.0;
                RET   = RTT =  1.0;
                // --- Diffusion coefficients
                Dr_WO = Dr_OE = Dr_SO = Dr_ON = 1.0;
                DT_WO = DT_OE = DT_SO = DT_ON = 1.0;
            }    
            // Scale coefficients by constants
            // Reaction
            REE *= -k12, RET *= -k12;
            RTE *=  k22, RTT *=  k22;
            // Diffusion
            Dr_WO *= -k11, Dr_OE *= -k11, Dr_SO *= -k11, Dr_ON *= -k11;
            DT_WO *= -k21, DT_OE *= -k21, DT_SO *= -k21, DT_ON *= -k21;
            
            /* ------------------------------------------------------------------------
            The nonlinear discretization is then as follows

            // Apply diffusion operators
            double dif_E_action = Dr_OE*( E_E - E_O )*rh2 - Dr_WO*( E_O - E_W )*rh2 \
                    + Dr_ON*( E_N - E_O )*rh2 - Dr_SO*( E_O - E_S )*rh2;
            double dif_T_action = DT_OE*( T_E - T_O )*rh2 - DT_WO*( T_O - T_W )*rh2 \
                    + DT_ON*( T_N - T_O )*rh2 - DT_SO*( T_O - T_S )*rh2;
            // Sum diffusion and reaction terms
            double LE = dif_E_action + ( REE*E_O + RET*T_O );
            double LT = dif_T_action + ( RTE*E_O + RTT*T_O );
            */

            /* --- Reaction terms --- */
            r_EE_vec->setValueByGlobalID<double>( dof_O, REE );
            r_ET_vec->setValueByGlobalID<double>( dof_O, RET );
            r_TE_vec->setValueByGlobalID<double>( dof_O, RTE );
            r_TT_vec->setValueByGlobalID<double>( dof_O, RTT );

            /* --- Diffusion terms --- */
            // Get 5-point stencils for interior DOFs: S, W, O, E, N
            // Energy
            std::vector<double> stnE = { rh2*Dr_SO, rh2*Dr_WO, 
                                    -rh2*(Dr_SO + Dr_WO + Dr_OE + Dr_ON), 
                                         rh2*Dr_OE, rh2*Dr_ON };
            // Temperature
            std::vector<double> stnT = { rh2*DT_SO, rh2*DT_WO, 
                                    -rh2*(DT_SO + DT_WO + DT_OE + DT_ON), 
                                         rh2*DT_OE, rh2*DT_ON };

            // The current DOF is on a boundary. The ghost connection in the stencil gets added into the into the diagonal. In the temperature operator it's with coefficient 1, and in the energy operator it's with coefficient alpha_k 
            if ( onBoundary ) {

                // For readibility, just get Picard coefficients on all boundaries
                double c1 = k11 * d_RadDifOp->diffusionCoefficientE( 0.5*(T_W+T_O), z );
                double c2 = k11 * d_RadDifOp->diffusionCoefficientE( 0.5*(T_O+T_E), z );
                double c3 = k11 * d_RadDifOp->diffusionCoefficientE( 0.5*(T_S+T_O), z );
                double c4 = k11 * d_RadDifOp->diffusionCoefficientE( 0.5*(T_O+T_N), z );
                // ak == alphak (abuse of notation; these are not the ak constants in Robin BCs)
                double aW = d_RadDifOp->PicardCorrectionRobinE( c1, 1 );
                double aE = d_RadDifOp->PicardCorrectionRobinE( c2, 2 );
                double aS = d_RadDifOp->PicardCorrectionRobinE( c3, 3 );
                double aN = d_RadDifOp->PicardCorrectionRobinE( c4, 4 );

                // Vectors we're going to fill
                std::vector<size_t> cols;
                std::vector<double> valsT;
                std::vector<double> valsE;

                // On WEST boundary; DOF to our WEST is a ghost. 
                // S, W, O, E, N
                // 0, 1, 2, 3, 4
                if ( i == d_RadDifOp->d_globalBox->first[0] ) {
                    // At SOUTH-WEST corner; our WEST and SOUTH neighbors are ghosts
                    // So, stencil entries 0,1 get lumped in with stencil entry 2
                    if ( j == d_RadDifOp->d_globalBox->first[1] ) {
                        cols.assign(  {                           dofs[2], dofs[3], dofs[4] } );
                        valsE.assign( { aS*stnE[0] + aW*stnE[1] + stnE[2], stnE[3], stnE[4] } );
                        valsT.assign( {    stnT[0] +    stnT[1] + stnT[2], stnT[3], stnT[4] } );

                    // At NORTH-WEST corner; our WEST and NORTH neighbors are ghosts
                    // So, stencil entries 1,4 get lumped in with stencil entry 2
                    } else if ( j == d_RadDifOp->d_globalBox->last[1] ) {
                        cols.assign(  { dofs[0],              dofs[2],              dofs[3] } );
                        valsE.assign( { stnE[0], aW*stnE[1] + stnE[2] + aN*stnE[4], stnE[3] } );
                        valsT.assign( { stnT[0],    stnT[1] + stnT[2] +    stnT[4], stnT[3] } );

                    // On interior of WEST boundary; our WEST neighbor is a ghost
                    // So, stencil entry 1 gets lumped in with stencil entry 2
                    } else {
                        cols.assign(  { dofs[0],              dofs[2], dofs[3], dofs[4] } );
                        valsE.assign( { stnE[0], aW*stnE[1] + stnE[2], stnE[3], stnE[4] } );
                        valsT.assign( { stnT[0],    stnT[1] + stnT[2], stnT[3], stnT[4] } );
                    }    
                    
                // On EAST boundary; DOF to our EAST is a ghost.
                // S, W, O, E, N
                // 0, 1, 2, 3, 4
                } else if ( i == d_RadDifOp->d_globalBox->last[0] ) {
                    // At SOUTH-EAST corner; our EAST and SOUTH neighbors are ghosts
                    // So, stencil entries 0,3 get lumped in with stencil entry 2
                    if ( j == d_RadDifOp->d_globalBox->first[1] ) {
                        cols.assign(  { dofs[1],              dofs[2],              dofs[4] } );
                        valsE.assign( { stnE[1], aS*stnE[0] + stnE[2] + aE*stnE[3], stnE[4] } );
                        valsT.assign( { stnT[1],    stnT[0] + stnT[2] +    stnT[3], stnT[4] } );

                    // At NORTH-EAST corner; our EAST and NORTH neighbors are ghosts
                    // So, stencil entries 3,4 get lumped in with stencil entry 2
                    } else if ( j == d_RadDifOp->d_globalBox->last[1] ) {
                        cols.assign(  { dofs[0], dofs[1], dofs[2]                           } );
                        valsE.assign( { stnE[0], stnE[1], stnE[2] + aE*stnE[3] + aN*stnE[4] } );
                        valsT.assign( { stnT[0], stnT[1], stnT[2] +    stnT[3] +    stnT[4] } );

                    // On interior of EAST boundary; our EAST neighbor is a ghost
                    // So, stencil entry 3 gets lumped in with stencil entry 2
                    } else {
                        cols.assign(  { dofs[0], dofs[1], dofs[2],              dofs[4] } );
                        valsE.assign( { stnE[0], stnE[1], stnE[2] + aE*stnE[3], stnE[4] } );
                        valsT.assign( { stnT[0], stnT[1], stnT[2] +    stnT[3], stnT[4] } );
                    }
                    
                // On SOUTH *interior* boundary; DOF to our SOUTH is a ghost.
                // S, W, O, E, N
                // 0, 1, 2, 3, 4
                } else if ( j == d_RadDifOp->d_globalBox->first[1] ) {
                    // On interior of SOUTH boundary; our SOUTH neighbor is a ghost
                    // So, stencil entry 0 gets lumped in with stencil entry 2
                    cols.assign(  { dofs[1],              dofs[2], dofs[3], dofs[4] } );
                    valsE.assign( { stnE[1], aS*stnE[0] + stnE[2], stnE[3], stnE[4] } );
                    valsT.assign( { stnT[1],    stnT[0] + stnT[2], stnT[3], stnT[4] } );

                // On NORTH *interior* boundary; DOF to our NORTH is a ghost.
                // S, W, O, E, N
                // 0, 1, 2, 3, 4
                } else if ( j == d_RadDifOp->d_globalBox->last[1] ) {
                    // On interior of NORTH boundary; our NORTH neighbor is a ghost
                    // So, stencil entry 4 gets lumped in with stencil entry 2
                    cols.assign(  { dofs[0], dofs[1], dofs[2],              dofs[3] } );
                    valsE.assign( { stnE[0], stnE[1], stnE[2] + aN*stnE[4], stnE[3] } );
                    valsT.assign( { stnT[0], stnT[1], stnT[2] +    stnT[4], stnT[3] } );
                }

                // Pack local CSR data for boundary
                localCSRData_d_E[dof_O] = { cols, valsE };
                localCSRData_d_T[dof_O] = { cols, valsT };
                continue;
            }

            // At an interior DOF, both neighbors are active DOFs
            std::vector<size_t> cols = dofs;
            localCSRData_d_E[dof_O] = { cols, stnE };
            localCSRData_d_T[dof_O] = { cols, stnT };
        }
    }

    // Finalize vector construction
    r_EE_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_ET_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_TE_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_TT_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // --- Create matrices
    // Create Lambdas to return col inds from a given row ind
    auto getColumnIDs_d_E = [&](int row) { return localCSRData_d_E[row].cols; };
    auto getColumnIDs_d_T = [&](int row) { return localCSRData_d_T[row].cols; };
    // Create CSR matrices
    auto inVec = r_EE_vec, outVec = r_EE_vec;
    auto d_E_mat = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix", getColumnIDs_d_E );
    auto d_T_mat = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix", getColumnIDs_d_T );
    // Fill matrices with data
    fillMatWithLocalCSRData( d_E_mat, d_RadDifOp->d_scalarDOFMan, localCSRData_d_E );
    fillMatWithLocalCSRData( d_T_mat, d_RadDifOp->d_scalarDOFMan, localCSRData_d_T );
    // Finalize matrix construction
    d_E_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_T_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    d_data->d_E = d_E_mat;
    d_data->d_T = d_T_mat;
}
#endif


#endif