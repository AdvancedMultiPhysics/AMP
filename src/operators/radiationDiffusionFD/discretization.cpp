// Class implementations for discretizations and solvers

#include "discretization.hpp"








#if 1

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
    auto multiDOF = d_RadDifOpJac->d_RadDifOp->d_multiDOFMan;

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







/*
next thing to do is neaten up setData1D in terms of how i overwrite the exiting data etc... then i could make the 2D version, or i could work on the operator split prec. it is probably best to do the operator split stuff in case it requires coming back to rework the stuff implementation of setData (e.g., this business of the dt...)

I + dt*L + dt*R = (I + dt*L)*( I + dt*R ) + O(dt^2)

Suppose I write create 
    A == (I + dt*L) 
    B == dt*R
Then I can still apply A + B, and I can still apply A^{-1} and (I + B)^{-1}

I should have a flag in the jacobian that checks somehow whether we have overwritten L with I + dt*L such that it's apply is not going to work any longer or only if we call it in a special way or something.
*/



/* --------------------------------------
    Implementation of RadDifOpJac 
----------------------------------------- */

std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOpJac::createInputVector() const {
    return d_RadDifOp->createInputVector();
};

// Apply action of the operator utilizing its representation in d_data
/* Picard linearization of a RadDifOp. This LinearOperator has the following structure: 
[ d_E 0   ]   [ diag(r_EE) diag(r_ET) ]
[ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
*/
void RadDifOpJac::applyFromData( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_, std::shared_ptr<AMP::LinearAlgebra::Vector> LET_  ) {

    if ( d_iDebugPrintInfoLevel > 1 )
        AMP::pout << "BERadDifOpJac::applyFromData() " << std::endl;

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
}


/* Populate our Jacobian data, d_data 

Data required to store the Picard linearization of a RadDifOp. The data is stored as two matrices and 4 vectors.
The underlying LinearOperator has the following structure: 
[ d_E 0   ]   [ diag(r_EE) diag(r_ET) ]
[ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
*/
void RadDifOpJac::setData1D(  ) 
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
    double z   = PDE_db->getScalar<double>( "z" );  

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
        d_data       = std::make_shared<RadDifOpJacData>();
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
    auto localBox  = getLocalNodeBox( d_RadDifOp->d_BoxMesh );
    auto globalBox = getGlobalNodeBox( d_RadDifOp->d_BoxMesh );

    // Iterate over local box
    for ( auto i = localBox.first[0]; i <= localBox.last[0]; i++ ) {

        // Get values in the stencil
        auto   ET  = d_RadDifOp->unpackLocalData( E_vec, T_vec, globalBox, i, dofs, onBoundary );
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
        double dif_E = Dr_OE*( E_E - E_O )*rh2 - Dr_WO*( E_O - E_W )*rh2;
        double dif_T = DT_OE*( T_E - T_O )*rh2 - DT_WO*( T_O - T_W )*rh2;
        // Sum diffusion and reaction terms
        double LE = dif_E + ( REE*E_O + RET*T_O );
        double LT = dif_T + ( RTE*E_O + RTT*T_O );
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
            if ( i == globalBox.first[0] ) {
                // Compute energy diffusion coefficient on boundary
                double Dr_WO  = d_RadDifOp->energyDiffusionCoefficientAtMidPoint( T_W , T_O ); // T_{i-1/2}
                double c1     = k11 * Dr_WO;
                // Get Picard coefficient
                double alpha1 = d_RadDifOp->ghostValueRobinEPicardCoefficient( c1, 1 );

                std::vector<size_t> cols  = {                  dofs[1], dofs[2] };
                std::vector<double> valsE = { alpha1*stnE[0] + stnE[1], stnE[2] };
                std::vector<double> valsT = {        stnT[0] + stnT[1], stnT[2] };
                localCSRData_d_E[dof_O] = { cols, valsE };
                localCSRData_d_T[dof_O] = { cols, valsT };
                continue;
            }

            // On EAST boundary; DOF to our EAST is ghost.
            if ( i == globalBox.last[0] ) {
                // Compute energy diffusion coefficient on boundary
                double Dr_OE  = d_RadDifOp->energyDiffusionCoefficientAtMidPoint( T_O , T_E ); // T_{i+1/2}
                double c2     = k11 * Dr_OE;
                // Get Picard coefficient
                double alpha2 = d_RadDifOp->ghostValueRobinEPicardCoefficient( c2, 2 );

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



void RadDifOpJac::setData2D(  ) 
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
    double z   = PDE_db->getScalar<double>( "z" );  

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
        d_data       = std::make_shared<RadDifOpJacData>();
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
    auto localBox  = getLocalNodeBox( d_RadDifOp->d_BoxMesh );
    auto globalBox = getGlobalNodeBox( d_RadDifOp->d_BoxMesh );

    // Iterate over local box
    for ( auto j = localBox.first[1]; j <= localBox.last[1]; j++ ) {
        for ( auto i = localBox.first[0]; i <= localBox.last[0]; i++ ) {

            // Get values in the stencil; S, W, O, E, N
            auto   ET  = d_RadDifOp->unpackLocalData( E_vec, T_vec, globalBox, i, j, dofs, onBoundary );
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
            double dif_E = Dr_OE*( E_E - E_O )*rh2 - Dr_WO*( E_O - E_W )*rh2 \
                    + Dr_ON*( E_N - E_O )*rh2 - Dr_SO*( E_O - E_S )*rh2;
            double dif_T = DT_OE*( T_E - T_O )*rh2 - DT_WO*( T_O - T_W )*rh2 \
                    + DT_ON*( T_N - T_O )*rh2 - DT_SO*( T_O - T_S )*rh2;
            // Sum diffusion and reaction terms
            double LE = dif_E + ( REE*E_O + RET*T_O );
            double LT = dif_T + ( RTE*E_O + RTT*T_O );
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
                double c1 = k11 * d_RadDifOp->energyDiffusionCoefficientAtMidPoint( T_W, T_O );
                double c2 = k11 * d_RadDifOp->energyDiffusionCoefficientAtMidPoint( T_O, T_E );
                double c3 = k11 * d_RadDifOp->energyDiffusionCoefficientAtMidPoint( T_S, T_O );
                double c4 = k11 * d_RadDifOp->energyDiffusionCoefficientAtMidPoint( T_O, T_N );
                // ak == alphak (abuse of notation; these are not the ak constants in Robin BCs)
                double aW = d_RadDifOp->ghostValueRobinEPicardCoefficient( c1, 1 );
                double aE = d_RadDifOp->ghostValueRobinEPicardCoefficient( c2, 2 );
                double aS = d_RadDifOp->ghostValueRobinEPicardCoefficient( c3, 3 );
                double aN = d_RadDifOp->ghostValueRobinEPicardCoefficient( c4, 4 );

                // Vectors we're going to fill
                std::vector<size_t> cols;
                std::vector<double> valsT;
                std::vector<double> valsE;

                // On WEST boundary; DOF to our WEST is a ghost. 
                // S, W, O, E, N
                // 0, 1, 2, 3, 4
                if ( i == globalBox.first[0] ) {
                    // At SOUTH-WEST corner; our WEST and SOUTH neighbors are ghosts
                    // So, stencil entries 0,1 get lumped in with stencil entry 2
                    if ( j == globalBox.first[1] ) {
                        cols.assign(  {                           dofs[2], dofs[3], dofs[4] } );
                        valsE.assign( { aS*stnE[0] + aW*stnE[1] + stnE[2], stnE[3], stnE[4] } );
                        valsT.assign( {    stnT[0] +    stnT[1] + stnT[2], stnT[3], stnT[4] } );

                    // At NORTH-WEST corner; our WEST and NORTH neighbors are ghosts
                    // So, stencil entries 1,4 get lumped in with stencil entry 2
                    } else if ( j == globalBox.last[1] ) {
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
                } else if ( i == globalBox.last[0] ) {
                    // At SOUTH-EAST corner; our EAST and SOUTH neighbors are ghosts
                    // So, stencil entries 0,3 get lumped in with stencil entry 2
                    if ( j == globalBox.first[1] ) {
                        cols.assign(  { dofs[1],              dofs[2],              dofs[4] } );
                        valsE.assign( { stnE[1], aS*stnE[0] + stnE[2] + aE*stnE[3], stnE[4] } );
                        valsT.assign( { stnT[1],    stnT[0] + stnT[2] +    stnT[3], stnT[4] } );

                    // At NORTH-EAST corner; our EAST and NORTH neighbors are ghosts
                    // So, stencil entries 3,4 get lumped in with stencil entry 2
                    } else if ( j == globalBox.last[1] ) {
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
                } else if ( j == globalBox.first[1] ) {
                    // On interior of SOUTH boundary; our SOUTH neighbor is a ghost
                    // So, stencil entry 0 gets lumped in with stencil entry 2
                    cols.assign(  { dofs[1],              dofs[2], dofs[3], dofs[4] } );
                    valsE.assign( { stnE[1], aS*stnE[0] + stnE[2], stnE[3], stnE[4] } );
                    valsT.assign( { stnT[1],    stnT[0] + stnT[2], stnT[3], stnT[4] } );

                // On NORTH *interior* boundary; DOF to our NORTH is a ghost.
                // S, W, O, E, N
                // 0, 1, 2, 3, 4
                } else if ( j == globalBox.last[1] ) {
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


/* Updates frozen value of current solution. I.e., when this is called, the d_frozenSolution vector in params is the current approximation to the outer nonlinear problem.
    Also direct d_data to null, indicating it's out of date
*/
void RadDifOpJac::reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ ) {

    if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "RadDifOpJac::reset() " << std::endl;

    //AMP_WARNING( "Come back here... Either is this or the BE wrapper I need to implement a check to see if the vector has actually changed since multiple resets per Newton iteration are called... Maybe in the BEWrapper makes more sense, since there I could check if the time-step has changed too? Well. How am I going to check if the solution vector has been updated because it's pointer is probably always the same..." );

    AMP_ASSERT( params_ );
    // Downcast OperatorParameters to their derived class
    auto params = std::dynamic_pointer_cast<const RadDifOpJacParameters>( params_ );
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


/* --------------------------------------
        Implementation of RadDifOp 
----------------------------------------- */

/* Populate vector with function that takes an int representing the component and a reference to a MeshElement and returns a double.  */
void RadDifOp::fillMultiVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> vec_, std::function<double( int, AMP::Mesh::MeshElement & )> fun ) {

    // Unpack multiVector
    auto vec = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( vec_ );
    AMP_INSIST( vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto vec0 = vec->getVector(0);
    auto vec1 = vec->getVector(1);

    double u0, u1;
    auto it = d_BoxMesh->getIterator( d_geomType ); // Mesh iterator
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

/* Build and set d_multiDOFMan and d_scalarDOFMan */
void RadDifOp::setDOFManagers() {

    // Specify mesh is to use cell-based geometry
    AMP::Mesh::GeomType myGeomType = AMP::Mesh::GeomType::Vertex;
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

    #if 0
    comm.barrier();
    // This demonstrates how DOFs are organized on the mesh by multiDOFManager 
    // Iterate through the mesh, and pull out DOFs associated with each mesh element from the multiDOF
    auto iter = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
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


    // This is how we get nodal ordering rather than variable ordering
    d_nodalDOFMan = AMP::Discretization::boxMeshDOFManager::create(mesh, myGeomType, gcw, 2);
    #if 0
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

/*
I could get an index, map it to a MeshElementID, and then get DOFs from that MeshElementID
*/


// E and T must be non-negative
bool RadDifOp::isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET ) {
    // auto min = ET->min();
    // AMP::pout << "RadDifOp::isValidVector: min=" << min << "\n";
    return ( ET->min() >= 0.0 ); 
}


// Apply operator L to vector ET. This holds for both linear and nonlinear RadDifOp
/* The operator is computed as
            diffusion + reaction == 
            [ d_E 0   ][E]   [ diag(r_EE) diag(r_ET) ][E]
            [ 0   d_T ][T] + [ diag(r_TE) diag(r_TT) ][T]
        where: 
            1. d_E*E        = -k11 * div grad( DE*E ) 
            2. diag(r_EE)*E = -k12 * diag(REE) * E
            3. diag(r_ET)*T = -k12 * diag(RET) * T
            
            4. d_T*T        = -k21 * div grad( DT*T ) 
            5. diag(r_TE)*E = +k22 * diag(RTE) * E
            6. diag(r_TT)*T = +k22 * diag(RTT) * T

        In the nonlinear model:
            REE = RTE = -sigma
            RET = RTT = +sigma*T^3
        In the linear model:
            REE = RTE = -1
            RET = RTT = +1
*/
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
    double z   = PDE_db->getScalar<double>( "z" );  

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
    auto localBox  = getLocalNodeBox( d_BoxMesh );
    auto globalBox = getGlobalNodeBox( d_BoxMesh );

    // Iterate over local box
    for ( auto i = localBox.first[0]; i <= localBox.last[0]; i++ ) {

        // Get values in the stencil
        auto   ET  = unpackLocalData( E_vec, T_vec, globalBox, i );
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
        double dif_E = Dr_OE*( E_E - E_O )*rh2 - Dr_WO*( E_O - E_W )*rh2;
        double dif_T = DT_OE*( T_E - T_O )*rh2 - DT_WO*( T_O - T_W )*rh2;
        // Sum diffusion and reaction terms
        double LE = dif_E + ( REE*E_O + RET*T_O );
        double LT = dif_T + ( RTE*E_O + RTT*T_O );

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
    double z   = PDE_db->getScalar<double>( "z" );  

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
    auto localBox  = getLocalNodeBox( d_BoxMesh );
    auto globalBox = getGlobalNodeBox( d_BoxMesh );

    // Iterate over local box
    for ( auto j = localBox.first[1]; j <= localBox.last[1]; j++ ) {
        for ( auto i = localBox.first[0]; i <= localBox.last[0]; i++ ) {

            // Get values in the stencil
            auto   ET  = unpackLocalData( E_vec, T_vec, globalBox, i, j );
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
            double dif_E = Dr_OE*( E_E - E_O )*rh2 - Dr_WO*( E_O - E_W )*rh2 \
                    + Dr_ON*( E_N - E_O )*rh2 - Dr_SO*( E_O - E_S )*rh2;
            double dif_T = DT_OE*( T_E - T_O )*rh2 - DT_WO*( T_O - T_W )*rh2 \
                    + DT_ON*( T_N - T_O )*rh2 - DT_SO*( T_O - T_S )*rh2;
            // Sum diffusion and reaction terms
            double LE = dif_E + ( REE*E_O + RET*T_O );
            double LT = dif_T + ( RTE*E_O + RTT*T_O );

            // Insert values into the vectors
            LE_vec->setValueByGlobalID<double>( dof_O, LE );
            LT_vec->setValueByGlobalID<double>( dof_O, LT );
        }
    }
    LET_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

/* Get the ghost values on the given boundary at the given node given the interior value Eint and Tint. This is assuming Robin on E and pseudo-Neumann on T */
std::vector<double> RadDifOp::getGhostValues( int boundary, AMP::Mesh::MeshElement &node, double Eint, double Tint )
{

    // Get the Robin constants for the given boundary
    double ak, bk; 
    if ( boundary == 1 ) {
        ak = d_db->getDatabase( "PDE" )->getScalar<double>( "a1" );
        bk = d_db->getDatabase( "PDE" )->getScalar<double>( "b1" );
    } else if ( boundary == 2 ) {
        ak = d_db->getDatabase( "PDE" )->getScalar<double>( "a2" );
        bk = d_db->getDatabase( "PDE" )->getScalar<double>( "b2" );
    } else if ( boundary == 3 ) {
        ak = d_db->getDatabase( "PDE" )->getScalar<double>( "a3" );
        bk = d_db->getDatabase( "PDE" )->getScalar<double>( "b3" );
    } else if ( boundary == 4 ) {
        ak = d_db->getDatabase( "PDE" )->getScalar<double>( "a4" );
        bk = d_db->getDatabase( "PDE" )->getScalar<double>( "b4" );
    } else {
        AMP_ERROR( "Invalid boundary" );
    }
    // Now get the corresponding Robin value
    double rk = d_robinFunctionE( boundary, ak, bk, node );
    // Get Neumann value
    double nk = d_pseudoNeumannFunctionT( boundary, node );
    
    // Solve for ghost values given these constants
    return ghostValuesSolve( ak, bk, rk, nk, Eint, Tint );
}


double RadDifOp::pseudoNeumannFunctionTDefault( int boundary ){
    if ( boundary == 1 ) {
        return d_db->getDatabase( "PDE" )->getScalar<double>( "n1" );
    } else if ( boundary == 2 ) {
        return d_db->getDatabase( "PDE" )->getScalar<double>( "n2" );
    } else if ( boundary == 3 ) {
        return d_db->getDatabase( "PDE" )->getScalar<double>( "n3" );
    } else if ( boundary == 4 ) {
        return d_db->getDatabase( "PDE" )->getScalar<double>( "n4" );
    } else { 
        AMP_ERROR( "Invalid boundary" );
    }
}

double RadDifOp::robinFunctionEDefault( int boundary ){
    if ( boundary == 1 ) {
        return d_db->getDatabase( "PDE" )->getScalar<double>( "r1" );
    } else if ( boundary == 2 ) {
        return d_db->getDatabase( "PDE" )->getScalar<double>( "r2" );
    } else if ( boundary == 3 ) {
        return d_db->getDatabase( "PDE" )->getScalar<double>( "r3" );
    } else if ( boundary == 4 ) {
        return d_db->getDatabase( "PDE" )->getScalar<double>( "r4" );
    } else { 
        AMP_ERROR( "Invalid boundary" );
    }
}


/* Suppose on boundary k we have the two equations:

    ak*E + bk * \hat{n}_k \dot k11 D_E(T) \grad E = rk
                \hat{n}_k \dot            \grad T = nK,

where ak, bk, rk, and nk are all known constants; \hat{n}_k is the outward-facing normal vector at the boundary.

The discretization of these conditions involves one ghost point for E and T (Eg and Tg), and one interior point (Eint and Tint). Here we solve for the ghost points and return them. Note that this system, although nonlinear, can be solved by forward substitution 

Note that the actual boundary does not matter here once the constants a, b, r, and n have been specified.
    */
std::vector<double> RadDifOp::ghostValuesSolve( double a, double b, double r, double n, double Eint, double Tint ){

    // Unpack parameters
    auto PDE_db = d_db->getDatabase( "PDE" );
    double k11  = PDE_db->getScalar<double>( "k11" );
    double z    = PDE_db->getScalar<double>( "z" );

    // Solve for Tg
    double Tg = ghostValuePseudoNeumannTSolve( n, Tint );

    // Compute energy diffusion coefficient on the boundary
    double D_E = energyDiffusionCoefficientAtMidPoint( Tg, Tint );
    double c   = k11 * D_E;  

    // Solve for Eg
    double Eg = ghostValueRobinESolve( a, b, r, c, Eint );

    std::vector<double> ETg = { Eg, Tg };
    return ETg;
}

/* Compute the energy diffusion coefficient D_E at the mid-point between T_left and T_right.
For the nonlinear problem:
    D_E = 1/3*sigma(T_midpoint), where sigma(T) = (z/T)^3
For the linear    problem:
    D_E = 1
This function is not used everywhere in the code, but is in somplaces for readability.
*/
double RadDifOp::energyDiffusionCoefficientAtMidPoint( double T_left, double T_right) {
    auto PDE_db = d_db->getDatabase( "PDE" );
    double z    = PDE_db->getScalar<double>( "z" );
    double D_E; 
    if ( PDE_db->getScalar<std::string>( "model" ) == "nonlinear" ) {
        double sigma = std::pow( z/( 0.5*( T_left + T_right ) ), 3.0 ); // Sample T at mid point, i.e., boundary
        D_E = 1.0/(3*sigma);
    } else {
        D_E = 1.0;
    }
    return D_E; 
}

/* Suppose on boundary k we have the equation:
        \hat{n}_k \dot \grad T = nK,
where nk is a known constant and \hat{n}_k is the outward-facing normal vector at the boundary.

The discretization of this conditions involves one ghost point for T (Tg), and one interior point (Tint). Here we solve for the ghost point and return it
*/
double RadDifOp::ghostValuePseudoNeumannTSolve( double n, double Tint ) {
    // Unpack parameters
    double h = d_db->getDatabase( "mesh" )->getScalar<double>( "h" );
    /* Independent of the boundary, we get Tg = Tint * h*n, as follows:
    West boundary: \hat{n}_k = -\hat{x} 
    -( T1j - Tg )/h = n ---> Tg = Ti1 + h*n;
    East boundary: \hat{n}_k = +\hat{x}
    +( Tg - T2j )/h = n ---> Tg = Ti2 + h*n;
    South boundary: \hat{n}_k = -\hat{y}
    -( Ti3 - Tg )/h = n ---> Tg = Ti3 + h*n;
    North boundary: \hat{n}_k = +\hat{y}
    -( Ti4 - Tg )/h = n ---> Tg = Ti4 + h*n; */
    return Tint + h*n;
}

/* Suppose on boundary k we have the equation:
    ak*E + bk * \hat{n}_k \dot ck \grad E = rk
where ak, bk, rk, nk, and ck are all known constants; \hat{n}_k is the outward-facing normal vector at the boundary.

The discretization of these conditions involves one ghost point for E (Eg), and one interior point (Eint). Here we solve for the ghost point and return it.
*/
double RadDifOp::ghostValueRobinESolve( double a, double b, double r, double c, double Eint ) {

    /* Independent of the boundary we get the same result */
    double h  = d_db->getDatabase( "mesh" )->getScalar<double>( "h" );
    double Eg = (2*Eint*c*b - Eint*a*h + 2*h*r)/(2*c*b + a*h);
    return Eg;
}

/* In the Robin BC for energy we get Eghost = alpha*Eint + beta, this function returns the coefficient alpha, which is the Picard linearization of this equation w.r.t Eint */
// ck = k11*DE( 0.5*( Tghost + Tint ) )
double RadDifOp::ghostValueRobinEPicardCoefficient( double ck, size_t boundary ) const {

    double h  = d_db->getDatabase( "mesh" )->getScalar<double>( "h" );
    // Get the Robin constants for the given boundary
    double ak, bk; 
    if ( boundary == 1 ) {
        ak = d_db->getDatabase( "PDE" )->getScalar<double>( "a1" );
        bk = d_db->getDatabase( "PDE" )->getScalar<double>( "b1" );
    } else if ( boundary == 2 ) {
        ak = d_db->getDatabase( "PDE" )->getScalar<double>( "a2" );
        bk = d_db->getDatabase( "PDE" )->getScalar<double>( "b2" );
    } else if ( boundary == 3 ) {
        ak = d_db->getDatabase( "PDE" )->getScalar<double>( "a3" );
        bk = d_db->getDatabase( "PDE" )->getScalar<double>( "b3" );
    } else if ( boundary == 4 ) {
        ak = d_db->getDatabase( "PDE" )->getScalar<double>( "a4" );
        bk = d_db->getDatabase( "PDE" )->getScalar<double>( "b4" );
    } else {
        AMP_ERROR( "Invalid boundary" );
    }

    double alpha = ( 2 * ck * bk - ak * h ) / ( 2 * ck * bk + ak * h );
    return alpha;
}



// Get all values in the 3-point stencil centered at grid index i
std::vector<double> RadDifOp::unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, AMP::Mesh::BoxMesh::Box &globalBox, int i ) {

    double E_W, E_O, E_E; 
    double T_W, T_O, T_E;

    // The current DOF
    size_t dof_O = gridIndsToScalarDOF( i );
    E_O = E_vec->getValueByGlobalID<double>( dof_O );
    T_O = T_vec->getValueByGlobalID<double>( dof_O );

    // At WEST boundary (1), so WEST neighbor is a ghost
    if ( i == globalBox.first[0] ) {
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
    if ( i == globalBox.last[0] ) {
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


/* Get all values in the 3-point stencil centered at grid index i
 dofs will be returned with W, O, E
 onBoundary is a flag indicating whether the DOF is on a boundary. 
 In the case that it is on a boundary, values in dofs corresponding to the boundaries are not meaningful
 */
std::vector<double> RadDifOp::unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, AMP::Mesh::BoxMesh::Box &globalBox, int i, std::vector<size_t> &dofs, bool &onBoundary ) {

    onBoundary = false;
    double E_W, E_O, E_E; 
    double T_W, T_O, T_E;

    // The current DOF
    dofs[1] = gridIndsToScalarDOF( i );
    E_O = E_vec->getValueByGlobalID<double>( dofs[1] );
    T_O = T_vec->getValueByGlobalID<double>( dofs[1] );

    // At WEST boundary (1), so WEST neighbor is a ghost
    if ( i == globalBox.first[0] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i );
        auto ETg   = getGhostValues( 1, node, E_O, T_O );
        E_W        = ETg[0]; 
        T_W        = ETg[1];
        

    // At interior DOF; WEST neighbor is an interior DOF
    } else {
        dofs[0] = gridIndsToScalarDOF( i-1 );
        E_W     = E_vec->getValueByGlobalID( dofs[0] ); 
        T_W     = T_vec->getValueByGlobalID( dofs[0] );
    }

    // At EAST boundary (2), so EAST neighbor is a ghost
    if ( i == globalBox.last[0] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i );
        auto ETg   = getGhostValues( 2, node, E_O, T_O );
        E_E        = ETg[0]; 
        T_E        = ETg[1];

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
std::vector<double> RadDifOp::unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, AMP::Mesh::BoxMesh::Box &globalBox, int i, int j ) {

    double E_S, E_W, E_O, E_E, E_N; 
    double T_S, T_W, T_O, T_E, T_N;

    // The current DOF
    size_t dof_O = gridIndsToScalarDOF( i, j );
    E_O = E_vec->getValueByGlobalID<double>( dof_O );
    T_O = T_vec->getValueByGlobalID<double>( dof_O );

    // Get WEST neighboring value
    // At WEST boundary (1), so WEST neighbor is a ghost
    if ( i == globalBox.first[0] ) {
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
    if ( i == globalBox.last[0] ) {
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
    if ( j == globalBox.first[1] ) {
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
    if ( j == globalBox.last[1] ) {
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



// Get all values in the 5-point stencil centered at grid index i,j
// S, W, O, E, N
// 0, 1, 2, 3, 4
std::vector<double> RadDifOp::unpackLocalData( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec, std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec, AMP::Mesh::BoxMesh::Box &globalBox, int i, int j, std::vector<size_t> &dofs, bool &onBoundary ) {

    onBoundary = false;
    double E_S, E_W, E_O, E_E, E_N; 
    double T_S, T_W, T_O, T_E, T_N;

    // The current DOF
    dofs[2] = gridIndsToScalarDOF( i, j );
    E_O = E_vec->getValueByGlobalID<double>( dofs[2] );
    T_O = T_vec->getValueByGlobalID<double>( dofs[2] );

    // Get WEST neighboring value
    // At WEST boundary (1), so WEST neighbor is a ghost
    if ( i == globalBox.first[0] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i, j );
        auto ETg   = getGhostValues( 1, node, E_O, T_O );
        E_W        = ETg[0];
        T_W        = ETg[1];

    // At interior DOF; WEST neighbor is an interior DOF
    } else {
        dofs[1] = gridIndsToScalarDOF( i-1, j );
        E_W     = E_vec->getValueByGlobalID( dofs[1] ); 
        T_W     = T_vec->getValueByGlobalID( dofs[1] );
    }

    // Get EAST neighboring value
    // At EAST boundary (2), so EAST neighbor is a ghost
    if ( i == globalBox.last[0] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i, j );
        auto ETg   = getGhostValues( 2, node, E_O, T_O );
        E_E        = ETg[0]; 
        T_E        = ETg[1];

    // At interior DOF; EAST neighbor is an interior DOF
    } else {
        dofs[3] = gridIndsToScalarDOF( i+1, j );
        E_E     = E_vec->getValueByGlobalID( dofs[3] ); 
        T_E     = T_vec->getValueByGlobalID( dofs[3] );
    }

    // Get SOUTH neighboring value
    // At SOUTH boundary (3), so SOUTH neighbor is a ghost
    if ( j == globalBox.first[1] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i, j );
        auto ETg   = getGhostValues( 3, node, E_O, T_O );
        E_S        = ETg[0]; 
        T_S        = ETg[1];

    // At interior DOF; SOUTH neighbor is an interior DOF
    } else {
        dofs[0] = gridIndsToScalarDOF( i, j-1 );
        E_S     = E_vec->getValueByGlobalID( dofs[0] ); 
        T_S     = T_vec->getValueByGlobalID( dofs[0] );
    }

    // Get NORTH neighboring value
    // At NORTH boundary (4), so NORTH neighbor is a ghost
    if ( j == globalBox.last[1] ) {
        onBoundary = true;
        auto node  = gridIndsToMeshElement( i, j );
        auto ETg   = getGhostValues( 4, node, E_O, T_O );
        E_N        = ETg[0]; 
        T_N        = ETg[1];

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
