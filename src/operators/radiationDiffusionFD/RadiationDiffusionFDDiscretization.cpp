#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"

namespace AMP::Operator {


/** ---------------------------------------------------------- *
 *  ----------- Implementation of RadDifCoefficients --------- *
 *  ---------------------------------------------------------- */

inline double RadDifCoefficients::diffusionE( double k11, double T, double zatom )
{
    if constexpr ( IsNonlinear ) {
        double zByT  = zatom / T;
        double sigma = zByT * zByT * zByT;
        return k11 / ( 3.0 * sigma );
    } else {
        return k11;
    }
}

inline double RadDifCoefficients::diffusionT( double k21, double T )
{
    if constexpr ( IsNonlinear ) {
        return k21 * T * T * std::sqrt( T ); // T^2.5
    } else {
        return k21;
    }
}

inline void RadDifCoefficients::reaction( double k12,
                                          double k22,
                                          double T,
                                          double zatom,
                                          double &REE,
                                          double &RET,
                                          double &RTE,
                                          double &RTT )
{
    if constexpr ( IsNonlinear ) {
        double Tcube = T * T * T;
        double sigma = std::pow( zatom / T, 3.0 );
        REE = RTE = -sigma;
        RET = RTT = +sigma * Tcube;
    } else {
        REE = RTE = -1.0;
        RET = RTT = +1.0;
    }

    REE *= -k12;
    RET *= -k12;
    RTE *= k22;
    RTT *= k22;
}


/** ------------------------------------------------- *
 *  ----------- Implementation of FDMeshOps --------- *
 *  ------------------------------------------------- */
void FDMeshOps::createMeshData(
    std::shared_ptr<AMP::Mesh::Mesh> mesh,
    std::shared_ptr<AMP::Mesh::BoxMesh> &d_BoxMesh,
    size_t &d_dim,
    AMP::Mesh::GeomType &d_CellCenteredGeom,
    std::shared_ptr<AMP::Discretization::DOFManager> &d_scalarDOFMan,
    std::shared_ptr<AMP::Discretization::multiDOFManager> &d_multiDOFMan,
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> &d_globalBox,
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> &d_localBox,
    std::shared_ptr<AMP::ArraySize> &d_localArraySize,
    std::vector<double> &d_h,
    std::vector<double> &d_rh2 )
{

    // Keep a pointer to my BoxMesh to save having to do this downcast repeatedly
    d_BoxMesh = std::dynamic_pointer_cast<AMP::Mesh::BoxMesh>( mesh );
    AMP_INSIST( d_BoxMesh, "Mesh must be a AMP::Mesh::BoxMesh" );

    d_dim = d_BoxMesh->getDim();
    AMP_INSIST( d_dim == 1 || d_dim == 2 || d_dim == 3,
                "Invalid dimension: dim=" + std::to_string( d_dim ) +
                    std::string( " !in {1,2,3}" ) );

    // Set the geometry type that gives us cell centered data
    if ( d_dim == 1 ) {
        d_CellCenteredGeom = AMP::Mesh::GeomType::Edge;
    } else if ( d_dim == 2 ) {
        d_CellCenteredGeom = AMP::Mesh::GeomType::Face;
    } else {
        d_CellCenteredGeom = AMP::Mesh::GeomType::Cell;
    }


    // Set DOFManagers
    createDOFManagers( d_CellCenteredGeom, d_BoxMesh, d_scalarDOFMan, d_multiDOFMan );
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
        AMP_INSIST( boundaryIDs[2 * dim] == int( 2 * dim + 1 ), "Invalid boundaryID" );
        AMP_INSIST( boundaryIDs[2 * dim + 1] == int( 2 * dim + 2 ), "Invalid boundaryID" );
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
    // Set local array size
    d_localArraySize = std::make_shared<AMP::ArraySize>( d_localBox->size() );

    // There are nk cells in dimension k, nk = d_globalBox.last[k] - d_globalBox.first[k]+1, such
    // that the mesh spacing is hk = (xkMax - xkMin)/nk
    d_h.resize( 0 );
    for ( size_t k = 0; k < d_dim; k++ ) {
        auto nk    = d_globalBox->last[k] - d_globalBox->first[k] + 1;
        auto xkMin = range[2 * k];
        auto xkMax = range[2 * k + 1];
        d_h.push_back( ( xkMax - xkMin ) / nk );
    }
    d_rh2.resize( 0 );
    // Compute reciprocal square of mesh spacing
    for ( auto h : d_h ) {
        d_rh2.push_back( 1.0 / ( h * h ) );
    }
}

void FDMeshOps::createDOFManagers(
    const AMP::Mesh::GeomType &Geom,
    std::shared_ptr<AMP::Mesh::BoxMesh> &mesh,
    std::shared_ptr<AMP::Discretization::DOFManager> &scalarDOFMan,
    std::shared_ptr<AMP::Discretization::multiDOFManager> &multiDOFMan )
{

    // Number of DOFs per mesh element (make 1, even though we have two variables. We'll create
    // separate DOF managers for them)
    int myDOFsPerElement = 1;
    int gcw              = 1; // Ghost-cell width; stencils are 3-point
    auto comm            = mesh->getComm();

    // E and T use the same DOFManager under the hood
    scalarDOFMan =
        AMP::Discretization::boxMeshDOFManager::create( mesh, Geom, gcw, myDOFsPerElement );
    auto T_DOFManager = scalarDOFMan;
    auto E_DOFManager = scalarDOFMan;

    // Create a multiDOFManager that wraps both DOF managers
    std::vector<std::shared_ptr<AMP::Discretization::DOFManager>> DOFManagersVec = { E_DOFManager,
                                                                                     T_DOFManager };
    multiDOFMan =
        std::make_shared<AMP::Discretization::multiDOFManager>( comm, DOFManagersVec, mesh );
}

template<bool computeE, bool computeT>
void FDMeshOps::FaceDiffusionCoefficients( std::array<double, 3> &ELoc3,
                                           std::array<double, 3> &TLoc3,
                                           double k11,
                                           double k21,
                                           double zatom,
                                           double h,
                                           double *Dr_WO,
                                           double *Dr_OE,
                                           double *DT_WO,
                                           double *DT_OE )
{

    // Compute temp at mid points
    double T_WO = 0.5 * ( TLoc3[WEST] + TLoc3[ORIGIN] ); // T_{i-1/2}
    double T_OE = 0.5 * ( TLoc3[ORIGIN] + TLoc3[EAST] ); // T_{i+1/2}

    // Get diffusion coefficients at cell faces, i.e., mid points
    // Energy
    if constexpr ( computeE ) {
        *Dr_WO = RadDifCoefficients::diffusionE( k11, T_WO, zatom );
        *Dr_OE = RadDifCoefficients::diffusionE( k11, T_OE, zatom );
        // Limit the energy flux if need be, eq. (17)
        // This is also the same as eq. (9) in "An efficient nonlinear solution method for
        // non-equilibrium radiation diffusion by D.A. Knoll, W.J. Rider, G.L. Olson"
        if constexpr ( IsFluxLimited ) {
            double DE_WO =
                *Dr_WO / ( 1.0 + *Dr_WO * ( abs( ELoc3[ORIGIN] - ELoc3[WEST] ) /
                                            ( h * 0.5 * ( ELoc3[ORIGIN] + ELoc3[WEST] ) ) ) );
            double DE_OE =
                *Dr_OE / ( 1.0 + *Dr_OE * ( abs( ELoc3[EAST] - ELoc3[ORIGIN] ) /
                                            ( h * 0.5 * ( ELoc3[EAST] + ELoc3[ORIGIN] ) ) ) );
            *Dr_WO = DE_WO;
            *Dr_OE = DE_OE;
        }
    }

    // Temperature
    if constexpr ( computeT ) {
        *DT_WO = RadDifCoefficients::diffusionT( k21, T_WO );
        *DT_OE = RadDifCoefficients::diffusionT( k21, T_OE );
    }
}


/** --------------------------------------------------------------- *
 *  ----------- Implementation of FDMeshGlobalIndexingOps --------- *
 *  --------------------------------------------------------------- */
FDMeshGlobalIndexingOps::FDMeshGlobalIndexingOps(
    std::shared_ptr<AMP::Mesh::BoxMesh> BoxMesh,
    AMP::Mesh::GeomType &geom,
    std::shared_ptr<AMP::Discretization::DOFManager> scalarDOFMan,
    std::shared_ptr<AMP::Discretization::multiDOFManager> multiDOFMan )
    : d_BoxMesh( BoxMesh ),
      d_geom( geom ),
      d_scalarDOFMan( scalarDOFMan ),
      d_multiDOFMan( multiDOFMan ){};


//! Map from grid index to a the corresponding DOF
size_t FDMeshGlobalIndexingOps::gridIndsToScalarDOF( const std::array<size_t, 3> &ijk ) const
{
    AMP::Mesh::BoxMesh::MeshElementIndex ind( d_geom, 0, ijk[0], ijk[1], ijk[2] );
    AMP::Mesh::MeshElementID id = d_BoxMesh->convert( ind );
    std::vector<size_t> dof;
    d_scalarDOFMan->getDOFs( id, dof );
    return dof[0];
}

//! Map from grid index to a MeshElement
AMP::Mesh::structuredMeshElement
FDMeshGlobalIndexingOps::gridIndsToMeshElement( const std::array<size_t, 3> &ijk ) const
{
    AMP::Mesh::BoxMesh::MeshElementIndex ind( d_geom, 0, ijk[0], ijk[1], ijk[2] );
    return d_BoxMesh->getElement( ind );
}

//! Map from scalar DOF to grid indices i, j, k
std::array<size_t, 3> FDMeshGlobalIndexingOps::scalarDOFToGridInds( size_t dof ) const
{
    // Get ElementID
    AMP::Mesh::MeshElementID id = d_scalarDOFMan->getElementID( dof );
    // Convert ElementID into a MeshElementIndex
    AMP::Mesh::BoxMesh::MeshElementIndex ind = d_BoxMesh->convert( id );
    // Get grid index along each component direction
    std::array<int, 3> idx = ind.index();
    // Note that grid is not periodic, so indices are non-negative
    return { static_cast<size_t>( idx[0] ),
             static_cast<size_t>( idx[1] ),
             static_cast<size_t>( idx[2] ) };
}


/** ------------------------------------------------------- *
 *  ----------- Implementation of FDBoundaryUtils --------- *
 *  ------------------------------------------------------- */
size_t FDBoundaryUtils::getBoundaryIDFromDim( size_t dim, BoundarySide side )
{
    if ( side == BoundarySide::WEST ) {
        return 2 * dim + 1;
    } else if ( side == BoundarySide::EAST ) {
        return 2 * dim + 2;
    } else {
        AMP_ERROR( "Invalid side" );
    }
}

size_t FDBoundaryUtils::getDimFromBoundaryID( size_t boundaryID )
{
    AMP_INSIST( boundaryID >= 1 && boundaryID <= 6, "boundaryID not recognised" );
    return ( boundaryID - 1 ) / 2; // Note the integer division
}

double FDBoundaryUtils::ghostValueSolveT( double n, double h, double Tint )
{
    double alpha = 1.0;
    double beta  = h * n;
    double Tg    = alpha * Tint + beta;
    return Tg;
}

double
FDBoundaryUtils::ghostValueSolveE( double a, double b, double r, double c, double h, double Eint )
{
    double alpha = ( 2 * c * b - a * h ) / ( 2 * c * b + a * h );
    double beta  = 2 * h * r / ( 2 * c * b + a * h );
    double Eg    = alpha * Eint + beta;
    return Eg;
}

void FDBoundaryUtils::ghostValuesSolve( double a,
                                        double b,
                                        const std::function<double( double T )> &cHandle,
                                        double r,
                                        double n,
                                        double h,
                                        double Eint,
                                        double Tint,
                                        double &Eg,
                                        double &Tg )
{

    // Solve for Tg
    Tg = FDBoundaryUtils::ghostValueSolveT( n, h, Tint );

    // Compute energy diffusion coefficient on the boundary, i.e., the mid-point between Tg and Tint
    double T_midpoint = 0.5 * ( Tg + Tint );
    auto c            = cHandle( T_midpoint );

    // Solve for Eg
    Eg = FDBoundaryUtils::ghostValueSolveE( a, b, r, c, h, Eint );
}

void FDBoundaryUtils::getBCConstantsFromDB(
    const AMP::Database &db, size_t boundaryID, double &ak, double &bk, double &rk, double &nk )
{
    AMP_INSIST( boundaryID >= 1 && boundaryID <= 6, "Invalid boundaryID" );
    ak = db.getScalar<double>( a_keys[boundaryID - 1] );
    bk = db.getScalar<double>( b_keys[boundaryID - 1] );
    rk = db.getWithDefault<double>( r_keys[boundaryID - 1], 0.0 );
    nk = db.getWithDefault<double>( n_keys[boundaryID - 1], 0.0 );
}


/** -------------------------------------------------------- *
 *  ----------- Implementation of RadDifOpPJacData --------- *
 *  -------------------------------------------------------- */
std::tuple<std::shared_ptr<AMP::LinearAlgebra::Matrix>,
           std::shared_ptr<AMP::LinearAlgebra::Matrix>,
           std::shared_ptr<AMP::LinearAlgebra::Vector>,
           std::shared_ptr<AMP::LinearAlgebra::Vector>,
           std::shared_ptr<AMP::LinearAlgebra::Vector>,
           std::shared_ptr<AMP::LinearAlgebra::Vector>>
RadDifOpPJacData::get()
{

    AMP_INSIST( d_E, "E is null before packing" );

    d_dataMaybeOverwritten = true;
    return std::make_tuple( d_E, d_T, r_EE, r_ET, r_TE, r_TT );
}


/** -------------------------------------------------------- *
 *  ------------ Implementation of RadDifOpPJac ------------ *
 *  -------------------------------------------------------- */
RadDifOpPJac::RadDifOpPJac( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ )
    : AMP::Operator::LinearOperator( params_ ),
      d_k11( params_->d_db->getScalar<double>( "k11" ) ),
      d_k12( params_->d_db->getScalar<double>( "k12" ) ),
      d_k21( params_->d_db->getScalar<double>( "k21" ) ),
      d_k22( params_->d_db->getScalar<double>( "k22" ) )
{
    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "RadDifOpPJac::RadDifOpPJac() " << std::endl;
    }

    auto params = std::dynamic_pointer_cast<const RadDifOpPJacParameters>( params_ );
    AMP_INSIST( params, "params must be of type RadDifOpPJacParameters" );

    // Unpack parameter database
    d_db = params->d_db;

    // Create all of our required mesh data
    FDMeshOps::createMeshData( this->getMesh(),
                               d_BoxMesh,
                               d_dim,
                               CellCenteredGeom,
                               d_scalarDOFMan,
                               d_multiDOFMan,
                               d_globalBox,
                               d_localBox,
                               d_localArraySize,
                               d_h,
                               d_rh2 );
    // Create our FDMeshOps
    d_meshIndexingOps = std::make_shared<FDMeshGlobalIndexingOps>(
        d_BoxMesh, CellCenteredGeom, d_scalarDOFMan, d_multiDOFMan );

    // Handle boundary conditions
    // Unpack boundary condition constants into member vectors
    for ( auto boundaryID : d_BoxMesh->getBoundaryIDs() ) {
        FDBoundaryUtils::getBCConstantsFromDB( *d_db,
                                               boundaryID,
                                               d_ak[boundaryID - 1],
                                               d_bk[boundaryID - 1],
                                               d_rk[boundaryID - 1],
                                               d_nk[boundaryID - 1] );
    }
    // Copy so rk and nk so lambdas capture by value
    auto rk = this->d_rk;
    auto nk = this->d_nk;
    // Set boundary condition functions to the default of retrieving constants from member vectors
    // d_rk and d_nk
    d_robinFunctionE = [rk]( size_t boundaryID, const AMP::Mesh::Point & ) {
        return rk[boundaryID - 1];
    };
    d_pseudoNeumannFunctionT = [nk]( size_t boundaryID, const AMP::Mesh::Point & ) {
        return nk[boundaryID - 1];
    };


    // Unpack frozen vector
    d_frozenVec = params->d_frozenSolution;

    // Update boundary functions for our stencil operations if boundary functions were given
    if ( params->d_robinFunctionE ) {
        d_robinFunctionE = params->d_robinFunctionE;
    }
    if ( params->d_pseudoNeumannFunctionT ) {
        d_pseudoNeumannFunctionT = params->d_pseudoNeumannFunctionT;
    }

    setData();
};

void RadDifOpPJac::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET,
                          std::shared_ptr<AMP::LinearAlgebra::Vector> rET )
{

    PROFILE( "RadDifOpPJac::apply" );

    if ( d_iDebugPrintInfoLevel > 1 )
        AMP::pout << "RadDifOpPJac::apply() " << std::endl;

    // If the data has been overwritten by a BDFOper, then this apply will be an apply of that
    // operator. That's fine, but so as to not cause any confusion about the state of the data the
    // BDFOper must acknowledge before every apply that's indeed what they're trying to do.
    if ( d_data->d_dataMaybeOverwritten ) {
        AMP_INSIST( d_applyWithOverwrittenDataIsValid,
                    "This apply is invalid because the data has been mutated by a BDFOper; you "
                    "must first set the flag 'applyWithOverwrittenBDFOperDataIsValid' to true if "
                    "you really want to do an apply" );
    }

    applyFromData( ET, rET );

    // Reset flag
    d_applyWithOverwrittenDataIsValid = false;
};


std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOpPJac::createInputVector() const
{
    auto ET_var = std::make_shared<AMP::LinearAlgebra::Variable>( "ET" );
    auto ET_vec = AMP::LinearAlgebra::createVector<double>( d_multiDOFMan, ET_var );
    return ET_vec;
};


/** Picard linearization of a RadDifOp. This LinearOperator has the following structure:
 * [ d_E 0   ]   [ diag(r_EE) diag(r_ET) ]
 * [ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
 */
void RadDifOpPJac::applyFromData( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_,
                                  std::shared_ptr<AMP::LinearAlgebra::Vector> LET_ )
{

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "RadDifOpJac::applyFromData() " << std::endl;
    }

    // Check that if Jacobian data has been modified that the caller of this function really intends
    // to do the apply with modified data.
    if ( d_data->d_dataMaybeOverwritten ) {
        AMP_INSIST(
            d_applyWithOverwrittenDataIsValid,
            "Jacobian data may have been modified. If you understand what that means for an apply "
            "with it, you must first call RadDifOpPJac::applyWithOverwrittenDataIsValid()" );
    }

    // Downcast input Vectors to MultiVectors
    auto ET  = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( ET_ );
    auto LET = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( LET_ );
    AMP_INSIST( ET, "ET downcast to MultiVector unsuccessful" );
    AMP_INSIST( LET, "LET downcast to MultiVector unsuccessful" );

    // Unpack scalar vectors from multivectors
    auto E  = ET->getVector( 0 );
    auto T  = ET->getVector( 1 );
    auto LE = LET->getVector( 0 );
    auto LT = LET->getVector( 1 );

    auto temp_ = this->createInputVector();
    auto temp  = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( temp_ );
    auto temp1 = temp->getVector( 0 );
    auto temp2 = temp->getVector( 1 );

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


void RadDifOpPJac::ghostValuesSolveWrapper( size_t boundaryID,
                                            const AMP::Mesh::Point &boundaryPoint,
                                            double Eint,
                                            double Tint,
                                            double &Eg,
                                            double &Tg )
{

    auto zatom = d_db->getWithDefault<double>( "zatom", 1.0 );
    // Create a handle for evaluating the diffusion coefficient
    auto cHandle = [&]( double T_midpoint ) {
        return RadDifCoefficients::diffusionE( d_k11, T_midpoint, zatom );
    };

    auto ak   = d_ak[boundaryID - 1];
    auto bk   = d_bk[boundaryID - 1];
    double rk = d_robinFunctionE( boundaryID, boundaryPoint );
    double nk = d_pseudoNeumannFunctionT( boundaryID, boundaryPoint );

    // Spatial mesh size
    double hk = d_h[FDBoundaryUtils::getDimFromBoundaryID( boundaryID )];

    // Solve for ghost values given these constants, storing results in Eg and Tg
    FDBoundaryUtils::ghostValuesSolve( ak, bk, cHandle, rk, nk, hk, Eint, Tint, Eg, Tg );
}


void RadDifOpPJac::getNNDataBoundary(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
    std::array<size_t, 3> &ijk,
    size_t dim,
    std::array<double, 3> &ELoc3,
    std::array<double, 3> &TLoc3,
    std::array<size_t, 3> &dofsLoc3,
    std::optional<FDBoundaryUtils::BoundarySide> &boundaryIntersection )
{

    // Get WEST neighboring value
    // At WEST boundary, so WEST neighbor is a ghost
    if ( ijk[dim] == static_cast<size_t>( d_globalBox->first[dim] ) ) {
        auto boundaryID =
            FDBoundaryUtils::getBoundaryIDFromDim( dim, FDBoundaryUtils::BoundarySide::WEST );
        // Get point on the boundary
        auto boundaryPoint =
            d_meshIndexingOps->gridIndsToMeshElement( ijk ).centroid(); // Point in cell center
        boundaryPoint[dim] -= d_h[dim] / 2; // The boundary is h/2 WEST of the cell center
        ghostValuesSolveWrapper(
            boundaryID, boundaryPoint, ELoc3[ORIGIN], TLoc3[ORIGIN], ELoc3[WEST], TLoc3[WEST] );
        // Flag that we're on the WEST boundary
        boundaryIntersection = FDBoundaryUtils::BoundarySide::WEST;

        // At interior DOF; WEST neighbor is an interior DOF
    } else {
        ijk[dim] -= 1;
        size_t indWEST = d_meshIndexingOps->gridIndsToScalarDOF( ijk );
        ijk[dim] += 1; // reset to ORIGIN
        ELoc3[WEST] = E_vec->getValueByGlobalID( indWEST );
        TLoc3[WEST] = T_vec->getValueByGlobalID( indWEST );
        //
        dofsLoc3[WEST] = indWEST;
    }

    // Get EAST neighboring value
    // At EAST boundary, so EAST neighbor is a ghost
    if ( ijk[dim] == static_cast<size_t>( d_globalBox->last[dim] ) ) {
        auto boundaryID =
            FDBoundaryUtils::getBoundaryIDFromDim( dim, FDBoundaryUtils::BoundarySide::EAST );
        // Get point on the boundary
        auto boundaryPoint =
            d_meshIndexingOps->gridIndsToMeshElement( ijk ).centroid(); // Point in cell center
        boundaryPoint[dim] += d_h[dim] / 2; // The boundary is h/2 EAST of the cell center
        ghostValuesSolveWrapper(
            boundaryID, boundaryPoint, ELoc3[ORIGIN], TLoc3[ORIGIN], ELoc3[EAST], TLoc3[EAST] );
        // Flag that we're on the EAST boundary
        boundaryIntersection = FDBoundaryUtils::BoundarySide::EAST;

        // At interior DOF; EAST neighbor is an interior DOF
    } else {
        ijk[dim] += 1;
        size_t indEAST = d_meshIndexingOps->gridIndsToScalarDOF( ijk );
        ijk[dim] -= 1; // reset to ORIGIN
        ELoc3[EAST] = E_vec->getValueByGlobalID( indEAST );
        TLoc3[EAST] = T_vec->getValueByGlobalID( indEAST );
        //
        dofsLoc3[EAST] = indEAST;
    }
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
 *      i. when building the BDFRadDifOpPJacData because this just adds an identity perturbation to
 * these matrices (and hence assumes they exist)
 *      ii. that the matrix doesn't exist during the apply (which has implications for the
 * BDFRadDifOpPJacData since this uses the apply)
 *      iii. in the operator-split preconditioner since this assumes the matrices exist
 * 2. The sparsity pattern of both matrices is the same, and is independent of the linearization
 * point. So, the sparsity pattern should be computed once, and then the values in the matrices
 * reset when ever the linearization point changes (currently fresh matrices are built every time
 * this function is called).
 */
void RadDifOpPJac::setData()
{

    PROFILE( "RadDifOpPJac::setData" );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "RadDifOpJac::setData() " << std::endl;
    }

    // --- Unpack frozen vector ---
    auto ET_vec =
        std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( this->d_frozenVec );
    AMP_INSIST( ET_vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto E_vec = ET_vec->getVector( 0 );
    auto T_vec = ET_vec->getVector( 1 );

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
    // Extract local raw data
    const double *E_rawData = E_vec->getRawDataBlock<double>();
    const double *T_rawData = T_vec->getRawDataBlock<double>();
    // Place-holders for CSR data in each row
    std::vector<size_t> cols_dE;
    std::vector<double> data_dE;
    std::vector<size_t> cols_dT;
    std::vector<double> data_dT;
    // Create wrapper around CSR data function that sets cols and data
    std::function<void( size_t dof )> setColsAndData_dE = [&]( size_t dof ) {
        getCSRDataDiffusionMatrix<0>( E_vec, T_vec, E_rawData, T_rawData, dof, cols_dE, data_dE );
    };
    std::function<void( size_t dof )> setColsAndData_dT = [&]( size_t dof ) {
        getCSRDataDiffusionMatrix<1>( E_vec, T_vec, E_rawData, T_rawData, dof, cols_dT, data_dT );
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
    fillDiffusionMatrixWithData<0>( d_E_mat );
    fillDiffusionMatrixWithData<1>( d_T_mat );
    // Finalize matrix construction
    d_E_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_T_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    d_data->d_E = d_E_mat;
    d_data->d_T = d_T_mat;
}


template<size_t Component>
void RadDifOpPJac::fillDiffusionMatrixWithData( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
{
    PROFILE( "RadDifOpPJac::fillDiffusionMatricesWithData" );

    // --- Unpack frozen vector ---
    auto ET_vec =
        std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( this->d_frozenVec );
    AMP_INSIST( ET_vec, "d_frozenVec downcast to MultiVector unsuccessful" );
    auto E_vec = ET_vec->getVector( 0 );
    auto T_vec = ET_vec->getVector( 1 );

    // Extract local raw data
    const double *E_rawData = E_vec->getRawDataBlock<double>();
    const double *T_rawData = T_vec->getRawDataBlock<double>();
    // Place-holders for CSR data in each row
    std::vector<size_t> cols;
    std::vector<double> data;
    // Create wrapper around CSR data function that sets cols and data
    std::function<void( size_t dof )> setColsAndData = [&]( size_t dof ) {
        getCSRDataDiffusionMatrix<Component>( E_vec, T_vec, E_rawData, T_rawData, dof, cols, data );
    };

    // Iterate through local rows in matrix
    size_t nrows = 1;
    for ( size_t dof = d_scalarDOFMan->beginDOF(); dof != d_scalarDOFMan->endDOF(); dof++ ) {
        setColsAndData( dof );
        matrix->setValuesByGlobalID<double>( nrows, cols.size(), &dof, cols.data(), data.data() );
    }
}


void RadDifOpPJac::setDataReaction( std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec )
{

    PROFILE( "RadDifOpPJac::setDataReaction" );

    // Unpack z
    auto zatom = d_db->getWithDefault<double>( "zatom", 1.0 );
    // Placeholder for current grid index
    std::array<size_t, 3> ijk;
    // Placeholder for ORIGIN dof
    size_t dof;
    // Placeholder for temperature
    double T;
    // Placeholders for reaction coefficients
    double REE, RET, RTE, RTT;

    // Compute upper local indices
    auto iLast = d_localBox->last[0] - d_localBox->first[0];
    auto jLast = d_localBox->last[1] - d_localBox->first[1];
    auto kLast = d_localBox->last[2] - d_localBox->first[2];

    // Get raw data arrays.
    const double *T_rawData = T_vec->getRawDataBlock<double>();
    double *r_EE_rawData    = d_data->r_EE->getRawDataBlock<double>();
    double *r_ET_rawData    = d_data->r_ET->getRawDataBlock<double>();
    double *r_TE_rawData    = d_data->r_TE->getRawDataBlock<double>();
    double *r_TT_rawData    = d_data->r_TT->getRawDataBlock<double>();

    // Iterate over local box
    for ( auto k = 0; k <= kLast; k++ ) {
        ijk[2] = k;
        for ( auto j = 0; j <= jLast; j++ ) {
            ijk[1] = j;
            for ( auto i = 0; i <= iLast; i++ ) {
                ijk[0] = i;

                // Compute coefficients to apply stencil in a quasi-linear fashion
                dof = d_localArraySize->index( ijk[0], ijk[1], ijk[2] );
                T   = T_rawData[dof];

                // Compute reaction coefficients at cell centers
                RadDifCoefficients::reaction( d_k12, d_k22, T, zatom, REE, RET, RTE, RTT );

                // Insert values into the vectors
                r_EE_rawData[dof] = REE;
                r_ET_rawData[dof] = RET;
                r_TE_rawData[dof] = RTE;
                r_TT_rawData[dof] = RTT;
            } // Loop over i
        }     // Loop over j
    }         // Loop over k
}


template<size_t Component>
void RadDifOpPJac::getCSRDataDiffusionMatrix(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
    const double *E_rawData,
    const double *T_rawData,
    size_t row,
    std::vector<size_t> &cols,
    std::vector<double> &data )
{
    PROFILE( "RadDifOpPJac::getCSRDataDiffusionMatrix" );

    AMP_INSIST( Component == 0 || Component == 1, "Invalid component" );


    // Work out if we're on process boundary or interior
    size_t globalOffset = d_scalarDOFMan->beginDOF();
    // Convert global DOF into local grid index
    size_t rowLocal = row - globalOffset;
    d_localArraySize->ijk( rowLocal, d_ijk.data() );

    // If DOF is on a processor boundary, parse it off to getCSRDataDiffusionMatrixBoundary since
    // the interior version of that function cannot handle ghost data (either on physical boundary
    // or inter-process boundary)
    for ( size_t dim = 0; dim < d_dim; dim++ ) {
        if ( d_ijk[dim] == 0 ||
             d_ijk[dim] == static_cast<size_t>( d_localBox->last[dim] - d_localBox->first[dim] ) ) {
            getCSRDataDiffusionMatrixBoundary<Component>( E_vec, T_vec, row, cols, data );
            return;
        }
    }

    // DOF is on interior of processor domain
    getCSRDataDiffusionMatrixInterior<Component>(
        E_rawData, T_rawData, rowLocal, d_ijk, cols, data );
    // Indices returned in cols are local, so promote them back to the global space
    for ( auto &col : cols ) {
        col += globalOffset;
    }
}


template<size_t Component>
void RadDifOpPJac::getCSRDataDiffusionMatrixInterior( const double *E_rawData,
                                                      const double *T_rawData,
                                                      size_t rowLocal,
                                                      std::array<size_t, 5> &ijkLocal,
                                                      std::vector<size_t> &colsLocal,
                                                      std::vector<double> &data )
{
    PROFILE( "RadDifOpPJac::getCSRDataDiffusionMatrixInterior" );

    // Unpack z
    auto zatom = d_db->getWithDefault<double>( "zatom", 1.0 );

    // Get ORIGIN DOFs
    auto indORIGIN  = rowLocal;
    d_ELoc3[ORIGIN] = E_rawData[indORIGIN];
    d_TLoc3[ORIGIN] = T_rawData[indORIGIN];

    /**
     * We sum over dimensions, resulting in columns ordered as
     * O, W, E, S, N, D, U
     * But note that in boundary-adjacent rows some of these connections can disappear
     * The number of DOFs per non-boundary row is 1 + 2*dim.
     */

    // Initial resize here
    colsLocal.resize( 1 + 2 * d_dim );
    data.resize( 1 + 2 * d_dim );
    // Initialize ORIGIN connection to zero since it's incremented in each dimension
    data[0]      = 0.0;
    colsLocal[0] = indORIGIN;
    // Counter for number connections set
    size_t nnz = 1;

    // Loop over each dimension, adding in its contribution to the total stencil
    double D_WO, D_OE; // Placeholders for coefficinets
    for ( size_t dim = 0; dim < d_dim; dim++ ) {

        // Get WEST and EAST ET data for given dimension
        ijkLocal[dim] -= 1;
        auto indWEST = d_localArraySize->index( ijkLocal[0], ijkLocal[1], ijkLocal[2] );
        ijkLocal[dim] += 2;
        auto indEAST = d_localArraySize->index( ijkLocal[0], ijkLocal[1], ijkLocal[2] );
        ijkLocal[dim] -= 1; // Reset to O

        d_ELoc3[WEST] = E_rawData[indWEST];
        d_ELoc3[EAST] = E_rawData[indEAST];
        d_TLoc3[WEST] = T_rawData[indWEST];
        d_TLoc3[EAST] = T_rawData[indEAST];

        // Get energy coefficients
        if constexpr ( Component == 0 ) {
            FDMeshOps::FaceDiffusionCoefficients<true, false>(
                d_ELoc3, d_TLoc3, d_k11, d_k21, zatom, d_h[dim], &D_WO, &D_OE, nullptr, nullptr );
            // Get temperature coefficients
        } else if constexpr ( Component == 1 ) {
            FDMeshOps::FaceDiffusionCoefficients<false, true>(
                d_ELoc3, d_TLoc3, d_k11, d_k21, zatom, d_h[dim], nullptr, nullptr, &D_WO, &D_OE );
        }

        /** Recall the stencil is applied in the following fashion:
         * dif_action +=
         *  [ -D_OE*(d_ELoc3[EAST]-d_ELoc3[ORIGIN]) + D_WO*(d_ELoc3[ORIGIN]-d_ELoc3[WEST]) ]*rh2
         *  ==
         * - [D_WO*rh2]*WEST + [(D_OE+D_WO)*rh2]*ORIGIN - [D_OE*rh2]*E
         */

        // Pack stencil and column data
        // Increment diagonal connection
        data[0] += ( D_OE + D_WO ) * d_rh2[dim];

        // WEST connection
        data[nnz]      = -D_WO * d_rh2[dim];
        colsLocal[nnz] = indWEST;
        nnz++;
        // EAST connection
        data[nnz]      = -D_OE * d_rh2[dim];
        colsLocal[nnz] = indEAST;
        nnz++;

    } // Loop over dimension
}

template<size_t Component>
void RadDifOpPJac::getCSRDataDiffusionMatrixBoundary(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
    std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
    size_t row,
    std::vector<size_t> &cols,
    std::vector<double> &data )
{
    PROFILE( "RadDifOpPJac::getCSRDataDiffusionMatrixBoundary" );

    // Unpack z
    auto zatom = d_db->getWithDefault<double>( "zatom", 1.0 );

    // Flag indicating which boundary the 3-point stencil intersects with (unset by default)
    std::optional<FDBoundaryUtils::BoundarySide> boundaryIntersection;

    // Convert DOF to a grid index
    auto ijk = d_meshIndexingOps->scalarDOFToGridInds( row );

    // Get ORIGIN DOFs
    auto indORIGIN  = row;
    d_ELoc3[ORIGIN] = E_vec->getValueByGlobalID<double>( indORIGIN );
    d_TLoc3[ORIGIN] = T_vec->getValueByGlobalID<double>( indORIGIN );

    /**
     * We sum over dimensions, resulting in columns ordered as
     * O, W, E, S, N, D, U
     * But note that in boundary-adjacent rows some of these connections can disappear
     * The number of DOFs per non-boundary row is 1 + 2*dim.
     */

    // Preemptive resize
    cols.resize( 1 + 2 * d_dim );
    data.resize( 1 + 2 * d_dim );
    // Initialize ORIGIN connection to zero since it's incremented in each dimension
    data[0] = 0.0;
    cols[0] = indORIGIN;
    // Counter for number of connections set
    size_t nnz = 1;

    // Loop over each dimension, adding in its contribution to the total stencil
    double D_WO, D_OE; // Placeholders for coefficinets
    for ( size_t dim = 0; dim < d_dim; dim++ ) {

        // Get WEST and EAST data for the given dimension
        getNNDataBoundary(
            E_vec, T_vec, ijk, dim, d_ELoc3, d_TLoc3, d_dofsLoc3, boundaryIntersection );

        // Compute diffusion coefficients for E or T
        // Get energy coefficients
        if constexpr ( Component == 0 ) {
            FDMeshOps::FaceDiffusionCoefficients<true, false>(
                d_ELoc3, d_TLoc3, d_k11, d_k21, zatom, d_h[dim], &D_WO, &D_OE, nullptr, nullptr );
            // Get temperature coefficients
        } else if constexpr ( Component == 1 ) {
            FDMeshOps::FaceDiffusionCoefficients<false, true>(
                d_ELoc3, d_TLoc3, d_k11, d_k21, zatom, d_h[dim], nullptr, nullptr, &D_WO, &D_OE );
        }

        /** Recall the stencil is applied in the following fashion:
         * dif_action +=
         *  [ -D_OE*(d_ELoc3[EAST]-d_ELoc3[ORIGIN]) + D_WO*(d_ELoc3[ORIGIN]-d_ELoc3[WEST]) ]*rh2
         *  ==
         * - [D_WO*rh2]*WEST + [(D_OE+D_WO)*rh2]*ORIGIN - [D_OE*rh2]*E
         */

        // Pack stencil and column data
        // Increment diagonal connection
        data[0] += ( D_OE + D_WO ) * d_rh2[dim];

        // Set off-diagonal connections
        // Case 1: Stencil does not intersect boundary
        if ( !boundaryIntersection.has_value() ) {
            // WEST connection
            data[nnz] = -D_WO * d_rh2[dim];
            cols[nnz] = d_dofsLoc3[WEST];
            nnz++;
            // EAST connection
            data[nnz] = -D_OE * d_rh2[dim];
            cols[nnz] = d_dofsLoc3[EAST];
            nnz++;

            /** Case 2: Stencil intersects a boundary. This means that one of the stencil
             * connections is to a ghost. However, recall that the ghost value is a function of the
             * ORIGIN point, as well as some other stuff. That is, the ghost point values satisfy Eg
             * = alpha_E*Eint + blah_E, Tg = alpha_T*Tint + blah_T. In the spirit of Picard
             * linearization when we linearize these equations w.r.t. Eint and Tint we ignore the
             * "blah_E" and "blah_T" (even though these terms are actually functions of Eint and
             * Tint). As such, the stencil connection to the ghost needs to get added into the
             * ORIGIN connection with weight alpha.
             */
            // Case 2a: Stencil intersects WEST boundary -> WEST neighbor is a ghost
        } else if ( boundaryIntersection.value() == FDBoundaryUtils::BoundarySide::WEST ) {
            boundaryIntersection.reset(); // Reset to have no value
            // Add WEST connection into diagonal with weight alpha
            size_t boundaryID =
                FDBoundaryUtils::getBoundaryIDFromDim( dim, FDBoundaryUtils::BoundarySide::WEST );
            double alpha = PicardCorrectionCoefficient( Component, boundaryID, D_WO );
            data[0] += alpha * -D_WO * d_rh2[dim];

            // Add in EAST connection
            data[nnz] = -D_OE * d_rh2[dim];
            cols[nnz] = d_dofsLoc3[EAST];
            nnz++;

            // Case 2b: Stencil intersects EAST boundary -> EAST neighbor is a ghost
        } else {
            boundaryIntersection.reset(); // Reset to have no value
            // Add in WEST connection
            data[nnz] = -D_WO * d_rh2[dim];
            cols[nnz] = d_dofsLoc3[WEST];
            nnz++;

            // Add EAST connection into diagonal with weight alpha
            size_t boundaryID =
                FDBoundaryUtils::getBoundaryIDFromDim( dim, FDBoundaryUtils::BoundarySide::EAST );
            double alpha = PicardCorrectionCoefficient( Component, boundaryID, D_OE );
            data[0] += alpha * -D_OE * d_rh2[dim];
        }
    } // Loop over dimension

    // Do a final resize to trim excess in the event we're in a physical boundary-adjacent row
    cols.resize( nnz );
    data.resize( nnz );
}


void RadDifOpPJac::reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ )
{

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "RadDifOpPJac::reset() " << std::endl;
    }

    Operator::reset( params_ );

    AMP_ASSERT( params_ );
    // Downcast OperatorParameters to its derived class
    auto params = std::dynamic_pointer_cast<const RadDifOpPJacParameters>( params_ );
    AMP_INSIST( ( params.get() ) != nullptr, "NULL parameters" );
    AMP_INSIST( ( ( params->d_db ).get() ) != nullptr, "NULL database" );
    AMP_INSIST( ( ( params->d_frozenSolution ).get() ) != nullptr, "NULL frozen solution" );

    // Unpack parameter database
    d_db = params->d_db;

    // Unpack frozen vector
    d_frozenVec = params->d_frozenSolution;

    // Point data to null, that way it will be reconstructed from new data during the next apply
    // call
    d_data = nullptr;
    setData();
};

double
RadDifOpPJac::PicardCorrectionCoefficient( size_t component, size_t boundaryID, double ck ) const
{
    // Energy
    if ( component == 0 ) {
        // Spatial mesh size
        double hk = d_h[FDBoundaryUtils::getDimFromBoundaryID( boundaryID )];

        auto ak = d_ak[boundaryID - 1];
        auto bk = d_bk[boundaryID - 1];

        // The value we require coincides with r=0 and Eint=1 in "ghostValueSolveE"
        return FDBoundaryUtils::ghostValueSolveE( ak, bk, 0.0, ck, hk, 1.0 );

        // Temperature; correction coefficient is 1.0, because we get Tg = 1.0*Tint + blah
    } else if ( component == 1 ) {

        // The value we require coincides with any n and h in "ghostValueSolveT"
        return FDBoundaryUtils::ghostValueSolveT( 0.0, 0.0, 1.0 );

    } else {
        AMP_ERROR( "Invalid component" );
    }
}


/** -------------------------------------------------------- *
 *  --------------- Implementation of RadDifOp ------------- *
 *  -------------------------------------------------------- */
RadDifOp::RadDifOp( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
    : AMP::Operator::Operator( params ),
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

    // Create all of our required mesh data
    FDMeshOps::createMeshData( this->getMesh(),
                               d_BoxMesh,
                               d_dim,
                               CellCenteredGeom,
                               d_scalarDOFMan,
                               d_multiDOFMan,
                               d_globalBox,
                               d_localBox,
                               d_localArraySize,
                               d_h,
                               d_rh2 );
    // Create our FDMeshOps
    d_meshIndexingOps = std::make_shared<FDMeshGlobalIndexingOps>(
        d_BoxMesh, CellCenteredGeom, d_scalarDOFMan, d_multiDOFMan );


    // Handle boundary conditions
    // Unpack boundary condition constants into member vectors
    for ( auto boundaryID : d_BoxMesh->getBoundaryIDs() ) {
        FDBoundaryUtils::getBCConstantsFromDB( *d_db,
                                               boundaryID,
                                               d_ak[boundaryID - 1],
                                               d_bk[boundaryID - 1],
                                               d_rk[boundaryID - 1],
                                               d_nk[boundaryID - 1] );
    }
    // Copy so rk and nk so lambdas capture by value
    auto rk = this->d_rk;
    auto nk = this->d_nk;
    // Set boundary condition functions to the default of retrieving constants from member vectors
    // d_rk and d_nk
    d_robinFunctionE = [rk]( size_t boundaryID, const AMP::Mesh::Point & ) {
        return rk[boundaryID - 1];
    };
    d_pseudoNeumannFunctionT = [nk]( size_t boundaryID, const AMP::Mesh::Point & ) {
        return nk[boundaryID - 1];
    };
};


const std::vector<double> &RadDifOp::getMeshSize() const { return d_h; }

AMP::Mesh::GeomType RadDifOp::getGeomType() const { return CellCenteredGeom; }

std::shared_ptr<const AMP::Discretization::DOFManager> RadDifOp::getScalarDOFManager() const
{
    return d_scalarDOFMan;
}

std::shared_ptr<AMP::LinearAlgebra::Vector> RadDifOp::createInputVector() const
{
    auto ET_var = std::make_shared<AMP::LinearAlgebra::Variable>( "ET" );
    auto ET_vec = AMP::LinearAlgebra::createVector<double>( d_multiDOFMan, ET_var );
    return ET_vec;
}

void RadDifOp::setBoundaryFunctionE(
    const std::function<double( size_t, const AMP::Mesh::Point & )> &fn_ )
{
    d_robinFunctionE = fn_;
}

void RadDifOp::setBoundaryFunctionT(
    const std::function<double( size_t, const AMP::Mesh::Point & )> &fn_ )
{
    d_pseudoNeumannFunctionT = fn_;
}

std::shared_ptr<AMP::Operator::OperatorParameters>
RadDifOp::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in )
{

    // Create a copy of d_db using Database copy constructor
    auto db = std::make_shared<AMP::Database>( *d_db );
    // OperatorParameters database must contain the "name" of the Jacobian operator that will be
    // created from this
    db->putScalar( "name", "RadDifOpPJac" );
    // Create derived OperatorParameters for Jacobian
    auto jacOpParams = std::make_shared<RadDifOpPJacParameters>( db );
    // Set its mesh
    jacOpParams->d_Mesh = this->getMesh();
    // Get frozen solution vector
    jacOpParams->d_frozenSolution = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( u_in );

    // Copy boundary functions
    jacOpParams->d_robinFunctionE         = d_robinFunctionE;
    jacOpParams->d_pseudoNeumannFunctionT = d_pseudoNeumannFunctionT;

    return jacOpParams;
}

void RadDifOp::ghostValuesSolveWrapper( size_t boundaryID,
                                        const AMP::Mesh::Point &boundaryPoint,
                                        double Eint,
                                        double Tint,
                                        double &Eg,
                                        double &Tg )
{

    auto zatom = d_db->getWithDefault<double>( "zatom", 1.0 );
    // Create a handle for evaluating the diffusion coefficient
    auto cHandle = [&]( double T_midpoint ) {
        return RadDifCoefficients::diffusionE( d_k11, T_midpoint, zatom );
    };

    auto ak   = d_ak[boundaryID - 1];
    auto bk   = d_bk[boundaryID - 1];
    double rk = d_robinFunctionE( boundaryID, boundaryPoint );
    double nk = d_pseudoNeumannFunctionT( boundaryID, boundaryPoint );

    // Spatial mesh size
    double hk = d_h[FDBoundaryUtils::getDimFromBoundaryID( boundaryID )];

    // Solve for ghost values given these constants, storing results in Eg and Tg
    FDBoundaryUtils::ghostValuesSolve( ak, bk, cHandle, rk, nk, hk, Eint, Tint, Eg, Tg );
}

bool RadDifOp::isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET )
{
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
 */
void RadDifOp::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> ET_vec_,
                      std::shared_ptr<AMP::LinearAlgebra::Vector> LET_vec_ )
{

    PROFILE( "RadDifOp::apply" );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "RadDifOp::apply() " << std::endl;
    }

    // Unpack inputs
    // Downcast input Vectors to MultiVectors
    auto ET_vec  = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( ET_vec_ );
    auto LET_vec = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( LET_vec_ );
    // Check to see if this is a multivector
    AMP_INSIST( ET_vec, "ET downcast to MultiVector unsuccessful" );
    AMP_INSIST( LET_vec, "LET downcast to MultiVector unsuccessful" );
    // Unpack vectors
    auto E_vec  = ET_vec->getVector( 0 );
    auto T_vec  = ET_vec->getVector( 1 );
    auto LE_vec = LET_vec->getVector( 0 );
    auto LT_vec = LET_vec->getVector( 1 );

    applyInterior( E_vec, T_vec, LE_vec, LT_vec );
    applyBoundary( E_vec, T_vec, LE_vec, LT_vec );

    LET_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

void RadDifOp::applyInterior( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                              std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                              std::shared_ptr<AMP::LinearAlgebra::Vector> LE_vec,
                              std::shared_ptr<AMP::LinearAlgebra::Vector> LT_vec )
{

    PROFILE( "RadDifOp::applyInterior" );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "RadDifOp::applyInterior() " << std::endl;
    }

    // If this process is empty no need to do anything
    if ( d_scalarDOFMan->numLocalDOF() == 0 ) {
        return;
    }

    // Unpack parameters
    double zatom = this->d_db->getWithDefault<double>( "zatom", 1.0 );
    // Placeholder array for grid indices
    std::array<size_t, 3> ijk;
    // Placeholder arrays for values used in 3-point stencils.
    std::array<double, 3> ELoc3;
    std::array<double, 3> TLoc3;

    // Number of DOFs in each dimension
    std::array<int, 3> domLen;
    for ( int dim = 0; dim < 3; dim++ ) {
        domLen[dim] = 1 + d_localBox->last[dim] - d_localBox->first[dim];
    }

    // Compute first and last index in each dimension.
    std::array<int, 3> ijkFirst;
    std::array<int, 3> ijkLast;

    // There are several cases:
    for ( int dim = 0; dim < 3; dim++ ) {

        // There's at least one interior DOF
        if ( domLen[dim] >= 3 ) {
            ijkFirst[dim] = 1;
            ijkLast[dim]  = domLen[dim] - 2; // This index is inclusive in the below loop


            // There's one or two interior DOFs (there cannot be zero because then the total number
            // of local DOFs is zero and we've already accounted for that)
        } else {

            // The given dimension is "empty" (as happens when the problem dimension is smaller than
            // dim+1). Really there's one DOF in this dimension, but there's no notion of interior
            // or boundary in this dimension. In this case we require start and end indices of 0 to
            // enter into the loop below.
            if ( d_globalBox->last[dim] == 0 ) {
                ijkFirst[dim] = 0;
                ijkLast[dim]  = 0;

                // The current dimension is not empty and has either 1 or 2 DOFs, and thus all of
                // this process' DOFs live on a boundary and will be handled by applyBoundary()
            } else {
                return;
            }
        }
    }

    /** Get raw data arrays.
     * We can index directly into these using indices from our ArraySize
     * insatnce, d_localArraySize.
     * Note that in principle we could create AMP::Array's providing views of these raw data
     * using AMP::Array::constView (or similar), but we ultimately need to use the index() function
     * from ArraySize to get indices (for the Jacobian), so we don't bother creating views
     */
    const double *E_rawData = E_vec->getRawDataBlock<double>();
    const double *T_rawData = T_vec->getRawDataBlock<double>();
    double *LE_rawData      = LE_vec->getRawDataBlock<double>();
    double *LT_rawData      = LT_vec->getRawDataBlock<double>();

    // Placeholders for diffusion coefficients
    double Dr_WO, Dr_OE, DT_WO, DT_OE;

    // Iterate over interior of local box
    for ( auto k = ijkFirst[2]; k <= ijkLast[2]; k++ ) {
        ijk[2] = k;
        for ( auto j = ijkFirst[1]; j <= ijkLast[1]; j++ ) {
            ijk[1] = j;
            for ( auto i = ijkFirst[0]; i <= ijkLast[0]; i++ ) {
                ijk[0] = i;

                // Compute coefficients to apply stencil in a quasi-linear fashion
                // Action of diffusion operator: Loop over each dimension, adding in its
                // contribution to the total
                double dif_E_action = 0.0; // d_E * E
                double dif_T_action = 0.0; // d_T * T

                // Get ORIGIN DOFs - these are independent of the dimension
                auto indORIGIN = d_localArraySize->index( ijk[0], ijk[1], ijk[2] );
                ELoc3[ORIGIN]  = E_rawData[indORIGIN];
                TLoc3[ORIGIN]  = T_rawData[indORIGIN];

                for ( size_t dim = 0; dim < d_dim; dim++ ) {

                    // Get WEST and EAST ET data for given dimension
                    ijk[dim] -= 1;
                    auto indWEST = d_localArraySize->index( ijk[0], ijk[1], ijk[2] );
                    ijk[dim] += 2;
                    auto indEAST = d_localArraySize->index( ijk[0], ijk[1], ijk[2] );
                    ijk[dim] -= 1; // Reset to O

                    ELoc3[WEST] = E_rawData[indWEST];
                    ELoc3[EAST] = E_rawData[indEAST];
                    TLoc3[WEST] = T_rawData[indWEST];
                    TLoc3[EAST] = T_rawData[indEAST];

                    // Get diffusion coefficients for both E and T
                    FDMeshOps::FaceDiffusionCoefficients<true, true>( ELoc3,
                                                                      TLoc3,
                                                                      d_k11,
                                                                      d_k21,
                                                                      zatom,
                                                                      d_h[dim],
                                                                      &Dr_WO,
                                                                      &Dr_OE,
                                                                      &DT_WO,
                                                                      &DT_OE );

                    // Apply diffusion operators
                    dif_E_action += ( -Dr_OE * ( ELoc3[EAST] - ELoc3[ORIGIN] ) +
                                      Dr_WO * ( ELoc3[ORIGIN] - ELoc3[WEST] ) ) *
                                    d_rh2[dim];
                    dif_T_action += ( -DT_OE * ( TLoc3[EAST] - TLoc3[ORIGIN] ) +
                                      DT_WO * ( TLoc3[ORIGIN] - TLoc3[WEST] ) ) *
                                    d_rh2[dim];
                } // Finished looping over dimensions for diffusion discretizations

                AMP_INSIST( TLoc3[ORIGIN] > 1e-14, "PDE coefficients ill-defined for T <= 0" );

                // Compute reaction coefficients at cell centers
                double REE, RET, RTE, RTT;
                RadDifCoefficients::reaction(
                    d_k12, d_k22, TLoc3[ORIGIN], zatom, REE, RET, RTE, RTT );

                // Sum diffusion and reaction terms
                double LE = dif_E_action + ( REE * ELoc3[ORIGIN] + RET * TLoc3[ORIGIN] );
                double LT = dif_T_action + ( RTE * ELoc3[ORIGIN] + RTT * TLoc3[ORIGIN] );

                // Insert values into the vectors
                LE_rawData[indORIGIN] = LE;
                LT_rawData[indORIGIN] = LT;

            } // Loop over i
        }     // Loop over j
    }         // Loop over k
}

void RadDifOp::applyBoundary( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                              std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                              std::shared_ptr<AMP::LinearAlgebra::Vector> LE_vec,
                              std::shared_ptr<AMP::LinearAlgebra::Vector> LT_vec )
{

    PROFILE( "RadDifOp::applyBoundary" );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "RadDifOp::applyBoundary() " << std::endl;
    }

    // If this process is empty no need to do anything
    if ( d_scalarDOFMan->numLocalDOF() == 0 ) {
        return;
    }

    // Unpack parameters
    double zatom = this->d_db->getWithDefault<double>( "zatom", 1.0 );
    // Placeholder array for grid indices
    std::array<size_t, 3> ijk;
    // Placeholder arrays for values used in 3-point stencils.
    std::array<double, 3> ELoc3;
    std::array<double, 3> TLoc3;

    // Placeholders for diffusion coefficients
    double Dr_WO, Dr_OE, DT_WO, DT_OE;

    /** Loop over processor boundary. Note that DOFs are (re)set a number of times equal to the
     * number of boundaries they live on
     */
    std::array<size_t, 3> dims = { 0, 1, 2 };
    // Loop over each dimension
    for ( auto boundaryDim : dims ) {

        // Create vector of free dimensions to loop over with current dim fixed
        std::vector<size_t> freeDims = {};
        for ( auto dim : dims ) {
            if ( dim != boundaryDim ) {
                freeDims.push_back( dim );
            }
        }

        // Loop over both boundaries of current dimension (these are the same if only one DOF in
        // this dimension)
        for ( auto boundaryInd :
              { d_localBox->first[boundaryDim], d_localBox->last[boundaryDim] } ) {
            ijk[boundaryDim] = boundaryInd;

            // Loop over all of second free dimension
            for ( auto freeInd2 = d_localBox->first[freeDims[1]];
                  freeInd2 <= d_localBox->last[freeDims[1]];
                  freeInd2++ ) {
                ijk[freeDims[1]] = freeInd2;

                // Loop over all of first free dimension
                for ( auto freeInd1 = d_localBox->first[freeDims[0]];
                      freeInd1 <= d_localBox->last[freeDims[0]];
                      freeInd1++ ) {
                    ijk[freeDims[0]] = freeInd1;

                    // Compute coefficients to apply stencil in a quasi-linear fashion
                    // Action of diffusion operator: Loop over each dimension, adding in its
                    // contribution to the total
                    double dif_E_action = 0.0; // d_E * E
                    double dif_T_action = 0.0; // d_T * T
                    // Get ORIGIN DOFs - these are independent of the dimension
                    auto indORIGIN = d_meshIndexingOps->gridIndsToScalarDOF( ijk );
                    ELoc3[ORIGIN]  = E_vec->getValueByGlobalID<double>( indORIGIN );
                    TLoc3[ORIGIN]  = T_vec->getValueByGlobalID<double>( indORIGIN );

                    for ( size_t dim = 0; dim < d_dim; dim++ ) {

                        // Get WEST and EAST data for the given dimension
                        getNNDataBoundary( E_vec, T_vec, ijk, dim, ELoc3, TLoc3 );
                        // Get diffusion coefficients for both E and T
                        FDMeshOps::FaceDiffusionCoefficients<true, true>( ELoc3,
                                                                          TLoc3,
                                                                          d_k11,
                                                                          d_k21,
                                                                          zatom,
                                                                          d_h[dim],
                                                                          &Dr_WO,
                                                                          &Dr_OE,
                                                                          &DT_WO,
                                                                          &DT_OE );

                        // Apply diffusion operators
                        dif_E_action += ( -Dr_OE * ( ELoc3[EAST] - ELoc3[ORIGIN] ) +
                                          Dr_WO * ( ELoc3[ORIGIN] - ELoc3[WEST] ) ) *
                                        d_rh2[dim];
                        dif_T_action += ( -DT_OE * ( TLoc3[EAST] - TLoc3[ORIGIN] ) +
                                          DT_WO * ( TLoc3[ORIGIN] - TLoc3[WEST] ) ) *
                                        d_rh2[dim];
                    } // Finished looping over dimensions for diffusion discretizations

                    AMP_INSIST( TLoc3[ORIGIN] > 1e-14, "PDE coefficients ill-defined for T <= 0" );

                    // Compute reaction coefficients at cell centers using T value set in the last
                    // iteration of the above loop
                    double REE, RET, RTE, RTT;
                    RadDifCoefficients::reaction(
                        d_k12, d_k22, TLoc3[ORIGIN], zatom, REE, RET, RTE, RTT );

                    // Sum diffusion and reaction terms
                    double LE = dif_E_action + ( REE * ELoc3[ORIGIN] + RET * TLoc3[ORIGIN] );
                    double LT = dif_T_action + ( RTE * ELoc3[ORIGIN] + RTT * TLoc3[ORIGIN] );

                    // Insert values into the vectors
                    LE_vec->setValueByGlobalID<double>( indORIGIN, LE );
                    LT_vec->setValueByGlobalID<double>( indORIGIN, LT );

                } // Loop first free dim
            }     // Loop over second free dim
        }         // Loop over boundary in frozen dim
    }             // Loop over frozen dim
}

void RadDifOp::getNNDataBoundary( std::shared_ptr<const AMP::LinearAlgebra::Vector> E_vec,
                                  std::shared_ptr<const AMP::LinearAlgebra::Vector> T_vec,
                                  std::array<size_t, 3> &ijk,
                                  size_t dim,
                                  std::array<double, 3> &ELoc3,
                                  std::array<double, 3> &TLoc3 )
{

    // Get WEST neighboring value
    // At WEST boundary, so WEST neighbor is a ghost
    if ( ijk[dim] == static_cast<size_t>( d_globalBox->first[dim] ) ) {
        auto boundaryID =
            FDBoundaryUtils::getBoundaryIDFromDim( dim, FDBoundaryUtils::BoundarySide::WEST );
        // Get point on the boundary
        auto boundaryPoint =
            d_meshIndexingOps->gridIndsToMeshElement( ijk ).centroid(); // Point in cell center
        boundaryPoint[dim] -= d_h[dim] / 2; // The boundary is h/2 WEST of the cell center
        ghostValuesSolveWrapper(
            boundaryID, boundaryPoint, ELoc3[ORIGIN], TLoc3[ORIGIN], ELoc3[WEST], TLoc3[WEST] );

        // At interior DOF; WEST neighbor is an interior DOF
    } else {
        ijk[dim] -= 1;
        size_t indWEST = d_meshIndexingOps->gridIndsToScalarDOF( ijk );
        ijk[dim] += 1; // reset to ORIGIN
        ELoc3[WEST] = E_vec->getValueByGlobalID( indWEST );
        TLoc3[WEST] = T_vec->getValueByGlobalID( indWEST );
    }

    // Get EAST neighboring value
    // At EAST boundary, so EAST neighbor is a ghost
    if ( ijk[dim] == static_cast<size_t>( d_globalBox->last[dim] ) ) {
        auto boundaryID =
            FDBoundaryUtils::getBoundaryIDFromDim( dim, FDBoundaryUtils::BoundarySide::EAST );
        // Get point on the boundary
        auto boundaryPoint =
            d_meshIndexingOps->gridIndsToMeshElement( ijk ).centroid(); // Point in cell center
        boundaryPoint[dim] += d_h[dim] / 2; // The boundary is h/2 EAST of the cell center
        ghostValuesSolveWrapper(
            boundaryID, boundaryPoint, ELoc3[ORIGIN], TLoc3[ORIGIN], ELoc3[EAST], TLoc3[EAST] );

        // At interior DOF; EAST neighbor is an interior DOF
    } else {
        ijk[dim] += 1;
        size_t indEAST = d_meshIndexingOps->gridIndsToScalarDOF( ijk );
        ijk[dim] -= 1; // reset to ORIGIN
        ELoc3[EAST] = E_vec->getValueByGlobalID( indEAST );
        TLoc3[EAST] = T_vec->getValueByGlobalID( indEAST );
    }
}
/** ------------------------------------------------------- *
 *  ----------- End of Implementation of RadDifOp --------- *
 *  ------------------------------------------------------- */


} // namespace AMP::Operator
