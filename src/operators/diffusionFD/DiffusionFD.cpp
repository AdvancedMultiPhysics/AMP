#include "AMP/operators/diffusionFD/DiffusionFD.h"

namespace AMP::Operator {


/* ------------------------------------------------------------------------------------------
    Implementation of a finite-difference discretization of a constant-coefficient diffusion
operator
------------------------------------------------------------------------------------------- */

DiffusionFDOperator::DiffusionFDOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params_ )
    : LinearOperator( params_ )
{

    AMP_INSIST( params_, "Non-null parameters required" );

    // Set my database
    d_db = params_->d_db;

    // Keep a pointer to my BoxMesh to save having to do this downcast repeatedly
    d_BoxMesh = std::dynamic_pointer_cast<AMP::Mesh::BoxMesh>( this->getMesh() );
    AMP_INSIST( d_BoxMesh, "Mesh must be a AMP::Mesh::BoxMesh" );

    d_dim = d_BoxMesh->getDim();
    AMP_INSIST( d_dim == 1 || d_dim == 2 || d_dim == 3,
                "Invalid dimension: dim=" + std::to_string( d_dim ) +
                    std::string( " !in {1,2,3}" ) );

    /* Now ensure that the geometry of the Mesh is a Box, which is what happens only when a "cube"
       generator is used, as opposed to any other generator; see the function:
            std::shared_ptr<AMP::Geometry::Geometry>
                Geometry::buildGeometry( std::shared_ptr<const AMP::Database> db )
        I.e., check whether down casting the geometry to a Box passes. Note that Box is a
       dimension-templated class, so we need to resolve each case separately.
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

    // Discretization assumes Dirichlet boundaries in all directions
    for ( auto periodic : d_BoxMesh->periodic() ) {
        AMP_INSIST( periodic == false, "Mesh cannot be periodic in any direction" );
    }

    // Set DOFManager
    this->setDOFManager();
    AMP_INSIST( d_DOFMan, "Requires non-null DOFManager" );

    // Compute mesh spacings in each dimension
    // [ x_min x_max y_min y_max z_min z_max ]
    auto range = d_BoxMesh->getBoundingBox();
    // Set boxes
    d_globalBox = std::make_shared<AMP::Mesh::BoxMesh::Box>( getGlobalNodeBox() );
    d_localBox  = std::make_shared<AMP::Mesh::BoxMesh::Box>( getLocalNodeBox() );
    // Set local size
    d_localArraySize = std::make_shared<AMP::ArraySize>( d_localBox->size() );

    // There are nk+1 grid points in dimension k, nk = globalBox.last[k] - globalBox.first[k], such
    // that the mesh spacing is hk = (xkMax - xkMin)/nk
    for ( size_t k = 0; k < d_dim; k++ ) {
        auto nk    = d_globalBox->last[k] - d_globalBox->first[k];
        auto xkMin = range[2 * k];
        auto xkMax = range[2 * k + 1];
        d_h.push_back( ( xkMax - xkMin ) / nk );
    }


    // Ensure PDE coefficients were parsed
    std::vector<std::string> requiredKeys;
    if ( d_dim == 1 ) {
        requiredKeys.assign( { "cxx" } );
    } else if ( d_dim == 2 ) {
        requiredKeys.assign( { "cxx", "cyy", "cyx" } );
    } else if ( d_dim == 3 ) {
        requiredKeys.assign( { "cxx", "cyy", "czz", "cyx", "czx", "czy" } );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
    auto coefficients = d_db->getDatabase( "DiffusionCoefficients" );
    AMP_INSIST( coefficients, "Input database must include a ''DiffusionCoefficients'' database" );
    for ( auto &key : requiredKeys ) {
        AMP_INSIST( coefficients->keyExists( key ), "Key ''" + key + "'' is missing!" );
    }

    // Set stencil
    d_stencil = std::make_shared<std::vector<double>>( createStencil() );

    // Get the matrix
    auto A = createDiscretizationMatrix();

    // Set linear operator's matrix
    this->setMatrix( A );
}

// Build and set d_DOFMan
void DiffusionFDOperator::setDOFManager()
{
    int DOFsPerElement = 1;
    int gcw            = 1; // Ghost-cell width (stencils are at most 3-point in each direction)
    d_DOFMan           = std::dynamic_pointer_cast<AMP::Discretization::boxMeshDOFManager>(
        AMP::Discretization::boxMeshDOFManager::create(
            this->getMesh(), VertexGeom, gcw, DOFsPerElement ) );
}


size_t DiffusionFDOperator::gridIndsToDOF( std::array<int, 3> ijk ) const
{
    AMP::Mesh::BoxMesh::MeshElementIndex ind( VertexGeom, 0, ijk[0], ijk[1], ijk[2] );
    AMP::Mesh::MeshElementID id = d_BoxMesh->convert( ind );
    std::vector<size_t> dof;
    d_DOFMan->getDOFs( id, dof );
    return dof[0];
}


// Map from DOF to a grid index i, j, or k
std::array<int, 3> DiffusionFDOperator::DOFToGridInds( size_t dof ) const
{
    // Get ElementID
    AMP::Mesh::MeshElementID id = d_DOFMan->getElementID( dof );
    // Convert ElementID into a MeshElementIndex
    AMP::Mesh::BoxMesh::MeshElementIndex ind = d_BoxMesh->convert( id );
    // Get grid index along given component direction
    return ind.index();
}


AMP::Mesh::BoxMesh::Box DiffusionFDOperator::getGlobalNodeBox() const
{
    auto global = d_BoxMesh->getGlobalBox();
    for ( int d = 0; d < 3; d++ ) {
        // Don't increment last index of box in empty dimensions
        if ( global.last[d] > 0 ) {
            global.last[d]++;
        }
    }
    return global;
}

AMP::Mesh::BoxMesh::Box DiffusionFDOperator::getLocalNodeBox() const
{
    auto global = d_BoxMesh->getGlobalBox();
    auto local  = d_BoxMesh->getLocalBox();
    for ( int d = 0; d < 3; d++ ) {
        // Don't increment last index of box in empty dimensions
        if ( global.last[d] > 0 ) {
            if ( local.last[d] == global.last[d] ) {
                local.last[d]++;
            }
        }
    }
    return local;
}


// Vector of hx, hy, hz
const std::vector<double> &DiffusionFDOperator::getMeshSize() const { return d_h; }


/* Populate vector with function that takes a reference to a point and returns a double.  */
void DiffusionFDOperator::fillVectorWithFunction(
    std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
    std::function<double( const AMP::Mesh::Point & )> fun ) const
{

    double u; // Placeholder for funcation evaluation

    // Fill in exact solution vector
    auto it = d_BoxMesh->getIterator( VertexGeom ); // Mesh iterator
    for ( auto elem = it.begin(); elem != it.end(); elem++ ) {
        u = fun( elem->coord() );
        std::vector<size_t> i;
        d_DOFMan->getDOFs( elem->globalID(), i );
        vec->setValueByGlobalID( i[0], u );
    }
}


/* Create RHS vector consistent with the linear operator.
   Here:
    1. PDESourceFun is a function returning the PDE source term
    2. boundaryFun  is a function returning the Dirichlet boundary value at the given location on
   the given boundary
*/
std::shared_ptr<AMP::LinearAlgebra::Vector> DiffusionFDOperator::createRHSVector(
    std::function<double( const AMP::Mesh::Point & )> PDESourceFun,
    std::function<double( const AMP::Mesh::Point &, int boundary_id )> boundaryFun )
{

    // Create a vector
    auto rhs = this->createOutputVector();
    // To start, fill all of it with PDE source term
    this->fillVectorWithFunction( rhs, PDESourceFun );

    // Now modify its values at boundary locations
    double boundaryValue; // Placeholder for funcation evaluation
    // Iterate over all boundaries. Note that DOFs that live on multiple boundaries will be set
    // several times.
    std::vector<int> boundary_ids = d_BoxMesh->getBoundaryIDs();
    for ( auto boundary_id : boundary_ids ) {
        // Get on-process iterator over current boundary
        auto it = d_BoxMesh->getBoundaryIDIterator( VertexGeom, boundary_id );
        for ( auto elem = it.begin(); elem != it.end(); elem++ ) {
            // Get DOF
            std::vector<size_t> dof;
            d_DOFMan->getDOFs( elem->globalID(), dof );
            // Get boundary value and multiply by corresponding entry in A
            boundaryValue = boundaryFun( elem->coord(), boundary_id );
            boundaryValue *= this->getMatrix()->getValueByGlobalID( dof[0], dof[0] );
            // Set value in vector
            rhs->setValueByGlobalID( dof[0], boundaryValue );
        }
    }

    return rhs;
}

/* Create stencil for discretization of the operators
    1D.   [-cxx*u_xx]
    2D.  [[-cxx*u_xx] - cyy*u_yy - cyx*u_yx]
    3D. [[[-cxx*u_xx] - cyy*u_yy - cyx*u_yx] - czz*u_zz - cyx*u_zx - czy*u_zy]
*/
std::vector<double> DiffusionFDOperator::createStencil() const
{

    // Unpack PDE coefficients
    // Note: Defaults values below are not actually used, they just allow this function to be
    // dimension agnostic (since those coefficients don't exist in lower dimensions)
    auto cxx = d_db->getDatabase( "DiffusionCoefficients" )->getScalar<double>( "cxx" );
    auto cyy = d_db->getDatabase( "DiffusionCoefficients" )->getWithDefault<double>( "cyy", 0.0 );
    auto czz = d_db->getDatabase( "DiffusionCoefficients" )->getWithDefault<double>( "czz", 0.0 );
    std::vector<double> cWhole = { cxx, cyy, czz };
    auto cyx = d_db->getDatabase( "DiffusionCoefficients" )->getWithDefault<double>( "cyx", 0.0 );
    auto czx = d_db->getDatabase( "DiffusionCoefficients" )->getWithDefault<double>( "czx", 0.0 );
    auto czy = d_db->getDatabase( "DiffusionCoefficients" )->getWithDefault<double>( "czy", 0.0 );
    std::vector<std::vector<double>> cMixed;
    cMixed.push_back( {} );
    cMixed.push_back( { cyx } );
    cMixed.push_back( { czx, czy } );

    // Stencil vector we will pack
    std::vector<double> stencil;
    // The diagonal connection O, in index 0, is the only connection that multiple individual terms'
    // discretization stencils contribute to, so we increment it as we iterate through stencils of
    // the individual terms.
    stencil.push_back( 0.0 );

    // Loop over each dimension
    for ( size_t dim = 0; dim < d_dim; dim++ ) {
        // Discretize -c{dim,dim}*u_{dim,dim} using standard 3-point stencil
        stencil.emplace_back( -1.0 * cWhole[dim] / ( d_h[dim] * d_h[dim] ) ); // W
        stencil[0] += +2.0 * cWhole[dim] / ( d_h[dim] * d_h[dim] );           // O
        stencil.emplace_back( -1.0 * cWhole[dim] / ( d_h[dim] * d_h[dim] ) ); // E

        // Discretize -c{dim,ell}*u_{dim,ell}, for ell=0,...,dim-1, using 4-point central stencil
        // We orient the 4 corners of each plane counter clockwise, starting in the bottom left
        for ( size_t ell = 0; ell < dim; ell++ ) {
            stencil.emplace_back( -0.25 * cMixed[dim][ell] /
                                  ( d_h[dim] * d_h[ell] ) ); // SW (or DW or DS)
            stencil.emplace_back( +0.25 * cMixed[dim][ell] /
                                  ( d_h[dim] * d_h[ell] ) ); // SE (or DE or DN)
            stencil.emplace_back( -0.25 * cMixed[dim][ell] /
                                  ( d_h[dim] * d_h[ell] ) ); // NE (or UE or UN)
            stencil.emplace_back( +0.25 * cMixed[dim][ell] /
                                  ( d_h[dim] * d_h[ell] ) ); // NW (or UN or US)
        }
    }

    return stencil;
}


void DiffusionFDOperator::getCSRData( size_t row,
                                      std::vector<size_t> &cols,
                                      std::vector<double> &data )
{

    PROFILE( "DiffusionFDOperator::getCSRData" );

    // We do local grid-index based computations using ArraySize, so we need to offset the global
    // index "row" to a local index.
    size_t globalOffset = d_DOFMan->beginDOF();

    // Convert global DOF into local grid index
    size_t rowLocal = row - globalOffset;
    d_localArraySize->ijk( rowLocal, d_ijk.data() );

    // If current DOF is on a processor boundary, parse it off to the other getCSRDataBoundary
    // function, which can handle such DOFs, albeit inefficiently.
    for ( size_t dim = 0; dim < d_dim; dim++ ) {
        if ( d_ijk[dim] == 0 ||
             d_ijk[dim] == static_cast<size_t>( d_localBox->last[dim] - d_localBox->first[dim] ) ) {
            getCSRDataBoundary( row, cols, data );
            return;
        }
    }

    // DOF is on interior of process
    getCSRDataInterior( d_ijk, rowLocal, cols, data );
    // Convert local indices back to global
    for ( auto &col : cols ) {
        col += globalOffset;
    }
}


void DiffusionFDOperator::getCSRDataInterior( std::array<size_t, 5> &ijkLocal,
                                              size_t rowLocal,
                                              std::vector<size_t> &cols,
                                              std::vector<double> &data ) const
{

    PROFILE( "DiffusionFDOperator::getCSRDataInterior" );

    // At an interior DOF
    cols.resize( d_stencil->size() );
    data.resize( d_stencil->size() );
    // Copy stencil contents into data
    data = *d_stencil;

    // Origin
    cols[0]      = rowLocal;
    size_t count = 1;
    // Loop over all mesh dimensions, skipping diagonal connection since we already set it
    for ( size_t dim = 0; dim < d_dim; dim++ ) {
        // --- Whole terms, u_{dim,dim}
        // W (or S or D)
        ijkLocal[dim] -= 1;
        cols[count] = d_localArraySize->index( ijkLocal[0], ijkLocal[1], ijkLocal[2] );
        count++;
        // E (or N or U)
        ijkLocal[dim] += 2;
        cols[count] = d_localArraySize->index( ijkLocal[0], ijkLocal[1], ijkLocal[2] );
        count++;
        // Reset ijkLocal to O.
        ijkLocal[dim] -= 1;

        // --- Mixed terms, u_{dim,ell}, for ell=0,...,dim-1
        // We orient the 4 corners of each plane counter clockwise, starting in the bottom left
        for ( size_t ell = 0; ell < dim; ell++ ) {
            // SW (or DW or DS)
            ijkLocal[dim] -= 1;
            ijkLocal[ell] -= 1;
            cols[count] = d_localArraySize->index( ijkLocal[0], ijkLocal[1], ijkLocal[2] );
            count++;
            // SE (or DE or DN)
            ijkLocal[ell] += 2;
            cols[count] = d_localArraySize->index( ijkLocal[0], ijkLocal[1], ijkLocal[2] );
            count++;
            // NE (or UE or UN)
            ijkLocal[dim] += 2;
            cols[count] = d_localArraySize->index( ijkLocal[0], ijkLocal[1], ijkLocal[2] );
            count++;
            // NW (UN or US)
            ijkLocal[ell] -= 2;
            cols[count] = d_localArraySize->index( ijkLocal[0], ijkLocal[1], ijkLocal[2] );
            count++;
            // Reset ijkLocal to O.
            ijkLocal[ell] += 1;
            ijkLocal[dim] -= 1;
        }
    }
}


void DiffusionFDOperator::getCSRDataBoundary( size_t row,
                                              std::vector<size_t> &cols,
                                              std::vector<double> &data ) const
{

    PROFILE( "DiffusionFDOperator::getCSRDataBoundary" );

    // Get grid indices in an array for ease of iterating through them
    std::array<int, 3> ijk = DOFToGridInds( row );

    // Determine if current DOF is on a physical domain boundary
    bool onBoundary = false;
    for ( size_t dim = 0; dim < d_dim; dim++ ) {
        if ( ijk[dim] == d_globalBox->first[dim] || ijk[dim] == d_globalBox->last[dim] ) {
            onBoundary = true;
            break;
        }
    }

    // At a boundary DOF
    if ( onBoundary ) {
        cols.resize( 1 );
        data.resize( 1 );
        cols[0] = row;
        data[0] = ( *d_stencil )[0];
        return;
    }

    cols.resize( d_stencil->size() );
    data.resize( d_stencil->size() );
    // Copy stencil contents into data
    data = *d_stencil;

    // At an interior DOF
    // Origin
    cols[0]      = gridIndsToDOF( ijk );
    size_t count = 1;
    // Loop over all mesh dimensions, skipping diagonal connection since we already set it
    for ( size_t dim = 0; dim < d_dim; dim++ ) {
        // --- Whole terms, u_{dim,dim}
        // W (or S or D)
        ijk[dim] -= 1;
        cols[count] = gridIndsToDOF( ijk );
        count++;
        // E (or N or U)
        ijk[dim] += 2;
        cols[count] = gridIndsToDOF( ijk );
        count++;
        // Reset ijk to O.
        ijk[dim] -= 1;

        // --- Mixed terms, u_{dim,ell}, for ell=0,...,dim-1
        // We orient the 4 corners of each plane counter clockwise, starting in the bottom left
        for ( size_t ell = 0; ell < dim; ell++ ) {
            // SW (or DW or DS)
            ijk[dim] -= 1;
            ijk[ell] -= 1;
            cols[count] = gridIndsToDOF( ijk );
            count++;
            // SE (or DE or DN)
            ijk[ell] += 2;
            cols[count] = gridIndsToDOF( ijk );
            count++;
            // NE (or UE or UN)
            ijk[dim] += 2;
            cols[count] = gridIndsToDOF( ijk );
            count++;
            // NW (UN or US)
            ijk[ell] -= 2;
            cols[count] = gridIndsToDOF( ijk );
            count++;
            // Reset ijk to O.
            ijk[ell] += 1;
            ijk[dim] -= 1;
        }
    }
}


// Helper function to fill matrix with CSR data
void DiffusionFDOperator::fillMatrixWithCSRData(
    std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
{

    PROFILE( "DiffusionFDOperator::fillMatrixWithCSRData" );

    // Place-holders for CSR data in each row
    std::vector<size_t> cols;
    std::vector<double> data;
    // Create wrapper around CSR data function that sets cols and data
    std::function<void( size_t dof )> setColsAndData = [&]( size_t dof ) {
        getCSRData( dof, cols, data );
    };

    // Iterate through local rows in matrix
    size_t nrows = 1;
    for ( size_t dof = d_DOFMan->beginDOF(); dof != d_DOFMan->endDOF(); dof++ ) {
        setColsAndData( dof );
        matrix->setValuesByGlobalID<double>( nrows, cols.size(), &dof, cols.data(), data.data() );
    }
}


/* Return a constructed CSR matrix corresponding to the discretized operator on the mesh */
std::shared_ptr<AMP::LinearAlgebra::Matrix> DiffusionFDOperator::createDiscretizationMatrix()
{

    PROFILE( "DiffusionFDOperator::createDiscretizationMatrix" );

    auto tempVar =
        std::make_shared<AMP::LinearAlgebra::Variable>( "AMP::Operator::DiffusionFD::x" );
    auto inVec  = AMP::LinearAlgebra::createVector( this->d_DOFMan, tempVar );
    auto outVec = AMP::LinearAlgebra::createVector( this->d_DOFMan, tempVar );

    d_inputVariable  = tempVar;
    d_outputVariable = tempVar;

    // Place-holders for CSR data in each row
    std::vector<size_t> cols;
    std::vector<double> data;
    // Create wrapper around CSR data function that sets cols and data
    std::function<void( size_t dof )> setColsAndData = [&]( size_t dof ) {
        getCSRData( dof, cols, data );
    };

    // Create Lambda to return col inds from a given row ind
    auto getColumnIDs = [&]( size_t row ) {
        setColsAndData( row );
        return cols;
    };

    // Create CSR matrix
    std::shared_ptr<AMP::LinearAlgebra::Matrix> A =
        AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix", getColumnIDs );
    fillMatrixWithCSRData( A );

    // Finalize A
    A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    return A;
}
} // namespace AMP::Operator
