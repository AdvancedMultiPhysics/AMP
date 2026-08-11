#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/trilinos/tpetra/TpetraMatrixData.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/VectorDataDefault.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVector.h"

DISABLE_WARNINGS
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_FECrsMatrix.hpp"
#include <Teuchos_Comm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Details_makeColMap_decl.hpp>
#include <Tpetra_RowMatrixTransposer.hpp>
ENABLE_WARNINGS

#include "AMP/discretization/DOF_Manager.h"

namespace AMP::LinearAlgebra {

template<typename LO, typename GO, typename NT>
static inline auto createNoGhostMap( std::shared_ptr<AMP::Discretization::DOFManager> DOFs )
{
#ifdef AMP_USE_MPI
    const auto &ampComm = DOFs->getComm().getCommunicator();
    auto tpComm         = Teuchos::rcp( new Teuchos::MpiComm<int>( ampComm ) );
#else
    auto tpComm = Tpetra::getDefaultComm();
#endif
    return Tpetra::createContigMapWithNode<LO, GO, NT>(
        DOFs->numGlobalDOF(), DOFs->numLocalDOF(), tpComm );
}

template<typename LO, typename GO, typename NT>
static inline auto createColumnMap( std::shared_ptr<AMP::Discretization::DOFManager> DOFs )
{
#ifdef AMP_USE_MPI
    const auto &ampComm = DOFs->getComm().getCommunicator();
    auto tpComm         = Teuchos::rcp( new Teuchos::MpiComm<int>( ampComm ) );
#else
    auto tpComm = Tpetra::getDefaultComm();
#endif
    const auto numLocal = static_cast<GO>( DOFs->numLocalDOF() );
    auto domainMap =
        Teuchos::rcp( new Tpetra::Map<LO, GO, NT>( DOFs->numGlobalDOF(), numLocal, 0, tpComm ) );

    // all globals are contiguous local indices shifted up by starting DOF
    // followed by all remote DOFs
    const auto startDOF  = static_cast<GO>( DOFs->beginDOF() );
    auto remoteDOFs      = DOFs->getRemoteDOFs();
    const auto numRemote = static_cast<GO>( remoteDOFs.size() );
    Kokkos::View<GO *, typename NT::memory_space> gids( "TpetraMatrixData::createColumnMap gids",
                                                        numLocal + numRemote );
    for ( GO n = 0; n < numLocal; ++n ) {
        gids( n ) = startDOF + n;
    }
    for ( GO n = 0; n < numRemote; ++n ) {
        gids( numLocal + n ) = static_cast<GO>( remoteDOFs[n] );
    }

    Teuchos::RCP<const Tpetra::Map<LO, GO, NT>> colMap;
    int err = Tpetra::Details::makeColMap<LO, GO, NT>( colMap, domainMap, gids );
    AMP_ASSERT( err == 0 );
    return colMap;
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraMatrixData<ST, LO, GO, NT>::TpetraMatrixData( std::shared_ptr<MatrixParametersBase> params )
    : MatrixData( params )
{
    // upcast to MatrixParameters and build Tpetra::Map over the rows
    auto matParams = std::dynamic_pointer_cast<MatrixParameters>( params );
    AMP_INSIST( matParams, "Must provide MatrixParameters object to build TpetraMatrixData" );

    const auto colDOFs = matParams->getRightDOFManager();
    const auto rowDOFs = matParams->getLeftDOFManager();
    AMP_INSIST( rowDOFs && colDOFs,
                "MatrixParameters must provide non-null DOFManagers to build TpetraMatrixData" );


    // range map and row map are the same regardless of MPI distribution
    d_RowMap = createNoGhostMap<LO, GO, NT>( rowDOFs );

    // Domain map and column map are not the same for >1 rank
    // domain is simple, just global and local counts
    // column map needs to know about entries outside of diagonal block, so specific ghost
    // information gets included
    d_DomainMap = createNoGhostMap<LO, GO, NT>( colDOFs );
    d_ColumnMap = createColumnMap<LO, GO, NT>( colDOFs );

    // count up entries per row and build matrix if the getRow function exists
    const auto &getRow = matParams->getRowFunction();

    if ( getRow ) {
        const auto nrows = rowDOFs->numLocalDOF();
        const auto srow  = rowDOFs->beginDOF();
        std::vector<size_t> entries( nrows, 0 );
        for ( size_t i = 0; i < nrows; ++i ) {
            const auto cols = getRow( i + srow );
            entries[i]      = static_cast<size_t>( cols.size() );
        }
        Teuchos::ArrayView<size_t> colView( entries.data(), entries.size() );
        d_tpetraMatrix =
            Teuchos::rcp( new Tpetra::CrsMatrix<ST, LO, GO, NT>( d_RowMap, d_ColumnMap, colView ) );
        // new Tpetra::CrsMatrix<ST, LO, GO, NT>( d_RowMap, colView ) );
        // Fill matrix and call fillComplete to set the nz structure
        // Without setting column id's Tpetra will not allocate any memory
        for ( size_t i = 0; i < nrows; ++i ) {
            const auto cols = getRow( i + srow );
            createValuesByGlobalID( i + srow, cols );
        }
        d_tpetraMatrix->setAllToScalar( 0.0 );
        d_tpetraMatrix->fillComplete( d_DomainMap, d_RowMap );
    } else {
        d_tpetraMatrix =
            Teuchos::rcp( new Tpetra::CrsMatrix<ST, LO, GO, NT>( d_RowMap, d_ColumnMap, 0 ) );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraMatrixData<ST, LO, GO, NT>::TpetraMatrixData( const TpetraMatrixData &rhs )
    : TpetraMatrixData( rhs.d_pParameters )
{
    using row_matrix_type = Tpetra::RowMatrix<ST, LO, GO, NT>;
    using nonconst_global_inds_host_view_type =
        typename row_matrix_type::nonconst_global_inds_host_view_type;
    using nonconst_values_host_view_type = typename row_matrix_type::nonconst_values_host_view_type;

    d_pParameters = rhs.d_pParameters;

    auto matParams = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );

    size_t firstRow = matParams->getLeftDOFManager()->beginDOF();

    for ( size_t i = firstRow; i != matParams->getLeftDOFManager()->endDOF(); i++ ) {

        size_t localRow = i - firstRow;
        auto numCols    = rhs.getTpetra_CrsMatrix().getNumEntriesInLocalRow( localRow );
        std::vector<GO> cols( numCols );
        std::vector<ST> vals( numCols );

        nonconst_global_inds_host_view_type tpetraColsView( cols.data(), numCols );
        nonconst_values_host_view_type tpetraValsView( vals.data(), numCols );
        d_tpetraMatrix->getGlobalRowCopy( i, tpetraColsView, tpetraValsView, numCols );

        VerifyTpetraReturn(
            d_tpetraMatrix->replaceGlobalValues( i, numCols, vals.data(), cols.data() ),
            "TpetraMatrixData copy constructor" );
    }
    d_RowMap    = rhs.d_RowMap;
    d_ColumnMap = rhs.d_ColumnMap;
    d_DomainMap = rhs.d_DomainMap;
    makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraMatrixData<ST, LO, GO, NT>::TpetraMatrixData(
    Teuchos::RCP<Tpetra::CrsMatrix<ST, LO, GO, NT>> inMatrix )
    : MatrixData(), d_tpetraMatrix( inMatrix )
{
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<MatrixData> TpetraMatrixData<ST, LO, GO, NT>::cloneMatrixData() const
{
    auto *r = new TpetraMatrixData<ST, LO, GO, NT>( *this );
    return std::shared_ptr<MatrixData>( r );
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<MatrixData> TpetraMatrixData<ST, LO, GO, NT>::transpose() const
{
    Tpetra::RowMatrixTransposer<ST, LO, GO, NT> transposer( d_tpetraMatrix );

    auto matTranspose = transposer.createTranspose();
    return std::shared_ptr<MatrixData>( new TpetraMatrixData<ST, LO, GO, NT>( matTranspose ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::VerifyTpetraReturn( int err, const char *func ) const
{
    std::stringstream error;
    error << func << ": " << err;
    if ( err < 0 )
        AMP_ERROR( error.str() );
    if ( err > 0 )
        AMP_ERROR( error.str() );
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraMatrixData<ST, LO, GO, NT>::~TpetraMatrixData()
{
}

template<typename ST, typename LO, typename GO, typename NT>
Tpetra::CrsMatrix<ST, LO, GO, NT> &TpetraMatrixData<ST, LO, GO, NT>::getTpetra_CrsMatrix()
{
    return *d_tpetraMatrix;
}

template<typename ST, typename LO, typename GO, typename NT>
const Tpetra::CrsMatrix<ST, LO, GO, NT> &
TpetraMatrixData<ST, LO, GO, NT>::getTpetra_CrsMatrix() const
{
    return *d_tpetraMatrix;
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<TpetraMatrixData<ST, LO, GO, NT>>
TpetraMatrixData<ST, LO, GO, NT>::createView( std::shared_ptr<MatrixData> in_matrix )
{
    auto mat = std::dynamic_pointer_cast<TpetraMatrixData<ST, LO, GO, NT>>( in_matrix );
    if ( !mat )
        AMP_ERROR( "Managed memory matrix is not well defined" );
    return mat;
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::createValuesByGlobalID( size_t row,
                                                               const std::vector<size_t> &cols )
{
    if ( cols.empty() )
        return;

    std::vector<ST> values( cols.size(), 0 );
    if constexpr ( std::is_same_v<GO, size_t> ) {
        d_tpetraMatrix->insertGlobalValues(
            static_cast<GO>( row ), static_cast<LO>( cols.size() ), values.data(), cols.data() );

    } else {

        std::vector<GO> indices( cols.size(), 0 );
        std::transform(
            cols.begin(), cols.end(), indices.begin(), []( size_t c ) -> GO { return c; } );

        d_tpetraMatrix->insertGlobalValues(
            static_cast<GO>( row ), static_cast<LO>( cols.size() ), values.data(), indices.data() );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Discretization::DOFManager>
TpetraMatrixData<ST, LO, GO, NT>::getRightDOFManager() const
{
    return std::dynamic_pointer_cast<MatrixParameters>( d_pParameters )->getRightDOFManager();
}
template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Discretization::DOFManager>
TpetraMatrixData<ST, LO, GO, NT>::getLeftDOFManager() const
{
    return std::dynamic_pointer_cast<MatrixParameters>( d_pParameters )->getLeftDOFManager();
}

/********************************************************
 * Get the left/right Vector/DOFManager                  *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Vector> TpetraMatrixData<ST, LO, GO, NT>::createInputVector() const
{

    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );

    auto vec = createTpetraVector( params->getRightCommList(), params->getRightDOFManager() );
    vec->setVariable( params->getRightVariable() );
    return vec;
}
template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Vector> TpetraMatrixData<ST, LO, GO, NT>::createOutputVector() const
{
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    auto vec = createTpetraVector( params->getLeftCommList(), params->getLeftDOFManager() );
    vec->setVariable( params->getLeftVariable() );
    return vec;
}

template<typename ST, typename LO, typename GO, typename NT>
size_t TpetraMatrixData<ST, LO, GO, NT>::numGlobalRows() const
{
    // annoyingly, Tpetra global row/column counts include mutually owned DOFs with multiplicity
    // to get actual (mathematical) size of matrix need to jump through some hoops
    const auto startRow = d_tpetraMatrix->getRangeMap()->getMinAllGlobalIndex();
    const auto endRow   = d_tpetraMatrix->getRangeMap()->getMaxAllGlobalIndex();
    return static_cast<size_t>( 1 + endRow - startRow );
}

template<typename ST, typename LO, typename GO, typename NT>
size_t TpetraMatrixData<ST, LO, GO, NT>::numGlobalColumns() const
{
    // same commentary as numGlobalRows
    const auto startCol = d_tpetraMatrix->getDomainMap()->getMinAllGlobalIndex();
    const auto endCol   = d_tpetraMatrix->getDomainMap()->getMaxAllGlobalIndex();
    return static_cast<size_t>( 1 + endCol - startCol );
}

template<typename ST, typename LO, typename GO, typename NT>
size_t TpetraMatrixData<ST, LO, GO, NT>::numLocalRows() const
{
    return std::dynamic_pointer_cast<MatrixParameters>( d_pParameters )->getLocalNumberOfRows();
}

template<typename ST, typename LO, typename GO, typename NT>
size_t TpetraMatrixData<ST, LO, GO, NT>::numLocalColumns() const
{
    return std::dynamic_pointer_cast<MatrixParameters>( d_pParameters )->getLocalNumberOfColumns();
}

template<typename ST, typename LO, typename GO, typename NT>
AMP::AMP_MPI TpetraMatrixData<ST, LO, GO, NT>::getComm() const
{
    return d_pParameters->getComm();
}

/********************************************************
 * Set/Add values by global id                           *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::addValuesByGlobalID( size_t num_rows,
                                                            size_t num_cols,
                                                            const size_t *rows,
                                                            const size_t *cols,
                                                            const void *vals,
                                                            const typeID &id )
{
    AMP_INSIST( d_tpetraMatrix->isFillActive(),
                "TpetraMatrixData::addValuesByGlobalID matrix modifications must be enabled" );

    AMP_INSIST( id == AMP::getTypeID<ST>(),
                "TpetraMatrixData::addValuesByGlobalID Must match scalar type" );

    std::vector<GO> tpetra_cols( num_cols );
    std::copy( cols, cols + num_cols, tpetra_cols.begin() );
    const ST *values = reinterpret_cast<const ST *>( vals );

    auto params             = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    const auto dofmStartRow = params->getLeftDOFManager()->beginDOF();
    const auto dofmEndRow   = params->getLeftDOFManager()->endDOF();
    const auto startRow = static_cast<size_t>( d_tpetraMatrix->getRangeMap()->getMinGlobalIndex() );
    const auto endRow =
        static_cast<size_t>( d_tpetraMatrix->getRangeMap()->getMaxGlobalIndex() + 1 );

    AMP_ASSERT( dofmStartRow == startRow );
    AMP_ASSERT( dofmEndRow == endRow );

    for ( size_t i = 0; i != num_rows; i++ ) {
        if ( rows[i] < startRow || rows[i] >= endRow ) {
            AMP_WARN_ONCE( "attempting to add to non-owned row" );
            continue;
        }
        d_tpetraMatrix->sumIntoGlobalValues( static_cast<GO>( rows[i] ),
                                             static_cast<LO>( num_cols ),
                                             values + i * num_cols,
                                             tpetra_cols.data() );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::setValuesByGlobalID( size_t num_rows,
                                                            size_t num_cols,
                                                            const size_t *rows,
                                                            const size_t *cols,
                                                            const void *vals,
                                                            const typeID &id )
{
    AMP_INSIST( d_tpetraMatrix->isFillActive(),
                "TpetraMatrixData::setValuesByGlobalID matrix modifications must be enabled" );

    AMP_INSIST( id == AMP::getTypeID<ST>(),
                "TpetraMatrixData::setValuesByGlobalID Must match scalar type" );

    std::vector<GO> tpetra_cols( num_cols );
    std::copy( cols, cols + num_cols, tpetra_cols.begin() );
    const ST *values = reinterpret_cast<const ST *>( vals );

    for ( size_t i = 0; i != num_rows; i++ ) {
        d_tpetraMatrix->replaceGlobalValues( static_cast<GO>( rows[i] ),
                                             static_cast<LO>( num_cols ),
                                             values + i * num_cols,
                                             tpetra_cols.data() );
    }
}


/********************************************************
 * Get values/row by global id                           *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::getValuesByGlobalID( size_t num_rows,
                                                            size_t num_cols,
                                                            const size_t *rows,
                                                            const size_t *cols,
                                                            void *vals,
                                                            const typeID &id ) const
{
    using row_matrix_type = Tpetra::RowMatrix<ST, LO, GO, NT>;
    using nonconst_global_inds_host_view_type =
        typename row_matrix_type::nonconst_global_inds_host_view_type;
    using nonconst_values_host_view_type = typename row_matrix_type::nonconst_values_host_view_type;
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    // Zero out the data in values
    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<double *>( vals );
        for ( size_t i = 0; i < num_rows * num_cols; i++ )
            values[i] = 0.0;
        // Get the data for each row
        size_t firstRow = params->getLeftDOFManager()->beginDOF();
        size_t numRows  = params->getLeftDOFManager()->numLocalDOF();
        std::vector<GO> row_cols;
        std::vector<ST> row_values;
        for ( size_t i = 0; i < num_rows; i++ ) {
            if ( rows[i] < firstRow || rows[i] >= firstRow + numRows )
                continue;
            size_t localRow = rows[i] - firstRow;
            size_t numCols  = d_tpetraMatrix->getNumEntriesInLocalRow( localRow );
            if ( numCols == 0 )
                continue;
            row_cols.resize( numCols );
            row_values.resize( numCols );
            nonconst_global_inds_host_view_type tpetraColsView( row_cols.data(), numCols );
            nonconst_values_host_view_type tpetraValsView( row_values.data(), numCols );
            size_t nCols;
            d_tpetraMatrix->getGlobalRowCopy( rows[i], tpetraColsView, tpetraValsView, nCols );
            AMP_ASSERT( nCols == numCols );
            for ( size_t j1 = 0; j1 < num_cols; j1++ ) {
                for ( size_t j2 = 0; j2 < numCols; j2++ ) {
                    if ( cols[j1] == static_cast<size_t>( row_cols[j2] ) )
                        values[i * num_cols + j1] = row_values[j2];
                }
            }
        }
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::getRowByGlobalID( size_t row,
                                                         std::vector<size_t> &cols,
                                                         std::vector<double> &values ) const
{
    using row_matrix_type = Tpetra::RowMatrix<ST, LO, GO, NT>;
    using nonconst_global_inds_host_view_type =
        typename row_matrix_type::nonconst_global_inds_host_view_type;
    using nonconst_values_host_view_type = typename row_matrix_type::nonconst_values_host_view_type;

    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    size_t firstRow = params->getLeftDOFManager()->beginDOF();
    size_t numRows  = params->getLeftDOFManager()->numLocalDOF();
    AMP_ASSERT( row >= firstRow );
    AMP_ASSERT( row < firstRow + numRows );

    size_t localRow = row - firstRow;
    auto numCols    = d_tpetraMatrix->getNumEntriesInLocalRow( localRow );
    cols.resize( numCols );
    values.resize( numCols );

    std::vector<GO> row_cols;
    std::vector<ST> row_vals;

    if ( numCols ) {
        GO *row_cols_ptr;
        ST *row_vals_ptr;
        if constexpr ( std::is_same_v<size_t, GO> ) {
            row_cols_ptr = cols.data();
        } else {
            row_cols.resize( numCols );
            row_cols_ptr = row_cols.data();
        }
        if constexpr ( std::is_same_v<double, ST> ) {
            row_vals_ptr = values.data();
        } else {
            row_vals.resize( numCols );
            row_vals_ptr = row_vals.data();
        }

        nonconst_global_inds_host_view_type tpetraColsView( row_cols_ptr, numCols );
        nonconst_values_host_view_type tpetraValsView( row_vals_ptr, numCols );
        size_t nCols;
        d_tpetraMatrix->getGlobalRowCopy( row, tpetraColsView, tpetraValsView, nCols );

        if constexpr ( !std::is_same_v<size_t, GO> ) {
            std::transform( row_cols.begin(), row_cols.end(), cols.begin(), []( GO c ) -> size_t {
                return c;
            } );
        }
        if constexpr ( !std::is_same_v<double, ST> ) {
            std::transform( row_vals.begin(), row_vals.end(), values.begin(), []( ST c ) -> double {
                return c;
            } );
        }
    }
}

template<typename ST, typename LO, typename GO, typename NT>
size_t TpetraMatrixData<ST, LO, GO, NT>::numberColumnIDs( size_t row ) const
{
    auto params         = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    const auto firstRow = params->getLeftDOFManager()->beginDOF();
    const auto localRow = static_cast<size_t>( row - firstRow );
    return static_cast<size_t>( d_tpetraMatrix->getNumEntriesInLocalRow( localRow ) );
}

template<typename ST, typename LO, typename GO, typename NT>
std::vector<size_t> TpetraMatrixData<ST, LO, GO, NT>::getColumnIDs( size_t row ) const
{
    using row_matrix_type = Tpetra::RowMatrix<ST, LO, GO, NT>;
    using nonconst_global_inds_host_view_type =
        typename row_matrix_type::nonconst_global_inds_host_view_type;
    using nonconst_values_host_view_type = typename row_matrix_type::nonconst_values_host_view_type;

    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    size_t firstRow = params->getLeftDOFManager()->beginDOF();
    size_t numRows  = params->getLeftDOFManager()->numLocalDOF();
    AMP_ASSERT( row >= firstRow );
    AMP_ASSERT( row < firstRow + numRows );

    const size_t localRow = row - firstRow;
    auto numCols          = d_tpetraMatrix->getNumEntriesInLocalRow( localRow );
    std::vector<size_t> cols( numCols );

    std::vector<GO> row_cols;

    if ( numCols ) {

        std::vector<ST> row_vals( numCols );
        GO *row_cols_ptr;
        ST *row_vals_ptr = row_vals.data();
        if constexpr ( std::is_same_v<size_t, GO> ) {
            row_cols_ptr = cols.data();
        } else {
            row_cols.resize( numCols );
            row_cols_ptr = row_cols.data();
        }

        nonconst_global_inds_host_view_type tpetraColsView( row_cols_ptr, numCols );
        nonconst_values_host_view_type tpetraValsView( row_vals_ptr, numCols );
        size_t nCols;
        d_tpetraMatrix->getGlobalRowCopy( row, tpetraColsView, tpetraValsView, nCols );

        if constexpr ( !std::is_same_v<size_t, GO> ) {
            std::transform( row_cols.begin(), row_cols.end(), cols.begin(), []( GO c ) -> size_t {
                return c;
            } );
        }
    }

    return cols;
}

/********************************************************
 * makeConsistent                                        *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::makeConsistent( AMP::LinearAlgebra::ScatterType )
{
    this->disableModifications();
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::enableModifications()
{
    d_tpetraMatrix->resumeFill();
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::disableModifications()
{
    if ( d_tpetraMatrix->isFillActive() )
        d_tpetraMatrix->fillComplete( d_DomainMap, d_RowMap );
}


} // namespace AMP::LinearAlgebra
