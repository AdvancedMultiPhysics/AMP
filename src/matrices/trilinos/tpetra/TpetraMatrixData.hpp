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
#include <Tpetra_RowMatrixTransposer.hpp>
ENABLE_WARNINGS

#include "AMP/discretization/DOF_Manager.h"

namespace AMP::LinearAlgebra {


template<typename ST, typename LO, typename GO, typename NT>
static inline auto createTpetraMap( std::shared_ptr<AMP::Discretization::DOFManager> dofManager,
                                    const AMP_MPI &ampComm )
{
    AMP_DEBUG_ASSERT( dofManager );
#ifdef AMP_USE_MPI
    const auto &mpiComm = ampComm.getCommunicator();
    auto comm           = Teuchos::rcp( new Teuchos::MpiComm<int>( mpiComm ) );
#else
    auto comm = Tpetra::getDefaultComm();
#endif

    return Teuchos::rcp( new Tpetra::Map<LO, GO, NT>(
        dofManager->numGlobalDOF(), dofManager->numLocalDOF(), 0, comm ) );
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraMatrixData<ST, LO, GO, NT>::TpetraMatrixData( std::shared_ptr<MatrixParametersBase> params )
    : MatrixData( params )
{
    // upcast to MatrixParameters and build Tpetra::Map over the rows
    auto matParams = std::dynamic_pointer_cast<MatrixParameters>( params );
    AMP_INSIST( matParams, "Must provide MatrixParameters object to build TpetraMatrixData" );
    const auto rowDOFs = matParams->getLeftDOFManager();
    const auto colDOFs = matParams->getRightDOFManager();
    AMP_INSIST( rowDOFs && colDOFs,
                "MatrixParameters must provide non-null DOFManagers to build TpetraMatrixData" );
    d_RangeMap  = createTpetraMap<ST, LO, GO, NT>( rowDOFs, params->getComm() );
    d_DomainMap = createTpetraMap<ST, LO, GO, NT>( colDOFs, params->getComm() );

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
        d_tpetraMatrix = Teuchos::rcp(
            new Tpetra::CrsMatrix<ST, LO, GO, NT>( d_RangeMap, d_DomainMap, colView ) );
        // Fill matrix and call fillComplete to set the nz structure
        // Without setting column id's Tpetra will not allocate any memory
        for ( size_t i = 0; i < nrows; ++i ) {
            const auto cols = getRow( i + srow );
            createValuesByGlobalID( i + srow, cols );
        }
        d_tpetraMatrix->setAllToScalar( 0.0 );
        d_tpetraMatrix->fillComplete( d_DomainMap, d_RangeMap );
        // d_tpetraMatrix->describe( *( Teuchos::getFancyOStream( Teuchos::rcpFromRef( std::cout ) )
        // ),
        //                           Teuchos::VERB_EXTREME );
    } else {
        d_tpetraMatrix = Teuchos::rcp( new Tpetra::CrsMatrix<ST, LO, GO, NT>( d_RangeMap, 0 ) );
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
    d_RangeMap  = rhs.d_RangeMap;
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
void TpetraMatrixData<ST, LO, GO, NT>::setTpetraMaps( std::shared_ptr<Vector> range,
                                                      std::shared_ptr<Vector> domain )
{
    if ( range ) {
#ifdef AMP_USE_MPI
        const auto &mpiComm = range->getComm().getCommunicator();
        auto comm           = Teuchos::rcp( new Teuchos::MpiComm<int>( mpiComm ) );
#else
        auto comm = Tpetra::getDefaultComm();
#endif
        auto N_global = static_cast<GO>( range->getGlobalSize() );
        auto N_local  = static_cast<LO>( range->getLocalSize() );
        d_RangeMap    = Teuchos::rcp( new Tpetra::Map<LO, GO, NT>( N_global, N_local, comm ) );
        if ( domain ) {
            N_global    = static_cast<GO>( domain->getGlobalSize() );
            N_local     = static_cast<LO>( domain->getLocalSize() );
            d_DomainMap = Teuchos::rcp( new Tpetra::Map<LO, GO, NT>( N_global, N_local, comm ) );
        }
    }
}


template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::fillComplete()
{
    if ( d_tpetraMatrix->isFillActive() )
        d_tpetraMatrix->fillComplete( d_DomainMap, d_RangeMap );
    //        d_tpetraMatrix->fillComplete();
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

/********************************************************
 * setOtherData                                          *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::setOtherData()
{
    AMP_MPI myComm = d_pParameters->getComm();
    auto ndxLen    = d_OtherData.size();
    auto totNdxLen = myComm.sumReduce( ndxLen );
    if ( totNdxLen == 0 ) {
        return;
    }
    size_t dataLen = 0;
    auto cur_row   = d_OtherData.begin();
    while ( cur_row != d_OtherData.end() ) {
        dataLen += cur_row->second.size();
        ++cur_row;
    }
    auto rows   = new size_t[dataLen + 1]; // Add one to have the new work
    auto cols   = new size_t[dataLen + 1];
    auto data   = new ST[dataLen + 1];
    int cur_ptr = 0;
    cur_row     = d_OtherData.begin();
    while ( cur_row != d_OtherData.end() ) {
        auto cur_elem = cur_row->second.begin();
        while ( cur_elem != cur_row->second.end() ) {
            rows[cur_ptr] = cur_row->first;
            cols[cur_ptr] = cur_elem->first;
            data[cur_ptr] = cur_elem->second;
            ++cur_ptr;
            ++cur_elem;
        }
        ++cur_row;
    }

    const auto totDataLen = myComm.sumReduce( dataLen );

    auto params     = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    auto MyFirstRow = params->getLeftDOFManager()->beginDOF();
    auto MyEndRow   = params->getLeftDOFManager()->endDOF();

    auto aggregateRows = new decltype( MyFirstRow )[totDataLen];
    auto aggregateCols = new decltype( MyFirstRow )[totDataLen];
    auto aggregateData = new ST[totDataLen];

    myComm.allGather( rows, dataLen, aggregateRows );
    myComm.allGather( cols, dataLen, aggregateCols );
    myComm.allGather( data, dataLen, aggregateData );

    AMP_ASSERT( params );
    for ( size_t i = 0; i != totDataLen; i++ ) {
        if ( ( aggregateRows[i] >= MyFirstRow ) && ( aggregateRows[i] < MyEndRow ) ) {
            setValuesByGlobalID( 1u,
                                 1u,
                                 (size_t *) &aggregateRows[i],
                                 (size_t *) &aggregateCols[i],
                                 &aggregateData[i],
                                 getTypeID<ST>() );
        }
    }

    d_OtherData.clear();
    delete[] rows;
    delete[] cols;
    delete[] data;
    delete[] aggregateRows;
    delete[] aggregateCols;
    delete[] aggregateData;
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

    const auto localSize  = params->getLocalNumberOfColumns();
    const auto globalSize = params->getGlobalNumberOfColumns();
    const auto localStart = params->getRightDOFManager()->beginDOF();
    auto buffer = std::make_shared<VectorDataDefault<ST>>( localStart, localSize, globalSize );
    auto vec =
        createTpetraVector( params->getRightCommList(), params->getRightDOFManager(), buffer );
    vec->setVariable( params->getRightVariable() );
    return vec;
}
template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Vector> TpetraMatrixData<ST, LO, GO, NT>::createOutputVector() const
{
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    const auto localSize  = params->getLocalNumberOfRows();
    const auto globalSize = params->getGlobalNumberOfRows();
    const auto localStart = params->getRightDOFManager()->beginDOF();
    auto buffer = std::make_shared<VectorDataDefault<ST>>( localStart, localSize, globalSize );
    auto vec = createTpetraVector( params->getLeftCommList(), params->getLeftDOFManager(), buffer );
    vec->setVariable( params->getLeftVariable() );
    return vec;
}

template<typename ST, typename LO, typename GO, typename NT>
size_t TpetraMatrixData<ST, LO, GO, NT>::numGlobalRows() const
{
    return d_tpetraMatrix->getGlobalNumRows();
}

template<typename ST, typename LO, typename GO, typename NT>
size_t TpetraMatrixData<ST, LO, GO, NT>::numGlobalColumns() const
{
    return d_tpetraMatrix->getGlobalNumCols();
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
void TpetraMatrixData<ST, LO, GO, NT>::addValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    // NOTE: this routine assumes the same number of cols per row!!!
    // This has to be fixed in ALL AMP matrix interfaces
    std::vector<GO> tpetra_cols( num_cols );
    std::copy( cols, cols + num_cols, tpetra_cols.begin() );

    if ( id == getTypeID<double>() ) {

        for ( size_t i = 0; i != num_rows; i++ ) {

            std::vector<ST> row_vals;
            ST *values;
            if constexpr ( std::is_same_v<double, ST> ) {
                const auto array_dptr = reinterpret_cast<const ST *>( vals );
                values                = const_cast<ST *>( &array_dptr[num_cols * i] );
            } else {
                auto incoming_values = reinterpret_cast<const double *>( vals );
                row_vals.resize( num_cols );
                std::transform( &incoming_values[num_cols * i],
                                &incoming_values[num_cols * i] + num_cols,
                                row_vals.begin(),
                                []( double c ) -> ST { return c; } );
                values = row_vals.data();
            }

            auto nvals = d_tpetraMatrix->sumIntoGlobalValues(
                rows[i], num_cols, values + num_cols * i, tpetra_cols.data() );
            AMP_ASSERT( nvals == static_cast<LO>( num_cols ) );
        }

    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::setValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    // NOTE: this routine assumes the same number of cols per row!!!
    // This has to be fixed in ALL AMP matrix interfaces
    std::vector<GO> tpetra_cols( num_cols );
    std::copy( cols, cols + num_cols, tpetra_cols.begin() );
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );

    size_t MyFirstRow = params->getLeftDOFManager()->beginDOF();
    size_t MyEndRow   = params->getLeftDOFManager()->endDOF();
    if ( id == getTypeID<double>() ) {

        for ( size_t i = 0; i != num_rows; i++ ) {

            std::vector<ST> row_vals;
            ST *values;
            if constexpr ( std::is_same_v<double, ST> ) {
                const auto array_dptr = reinterpret_cast<const ST *>( vals );
                values                = const_cast<ST *>( &array_dptr[num_cols * i] );
            } else {
                auto incoming_values = reinterpret_cast<const double *>( vals );
                row_vals.resize( num_cols );
                std::transform( &incoming_values[num_cols * i],
                                &incoming_values[num_cols * i] + num_cols,
                                row_vals.begin(),
                                []( double c ) -> ST { return c; } );
                values = row_vals.data();
            }

            auto nvals = d_tpetraMatrix->replaceGlobalValues(
                rows[i], num_cols, values + num_cols * i, tpetra_cols.data() );
            AMP_ASSERT( nvals == static_cast<LO>( num_cols ) );
            if ( rows[i] < MyFirstRow || rows[i] >= MyEndRow ) {
                for ( size_t j = 0; j != num_cols; j++ ) {
                    d_OtherData[rows[i]][cols[j]] = static_cast<ST>( values[num_cols * i + j] );
                }
            }
        }
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}


/********************************************************
 * Get values/row by global id                           *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::getValuesByGlobalID( size_t num_rows,
                                                            size_t num_cols,
                                                            size_t *rows,
                                                            size_t *cols,
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

    size_t localRow = row - firstRow;
    auto numCols    = d_tpetraMatrix->getNumEntriesInLocalRow( localRow );
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
    fillComplete();
    setOtherData();
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
        d_tpetraMatrix->fillComplete();
}


} // namespace AMP::LinearAlgebra
