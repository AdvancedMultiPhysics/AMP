#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/trilinos/tpetra/TpetraMatrixData.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/VectorDataDefault.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVector.h"

DISABLE_WARNINGS
#include "Tpetra_CrsMatrix_decl.hpp"
#include "Tpetra_FECrsMatrix_decl.hpp"
#include <Teuchos_Comm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_RowMatrixTransposer_decl.hpp>
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

    return std::make_shared<Tpetra::Map<LO, GO, NT>>(
        dofManager->numGlobalDOF(), dofManager->numLocalDOF(), comm );
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraMatrixData<ST, LO, GO, NT>::TpetraMatrixData( std::shared_ptr<MatrixParametersBase> params )
{
    d_pParameters = params;

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
        std::vector<int> entries( nrows, 0 );
        for ( size_t i = 0; i < nrows; ++i ) {
            const auto cols = getRow( i + srow );
            entries[i]      = static_cast<int>( cols.size() );
        }
        d_tpetraMatrix = new Tpetra::FECrsMatrix( Copy, *d_RangeMap, entries.data(), false );
        // Fill matrix and call fillComplete to set the nz structure
        for ( size_t i = 0; i < nrows; ++i ) {
            const auto cols = getRow( i + srow );
            createValuesByGlobalID( i + srow, cols );
        }
        fillComplete();
    } else {
        d_tpetraMatrix = new Tpetra::FECrsMatrix( Copy, *d_RangeMap, 0, false );
    }
    d_DeleteMatrix = true;
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraMatrixData<ST, LO, GO, NT>::TpetraMatrixData( const TpetraMatrixData &rhs )
    : TpetraMatrixData( rhs.d_pParameters )
{
    d_pParameters = rhs.d_pParameters;

    auto matParams = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );

    for ( size_t i = matParams->getLeftDOFManager()->beginDOF();
          i != matParams->getLeftDOFManager()->endDOF();
          i++ ) {
        std::vector<size_t> cols;
        std::vector<double> vals;
        rhs.getRowByGlobalID( i, cols, vals );

        // cast down to ints
        const int ii    = static_cast<int>( i );
        const int ncols = static_cast<int>( cols.size() );
        std::vector<int> ep_cols( ncols );
        std::transform(
            cols.begin(), cols.end(), ep_cols.begin(), []( size_t c ) -> int { return c; } );
        VerifyTpetraReturn(
            d_tpetraMatrix->ReplaceGlobalValues( ii, ncols, vals.data(), ep_cols.data() ),
            "TpetraMatrixData copy constructor" );
    }
    d_RangeMap  = rhs.d_RangeMap;
    d_DomainMap = rhs.d_DomainMap;
    makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraMatrixData<ST, LO, GO, NT>::TpetraMatrixData( Tpetra::CrsMatrix<ST, LO, GO, NT> *inMatrix,
                                                    bool dele )
    : MatrixData(), d_tpetraMatrix( inMatrix ), d_DeleteMatrix( dele )
{
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<MatrixData> TpetraMatrixData<ST, LO, GO, NT>::cloneMatrixData() const
{
    auto *r           = new TpetraMatrixData<ST, LO, GO, NT>( *this );
    r->d_DeleteMatrix = true;
    return std::shared_ptr<MatrixData>( r );
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<MatrixData> TpetraMatrixData<ST, LO, GO, NT>::transpose() const
{
    auto &matrix = const_cast<Tpetra::CrsMatrix<ST, LO, GO, NT> &>( *d_tpetraMatrix );
    Tpetra::RowMatrixTransposer<ST, LO, GO, NT> transposer( matrix );

    auto matTranspose = transposer.createTranspose();
    return std::shared_ptr<MatrixData>( new TpetraMatrixData<ST, LO, GO, NT>(
        dynamic_cast<Tpetra::CrsMatrix<ST, LO, GO, NT> *>( &matTranspose ), true ) );
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
    if ( d_DeleteMatrix )
        delete d_tpetraMatrix;
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
        d_RangeMap    = std::make_shared<Tpetra::Map<LO, GO, NT>>( N_global, N_local, comm );
        if ( domain ) {
            N_global    = static_cast<GO>( domain->getGlobalSize() );
            N_local     = static_cast<LO>( domain->getLocalSize() );
            d_DomainMap = std::make_shared<Tpetra::Map<LO, GO, NT>>( N_global, N_local, comm );
        }
    }
}


template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::fillComplete()
{
    d_tpetraMatrix->fillComplete();
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
    int ndxLen     = d_OtherData.size();
    int totNdxLen  = myComm.sumReduce( ndxLen );
    if ( totNdxLen == 0 ) {
        return;
    }
    int dataLen  = 0;
    auto cur_row = d_OtherData.begin();
    while ( cur_row != d_OtherData.end() ) {
        dataLen += cur_row->second.size();
        ++cur_row;
    }
    auto rows   = new int[dataLen + 1]; // Add one to have the new work
    auto cols   = new int[dataLen + 1];
    auto data   = new double[dataLen + 1];
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

    int totDataLen = myComm.sumReduce( dataLen );

    auto aggregateRows = new int[totDataLen];
    auto aggregateCols = new int[totDataLen];
    auto aggregateData = new double[totDataLen];

    myComm.allGather( rows, dataLen, aggregateRows );
    myComm.allGather( cols, dataLen, aggregateCols );
    myComm.allGather( data, dataLen, aggregateData );

    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    int MyFirstRow = params->getLeftDOFManager()->beginDOF();
    int MyEndRow   = params->getLeftDOFManager()->endDOF();
    for ( int i = 0; i != totDataLen; i++ ) {
        if ( ( aggregateRows[i] >= MyFirstRow ) && ( aggregateRows[i] < MyEndRow ) ) {
            setValuesByGlobalID( 1u,
                                 1u,
                                 (size_t *) &aggregateRows[i],
                                 (size_t *) &aggregateCols[i],
                                 &aggregateData[i],
                                 getTypeID<double>() );
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

    int localSize  = params->getLocalNumberOfColumns();
    int globalSize = params->getGlobalNumberOfColumns();
    int localStart = params->getRightDOFManager()->beginDOF();
    auto buffer = std::make_shared<VectorDataDefault<double>>( localStart, localSize, globalSize );
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
    int localSize  = params->getLocalNumberOfRows();
    int globalSize = params->getGlobalNumberOfRows();
    int localStart = params->getRightDOFManager()->beginDOF();
    auto buffer = std::make_shared<VectorDataDefault<double>>( localStart, localSize, globalSize );
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
    std::vector<GO> tpetra_cols( num_cols );
    std::copy( cols, cols + num_cols, tpetra_cols.begin() );

    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<const double *>( vals );
        for ( size_t i = 0; i != num_rows; i++ )
            VerifyTpetraReturn( d_tpetraMatrix->sumIntoGlobalValues(
                                    rows[i], num_cols, values + num_cols * i, tpetra_cols.data() ),
                                "addValuesByGlobalId" );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::setValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    std::vector<GO> tpetra_cols( num_cols );
    std::copy( cols, cols + num_cols, tpetra_cols.begin() );
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );

    size_t MyFirstRow = params->getLeftDOFManager()->beginDOF();
    size_t MyEndRow   = params->getLeftDOFManager()->endDOF();
    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<const double *>( vals );
        for ( size_t i = 0; i != num_rows; i++ ) {
            VerifyTpetraReturn( d_tpetraMatrix->replaceGlobalValues(
                                    rows[i], num_cols, values + num_cols * i, tpetra_cols.data() ),
                                "setValuesByGlobalID" );
            if ( rows[i] < MyFirstRow || rows[i] >= MyEndRow ) {
                for ( size_t j = 0; j != num_cols; j++ ) {
                    d_OtherData[rows[i]][cols[j]] = values[num_cols * i + j];
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
        std::vector<double> row_values;
        for ( size_t i = 0; i < num_rows; i++ ) {
            if ( rows[i] < firstRow || rows[i] >= firstRow + numRows )
                continue;
            size_t localRow = rows[i] - firstRow;
            auto numCols    = d_tpetraMatrix->getNumEntriesInLocalRow( localRow );
            if ( numCols == 0 )
                continue;
            row_cols.resize( numCols );
            row_values.resize( numCols );
            VerifyTpetraReturn(
                d_tpetraMatrix->getGlobalRowCopy(
                    rows[i], numCols, numCols, &( row_values[0] ), &( row_cols[0] ) ),
                "getValuesByGlobalID" );
            for ( size_t j1 = 0; j1 < num_cols; j1++ ) {
                for ( size_t j2 = 0; j2 < (size_t) numCols; j2++ ) {
                    if ( cols[j1] == (size_t) row_cols[j2] )
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

    if ( numCols ) {
        std::vector<GO> tpetra_cols( numCols );
        VerifyTpetraReturn( d_tpetraMatrix->getGlobalRowCopy(
                                row, numCols, numCols, &( values[0] ), &( tpetra_cols[0] ) ),
                            "getRowByGlobalID" );
        std::copy( tpetra_cols.begin(), tpetra_cols.end(), cols.begin() );
    }
}


template<typename ST, typename LO, typename GO, typename NT>
std::vector<size_t> TpetraMatrixData<ST, LO, GO, NT>::getColumnIDs( size_t row ) const
{
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    size_t firstRow = params->getLeftDOFManager()->beginDOF();
    size_t numRows  = params->getLeftDOFManager()->numLocalDOF();
    AMP_ASSERT( row >= firstRow );
    AMP_ASSERT( row < firstRow + numRows );

    size_t localRow = row - firstRow;
    auto numCols    = d_tpetraMatrix->getNumEntriesInLocalRow( localRow );
    std::vector<size_t> cols( numCols );

    if ( numCols ) {
        std::vector<double> values( numCols );
        std::vector<GO> tpetra_cols( numCols );
        VerifyTpetraReturn( d_tpetraMatrix->getGlobalRowCopy(
                                row, numCols, numCols, values.data(), tpetra_cols.data() ),
                            "getRowByGlobalID" );
        std::copy( tpetra_cols.begin(), tpetra_cols.end(), cols.begin() );
    }

    return cols;
}

/********************************************************
 * makeConsistent                                        *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixData<ST, LO, GO, NT>::makeConsistent( AMP::LinearAlgebra::ScatterType t )
{
    const auto mode = ( t == AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD ) ?
                          Tpetra::CombineMode::Add :
                          Tpetra::CombineMode::Insert;
    auto *mat       = dynamic_cast<Tpetra::FECrsMatrix<ST, LO, GO, NT> *>( d_tpetraMatrix );
    if ( mat ) {
        VerifyTpetraReturn( mat->globalAssemble( false, mode ), "makeParallelConsistent" );
        fillComplete();
    }
    setOtherData();
}

} // namespace AMP::LinearAlgebra
