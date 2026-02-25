#include "AMP/matrices/trilinos/epetra/EpetraMatrixData.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/VectorDataDefault.h"
#include "AMP/vectors/trilinos/epetra/EpetraVector.h"

DISABLE_WARNINGS
#include "Epetra_CrsMatrix.h"
#include "Epetra_FECrsMatrix.h"
#include <EpetraExt_Transpose_RowMatrix.h>
#ifdef AMP_USE_MPI
    #include <Epetra_MpiComm.h>
#else
    #include <Epetra_SerialComm.h>
#endif
ENABLE_WARNINGS

#include "AMP/discretization/DOF_Manager.h"

namespace AMP::LinearAlgebra {


static inline auto createEpetraMap( std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                    const AMP_MPI &comm )
{
#ifdef AMP_USE_MPI
    Epetra_MpiComm comm2 = comm.getCommunicator();
#else
    AMP_ASSERT( comm.getSize() == 1 );
    Epetra_SerialComm comm2;
#endif
    AMP_INSIST( DOFs->numGlobalDOF() < 0x80000000,
                "Epetra does not support vectors with global size greater than 2^31" );
    auto N_local  = static_cast<int>( DOFs->numLocalDOF() );
    auto N_global = static_cast<int>( DOFs->numGlobalDOF() );
    return std::make_shared<Epetra_Map>( N_global, N_local, 0, comm2 );
}

EpetraMatrixData::EpetraMatrixData( std::shared_ptr<MatrixParametersBase> params )
{
    d_pParameters = params;

    // upcast to MatrixParameters and build Epetra_Map over the rows
    auto matParams = std::dynamic_pointer_cast<MatrixParameters>( params );
    AMP_INSIST( matParams, "Must provide MatrixParameters object to build EpetraMatrixData" );
    const auto rowDOFs = matParams->getLeftDOFManager();
    const auto colDOFs = matParams->getRightDOFManager();
    AMP_INSIST( rowDOFs && colDOFs,
                "MatrixParameters must provide non-null DOFManagers to build EpetraMatrixData" );
    d_RangeMap  = createEpetraMap( rowDOFs, params->getComm() );
    d_DomainMap = createEpetraMap( colDOFs, params->getComm() );


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
        d_epetraMatrix = new Epetra_FECrsMatrix( Copy, *d_RangeMap, entries.data(), false );
        // Fill matrix and call fillComplete to set the nz structure
        for ( size_t i = 0; i < nrows; ++i ) {
            const auto cols = getRow( i + srow );
            createValuesByGlobalID( i + srow, cols );
        }
        fillComplete();
    } else {
        d_epetraMatrix = new Epetra_FECrsMatrix( Copy, *d_RangeMap, 0, false );
    }
    d_DeleteMatrix = true;
}

EpetraMatrixData::EpetraMatrixData( const EpetraMatrixData &rhs )
    : EpetraMatrixData( rhs.d_pParameters )
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
        VerifyEpetraReturn(
            d_epetraMatrix->ReplaceGlobalValues( ii, ncols, vals.data(), ep_cols.data() ),
            "EpetraMatrixData copy constructor" );
    }
    d_RangeMap  = rhs.d_RangeMap;
    d_DomainMap = rhs.d_DomainMap;
    makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

EpetraMatrixData::EpetraMatrixData( Epetra_CrsMatrix *inMatrix, bool dele )
    : MatrixData(), d_epetraMatrix( inMatrix ), d_DeleteMatrix( dele )
{
}

std::shared_ptr<MatrixData> EpetraMatrixData::cloneMatrixData() const
{
    auto *r           = new EpetraMatrixData( *this );
    r->d_DeleteMatrix = true;
    return std::shared_ptr<MatrixData>( r );
}

std::shared_ptr<MatrixData> EpetraMatrixData::transpose() const
{
    EpetraExt::RowMatrix_Transpose transposer;
    Epetra_CrsMatrix &matrix = const_cast<Epetra_CrsMatrix &>( *d_epetraMatrix );
    return std::shared_ptr<MatrixData>(
        new EpetraMatrixData( dynamic_cast<Epetra_CrsMatrix *>( &transposer( matrix ) ), true ) );
}

void EpetraMatrixData::VerifyEpetraReturn( int err, const char *func ) const
{
    std::stringstream error;
    error << func << ": " << err;
    if ( err < 0 )
        AMP_ERROR( error.str() );
    if ( err > 0 )
        AMP_ERROR( error.str() );
}

EpetraMatrixData::~EpetraMatrixData()
{
    if ( d_DeleteMatrix )
        delete d_epetraMatrix;
}

Epetra_CrsMatrix &EpetraMatrixData::getEpetra_CrsMatrix() { return *d_epetraMatrix; }

const Epetra_CrsMatrix &EpetraMatrixData::getEpetra_CrsMatrix() const { return *d_epetraMatrix; }

std::shared_ptr<EpetraMatrixData>
EpetraMatrixData::createView( std::shared_ptr<MatrixData> in_matrix )
{
    auto mat = std::dynamic_pointer_cast<EpetraMatrixData>( in_matrix );
    if ( !mat )
        AMP_ERROR( "Managed memory matrix is not well defined" );
    return mat;
}

void EpetraMatrixData::setEpetraMaps( std::shared_ptr<Vector> range,
                                      std::shared_ptr<Vector> domain )
{
    if ( range ) {
#ifdef AMP_USE_MPI
        Epetra_MpiComm comm = range->getComm().getCommunicator();
#else
        Epetra_SerialComm comm;
#endif
        AMP_INSIST( range->getGlobalSize() < 0x80000000,
                    "Epetra does not support vectors with global size greater than 2^31" );
        auto N_global = static_cast<int>( range->getGlobalSize() );
        auto N_local  = static_cast<int>( range->getLocalSize() );
        d_RangeMap    = std::make_shared<Epetra_Map>( N_global, N_local, 0, comm );
        if ( domain ) {
            AMP_INSIST( domain->getGlobalSize() < 0x80000000,
                        "Epetra does not support vectors with global size greater than 2^31" );
            N_global    = static_cast<int>( domain->getGlobalSize() );
            N_local     = static_cast<int>( domain->getLocalSize() );
            d_DomainMap = std::make_shared<Epetra_Map>( N_global, N_local, 0, comm );
        }
    }
}


void EpetraMatrixData::fillComplete()
{
    if ( d_RangeMap ) {
        if ( d_DomainMap )
            d_epetraMatrix->FillComplete( *d_DomainMap, *d_RangeMap );
        else
            d_epetraMatrix->FillComplete( *d_RangeMap, *d_RangeMap );
    } else {
        d_epetraMatrix->FillComplete();
    }
}

void EpetraMatrixData::createValuesByGlobalID( size_t row, const std::vector<size_t> &cols )
{
    if ( cols.empty() )
        return;
    std::vector<int> indices( cols.size(), 0 );
    std::transform(
        cols.begin(), cols.end(), indices.begin(), []( size_t c ) -> int { return c; } );

    std::vector<double> values( cols.size(), 0 );

    VerifyEpetraReturn(
        d_epetraMatrix->InsertGlobalValues( row, cols.size(), values.data(), indices.data() ),
        "createValuesByGlobalID" );
}

/********************************************************
 * setOtherData                                          *
 ********************************************************/
void EpetraMatrixData::setOtherData()
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

std::shared_ptr<Discretization::DOFManager> EpetraMatrixData::getRightDOFManager() const
{
    return std::dynamic_pointer_cast<MatrixParameters>( d_pParameters )->getRightDOFManager();
}
std::shared_ptr<Discretization::DOFManager> EpetraMatrixData::getLeftDOFManager() const
{
    return std::dynamic_pointer_cast<MatrixParameters>( d_pParameters )->getLeftDOFManager();
}

/********************************************************
 * Get the left/right Vector/DOFManager                  *
 ********************************************************/
std::shared_ptr<Vector> EpetraMatrixData::createInputVector() const
{

    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );

    int localSize  = params->getLocalNumberOfColumns();
    int globalSize = params->getGlobalNumberOfColumns();
    int localStart = params->getRightDOFManager()->beginDOF();
    auto buffer = std::make_shared<VectorDataDefault<double>>( localStart, localSize, globalSize );
    auto vec =
        createEpetraVector( params->getRightCommList(), params->getRightDOFManager(), buffer );
    vec->setVariable( params->getRightVariable() );
    return vec;
}
std::shared_ptr<Vector> EpetraMatrixData::createOutputVector() const
{
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    int localSize  = params->getLocalNumberOfRows();
    int globalSize = params->getGlobalNumberOfRows();
    int localStart = params->getRightDOFManager()->beginDOF();
    auto buffer = std::make_shared<VectorDataDefault<double>>( localStart, localSize, globalSize );
    auto vec = createEpetraVector( params->getLeftCommList(), params->getLeftDOFManager(), buffer );
    vec->setVariable( params->getLeftVariable() );
    return vec;
}

size_t EpetraMatrixData::numGlobalRows() const { return d_epetraMatrix->NumGlobalRows(); }

size_t EpetraMatrixData::numGlobalColumns() const { return d_epetraMatrix->NumGlobalCols(); }

size_t EpetraMatrixData::numLocalRows() const
{
    return std::dynamic_pointer_cast<MatrixParameters>( d_pParameters )->getLocalNumberOfRows();
}

size_t EpetraMatrixData::numLocalColumns() const
{
    return std::dynamic_pointer_cast<MatrixParameters>( d_pParameters )->getLocalNumberOfColumns();
}

AMP::AMP_MPI EpetraMatrixData::getComm() const { return d_pParameters->getComm(); }

/********************************************************
 * Set/Add values by global id                           *
 ********************************************************/
void EpetraMatrixData::addValuesByGlobalID( size_t num_rows,
                                            size_t num_cols,
                                            const size_t *rows,
                                            const size_t *cols,
                                            const void *vals,
                                            const typeID &id )
{
    std::vector<int> epetra_cols( num_cols );
    std::copy( cols, cols + num_cols, epetra_cols.begin() );

    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<const double *>( vals );
        for ( size_t i = 0; i != num_rows; i++ )
            VerifyEpetraReturn( d_epetraMatrix->SumIntoGlobalValues(
                                    rows[i], num_cols, values + num_cols * i, epetra_cols.data() ),
                                "addValuesByGlobalId" );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

void EpetraMatrixData::setValuesByGlobalID( size_t num_rows,
                                            size_t num_cols,
                                            const size_t *rows,
                                            const size_t *cols,
                                            const void *vals,
                                            const typeID &id )
{
    std::vector<int> epetra_cols( num_cols );
    std::copy( cols, cols + num_cols, epetra_cols.begin() );
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );

    size_t MyFirstRow = params->getLeftDOFManager()->beginDOF();
    size_t MyEndRow   = params->getLeftDOFManager()->endDOF();
    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<const double *>( vals );
        for ( size_t i = 0; i != num_rows; i++ ) {
            VerifyEpetraReturn( d_epetraMatrix->ReplaceGlobalValues(
                                    rows[i], num_cols, values + num_cols * i, epetra_cols.data() ),
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
void EpetraMatrixData::getValuesByGlobalID( size_t num_rows,
                                            size_t num_cols,
                                            const size_t *rows,
                                            const size_t *cols,
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
        std::vector<int> row_cols;
        std::vector<double> row_values;
        for ( size_t i = 0; i < num_rows; i++ ) {
            if ( rows[i] < firstRow || rows[i] >= firstRow + numRows )
                continue;
            size_t localRow = rows[i] - firstRow;
            int numCols     = d_epetraMatrix->NumMyEntries( localRow );
            if ( numCols == 0 )
                continue;
            row_cols.resize( numCols );
            row_values.resize( numCols );
            VerifyEpetraReturn(
                d_epetraMatrix->ExtractGlobalRowCopy(
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
void EpetraMatrixData::getRowByGlobalID( size_t row,
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
    int numCols     = d_epetraMatrix->NumMyEntries( localRow );
    cols.resize( numCols );
    values.resize( numCols );

    if ( numCols ) {
        std::vector<int> epetra_cols( numCols );
        VerifyEpetraReturn( d_epetraMatrix->ExtractGlobalRowCopy(
                                row, numCols, numCols, &( values[0] ), &( epetra_cols[0] ) ),
                            "getRowByGlobalID" );
        std::copy( epetra_cols.begin(), epetra_cols.end(), cols.begin() );
    }
}


std::vector<size_t> EpetraMatrixData::getColumnIDs( size_t row ) const
{
    auto params = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( params );
    size_t firstRow = params->getLeftDOFManager()->beginDOF();
    size_t numRows  = params->getLeftDOFManager()->numLocalDOF();
    AMP_ASSERT( row >= firstRow );
    AMP_ASSERT( row < firstRow + numRows );

    size_t localRow = row - firstRow;
    int numCols     = d_epetraMatrix->NumMyEntries( localRow );
    std::vector<size_t> cols( numCols );

    if ( numCols ) {
        std::vector<double> values( numCols );
        std::vector<int> epetra_cols( numCols );
        VerifyEpetraReturn( d_epetraMatrix->ExtractGlobalRowCopy(
                                row, numCols, numCols, values.data(), epetra_cols.data() ),
                            "getRowByGlobalID" );
        std::copy( epetra_cols.begin(), epetra_cols.end(), cols.begin() );
    }

    return cols;
}

/********************************************************
 * makeConsistent                                        *
 ********************************************************/
void EpetraMatrixData::makeConsistent( AMP::LinearAlgebra::ScatterType t )
{
    const auto mode = ( t == AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD ) ?
                          Epetra_CombineMode::Add :
                          Epetra_CombineMode::Insert;
    auto *mat       = dynamic_cast<Epetra_FECrsMatrix *>( d_epetraMatrix );
    if ( mat ) {
        VerifyEpetraReturn( mat->GlobalAssemble( false, mode ), "makeParallelConsistent" );
        fillComplete();
    }
    setOtherData();
}

} // namespace AMP::LinearAlgebra
