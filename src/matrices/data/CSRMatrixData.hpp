#ifndef included_AMP_CSRMatrixData_hpp
#define included_AMP_CSRMatrixData_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/AMPCSRMatrixParameters.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/MatrixParametersBase.h"
#include "AMP/matrices/RawCSRMatrixParameters.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Utilities.h"

#include <type_traits>

namespace AMP::LinearAlgebra {

/********************************************************
 * Constructors/Destructor                              *
 ********************************************************/
template<typename Policy, class Allocator, class DiagMatrixData>
CSRMatrixData<Policy, Allocator, DiagMatrixData>::CSRMatrixData()
    : d_memory_location( AMP::Utilities::getAllocatorMemoryType<Allocator>() )
{
    AMPManager::incrementResource( "CSRMatrixData" );
}

template<typename Policy, class Allocator, class DiagMatrixData>
CSRMatrixData<Policy, Allocator, DiagMatrixData>::CSRMatrixData(
    std::shared_ptr<MatrixParametersBase> params )
    : MatrixData( params ), d_memory_location( AMP::Utilities::getAllocatorMemoryType<Allocator>() )
{
    AMPManager::incrementResource( "CSRMatrixData" );

    AMP_INSIST(
        d_memory_location != AMP::Utilities::MemoryType::device,
        "CSRMatrixData and CSRSerialMatrixData do not support pure-device memory locations yet" );

    // Figure out what kind of parameters object we have
    // Note: matParams always true if ampCSRParams is by inheritance
    auto rawCSRParams = std::dynamic_pointer_cast<RawCSRMatrixParameters<Policy>>( d_pParameters );
    auto ampCSRParams = std::dynamic_pointer_cast<AMPCSRMatrixParameters<Policy>>( d_pParameters );
    auto matParams    = std ::dynamic_pointer_cast<MatrixParameters>( d_pParameters );

    if ( rawCSRParams ) {

        // Simplest initialization, extract row/column bounds and pass through to diag/offd
        d_first_row = rawCSRParams->d_first_row;
        d_last_row  = rawCSRParams->d_last_row;
        d_first_col = rawCSRParams->d_first_col;
        d_last_col  = rawCSRParams->d_last_col;

        // Construct on/off diag blocks
        d_diag_matrix = std::make_shared<DiagMatrixData>(
            params, d_memory_location, d_first_row, d_last_row, d_first_col, d_last_col, true );
        d_offd_matrix = std::make_shared<DiagMatrixData>(
            params, d_memory_location, d_first_row, d_last_row, d_first_col, d_last_col, false );

        // Only special item, raw data doesn't come with DOFManagers so need to make some
        resetDOFManagers();

    } else if ( matParams ) {
        // get row/column bounds from DOFManagers
        d_leftDOFManager  = matParams->getLeftDOFManager();
        d_rightDOFManager = matParams->getRightDOFManager();
        AMP_ASSERT( d_leftDOFManager && d_rightDOFManager );
        d_first_row = d_leftDOFManager->beginDOF();
        d_last_row  = d_leftDOFManager->endDOF();
        d_first_col = d_rightDOFManager->beginDOF();
        d_last_col  = d_rightDOFManager->endDOF();

        // Construct on/off diag blocks
        d_diag_matrix = std::make_shared<DiagMatrixData>(
            params, d_memory_location, d_first_row, d_last_row, d_first_col, d_last_col, true );
        d_offd_matrix = std::make_shared<DiagMatrixData>(
            params, d_memory_location, d_first_row, d_last_row, d_first_col, d_last_col, false );

        // If, more specifically, have ampCSRParams then blocks are not yet
        // filled. This consolidates calls to getRow{NNZ,Cols} for both blocks
        if ( ampCSRParams && ampCSRParams->getRowNNZFunction() ) {
            AMP_INSIST( ampCSRParams->getRowColsFunction(),
                        "ampCSRParams->getRowColsFunction() must give valid function" );

            // number of local rows
            const lidx_t nrows = static_cast<lidx_t>( d_last_row - d_first_row );

            // pull out row functions
            auto getRowNNZ  = ampCSRParams->getRowNNZFunction();
            auto getRowCols = ampCSRParams->getRowColsFunction();

            // get NNZ counts and trigger allocations in blocks
            std::vector<lidx_t> nnz_diag( nrows ), nnz_offd( nrows );
            for ( lidx_t n = 0; n < nrows; ++n ) {
                getRowNNZ( d_first_row + n, nnz_diag[n], nnz_offd[n] );
            }
            d_diag_matrix->setNNZ( nnz_diag );
            d_offd_matrix->setNNZ( nnz_offd );


            // Fill in column indices
            for ( lidx_t n = 0; n < nrows; ++n ) {
                const auto rs_diag = d_diag_matrix->d_row_starts[n];
                auto cols_diag     = &( d_diag_matrix->d_cols[rs_diag] );
                const auto rs_offd = d_offd_matrix->d_row_starts[n];
                auto cols_offd =
                    d_offd_matrix->d_is_empty ? nullptr : &( d_offd_matrix->d_cols[rs_offd] );
                getRowCols( d_first_row + n, cols_diag, cols_offd );
            }

            // trigger re-packing of columns and convert to local cols
            globalToLocalColumns();
        }

    } else {
        AMP_ERROR( "Check supplied MatrixParameters object" );
    }

    // get total nnz count
    d_nnz = d_diag_matrix->d_nnz + d_offd_matrix->d_nnz;

    d_is_square = ( d_leftDOFManager->numGlobalDOF() == d_rightDOFManager->numGlobalDOF() );
}

template<typename Policy, class Allocator, class DiagMatrixData>
CSRMatrixData<Policy, Allocator, DiagMatrixData>::~CSRMatrixData()
{
    AMPManager::decrementResource( "CSRMatrixData" );
}

template<typename Policy, class Allocator, class DiagMatrixData>
std::shared_ptr<MatrixData>
CSRMatrixData<Policy, Allocator, DiagMatrixData>::cloneMatrixData() const
{
    std::shared_ptr<CSRMatrixData> cloneData;

    cloneData = std::make_shared<CSRMatrixData<Policy, Allocator, DiagMatrixData>>();

    cloneData->d_is_square       = d_is_square;
    cloneData->d_first_row       = d_first_row;
    cloneData->d_last_row        = d_last_row;
    cloneData->d_first_col       = d_first_col;
    cloneData->d_last_col        = d_last_col;
    cloneData->d_nnz             = d_nnz;
    cloneData->d_leftDOFManager  = d_leftDOFManager;
    cloneData->d_rightDOFManager = d_rightDOFManager;
    cloneData->d_pParameters     = d_pParameters;

    cloneData->d_diag_matrix = d_diag_matrix->cloneMatrixData();
    cloneData->d_offd_matrix = d_offd_matrix->cloneMatrixData();

    return cloneData;
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::setNNZ( const std::vector<lidx_t> &nnz_diag,
                                                               const std::vector<lidx_t> &nnz_offd )
{
    // forward to internal blocks to get the internals allocated
    d_diag_matrix->setNNZ( nnz_diag );
    d_offd_matrix->setNNZ( nnz_offd );
    d_nnz = d_diag_matrix->d_nnz + d_offd_matrix->d_nnz;
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::globalToLocalColumns()
{
    d_diag_matrix->globalToLocalColumns();
    d_offd_matrix->globalToLocalColumns();
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::resetDOFManagers()
{
    std::vector<size_t> remoteDOFsRight;
    d_offd_matrix->getColumnMap( remoteDOFsRight );
    d_rightDOFManager = std::make_shared<Discretization::DOFManager>(
        d_last_col - d_first_col, getComm(), remoteDOFsRight );
    // Finding ghosts for leftDM hard to do and not currently needed
    // should think about how to approach it though
    if ( !d_leftDOFManager ) {
        d_leftDOFManager =
            std::make_shared<Discretization::DOFManager>( d_last_row - d_first_row, getComm() );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
std::shared_ptr<MatrixData> CSRMatrixData<Policy, Allocator, DiagMatrixData>::transpose() const
{
    AMP_ERROR( "Not implemented" );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::getRowByGlobalID(
    size_t row, std::vector<size_t> &cols, std::vector<double> &vals ) const
{
    AMP_DEBUG_INSIST( row >= static_cast<size_t>( d_first_row ) &&
                          row < static_cast<size_t>( d_last_row ),
                      "row must be owned by rank" );

    AMP_DEBUG_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                      "CSRMatrixData::getRowByGlobalID not implemented for device memory" );

    auto local_row = row - d_first_row;

    // Get portion of row from diagonal matrix
    d_diag_matrix->getRowByGlobalID( local_row, cols, vals );

    // Get portion from off diagonal and append
    std::vector<size_t> od_cols;
    std::vector<double> od_vals;
    d_offd_matrix->getRowByGlobalID( local_row, od_cols, od_vals );
    cols.insert( cols.end(), od_cols.begin(), od_cols.end() );
    vals.insert( vals.end(), od_vals.begin(), od_vals.end() );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::getValuesByGlobalID(
    size_t num_rows,
    size_t num_cols,
    size_t *rows,
    size_t *cols,
    void *vals,
    [[maybe_unused]] const typeID &id ) const
{
    AMP_DEBUG_INSIST( getTypeID<scalar_t>() == id,
                      "CSRMatrixData::getValuesByGlobalID called with inconsistent typeID" );

    AMP_DEBUG_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                      "CSRMatrixData::getValuesByGlobalID not implemented for device memory" );

    auto values = reinterpret_cast<scalar_t *>( vals );

    // zero out values
    for ( size_t i = 0; i < num_rows * num_cols; i++ ) {
        values[i] = 0.0;
    }

    // get values row-by-row from the enclosed blocks
    lidx_t start_pos = 0;
    for ( size_t nr = 0; nr < num_rows; ++nr ) {
        const auto local_row = static_cast<lidx_t>( rows[nr] - d_first_row );
        d_diag_matrix->getValuesByGlobalID(
            local_row, num_cols, &cols[start_pos], &values[start_pos] );
        d_offd_matrix->getValuesByGlobalID(
            local_row, num_cols, &cols[start_pos], &values[start_pos] );
        start_pos += num_cols;
    }
}

// The two getValues functions above can directly forward to the diag and off diag blocks
// The addValuesByGlobalID and setValuesByGlobalID functions can't do this since
// they need to also handle the other_data case
template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::addValuesByGlobalID(
    size_t num_rows,
    size_t num_cols,
    size_t *rows,
    size_t *cols,
    void *vals,
    [[maybe_unused]] const typeID &id )
{
    AMP_DEBUG_INSIST( getTypeID<scalar_t>() == id,
                      "CSRMatrixData::addValuesByGlobalID called with inconsistent typeID" );

    AMP_DEBUG_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                      "CSRMatrixData::addValuesByGlobalID not implemented for device memory" );

    auto values = reinterpret_cast<const scalar_t *>( vals );

    for ( size_t i = 0u; i != num_rows; i++ ) {
        if ( rows[i] >= static_cast<size_t>( d_first_row ) &&
             rows[i] < static_cast<size_t>( d_last_row ) ) {

            // Forward single row to diag and off diag blocks
            // auto lcols = &cols[num_cols * i];
            const auto local_row = rows[i] - d_first_row;
            auto lvals           = &values[num_cols * i];
            d_diag_matrix->addValuesByGlobalID( num_cols, local_row, cols, lvals );
            d_offd_matrix->addValuesByGlobalID( num_cols, local_row, cols, lvals );
        } else {
            for ( size_t icol = 0; icol < num_cols; ++icol ) {
                d_other_data[rows[i]][cols[icol]] += values[num_cols * i + icol];
            }
        }
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::setValuesByGlobalID(
    size_t num_rows,
    size_t num_cols,
    size_t *rows,
    size_t *cols,
    void *vals,
    [[maybe_unused]] const typeID &id )
{
    AMP_DEBUG_INSIST( getTypeID<scalar_t>() == id,
                      "CSRMatrixData::setValuesByGlobalID called with inconsistent typeID" );

    AMP_DEBUG_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                      "CSRMatrixData::setValuesByGlobalID not implemented for device memory" );

    auto values = reinterpret_cast<const scalar_t *>( vals );

    for ( size_t i = 0u; i != num_rows; i++ ) {

        if ( rows[i] >= static_cast<size_t>( d_first_row ) &&
             rows[i] < static_cast<size_t>( d_last_row ) ) {

            // Forward single row to diag and off diag blocks
            // auto lcols = &cols[num_cols * i];
            const auto local_row = rows[i] - d_first_row;
            auto lvals           = &values[num_cols * i];
            d_diag_matrix->setValuesByGlobalID( num_cols, local_row, cols, lvals );
            d_offd_matrix->setValuesByGlobalID( num_cols, local_row, cols, lvals );
        } else {
            for ( size_t icol = 0; icol < num_cols; ++icol ) {
                d_ghost_data[rows[i]][cols[icol]] = values[num_cols * i + icol];
            }
        }
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
std::vector<size_t>
CSRMatrixData<Policy, Allocator, DiagMatrixData>::getColumnIDs( size_t row ) const
{
    AMP_DEBUG_INSIST( row >= static_cast<size_t>( d_first_row ) &&
                          row < static_cast<size_t>( d_last_row ),
                      "CSRMatrixData::getColumnIDs row must be owned by rank" );

    AMP_DEBUG_INSIST( d_diag_matrix, "CSRMatrixData::getColumnIDs diag matrix must exist" );

    AMP_DEBUG_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                      "CSRMatrixData::getColumnIDs not implemented for device memory" );

    auto local_row              = row - d_first_row;
    std::vector<size_t> cols    = d_diag_matrix->getColumnIDs( local_row );
    std::vector<size_t> od_cols = d_offd_matrix->getColumnIDs( local_row );
    cols.insert( cols.end(), od_cols.begin(), od_cols.end() );
    return cols;
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::setOtherData(
    std::map<gidx_t, std::map<gidx_t, scalar_t>> &other_data, AMP::LinearAlgebra::ScatterType t )
{
    AMP_MPI comm   = getComm();
    auto ndxLen    = other_data.size();
    auto totNdxLen = comm.sumReduce( ndxLen );
    if ( totNdxLen == 0 ) {
        return;
    }
    auto dataLen = 0;
    auto cur_row = other_data.begin();
    while ( cur_row != other_data.end() ) {
        dataLen += cur_row->second.size();
        ++cur_row;
    }
    std::vector<gidx_t> rows( dataLen + 1 );   // Add one to have the new work
    std::vector<gidx_t> cols( dataLen + 1 );   // Add one to have the new work
    std::vector<scalar_t> data( dataLen + 1 ); // Add one to have the new work
    size_t cur_ptr = 0;
    cur_row        = other_data.begin();
    while ( cur_row != other_data.end() ) {
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

    auto totDataLen = comm.sumReduce( dataLen );

    std::vector<gidx_t> aggregateRows( totDataLen );
    std::vector<gidx_t> aggregateCols( totDataLen );
    std::vector<scalar_t> aggregateData( totDataLen );

    comm.allGather( rows.data(), dataLen, aggregateRows.data() );
    comm.allGather( cols.data(), dataLen, aggregateCols.data() );
    comm.allGather( data.data(), dataLen, aggregateData.data() );

    if ( t == AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD ) {
        for ( int i = 0; i != totDataLen; i++ ) {
            if ( ( aggregateRows[i] >= d_first_row ) && ( aggregateRows[i] < d_last_row ) ) {
                if constexpr ( std::is_same_v<gidx_t, size_t> ) {
                    addValuesByGlobalID( 1u,
                                         1u,
                                         &aggregateRows[i],
                                         &aggregateCols[i],
                                         &aggregateData[i],
                                         getTypeID<scalar_t>() );
                } else {
                    size_t row = static_cast<size_t>( aggregateRows[i] );
                    size_t col = static_cast<size_t>( aggregateCols[i] );
                    addValuesByGlobalID(
                        1u, 1u, &row, &col, &aggregateData[i], getTypeID<scalar_t>() );
                }
            }
        }
    } else {

        if ( t == AMP::LinearAlgebra::ScatterType::CONSISTENT_SET ) {
            for ( int i = 0; i != totDataLen; i++ ) {
                if ( ( aggregateRows[i] >= d_first_row ) && ( aggregateRows[i] < d_last_row ) ) {
                    if constexpr ( std::is_same_v<gidx_t, size_t> ) {
                        setValuesByGlobalID( 1u,
                                             1u,
                                             &aggregateRows[i],
                                             &aggregateCols[i],
                                             &aggregateData[i],
                                             getTypeID<scalar_t>() );
                    } else {
                        size_t row = static_cast<size_t>( aggregateRows[i] );
                        size_t col = static_cast<size_t>( aggregateCols[i] );
                        setValuesByGlobalID(
                            1u, 1u, &row, &col, &aggregateData[i], getTypeID<scalar_t>() );
                    }
                }
            }
        }
    }

    other_data.clear();
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixData<Policy, Allocator, DiagMatrixData>::makeConsistent(
    AMP::LinearAlgebra::ScatterType t )
{
    if ( t == AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD )
        setOtherData( d_other_data, AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    else
        setOtherData( d_ghost_data, AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

template<typename Policy, class Allocator, class DiagMatrixData>
std::shared_ptr<Discretization::DOFManager>
CSRMatrixData<Policy, Allocator, DiagMatrixData>::getRightDOFManager() const
{
    return d_rightDOFManager;
}

template<typename Policy, class Allocator, class DiagMatrixData>
std::shared_ptr<Discretization::DOFManager>
CSRMatrixData<Policy, Allocator, DiagMatrixData>::getLeftDOFManager() const
{
    return d_leftDOFManager;
}

/********************************************************
 * Get the number of rows/columns in the matrix          *
 ********************************************************/
template<typename Policy, class Allocator, class DiagMatrixData>
size_t CSRMatrixData<Policy, Allocator, DiagMatrixData>::numLocalRows() const
{
    return static_cast<size_t>( d_last_row - d_first_row );
}

template<typename Policy, class Allocator, class DiagMatrixData>
size_t CSRMatrixData<Policy, Allocator, DiagMatrixData>::numGlobalRows() const
{
    AMP_ASSERT( d_leftDOFManager );
    return d_leftDOFManager->numGlobalDOF();
}

template<typename Policy, class Allocator, class DiagMatrixData>
size_t CSRMatrixData<Policy, Allocator, DiagMatrixData>::numLocalColumns() const
{
    return static_cast<size_t>( d_last_col - d_first_col );
}

template<typename Policy, class Allocator, class DiagMatrixData>
size_t CSRMatrixData<Policy, Allocator, DiagMatrixData>::numGlobalColumns() const
{
    AMP_ASSERT( d_rightDOFManager );
    return d_rightDOFManager->numGlobalDOF();
}

/********************************************************
 * Get iterators                                         *
 ********************************************************/
template<typename Policy, class Allocator, class DiagMatrixData>
size_t CSRMatrixData<Policy, Allocator, DiagMatrixData>::beginRow() const
{
    return static_cast<size_t>( d_first_row );
}

template<typename Policy, class Allocator, class DiagMatrixData>
size_t CSRMatrixData<Policy, Allocator, DiagMatrixData>::endRow() const
{
    return static_cast<size_t>( d_last_row );
}

template<typename Policy, class Allocator, class DiagMatrixData>
size_t CSRMatrixData<Policy, Allocator, DiagMatrixData>::beginCol() const
{
    return static_cast<size_t>( d_first_col );
}

template<typename Policy, class Allocator, class DiagMatrixData>
size_t CSRMatrixData<Policy, Allocator, DiagMatrixData>::endCol() const
{
    return static_cast<size_t>( d_last_col );
}

} // namespace AMP::LinearAlgebra

#endif
