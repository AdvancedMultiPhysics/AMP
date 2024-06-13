#ifndef included_AMP_CSRMatrixData_hpp
#define included_AMP_CSRMatrixData_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/CSRMatrixParameters.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Utilities.h"

#ifdef AMP_USE_UMPIRE
    #include "umpire/Allocator.hpp"
    #include "umpire/ResourceManager.hpp"
#endif

#include <memory>
#include <numeric>
#include <set>
#include <type_traits>

namespace AMP::LinearAlgebra {

/********************************************************
 * Constructor/Destructor helper functions              *
 ********************************************************/
template<typename T>
static T *allocate( size_t N, AMP::Utilities::MemoryType mem_type )
{
    if ( mem_type == AMP::Utilities::MemoryType::host ) {
        std::allocator<T> alloc;
        return alloc.allocate( N );
    } else if ( mem_type == AMP::Utilities::MemoryType::managed ||
                mem_type == AMP::Utilities::MemoryType::device ) {
#ifdef AMP_USE_UMPIRE
        auto &resourceManager = umpire::ResourceManager::getInstance();
        auto alloc            = ( mem_type == AMP::Utilities::MemoryType::managed ) ?
                                    resourceManager.getAllocator( "UM" ) :
                                    resourceManager.getAllocator( "DEVICE" );
        return static_cast<T *>( alloc.allocate( N * sizeof( T ) ) );
#else
        AMP_ERROR( "CSRMatrixData: managed and device memory handling without Umpire has not been "
                   "implemented as yet" );
#endif
    } else {
        AMP_ERROR( "Memory type undefined" );
    }
    return nullptr; // Unreachable
}

template<typename T>
void deallocate( T **data, size_t N, AMP::Utilities::MemoryType mem_type )
{
    if ( mem_type == AMP::Utilities::MemoryType::host ) {
        std::allocator<T> alloc;
        if ( *data ) {
            alloc.deallocate( *data, N );
            *data = nullptr;
        }
    } else if ( mem_type == AMP::Utilities::MemoryType::managed ||
                mem_type == AMP::Utilities::MemoryType::device ) {
#ifdef AMP_USE_UMPIRE
        auto &resourceManager = umpire::ResourceManager::getInstance();
        auto alloc            = ( mem_type == AMP::Utilities::MemoryType::managed ) ?
                                    resourceManager.getAllocator( "UM" ) :
                                    resourceManager.getAllocator( "DEVICE" );
        if ( *data ) {
            alloc.deallocate( data );
            *data = nullptr;
        }
#else
        AMP_ERROR( "CSRMatrixData: managed and device memory handling without Umpire has not been "
                   "implemented as yet" );
#endif
    } else {
        AMP_ERROR( "Memory type undefined" );
    }
}

template<typename Policy>
bool isColValid( typename Policy::gidx_t col,
                 bool is_diag,
                 typename Policy::gidx_t first_col,
                 typename Policy::gidx_t last_col )
{
    bool dValid  = is_diag && ( first_col <= col && col < last_col );
    bool odValid = !is_diag && ( col < first_col || last_col <= col );
    return ( dValid || odValid );
}

/********************************************************
 * Constructors/Destructor                              *
 ********************************************************/
template<typename Policy>
CSRMatrixData<Policy>::CSRMatrixData()
{
    AMPManager::incrementResource( "CSRMatrixData" );
}

template<typename Policy>
CSRMatrixData<Policy>::CSRSerialMatrixData::CSRSerialMatrixData(
    const CSRMatrixData<Policy> &outer )
    : d_outer( outer )
{
    AMPManager::incrementResource( "CSRSerialMatrixData" );
}

template<typename Policy>
CSRMatrixData<Policy>::CSRMatrixData( std::shared_ptr<MatrixParametersBase> params )
    : MatrixData( params )
{
    AMPManager::incrementResource( "CSRMatrixData" );
    auto csrParams = std::dynamic_pointer_cast<CSRMatrixParameters<Policy>>( d_pParameters );
    auto matParams = std ::dynamic_pointer_cast<MatrixParameters>( d_pParameters );

    d_memory_location = d_pParameters->d_memory_location;

    if ( csrParams ) {

        // add check for memory location etc and migrate if necessary
        d_is_square = csrParams->d_is_square;
        d_first_row = csrParams->d_first_row;
        d_last_row  = csrParams->d_last_row;
        d_first_col = csrParams->d_first_col;
        d_last_col  = csrParams->d_last_col;

        size_t N = d_last_row - d_first_row;

        if ( d_memory_location != AMP::Utilities::MemoryType::device ) {
            // Construct on/off diag blocks
            d_diag_matrix     = std::make_shared<CSRSerialMatrixData>( *this, params, true );
            d_off_diag_matrix = std::make_shared<CSRSerialMatrixData>( *this, params, false );

            // get total nnz count
            d_nnz = d_diag_matrix->d_nnz + d_off_diag_matrix->d_nnz;

            // collect off-diagonal entries and create right dof manager
            std::vector<size_t> remote_dofs;
            for ( gidx_t i = 0; i < d_off_diag_matrix->d_nnz; ++i ) {
                remote_dofs.push_back( d_off_diag_matrix->d_cols[i] );
            }
            AMP::Utilities::unique( remote_dofs );
            const auto &comm = getComm();
            d_rightDOFManager =
                std::make_shared<AMP::Discretization::DOFManager>( N, comm, remote_dofs );

            if ( d_is_square ) {
                d_leftDOFManager = d_rightDOFManager;
            } else {
                AMP_ERROR( "Non-square matrices not handled at present" );
            }

        } else {
            AMP_WARNING( "CSRMatrixData: device memory handling has not been implemented yet" );
        }
    } else if ( matParams ) {
        // for now all matrix parameter data is assumed to be on host
        d_leftDOFManager  = matParams->getLeftDOFManager();
        d_rightDOFManager = matParams->getRightDOFManager();
        AMP_ASSERT( d_leftDOFManager && d_rightDOFManager );

        d_is_square = ( d_leftDOFManager->numGlobalDOF() == d_rightDOFManager->numGlobalDOF() );
        d_first_row = d_leftDOFManager->beginDOF();
        d_last_row  = d_leftDOFManager->endDOF();
        d_first_col = d_rightDOFManager->beginDOF();
        d_last_col  = d_rightDOFManager->endDOF();

        // send params forward to the on/off diagonal blocks
        d_diag_matrix     = std::make_shared<CSRSerialMatrixData>( *this, params, true );
        d_off_diag_matrix = std::make_shared<CSRSerialMatrixData>( *this, params, false );
        d_nnz             = d_diag_matrix->d_nnz + d_off_diag_matrix->d_nnz;

    } else {
        AMP_ERROR( "Check supplied MatrixParameter object" );
    }
}

template<typename Policy>
CSRMatrixData<Policy>::~CSRMatrixData()
{
    AMPManager::decrementResource( "CSRMatrixData" );
}

/********************************************************
 * Constructors/Destructor for nested class             *
 ********************************************************/
template<typename Policy>
CSRMatrixData<Policy>::CSRSerialMatrixData::CSRSerialMatrixData(
    const CSRMatrixData<Policy> &outer, std::shared_ptr<MatrixParametersBase> params, bool is_diag )
    : d_outer( outer )
{
    AMPManager::incrementResource( "CSRSerialMatrixData" );
    d_pParameters  = params;
    auto csrParams = std::dynamic_pointer_cast<CSRMatrixParameters<Policy>>( d_pParameters );
    auto matParams = std ::dynamic_pointer_cast<MatrixParameters>( d_pParameters );

    d_memory_location = d_pParameters->d_memory_location;

    // Number of rows owned by this rank
    d_num_rows = outer.d_last_row - outer.d_first_row;

    // row starts always internally allocated
    d_row_starts = allocate<lidx_t>( d_num_rows + 1, d_memory_location );

    if ( csrParams ) {
        d_is_diag = is_diag;

        // memory not managed here regardless of block type (except row starts)
        d_manage_nnz    = false;
        d_manage_coeffs = false;
        d_manage_cols   = false;

        // copy in appropriate data depending on block type
        if ( d_is_diag ) {
            d_nnz_per_row = csrParams->d_nnz_per_row_diag;
            d_cols        = csrParams->d_cols_diag;
            d_cols_loc    = csrParams->d_cols_loc_diag;
            d_coeffs      = csrParams->d_coeffs_diag;
        } else {
            d_nnz_per_row = csrParams->d_nnz_per_row_odiag;
            d_cols        = csrParams->d_cols_odiag;
            d_cols_loc    = csrParams->d_cols_loc_odiag;
            d_coeffs      = csrParams->d_coeffs_odiag;
        }

        // count nnz and decide if block is empty
        d_nnz      = std::accumulate( d_nnz_per_row, d_nnz_per_row + d_num_rows, 0 );
        d_is_empty = ( d_nnz == 0 );

    } else if ( matParams ) {

        // for now all matrix parameter data is assumed to be on host

        auto leftDOFManager  = matParams->getLeftDOFManager();
        auto rightDOFManager = matParams->getRightDOFManager();
        AMP_ASSERT( leftDOFManager && rightDOFManager );
        AMP_ASSERT( matParams->d_CommListLeft && matParams->d_CommListRight );

        d_is_diag  = is_diag;
        d_is_empty = false;

        auto *nnzPerRowAll = matParams->entryList();
        auto &cols         = matParams->getColumns();
        AMP_INSIST(
            !cols.empty(),
            "CSRSerialMatrixData not constructable from MatrixParameters with emtpy columns" );

        // Count number of nonzeros depending on block type
        // also track un-referenced columns if off-diagonal
        std::vector<gidx_t> colPad;
        std::set<gidx_t> colSet;
        d_nnz = 0;
        for ( size_t i = 0; i < cols.size(); ++i ) {
            if ( isColValid<Policy>( cols[i], d_is_diag, outer.d_first_col, outer.d_last_col ) ) {
                d_nnz++;
                if ( !d_is_diag ) {
                    colSet.insert( cols[i] );
                }
            }
        }
        // attempt to insert all remote dofs into colSet to see which are un-referenced
        if ( !d_is_diag ) {
            auto remoteDOFs = rightDOFManager->getRemoteDOFs();
            for ( auto rdof : remoteDOFs ) {
                auto cs = colSet.insert( rdof );
                if ( cs.second ) {
                    // insertion success means this DOF is un-referenced
                    // add it to the padding list
                    colPad.push_back( rdof );
                }
            }
            // colPad now holds the un-referenced global ghost
            // indices that get padded in below
            d_nnz += colPad.size();
        }

        // bail out for degenerate case with no nnz
        // may happen in off-diagonal blocks
        if ( d_nnz == 0 ) {
            d_is_empty      = true;
            d_manage_nnz    = false;
            d_manage_coeffs = false;
            d_manage_cols   = false;
            return;
        }

        // allocate internal arrays either directly or through Umpire
        d_manage_nnz    = true;
        d_manage_coeffs = true;
        d_manage_cols   = true;

        d_nnz_per_row = allocate<lidx_t>( d_num_rows, d_memory_location );
        d_cols        = allocate<gidx_t>( d_nnz, d_memory_location );
        d_cols_loc    = allocate<lidx_t>( d_nnz, d_memory_location );
        d_coeffs      = allocate<scalar_t>( d_nnz, d_memory_location );

        // Fill cols and nnz based on local row extents and on/off diag status
        lidx_t cgi = 0, cli = 0; // indices into global and local arrays of columns
        gidx_t nnzFilled = 0;
        for ( gidx_t i = 0; i < d_num_rows; ++i ) {
            d_nnz_per_row[i] = 0;
            for ( lidx_t j = 0; j < nnzPerRowAll[i]; ++j ) {
                auto col = cols[cgi++];
                if ( isColValid<Policy>( col, d_is_diag, outer.d_first_col, outer.d_last_col ) ) {
                    d_nnz_per_row[i]++;
                    d_cols[cli] = col;
                    if ( d_is_diag ) {
                        d_cols_loc[cli] = static_cast<lidx_t>( col - outer.d_first_col );
                    } else {
                        d_cols_loc[cli] = static_cast<lidx_t>(
                            matParams->d_CommListRight->getLocalGhostID( col ) );
                    }
                    d_coeffs[cli] = 0.0;
                    cli++;
                    nnzFilled++;
                }
            }
        }

        // If off-diag pad in the un-referenced ghosts to the final row
        if ( !d_is_diag ) {
            lidx_t nPad = colPad.size();
            d_nnz_per_row[d_num_rows - 1] += nPad;
            for ( auto col : colPad ) {
                d_cols[cli] = col;
                d_cols_loc[cli] =
                    static_cast<lidx_t>( matParams->d_CommListRight->getLocalGhostID( col ) );
                d_coeffs[cli] = 0.0;
                cli++;
                nnzFilled++;
            }
        }

        // Ensure that the right number of nnz were actually filled in
        AMP_DEBUG_ASSERT( nnzFilled == d_nnz );
    } else {
        AMP_ERROR( "Check supplied MatrixParameter object" );
    }

    // Fill in row starts
    if ( !d_is_empty ) {
        // scan nnz counts to get starting index of each row
        std::exclusive_scan( d_nnz_per_row, d_nnz_per_row + d_num_rows, d_row_starts, 0 );
        d_row_starts[d_num_rows] = d_row_starts[d_num_rows - 1] + d_nnz_per_row[d_num_rows - 1];
    }
}

template<typename Policy>
CSRMatrixData<Policy>::CSRSerialMatrixData::~CSRSerialMatrixData()
{
    AMPManager::decrementResource( "CSRSerialMatrixData" );
    auto matParams = std ::dynamic_pointer_cast<MatrixParameters>( d_pParameters );

    if ( matParams ) {
        deallocate<lidx_t>( &d_row_starts, d_num_rows + 1, d_memory_location );
        if ( d_manage_cols ) {
            deallocate<gidx_t>( &d_cols, d_nnz, d_memory_location );
            deallocate<lidx_t>( &d_cols_loc, d_nnz, d_memory_location );
        }
        if ( d_manage_nnz ) {
            deallocate<lidx_t>( &d_nnz_per_row, d_num_rows, d_memory_location );
        }
        if ( d_manage_coeffs ) {
            deallocate<scalar_t>( &d_coeffs, d_nnz, d_memory_location );
        }
    }
}

template<typename Policy>
std::shared_ptr<MatrixData> CSRMatrixData<Policy>::cloneMatrixData() const
{
    std::shared_ptr<CSRMatrixData> cloneData;

    cloneData = std::make_shared<CSRMatrixData<Policy>>();

    cloneData->d_memory_location = d_memory_location;
    cloneData->d_is_square       = d_is_square;
    cloneData->d_first_row       = d_first_row;
    cloneData->d_last_row        = d_last_row;
    cloneData->d_first_col       = d_first_col;
    cloneData->d_last_col        = d_last_col;
    cloneData->d_nnz             = d_nnz;
    cloneData->d_leftDOFManager  = d_leftDOFManager;
    cloneData->d_rightDOFManager = d_rightDOFManager;
    cloneData->d_pParameters     = d_pParameters;

    cloneData->d_diag_matrix     = d_diag_matrix->cloneMatrixData( *cloneData );
    cloneData->d_off_diag_matrix = d_off_diag_matrix->cloneMatrixData( *cloneData );

    return cloneData;
}

template<typename Policy>
std::shared_ptr<typename CSRMatrixData<Policy>::CSRSerialMatrixData>
CSRMatrixData<Policy>::CSRSerialMatrixData::cloneMatrixData( const CSRMatrixData<Policy> &outer )
{
    std::shared_ptr<CSRSerialMatrixData> cloneData;

    cloneData = std::make_shared<CSRSerialMatrixData>( outer );

    cloneData->d_is_diag         = d_is_diag;
    cloneData->d_is_empty        = d_is_empty;
    cloneData->d_num_rows        = d_num_rows;
    cloneData->d_nnz             = d_nnz;
    cloneData->d_memory_location = d_memory_location;
    cloneData->d_pParameters     = d_pParameters;

    if ( !d_is_empty ) {
        cloneData->d_manage_nnz    = true;
        cloneData->d_manage_coeffs = true;
        cloneData->d_manage_cols   = true;

        cloneData->d_nnz_per_row = allocate<lidx_t>( d_num_rows, d_memory_location );
        cloneData->d_cols        = allocate<gidx_t>( d_nnz, d_memory_location );
        cloneData->d_cols_loc    = allocate<lidx_t>( d_nnz, d_memory_location );
        cloneData->d_coeffs      = allocate<scalar_t>( d_nnz, d_memory_location );
        cloneData->d_row_starts  = allocate<lidx_t>( d_num_rows + 1, d_memory_location );

        if ( d_memory_location < AMP::Utilities::MemoryType::device ) {
            std::copy( d_nnz_per_row, d_nnz_per_row + d_num_rows, cloneData->d_nnz_per_row );
            std::copy( d_row_starts, d_row_starts + d_num_rows + 1, cloneData->d_row_starts );
            std::copy( d_cols, d_cols + d_nnz, cloneData->d_cols );
            std::copy( d_cols_loc, d_cols_loc + d_nnz, cloneData->d_cols_loc );
        } else {
            AMP_ERROR( "Device memory copies not implemented as yet" );
        }
    } else {
        cloneData->d_manage_nnz    = false;
        cloneData->d_manage_coeffs = false;
        cloneData->d_manage_cols   = false;

        cloneData->d_nnz_per_row = nullptr;
        cloneData->d_cols        = nullptr;
        cloneData->d_cols_loc    = nullptr;
        cloneData->d_coeffs      = nullptr;
        cloneData->d_row_starts  = nullptr;
    }

    return cloneData;
}

// This function generates, but does not own, a column map that is usable by hypre
// This is only useful if called on the off-diagonal block
template<typename Policy>
void CSRMatrixData<Policy>::CSRSerialMatrixData::generateColumnMap(
    std::vector<gidx_t> &colMap ) const
{
    // Don't do anything if empty
    if ( d_is_empty ) {
        return;
    }

    // Find number of unique columns
    std::set<gidx_t> colSet( d_cols, d_cols + d_nnz );

    // Resize and fill colMap
    colMap.resize( colSet.size() );
    std::copy( colSet.begin(), colSet.end(), colMap.begin() );
}

template<typename Policy>
std::shared_ptr<MatrixData> CSRMatrixData<Policy>::transpose() const
{
    AMP_ERROR( "Not implemented" );
}

template<typename Policy>
void CSRMatrixData<Policy>::extractDiagonal( std::shared_ptr<Vector> buf ) const
{
    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 ); // temporary constraint
    AMP_ASSERT( buf->isType<scalar_t>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<scalar_t>();
    auto memType     = AMP::Utilities::getMemoryType( rawVecData );
    if ( memType < AMP::Utilities::MemoryType::device ) {

        const size_t N = d_last_row - d_first_row;
        for ( size_t i = 0; i < N; ++i ) {
            const auto start = d_diag_matrix->d_row_starts[i];
            const auto end   = d_diag_matrix->d_row_starts[i + 1];
            // colums are unordered at present
            for ( lidx_t j = start; j < end; ++j ) {
                if ( d_diag_matrix->d_cols[j] == static_cast<gidx_t>( d_first_col + i ) ) {
                    rawVecData[i] = d_diag_matrix->d_coeffs[j];
                    break;
                }
            }
        }
    } else {
        AMP_ERROR(
            "CSRSerialMatrixData<Policy>::extractDiagonal not implemented for vec and matrix in "
            "different memory spaces" );
    }
}
template<typename Policy>
void CSRMatrixData<Policy>::getRowByGlobalID( size_t row,
                                              std::vector<size_t> &cols,
                                              std::vector<double> &vals ) const
{
    AMP_INSIST( row >= static_cast<size_t>( d_first_row ) &&
                    row < static_cast<size_t>( d_last_row ),
                "row must be owned by rank" );

    auto local_row = row - d_first_row;

    // Get portion of row from diagonal matrix
    d_diag_matrix->getRowByGlobalID( local_row, cols, vals );

    // Get portion from off diagonal and append
    std::vector<size_t> od_cols;
    std::vector<double> od_vals;
    d_off_diag_matrix->getRowByGlobalID( local_row, od_cols, od_vals );
    cols.insert( cols.end(), od_cols.begin(), od_cols.end() );
    vals.insert( vals.end(), od_vals.begin(), od_vals.end() );
}

template<typename Policy>
void CSRMatrixData<Policy>::CSRSerialMatrixData::getRowByGlobalID(
    const size_t local_row, std::vector<size_t> &cols, std::vector<double> &values ) const
{
    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return;
    }
    auto memType = AMP::Utilities::getMemoryType( d_cols );

    if ( memType < AMP::Utilities::MemoryType::device ) {
        const auto row_offset = static_cast<size_t>( local_row );
        const auto offset     = std::accumulate( d_nnz_per_row, d_nnz_per_row + row_offset, 0 );
        const auto n          = d_nnz_per_row[row_offset];

        cols.resize( n );
        values.resize( n );

        if constexpr ( std::is_same_v<size_t, gidx_t> ) {
            std::copy( &d_cols[offset], &d_cols[offset] + n, cols.begin() );
        } else {
            std::transform( &d_cols[offset],
                            &d_cols[offset] + n,
                            cols.begin(),
                            []( size_t col ) -> gidx_t { return col; } );
        }

        if constexpr ( std::is_same_v<double, scalar_t> ) {
            std::copy( &d_coeffs[offset], &d_coeffs[offset] + n, values.begin() );
        } else {
            std::transform( &d_coeffs[offset],
                            &d_coeffs[offset] + n,
                            values.begin(),
                            []( size_t val ) -> scalar_t { return val; } );
        }
    } else {
        AMP_ERROR( "CSRSerialMatrixData::getRowByGlobalID not implemented for device memory" );
    }
}

template<typename Policy>
void CSRMatrixData<Policy>::getValuesByGlobalID( size_t num_rows,
                                                 size_t num_cols,
                                                 size_t *rows,
                                                 size_t *cols,
                                                 void *values,
                                                 const typeID &id ) const
{
    if ( getTypeID<scalar_t>() == id ) {
        if ( d_memory_location < AMP::Utilities::MemoryType::device ) {

            if ( num_rows == 1 && num_cols == 1 ) {

                const auto local_row = rows[0] - d_first_row;
                // Forward to internal matrices, nothing will happen if not found
                d_diag_matrix->getValuesByGlobalID( local_row, cols[0], values, id );
                d_off_diag_matrix->getValuesByGlobalID( local_row, cols[0], values, id );
            } else {
                AMP_ERROR(
                    "CSRSerialMatrixData::getValuesByGlobalID not implemented for num_rows>1 || "
                    "num_cols > 1" );
            }

        } else {
            AMP_ERROR(
                "CSRSerialMatrixData::getValuesByGlobalID not implemented for device memory" );
        }
    } else {
        AMP_ERROR( "Not implemented" );
    }
}

template<typename Policy>
void CSRMatrixData<Policy>::CSRSerialMatrixData::getValuesByGlobalID( const size_t local_row,
                                                                      const size_t col,
                                                                      void *values,
                                                                      const typeID &id ) const
{
    if ( getTypeID<scalar_t>() != id ) {
        AMP_ERROR( "Conversion not implemented" );
    }
    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return;
    }

    const auto start = d_row_starts[local_row];
    const auto end   = d_row_starts[local_row + 1];

    for ( lidx_t i = start; i < end; ++i ) {
        if ( d_cols[i] == static_cast<gidx_t>( col ) ) {
            *( reinterpret_cast<scalar_t *>( values ) ) = d_coeffs[i];
        }
    }
}

// The two getValues functions above can directly forward to the diag and off diag blocks
// The addValuesByGlobalID and setValuesByGlobalID functions can't do this since
// they need to also handle the other_data case
template<typename Policy>
void CSRMatrixData<Policy>::addValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    if ( getTypeID<scalar_t>() != id ) {
        AMP_ERROR( "Conversion not implemented" );
    }

    if ( d_memory_location < AMP::Utilities::MemoryType::device ) {
        
        auto values = reinterpret_cast<const scalar_t *>( vals );
	
	for ( size_t i = 0u; i != num_rows; i++ ) {
	    if ( rows[i] >= static_cast<size_t>( d_first_row ) &&
		 rows[i] < static_cast<size_t>( d_last_row ) ) {
	      
	        // Forward single row to diag and off diag blocks
	        // auto lcols = &cols[num_cols * i];
	        const auto local_row = rows[i] - d_first_row;
		auto lvals           = &values[num_cols * i];
		d_diag_matrix->addValuesByGlobalID( num_cols, local_row, cols, lvals, id );
		d_off_diag_matrix->addValuesByGlobalID( num_cols, local_row, cols, lvals, id );
	    } else {
	        for ( size_t icol = 0; icol < num_cols; ++icol ) {
		    d_other_data[rows[i]][cols[icol]] += values[num_cols * i + icol];
		}
	    }
	}
	
    } else {
      AMP_ERROR( "CSRMatrixData::addValuesByGlobalID not implemented for device memory" );
    }
}

template<typename Policy>
void CSRMatrixData<Policy>::CSRSerialMatrixData::addValuesByGlobalID( const size_t num_cols,
                                                                      const size_t local_row,
                                                                      const size_t *cols,
                                                                      const scalar_t *vals,
                                                                      const typeID &id )
{
    if ( d_is_empty ) {
        return;
    }

    if ( getTypeID<scalar_t>() != id ) {
        AMP_ERROR( "Conversion not implemented" );
    }
    
    const auto start = d_row_starts[local_row];
    const auto end   = d_row_starts[local_row + 1];
    // Inefficient because we don't assume order
    // not sure it's worth optimizing for our use cases
    for ( size_t icol = 0; icol < num_cols; ++icol ) {
        for ( lidx_t j = start; j < end; ++j ) {
            if ( d_cols[j] == static_cast<gidx_t>( cols[icol] ) ) {
                d_coeffs[j] += vals[icol];
            }
        }
    }
}

template<typename Policy>
void CSRMatrixData<Policy>::setValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    if ( getTypeID<scalar_t>() != id ) {
        AMP_ERROR( "Conversion not implemented" );
    }
    
    if ( d_memory_location < AMP::Utilities::MemoryType::device ) {
      
        auto values = reinterpret_cast<const scalar_t *>( vals );
	
	for ( size_t i = 0u; i != num_rows; i++ ) {
	  
	    if ( rows[i] >= static_cast<size_t>( d_first_row ) &&
		 rows[i] < static_cast<size_t>( d_last_row ) ) {
	      
	        // Forward single row to diag and off diag blocks
	        // auto lcols = &cols[num_cols * i];
	        const auto local_row = rows[i] - d_first_row;
		auto lvals           = &values[num_cols * i];
		d_diag_matrix->setValuesByGlobalID( num_cols, local_row, cols, lvals, id );
		d_off_diag_matrix->setValuesByGlobalID( num_cols, local_row, cols, lvals, id );
	    } else {
	        for ( size_t icol = 0; icol < num_cols; ++icol ) {
		    d_ghost_data[rows[i]][cols[icol]] = values[num_cols * i + icol];
		}
	    }
	}
	
    } else {
      AMP_ERROR( "CSRMatrixData::addValuesByGlobalID not implemented for device memory" );
    }
}

template<typename Policy>
void CSRMatrixData<Policy>::CSRSerialMatrixData::setValuesByGlobalID( const size_t num_cols,
                                                                      const size_t local_row,
                                                                      const size_t *cols,
                                                                      const scalar_t *vals,
                                                                      const typeID &id )
{
    if ( d_is_empty ) {
        return;
    }
    
    if ( getTypeID<scalar_t>() != id ) {
        AMP_ERROR( "Conversion not implemented" );
    }

    const auto start = d_row_starts[local_row];
    const auto end   = d_row_starts[local_row + 1];
    // Inefficient because we don't assume order
    // not sure it's worth optimizing for our use cases
    for ( size_t icol = 0; icol < num_cols; ++icol ) {
        for ( lidx_t j = start; j < end; ++j ) {
            if ( d_cols[j] == static_cast<gidx_t>( cols[icol] ) ) {
                d_coeffs[j] = vals[icol];
            }
        }
    }
}

template<typename Policy>
void CSRMatrixData<Policy>::setOtherData( std::map<gidx_t, std::map<gidx_t, scalar_t>> &other_data,
                                          AMP::LinearAlgebra::ScatterType t )
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
                addValuesByGlobalID( 1u,
                                     1u,
                                     (size_t *) &aggregateRows[i],
                                     (size_t *) &aggregateCols[i],
                                     &aggregateData[i],
                                     getTypeID<scalar_t>() );
            }
        }
    } else {

        if ( t == AMP::LinearAlgebra::ScatterType::CONSISTENT_SET ) {
            for ( int i = 0; i != totDataLen; i++ ) {
                if ( ( aggregateRows[i] >= d_first_row ) && ( aggregateRows[i] < d_last_row ) ) {
                    setValuesByGlobalID( 1u,
                                         1u,
                                         (size_t *) &aggregateRows[i],
                                         (size_t *) &aggregateCols[i],
                                         &aggregateData[i],
                                         getTypeID<scalar_t>() );
                }
            }
        }
    }

    other_data.clear();
}

template<typename Policy>
void CSRMatrixData<Policy>::makeConsistent( AMP::LinearAlgebra::ScatterType t )
{
    if ( t == AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD )
        setOtherData( d_other_data, AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    else
        setOtherData( d_ghost_data, AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


template<typename Policy>
std::vector<size_t> CSRMatrixData<Policy>::getColumnIDs( size_t row ) const
{
    AMP_INSIST( row >= static_cast<size_t>( d_first_row ) &&
                    row < static_cast<size_t>( d_last_row ),
                "row must be owned by rank" );
    AMP_INSIST( d_diag_matrix, "diag matrix must exist" );
    auto local_row              = row - d_first_row;
    std::vector<size_t> cols    = d_diag_matrix->getColumnIDs( local_row );
    std::vector<size_t> od_cols = d_off_diag_matrix->getColumnIDs( local_row );
    cols.insert( cols.end(), od_cols.begin(), od_cols.end() );
    return cols;
}

template<typename Policy>
std::vector<size_t>
CSRMatrixData<Policy>::CSRSerialMatrixData::getColumnIDs( const size_t local_row ) const
{
    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return std::vector<size_t>();
    }

    AMP_INSIST( d_cols && d_nnz_per_row, "Must be initialized" );

    auto memType = AMP::Utilities::getMemoryType( d_cols );

    if ( memType < AMP::Utilities::MemoryType::device ) {

        std::vector<size_t> cols;
        const auto row_offset = static_cast<size_t>( local_row );
        const auto offset     = std::accumulate( d_nnz_per_row, d_nnz_per_row + row_offset, 0 );
        const auto n          = d_nnz_per_row[row_offset];

        if constexpr ( std::is_same_v<size_t, gidx_t> ) {
            std::copy( &d_cols[offset], &d_cols[offset] + n, std::back_inserter( cols ) );
        } else {
            std::transform( &d_cols[offset],
                            &d_cols[offset] + n,
                            std::back_inserter( cols ),
                            []( size_t col ) -> gidx_t { return col; } );
        }
        return cols;
    } else {
        AMP_ERROR( "CSRSerialMatrixData:getColumnIDs not implemented for device memory" );
    }
}

template<typename Policy>
std::shared_ptr<Discretization::DOFManager> CSRMatrixData<Policy>::getRightDOFManager() const
{
    return d_rightDOFManager;
}

template<typename Policy>
std::shared_ptr<Discretization::DOFManager> CSRMatrixData<Policy>::getLeftDOFManager() const
{
    return d_leftDOFManager;
}

/********************************************************
 * Get the number of rows/columns in the matrix          *
 ********************************************************/
template<typename Policy>
size_t CSRMatrixData<Policy>::numLocalRows() const
{
    return static_cast<size_t>( d_last_row - d_first_row );
}

template<typename Policy>
size_t CSRMatrixData<Policy>::numGlobalRows() const
{
    AMP_ASSERT( d_leftDOFManager );
    return d_leftDOFManager->numGlobalDOF();
}

template<typename Policy>
size_t CSRMatrixData<Policy>::numLocalColumns() const
{
    return static_cast<size_t>( d_last_col - d_first_col );
}

template<typename Policy>
size_t CSRMatrixData<Policy>::numGlobalColumns() const
{
    AMP_ASSERT( d_rightDOFManager );
    return d_rightDOFManager->numGlobalDOF();
}

/********************************************************
 * Get iterators                                         *
 ********************************************************/
template<typename Policy>
size_t CSRMatrixData<Policy>::beginRow() const
{
    return static_cast<size_t>( d_first_row );
}

template<typename Policy>
size_t CSRMatrixData<Policy>::endRow() const
{
    return static_cast<size_t>( d_last_row );
}

} // namespace AMP::LinearAlgebra

#endif
