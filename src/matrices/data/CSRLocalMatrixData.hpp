#ifndef included_AMP_CSRLocalMatrixData_hpp
#define included_AMP_CSRLocalMatrixData_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/AMPCSRMatrixParameters.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/RawCSRMatrixParameters.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixDataHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Utilities.h"

#include <numeric>
#include <set>
#include <type_traits>

namespace AMP::LinearAlgebra {

template<typename Config>
bool isColValid( typename Config::gidx_t col,
                 bool is_diag,
                 typename Config::gidx_t first_col,
                 typename Config::gidx_t last_col )
{
    bool dValid  = is_diag && ( first_col <= col && col < last_col );
    bool odValid = !is_diag && ( col < first_col || last_col <= col );
    return ( dValid || odValid );
}

template<typename data_type>
std::shared_ptr<data_type[]> sharedArrayWrapper( data_type *raw_array )
{
    return std::shared_ptr<data_type[]>( raw_array, []( auto p ) -> void { (void) p; } );
}

template<typename Config>
CSRLocalMatrixData<Config>::CSRLocalMatrixData( std::shared_ptr<MatrixParametersBase> params,
                                                typename Config::gidx_t first_row,
                                                typename Config::gidx_t last_row,
                                                typename Config::gidx_t first_col,
                                                typename Config::gidx_t last_col,
                                                bool is_diag,
                                                bool is_symbolic,
                                                uint64_t hash )
    : d_first_row( first_row ),
      d_last_row( last_row ),
      d_first_col( first_col ),
      d_last_col( last_col ),
      d_is_diag( is_diag ),
      d_is_symbolic( is_symbolic ),
      d_num_rows( last_row - first_row ),
      d_hash( hash )
{
    AMPManager::incrementResource( "CSRLocalMatrixData" );
    PROFILE( "CSRLocalMatrixData::constructor" );

    // Figure out what kind of parameters object we have
    // Note: matParams always true if ampCSRParams is by inheritance
    auto rawCSRParams = std::dynamic_pointer_cast<RawCSRMatrixParameters<Config>>( params );
    auto ampCSRParams = std::dynamic_pointer_cast<AMPCSRMatrixParameters<Config>>( params );
    auto matParams    = std ::dynamic_pointer_cast<MatrixParameters>( params );

    if ( rawCSRParams ) {
        AMP_INSIST( !d_is_symbolic,
                    "CSRLocalMatrixData: Can not construct symbolic matrix from raw parameters" );

        // Pull out block specific parameters
        auto &blParams = d_is_diag ? rawCSRParams->d_diag : rawCSRParams->d_off_diag;

        // we guarantee that row_starts always exists, even for empty matrices
        if ( blParams.d_row_starts == nullptr ) {
            d_is_empty   = true;
            d_nnz        = 0;
            d_row_starts = makeLidxArray( d_num_rows + 1 );
            AMP::Utilities::Algorithms::zero_n(
                d_row_starts.get(), d_num_rows + 1, Config::mem_loc );
            return;
        }

        // count nnz and decide if block is empty
        // row starts may not be host-accessible, so do a copy to get last entry
        lidx_t nnz;
        AMP::Utilities::Algorithms::copy_n( &nnz,
                                            AMP::Utilities::MemoryType::host,
                                            &blParams.d_row_starts[d_num_rows],
                                            Config::mem_loc,
                                            1 );
        d_nnz      = nnz;
        d_is_empty = ( d_nnz == 0 );

        // Wrap raw pointers from blParams to match internal
        // shared_ptr<T[]> type
        d_row_starts = sharedArrayWrapper( blParams.d_row_starts );
        d_cols       = sharedArrayWrapper( blParams.d_cols );
        d_coeffs     = sharedArrayWrapper( blParams.d_coeffs );
    } else if ( matParams ) {
        // can always allocate row starts without external information
        d_row_starts = makeLidxArray( d_num_rows + 1 );
        AMP::Utilities::Algorithms::zero_n( d_row_starts.get(), d_num_rows + 1, Config::mem_loc );

        const auto &getRow = matParams->getRowFunction();

        if ( !getRow || ampCSRParams ) {
            // Initialization not desired or not possible
            // can be set later by calling setNNZ and filling d_cols in some fashion
            d_nnz      = 0;
            d_is_empty = true;
            return;
        }

        AMP_INSIST( d_memory_location != AMP::Utilities::MemoryType::device,
                    "CSRLocalMatrixData: construction from MatrixParameters on device not yet "
                    "supported. Try building from AMPCSRMatrixParameters." );

        // Count number of nonzeros per row and total
        d_nnz = 0;
        for ( gidx_t i = d_first_row; i < d_last_row; ++i ) {
            lidx_t valid_nnz = 0;
            for ( auto &&col : getRow( i ) ) {
                if ( isColValid<Config>( col, d_is_diag, d_first_col, d_last_col ) ) {
                    ++valid_nnz;
                }
            }
            d_row_starts[i - d_first_row] = valid_nnz;
            d_nnz += valid_nnz;
        }

        // bail out for degenerate case with no nnz
        // may happen in off-diagonal blocks
        if ( d_nnz == 0 ) {
            d_is_empty = true;
            return;
        }
        d_is_empty = false;

        // Allocate internal arrays
        d_cols = makeGidxArray( d_nnz );
        if ( !d_is_symbolic ) {
            d_coeffs = makeScalarArray( d_nnz );
        }

        // Fill cols and nnz based on local row extents and on/off diag status
        lidx_t nnzFilled = 0;
        lidx_t nnzCached = d_row_starts[0];
        d_row_starts[0]  = 0;
        for ( lidx_t row = 0; row < d_num_rows; ++row ) {
            // do exclusive scan on nnz per row along the way
            lidx_t rs             = d_row_starts[row] + nnzCached;
            nnzCached             = d_row_starts[row + 1];
            d_row_starts[row + 1] = rs;
            // fill in valid columns from getRow function
            auto cols = getRow( d_first_row + row );
            for ( auto &&col : cols ) {
                if ( isColValid<Config>( col, d_is_diag, d_first_col, d_last_col ) ) {
                    d_cols[nnzFilled] = col;
                    if ( !d_is_symbolic ) {
                        d_coeffs[nnzFilled] = 0.0;
                    }
                    ++nnzFilled;
                }
            }
        }

        // Ensure that the right number of nnz were actually filled in
        AMP_DEBUG_ASSERT( nnzFilled == d_nnz );
    } else {
        // The parameters object is allowed to be null
        // In this case matrices will stay purely local (e.g. not parts of
        // an enclosing CSRMatrixData object). This is used for remote blocks
        // in SpGEMM
        d_nnz        = 0;
        d_is_empty   = true;
        d_row_starts = makeLidxArray( d_num_rows + 1 );
        AMP::Utilities::Algorithms::zero_n( d_row_starts.get(), d_num_rows + 1, Config::mem_loc );
        return;
    }

    // fill in local column indices
    globalToLocalColumns();
}

template<typename Config>
size_t *CSRLocalMatrixData<Config>::getColumnMapSizeT() const
{
    if ( d_is_diag ) {
        return nullptr;
    }
    if ( !d_cols_unq_size_t ) {
        d_cols_unq_size_t = sharedArrayBuilder<size_t>( d_ncols_unq );
        AMP::Utilities::Algorithms::copyCast( d_cols_unq_size_t.get(),
                                              Config::mem_loc,
                                              d_cols_unq.get(),
                                              Config::mem_loc,
                                              d_ncols_unq );
    }
    return d_cols_unq_size_t.get();
}

template<typename Config>
typename Config::scalar_t *CSRLocalMatrixData<Config>::getGhostCache() const
{
    if ( d_is_diag ) {
        return nullptr;
    }
    if ( !d_ghost_cache ) {
        d_ghost_cache = makeScalarArray( d_ncols_unq );
    }
    return d_ghost_cache.get();
}

template<typename Config>
std::string CSRLocalMatrixData<Config>::type() const
{
    std::string tname = "CSRLocalMatrixData<";
    std::string c     = ", ";
    std::string s0    = getTypeID<allocator_type>().name;
    std::string s1    = getTypeID<gidx_t>().name;
    std::string s2    = getTypeID<lidx_t>().name;
    std::string s3    = getTypeID<scalar_t>().name;
    return tname + c + s0 + c + s1 + c + s2 + c + s3 + ">";
}

template<typename Config>
std::shared_ptr<CSRLocalMatrixData<Config>> CSRLocalMatrixData<Config>::ConcatHorizontal(
    std::shared_ptr<MatrixParametersBase> params,
    std::map<int, std::shared_ptr<CSRLocalMatrixData<Config>>> blocks )
{
    PROFILE( "CSRLocalMatrixData::ConcatHorizontal" );

    AMP_INSIST( blocks.size() > 0, "Attempted to concatenate empty set of blocks" );

    // Verify that all have matching row/col starts/stops
    // Blocks must have valid global columns present
    // Count total number of non-zeros in each row from combination.
    auto block           = ( *blocks.begin() ).second;
    const auto mem_loc   = block->d_memory_location;
    const auto first_row = block->d_first_row;
    const auto last_row  = block->d_last_row;
    const auto nrows     = static_cast<lidx_t>( last_row - first_row );
    const auto first_col = block->d_first_col;
    const auto last_col  = block->d_last_col;
    bool all_empty       = block->isEmpty();
    for ( auto it : blocks ) {
        block = it.second;
        if ( block->isEmpty() ) {
            continue;
        } else {
            all_empty = false;
        }
        AMP_INSIST( first_row == block->d_first_row && last_row == block->d_last_row &&
                        first_col == block->d_first_col && last_col == block->d_last_col,
                    "Blocks to concatenate must have compatible layouts" );
        AMP_INSIST( block->d_cols.get(), "Blocks to concatenate must have global columns" );
        AMP_INSIST( mem_loc == block->d_memory_location,
                    "Blocks to concatenate must be in same memory space" );
        AMP_INSIST( !block->d_is_symbolic, "Blocks to concatenate can't be symbolic" );
    }

    // extreme edge case where every block happened to be empty
    if ( all_empty ) {
        return nullptr;
    }

    // Create output matrix
    auto concat_matrix = std::make_shared<CSRLocalMatrixData<Config>>(
        params, first_row, last_row, first_col, last_col, false );

    // count number of non-zeros in each row
    for ( auto it : blocks ) {
        block = it.second;
        if ( block->isEmpty() ) {
            continue;
        }
        CSRMatrixDataHelpers<Config>::ConcatHorizontalCountNNZ(
            block->d_row_starts.get(), nrows, concat_matrix->d_row_starts.get() );
    }

    // trigger allocations
    concat_matrix->setNNZ( true );

    // Create counters for non-zeros entered into each row
    auto row_nnz_ctrs = makeLidxArray( nrows );
    AMP::Utilities::Algorithms::zero_n( row_nnz_ctrs.get(), nrows, Config::mem_loc );

    // loop back over blocks and write into new matrix
    for ( auto it : blocks ) {
        block = it.second;
        if ( block->isEmpty() ) {
            continue;
        }
        CSRMatrixDataHelpers<Config>::ConcatHorizontalFill( block->d_row_starts.get(),
                                                            block->d_cols.get(),
                                                            block->d_coeffs.get(),
                                                            nrows,
                                                            concat_matrix->d_row_starts.get(),
                                                            row_nnz_ctrs.get(),
                                                            concat_matrix->d_cols.get(),
                                                            concat_matrix->d_coeffs.get() );
    }

    return concat_matrix;
}

template<typename Config>
std::shared_ptr<CSRLocalMatrixData<Config>> CSRLocalMatrixData<Config>::ConcatVertical(
    std::shared_ptr<MatrixParametersBase> params,
    std::map<int, std::shared_ptr<CSRLocalMatrixData<Config>>> blocks,
    const gidx_t first_col,
    const gidx_t last_col,
    const bool is_diag )
{
    PROFILE( "CSRLocalMatrixData::ConcatVertical" );

    AMP_INSIST( blocks.size() > 0, "Attempted to concatenate empty set of blocks" );

    // count number of rows and check compatibility of blocks
    auto block      = ( *blocks.begin() ).second;
    lidx_t num_rows = 0;
    bool all_empty  = block->isEmpty();
    for ( auto it : blocks ) {
        block = it.second;
        AMP_INSIST( !block->d_is_symbolic, "Blocks to concatenate can't be symbolic" );
        num_rows += block->d_num_rows;
        all_empty = all_empty && block->isEmpty();
    }

    // extreme edge case where every block happened to be empty
    if ( all_empty ) {
        return nullptr;
    }

    // create output matrix
    auto concat_matrix = std::make_shared<CSRLocalMatrixData<Config>>(
        params, 0, num_rows, first_col, last_col, is_diag );
    // Count total number of non-zeros in each row from combination.
    lidx_t cat_row = 0; // counter for which row we are on in concat_matrix
    for ( auto it : blocks ) {
        block = it.second;
        CSRMatrixDataHelpers<Config>::ConcatVerticalCountNNZ(
            block->d_row_starts.get(),
            block->d_cols.get(),
            block->d_num_rows,
            first_col,
            last_col,
            is_diag,
            &concat_matrix->d_row_starts[cat_row] );
        cat_row += block->d_num_rows;
    }

    // Trigger allocations
    concat_matrix->setNNZ( true );

    // loop over blocks again and write into new matrix
    cat_row = 0;
    for ( auto it : blocks ) {
        block = it.second;
        if ( !block->d_is_empty ) {
            CSRMatrixDataHelpers<Config>::ConcatVerticalFill( block->d_row_starts.get(),
                                                              block->d_cols.get(),
                                                              block->d_coeffs.get(),
                                                              block->d_num_rows,
                                                              first_col,
                                                              last_col,
                                                              is_diag,
                                                              cat_row,
                                                              concat_matrix->d_row_starts.get(),
                                                              concat_matrix->d_cols.get(),
                                                              concat_matrix->d_coeffs.get() );
        }
        cat_row += block->d_num_rows;
    }

    return concat_matrix;
}

template<typename Config>
void CSRLocalMatrixData<Config>::swapDataFields( CSRLocalMatrixData<Config> &other )
{
    AMP_ASSERT( d_is_symbolic == other.d_is_symbolic );
    // swap metadata
    const auto o_is_empty  = other.d_is_empty;
    const auto o_nnz       = other.d_nnz;
    const auto o_ncols_unq = other.d_ncols_unq;
    other.d_is_empty       = d_is_empty;
    other.d_nnz            = d_nnz;
    other.d_ncols_unq      = d_ncols_unq;
    d_is_empty             = o_is_empty;
    d_nnz                  = o_nnz;
    d_ncols_unq            = o_ncols_unq;
    // swap fields
    d_row_starts.swap( other.d_row_starts );
    d_cols.swap( other.d_cols );
    d_cols_loc.swap( other.d_cols_loc );
    d_cols_unq.swap( other.d_cols_unq );
    d_coeffs.swap( other.d_coeffs );
}

template<typename Config>
void CSRLocalMatrixData<Config>::globalToLocalColumns()
{
    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::globalToLocalColumns not implemented for symbolic matrices" );

    PROFILE( "CSRLocalMatrixData::globalToLocalColumns" );

    if ( d_is_empty || d_cols.get() == nullptr ) {
        // gToL either trivially not needed or has already been called
        return;
    }

    // Columns easier to sort before converting to local
    // and defining unq cols in offd easier if globals are sorted
    sortColumns();

    // local columns always owned internally
    d_cols_loc = makeLidxArray( d_nnz );

    if ( d_is_diag ) {
        CSRMatrixDataHelpers<Config>::GlobalToLocalDiag(
            d_cols.get(), d_nnz, d_first_col, d_cols_loc.get() );
    } else {
        // for offd setup column map as part of the process

        // first make a copy of the global columns and sort them
        // as a whole. This is different from the sortColumns call
        // that acts within a row. This jumbles all rows together.
        auto cols_tmp = makeGidxArray( d_nnz );
        AMP::Utilities::Algorithms::copy_n( cols_tmp.get(), d_cols.get(), d_nnz, Config::mem_loc );
        AMP::Utilities::Algorithms::sort( cols_tmp.get(), d_nnz, Config::mem_loc );

        // make sorted entries unique and copy
        d_ncols_unq = static_cast<lidx_t>(
            AMP::Utilities::Algorithms::unique( cols_tmp.get(), d_nnz, Config::mem_loc ) );
        d_cols_unq = makeGidxArray( d_ncols_unq );
        AMP::Utilities::Algorithms::copy_n(
            d_cols_unq.get(), cols_tmp.get(), d_ncols_unq, Config::mem_loc );
        cols_tmp.reset();

        CSRMatrixDataHelpers<Config>::GlobalToLocalOffd(
            d_cols.get(), d_nnz, d_cols_unq.get(), d_ncols_unq, d_cols_loc.get() );
    }

    // free global cols as they should not be used from here on out
    d_cols.reset();
}

template<typename Config>
void CSRLocalMatrixData<Config>::sortColumns()
{
    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::sortColumns not implemented for symbolic matrices" );

    PROFILE( "CSRLocalMatrixData::sortColumns" );

    if ( d_is_empty ) {
        return;
    }

    AMP_DEBUG_INSIST( d_row_starts.get() != nullptr,
                      "CSRLocalMatrixData::sortColumns Row starts must be allocated" );
    AMP_DEBUG_INSIST( d_cols.get() != nullptr,
                      "CSRLocalMatrixData::sortColumns Access to global columns required" );

    if ( d_is_diag ) {
        CSRMatrixDataHelpers<Config>::SortColumnsDiag(
            d_row_starts.get(), d_cols.get(), d_coeffs.get(), d_num_rows, d_first_col );
    } else {
        CSRMatrixDataHelpers<Config>::SortColumnsOffd(
            d_row_starts.get(), d_cols.get(), d_coeffs.get(), d_num_rows );
    }
}


template<typename Config>
CSRLocalMatrixData<Config>::~CSRLocalMatrixData()
{
    AMPManager::decrementResource( "CSRLocalMatrixData" );
}

template<typename Config>
std::shared_ptr<CSRLocalMatrixData<Config>> CSRLocalMatrixData<Config>::cloneMatrixData()
{
    // cloning is just migration with the input/output configurations being the same
    return migrate<Config>();
}

template<typename Config>
std::shared_ptr<CSRLocalMatrixData<Config>>
CSRLocalMatrixData<Config>::maskMatrixData( const typename CSRLocalMatrixData<Config>::mask_t *mask,
                                            const bool is_symbolic ) const
{
    using outdata_t = CSRLocalMatrixData<Config>;
    PROFILE( "CSRLocalMatrixData::maskMatrixData" );

    AMP_INSIST( d_is_diag,
                "CSRLocalMatrixData::maskMatrixData not implemented for off-diag blocks" );

    AMP_INSIST( d_cols.get() == nullptr,
                "CSRLocalMatrixData::maskMatrixData can only be applied to assembled matrices" );

    if ( !is_symbolic ) {
        AMP_INSIST( !d_is_symbolic,
                    "CSRLocalMatrixData::maskMatrixData can not produce a numeric matrix from a "
                    "symbolic matrix" );
    }

    // create matrix with same layout and location, possibly now symbolic
    auto outData = std::make_shared<outdata_t>(
        nullptr, d_first_row, d_last_row, d_first_col, d_last_col, d_is_diag, is_symbolic );

    // count entries in mask and allocate output
    const auto num_rows = numLocalRows();
    auto rs_out         = outData->d_row_starts.get();
    CSRMatrixDataHelpers<Config>::MaskCountNNZ(
        d_row_starts.get(), mask, d_is_diag, num_rows, rs_out );
    outData->setNNZ( true );

    // get output data fields and copy over masked out information
    if ( d_is_diag ) {
        CSRMatrixDataHelpers<Config>::MaskFillDiag( d_row_starts.get(),
                                                    d_cols_loc.get(),
                                                    d_coeffs.get(),
                                                    mask,
                                                    d_is_diag,
                                                    num_rows,
                                                    rs_out,
                                                    outData->d_cols_loc.get(),
                                                    outData->d_coeffs.get() );
        outData->d_cols.reset();
    } else {
        // unreachable for now
    }

    return outData;
}

template<typename Config>
template<typename ConfigOut>
std::shared_ptr<CSRLocalMatrixData<ConfigOut>> CSRLocalMatrixData<Config>::migrate() const
{
    PROFILE( "CSRLocalMatrixData::migrate" );

    using outdata_t = CSRLocalMatrixData<ConfigOut>;

    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::migrate not implemented for symbolic matrices" );

    auto outData = std::make_shared<outdata_t>(
        nullptr, d_first_row, d_last_row, d_first_col, d_last_col, d_is_diag );

    outData->d_is_empty  = d_is_empty;
    outData->d_nnz       = static_cast<typename outdata_t::lidx_t>( d_nnz );
    outData->d_ncols_unq = d_ncols_unq;

    outData->d_cols     = nullptr;
    outData->d_cols_loc = nullptr;
    outData->d_coeffs   = nullptr;

    if ( d_is_empty ) {
        return outData;
    }

    // row starts always allocated internally, so always copy across
    AMP::Utilities::Algorithms::copyCast( outData->d_row_starts.get(),
                                          ConfigOut::mem_loc,
                                          d_row_starts.get(),
                                          Config::mem_loc,
                                          d_num_rows + 1 );

    if constexpr ( Config::allocator == ConfigOut::allocator && false ) {
        // migrate is only being called for type casting
        // we can share fields that match on type and only allocate/cast
        // for mismatches
        if constexpr ( Config::lidx == ConfigOut::lidx ) {
            outData->d_cols_loc = d_cols_loc;
        } else {
            outData->d_cols_loc = outdata_t::makeLidxArray( d_nnz );
            AMP::Utilities::Algorithms::copyCast( outData->d_cols_loc.get(),
                                                  ConfigOut::mem_loc,
                                                  d_cols_loc.get(),
                                                  Config::mem_loc,
                                                  d_nnz );
        }
        if constexpr ( Config::scalar_id == ConfigOut::scalar_id ) {
            outData->d_coeffs = d_coeffs;
        } else {
            outData->d_coeffs = outdata_t::makeScalarArray( d_nnz );
            AMP::Utilities::Algorithms::copyCast( outData->d_coeffs.get(),
                                                  ConfigOut::mem_loc,
                                                  d_coeffs.get(),
                                                  Config::mem_loc,
                                                  d_nnz );
        }
        if constexpr ( Config::gidx == ConfigOut::gidx ) {
            outData->d_cols     = d_cols;
            outData->d_cols_unq = d_cols_unq;
        } else {
            if ( d_cols.get() != nullptr ) {
                outData->d_cols = outdata_t::makeGidxArray( d_nnz );
                AMP::Utilities::Algorithms::copyCast( outData->d_cols.get(),
                                                      ConfigOut::mem_loc,
                                                      d_cols.get(),
                                                      Config::mem_loc,
                                                      d_nnz );
            }
            if ( d_cols_unq.get() != nullptr ) {
                outData->d_cols_unq = outdata_t::makeGidxArray( d_ncols_unq );
                AMP::Utilities::Algorithms::copyCast( outData->d_cols_unq.get(),
                                                      ConfigOut::mem_loc,
                                                      d_cols_unq.get(),
                                                      Config::mem_loc,
                                                      d_ncols_unq );
            }
        }
    } else {
        // different allocators, so migrate used to actually move across
        // memory spaces, and deep copies required for all fields
        outData->d_cols_loc = outdata_t::makeLidxArray( d_nnz );
        outData->d_coeffs   = outdata_t::makeScalarArray( d_nnz );
        AMP::Utilities::Algorithms::copyCast( outData->d_cols_loc.get(),
                                              ConfigOut::mem_loc,
                                              d_cols_loc.get(),
                                              Config::mem_loc,
                                              d_nnz );
        AMP::Utilities::Algorithms::copyCast(
            outData->d_coeffs.get(), ConfigOut::mem_loc, d_coeffs.get(), Config::mem_loc, d_nnz );

        if ( d_cols.get() != nullptr ) {
            outData->d_cols = outdata_t::makeGidxArray( d_nnz );
            AMP::Utilities::Algorithms::copyCast(
                outData->d_cols.get(), ConfigOut::mem_loc, d_cols.get(), Config::mem_loc, d_nnz );
        }
        if ( d_cols_unq.get() != nullptr ) {
            outData->d_cols_unq = outdata_t::makeGidxArray( d_ncols_unq );
            AMP::Utilities::Algorithms::copyCast( outData->d_cols_unq.get(),
                                                  ConfigOut::mem_loc,
                                                  d_cols_unq.get(),
                                                  Config::mem_loc,
                                                  d_ncols_unq );
        }
    }

    return outData;
}

template<typename Config>
std::shared_ptr<CSRLocalMatrixData<Config>>
CSRLocalMatrixData<Config>::transpose( std::shared_ptr<MatrixParametersBase> params ) const
{
    PROFILE( "CSRLocalMatrixData::transpose" );

    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::transpose not implemented for symbolic matrices" );

    // create new data, note swapped rows and cols
    auto transposeData = std::make_shared<CSRLocalMatrixData>(
        params, d_first_col, d_last_col, d_first_row, d_last_row, d_is_diag );

    // handle edge case of empty diagonal block
    if ( d_is_empty ) {
        return transposeData;
    }

    // allocate fully since total number of NZs doesn't change
    // transpose helpers resposible for setting up row_starts
    transposeData->setNNZ( d_nnz );

    // both host and device implementations require some workspace
    // host needs out_num_rows worth of lidx_t's
    // device needs two buffers of lidx_t's with nnz entries each
    const auto worksize =
        d_memory_location == AMP::Utilities::MemoryType::host ? transposeData->d_num_rows : d_nnz;
    auto counters = makeLidxArray( worksize );
    auto reduce_space =
        d_memory_location == AMP::Utilities::MemoryType::host ? nullptr : makeLidxArray( worksize );

    if ( d_is_diag ) {
        AMP_INSIST( d_cols_loc.get(),
                    "CSRLocalMatrixData::transpose Diag block must have accessible local columns" );
        CSRMatrixDataHelpers<Config>::TransposeDiag( d_row_starts.get(),
                                                     d_cols_loc.get(),
                                                     d_coeffs.get(),
                                                     d_num_rows,
                                                     transposeData->d_num_rows,
                                                     transposeData->d_first_col,
                                                     d_nnz,
                                                     transposeData->d_row_starts.get(),
                                                     transposeData->d_cols_loc.get(),
                                                     transposeData->d_cols.get(),
                                                     transposeData->d_coeffs.get(),
                                                     counters.get(),
                                                     reduce_space.get() );
    } else {
        AMP_INSIST(
            d_cols.get(),
            "CSRLocalMatrixData::transpose Offd block must have global columns accessible" );
        CSRMatrixDataHelpers<Config>::TransposeOffd( d_row_starts.get(),
                                                     d_cols.get(),
                                                     d_coeffs.get(),
                                                     d_num_rows,
                                                     d_first_col,
                                                     transposeData->d_num_rows,
                                                     transposeData->d_first_col,
                                                     d_nnz,
                                                     transposeData->d_row_starts.get(),
                                                     transposeData->d_cols_loc.get(),
                                                     transposeData->d_cols.get(),
                                                     transposeData->d_coeffs.get(),
                                                     counters.get(),
                                                     reduce_space.get() );
    }

    return transposeData;
}

template<typename Config>
void CSRLocalMatrixData<Config>::setNNZ( lidx_t tot_nnz )
{
    PROFILE( "CSRLocalMatrixData::setNNZ" );

    d_nnz = tot_nnz;

    if ( d_nnz == 0 ) {
        d_is_empty = true;
        // nothing to do, block stays empty
        return;
    }

    // allocate and fill remaining arrays
    d_is_empty = false;
    d_cols     = makeGidxArray( d_nnz );
    d_cols_loc = makeLidxArray( d_nnz );
    if ( !d_is_symbolic ) {
        d_coeffs = makeScalarArray( d_nnz );
    }

    AMP::Utilities::Algorithms::zero_n( d_cols.get(), d_nnz, Config::mem_loc );
    AMP::Utilities::Algorithms::zero_n( d_cols_loc.get(), d_nnz, Config::mem_loc );
    if ( !d_is_symbolic ) {
        AMP::Utilities::Algorithms::zero_n( d_coeffs.get(), d_nnz, Config::mem_loc );
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::setNNZ( bool do_accum )
{
    PROFILE( "CSRLocalMatrixData::setNNZ" );

    if ( do_accum ) {
        AMP::Utilities::Algorithms::exclusive_scan(
            d_row_starts.get(), d_num_rows + 1, d_row_starts.get(), 0, Config::mem_loc );
    }

    if ( d_memory_location == AMP::Utilities::MemoryType::device ) {
        const lidx_t *ptr_loc = d_row_starts.get() + d_num_rows;
        AMP::Utilities::Algorithms::copy_n(
            &d_nnz, AMP::Utilities::MemoryType::host, ptr_loc, Config::mem_loc, 1 );
    } else {
        // total nnz in all rows of block is last entry
        d_nnz = d_row_starts[d_num_rows];
    }
    if ( d_nnz == 0 ) {
        d_is_empty = true;
        // nothing to do, block stays empty
        return;
    }

    // allocate and fill remaining arrays
    d_is_empty = false;
    d_cols     = makeGidxArray( d_nnz );
    d_cols_loc = makeLidxArray( d_nnz );
    if ( !d_is_symbolic ) {
        d_coeffs = makeScalarArray( d_nnz );
    }

    AMP::Utilities::Algorithms::zero_n( d_cols.get(), d_nnz, Config::mem_loc );
    AMP::Utilities::Algorithms::zero_n( d_cols_loc.get(), d_nnz, Config::mem_loc );
    if ( !d_is_symbolic ) {
        AMP::Utilities::Algorithms::zero_n( d_coeffs.get(), d_nnz, Config::mem_loc );
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::setNNZ( const lidx_t *nnz,
                                         const AMP::Utilities::MemoryType nnz_loc )
{
    // copy passed nnz vector into row_starts and call internal setNNZ
    AMP::Utilities::Algorithms::copy_n(
        d_row_starts.get(), Config::mem_loc, nnz, nnz_loc, d_num_rows );
    setNNZ( true );
}

template<typename Config>
void CSRLocalMatrixData<Config>::removeRange( const scalar_t bnd_lo, const scalar_t bnd_up )
{
    PROFILE( "CSRLocalMatrixData::removeRange" );

    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::removeRange not defined for symbolic matrices" );

    if ( d_is_empty ) {
        return;
    }

    // count coeffs that lie within range and zero them along the way
    auto delete_per_row = makeLidxArray( d_num_rows );
    lidx_t num_delete   = CSRMatrixDataHelpers<Config>::RemoveRangeCountDel(
        d_row_starts.get(), d_coeffs.get(), d_num_rows, bnd_lo, bnd_up, delete_per_row.get() );

    // if none to delete then done
    if ( num_delete == 0 ) {
        return;
    }

    // if all entries will be deleted throw a warning and set the matrix
    // as empty
    if ( d_nnz == num_delete ) {
        AMP::Utilities::Algorithms::zero_n( d_row_starts.get(), d_num_rows + 1, Config::mem_loc );
        d_cols.reset();
        d_cols_unq.reset();
        d_cols_loc.reset();
        d_coeffs.reset();
        d_nnz       = 0;
        d_ncols_unq = 0;
        d_is_empty  = true;
        return;
    }

    // allocate space for new data fields and copy over parts to keep
    d_nnz -= num_delete;
    auto new_row_starts = makeLidxArray( d_num_rows + 1 );
    auto new_coeffs     = makeScalarArray( d_nnz );
    std::shared_ptr<lidx_t[]> new_cols_loc;
    std::shared_ptr<gidx_t[]> new_cols;
    if ( d_is_diag ) {
        new_cols_loc = makeLidxArray( d_nnz );
    } else {
        new_cols = makeGidxArray( d_nnz );
    }

    // new row starts is old minus running total of deleted entries
    CSRMatrixDataHelpers<Config>::RemoveRangeUpdateRowStart(
        d_row_starts.get(), delete_per_row.get(), d_num_rows, new_row_starts.get() );

    // coeffs is a masked copy
    // cols_loc is masked copy if this is diag block, otherwise
    // build cols from cols_unq and call globalToLocal
    if ( d_is_diag ) {
        CSRMatrixDataHelpers<Config>::RemoveRangeFillDiag( d_row_starts.get(),
                                                           d_cols_loc.get(),
                                                           d_coeffs.get(),
                                                           d_num_rows,
                                                           bnd_lo,
                                                           bnd_up,
                                                           new_row_starts.get(),
                                                           new_cols_loc.get(),
                                                           new_coeffs.get() );
    } else {
        CSRMatrixDataHelpers<Config>::RemoveRangeFillOffd( d_row_starts.get(),
                                                           d_cols_loc.get(),
                                                           d_cols_unq.get(),
                                                           d_coeffs.get(),
                                                           d_num_rows,
                                                           bnd_lo,
                                                           bnd_up,
                                                           new_row_starts.get(),
                                                           new_cols.get(),
                                                           new_coeffs.get() );
    }

    d_cols_unq.reset();

    d_row_starts.swap( new_row_starts );
    d_cols.swap( new_cols );
    d_cols_loc.swap( new_cols_loc );
    d_coeffs.swap( new_coeffs );

    new_row_starts.reset();
    new_cols.reset();
    new_cols_loc.reset();
    new_coeffs.reset();

    globalToLocalColumns();
}

template<typename Config>
void CSRLocalMatrixData<Config>::getColPtrs( std::vector<gidx_t *> &col_ptrs )
{
    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::getColPtrs not implemented on device yet" );

    if ( !d_is_empty ) {
        for ( lidx_t row = 0; row < d_num_rows; ++row ) {
            col_ptrs[row] = &d_cols[d_row_starts[row]];
        }
    } else {
        for ( lidx_t row = 0; row < d_num_rows; ++row ) {
            col_ptrs[row] = nullptr;
        }
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::printStats( bool verbose, bool show_zeros ) const
{
    AMP::plog << ( d_is_diag ? "  diag block:" : "  offd block:" ) << std::endl;
    if ( d_is_empty ) {
        AMP::plog << "    EMPTY" << std::endl;
        return;
    }
    AMP::plog << "    first | last row: " << d_first_row << " | " << d_last_row << std::endl;
    AMP::plog << "    first | last col: " << d_first_col << " | " << d_last_col << std::endl;

    if ( d_cols.get() ) {
        AMP::plog << "    min | max col: "
                  << AMP::Utilities::Algorithms::min_element( d_cols.get(), d_nnz, Config::mem_loc )
                  << " | "
                  << AMP::Utilities::Algorithms::max_element( d_cols.get(), d_nnz, Config::mem_loc )
                  << std::endl;
    }

    AMP::plog << "    num unique: " << d_ncols_unq << std::endl;
    scalar_t avg_nnz = static_cast<scalar_t>( d_nnz ) / static_cast<scalar_t>( d_num_rows );
    AMP::plog << "    avg nnz per row: " << avg_nnz << std::endl;
    AMP::plog << "    tot nnz: " << d_nnz << std::endl;
    if ( verbose && d_memory_location < AMP::Utilities::MemoryType::device ) {
        AMP::plog << "    row 0: ";
        for ( auto n = d_row_starts[0]; n < d_row_starts[1]; ++n ) {
            if ( d_coeffs.get() && ( d_coeffs[n] != 0 || show_zeros ) ) {
                AMP::plog << "("
                          << ( d_cols.get() ? static_cast<long long>( d_cols[n] ) :
                                              static_cast<long long>( d_cols_loc[n] ) )
                          << "," << d_coeffs[n] << "), ";
            } else if ( show_zeros ) {
                AMP::plog << "("
                          << ( d_cols.get() ? static_cast<long long>( d_cols[n] ) :
                                              static_cast<long long>( d_cols_loc[n] ) )
                          << ",--), ";
            }
        }
        AMP::plog << "\n    row last: ";
        for ( auto n = d_row_starts[d_num_rows - 1]; n < d_row_starts[d_num_rows]; ++n ) {
            if ( d_coeffs.get() && ( d_coeffs[n] != 0 || show_zeros ) ) {
                AMP::plog << "("
                          << ( d_cols.get() ? static_cast<long long>( d_cols[n] ) :
                                              static_cast<long long>( d_cols_loc[n] ) )
                          << "," << d_coeffs[n] << "), ";
            } else if ( show_zeros ) {
                AMP::plog << "("
                          << ( d_cols.get() ? static_cast<long long>( d_cols[n] ) :
                                              static_cast<long long>( d_cols_loc[n] ) )
                          << ",--), ";
            }
        }
        if ( d_ncols_unq > 0 && d_ncols_unq < 200 ) {
            AMP::plog << "\n    column map: ";
            for ( auto n = 0; n < d_ncols_unq; ++n ) {
                AMP::plog << "[" << n << "|" << d_cols_unq[n] << "], ";
            }
        }
    } else {
        AMP_WARNING(
            "CSRLocalMatrixData::printStats: verbose mode unsupported for device matrices" );
    }
    AMP::plog << std::endl << std::endl;
}


template<typename Config>
void CSRLocalMatrixData<Config>::printAll( bool force ) const
{
    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::printAll not implemented for symbolic matrices" );

    printStats( false, false );

    // bail if total entries too large and force output not enabled
    if ( d_nnz > 5000 && !force ) {
        AMP_WARN_ONCE( "CSRLocalMatrixData::printAll: Skipping print due to too many non-zeros. "
                       "Re-run with force=true to print anyway." );
        return;
    }

    // bail if empty, no warning needed
    if ( d_is_empty ) {
        return;
    }

    if constexpr ( Config::mem_loc == AMP::Utilities::MemoryType::device ) {
        AMP_WARNING( "" );
        return;
    }

    const bool have_loc = ( d_cols_loc.get() != nullptr );
    const bool have_gbl = ( d_cols.get() != nullptr );

    // print all unique columns
    if ( d_cols_unq ) {
        AMP::plog << "Unique cols: ";
        for ( lidx_t n = 0; n < d_ncols_unq; ++n ) {
            AMP::plog << "[" << n << "|" << d_cols_unq[n] << "] ";
        }
        AMP::plog << std::endl << std::endl;
    }

    // print all global columns and values row-by-row
    for ( lidx_t row = 0; row < d_num_rows; ++row ) {
        // skip empty rows to avoid a bunch of blank newlines
        if ( d_row_starts[row] < d_row_starts[row + 1] ) {
            AMP::plog << "Row " << row << ": ";
            for ( lidx_t c = d_row_starts[row]; c < d_row_starts[row + 1]; ++c ) {
                lidx_t cl = have_loc ? d_cols_loc[c] : -1;
                gidx_t cg;
                if ( d_is_diag ) {
                    cg = have_gbl ? d_cols[c] : d_first_col + static_cast<gidx_t>( d_cols_loc[c] );
                } else {
                    cg = have_gbl ? d_cols[c] : d_cols_unq[d_cols_loc[c]];
                }
                AMP::plog << "[" << cl << "|" << cg << "|" << d_coeffs[c] << "] ";
            }
            AMP::plog << std::endl;
        }
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::getRowByGlobalID( const size_t local_row,
                                                   std::vector<size_t> &cols,
                                                   std::vector<double> &values ) const
{
    PROFILE( "CSRLocalMatrixData::getRowByGlobalID" );

    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::getRowByGlobalId not implemented for symbolic matrices" );

    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return;
    }

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::getRowByGlobalID not implemented for device memory" );

    const auto start = d_row_starts[local_row];
    auto end         = d_row_starts[local_row + 1];
    cols.resize( end - start );
    values.resize( end - start );

    // don't store global ids, need to generate on the fly
    if ( d_is_diag ) {
        const auto first_col = d_first_col;
        std::transform( &d_cols_loc[start],
                        &d_cols_loc[end],
                        cols.begin(),
                        [&]( lidx_t col ) -> size_t { return col + first_col; } );
    } else {
        const auto cols_unq = d_cols_unq;
        std::transform( &d_cols_loc[start],
                        &d_cols_loc[end],
                        cols.begin(),
                        [&]( lidx_t col ) -> size_t { return cols_unq[col]; } );
    }

    if constexpr ( std::is_same_v<double, scalar_t> ) {
        std::copy( &d_coeffs[start], &d_coeffs[end], values.begin() );
    } else {
        std::transform( &d_coeffs[start],
                        &d_coeffs[end],
                        values.begin(),
                        []( scalar_t val ) -> double { return val; } );
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::getValuesByGlobalID( const size_t local_row,
                                                      const size_t num_cols,
                                                      const size_t *cols,
                                                      scalar_t *values ) const
{
    PROFILE( "CSRLocalMatrixData::getValuesByGlobalID" );

    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::getValuesByGlobalId not defined for symbolic matrices" );

    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return;
    }

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::getValuesByGlobalID not implemented for device memory" );

    const auto start = d_row_starts[local_row];
    auto end         = d_row_starts[local_row + 1];


    for ( size_t nc = 0; nc < num_cols; ++nc ) {
        auto query_col = cols[nc];
        for ( lidx_t i = start; i < end; ++i ) {
            auto icol = d_is_diag ? ( d_first_col + d_cols_loc[i] ) : ( d_cols_unq[d_cols_loc[i]] );
            if ( icol == static_cast<gidx_t>( query_col ) ) {
                values[nc] = d_coeffs[i];
            }
        }
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::addValuesByGlobalID( const size_t local_row,
                                                      const size_t num_cols,
                                                      const size_t *cols,
                                                      const scalar_t *vals )
{
    PROFILE( "CSRLocalMatrixData::addValuesByGlobalID" );

    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::addValuesByGlobalId not defined for symbolic matrices" );

    if ( d_is_empty ) {
        return;
    }

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::addValuesByGlobalID not implemented for device memory" );

    const auto start = d_row_starts[local_row];
    auto end         = d_row_starts[local_row + 1];

    if ( d_is_diag ) {
        for ( size_t icol = 0; icol < num_cols; ++icol ) {
            for ( lidx_t j = start; j < end; ++j ) {
                if ( d_first_col + d_cols_loc[j] == static_cast<gidx_t>( cols[icol] ) ) {
                    d_coeffs[j] += vals[icol];
                    break;
                }
            }
        }
    } else {
        for ( size_t icol = 0; icol < num_cols; ++icol ) {
            for ( lidx_t j = start; j < end; ++j ) {
                if ( d_cols_unq[d_cols_loc[j]] == static_cast<gidx_t>( cols[icol] ) ) {
                    d_coeffs[j] += vals[icol];
                    break;
                }
            }
        }
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::setValuesByGlobalID( const size_t local_row,
                                                      const size_t num_cols,
                                                      const size_t *cols,
                                                      const scalar_t *vals )
{
    PROFILE( "CSRLocalMatrixData::setValuesByGlobalID" );

    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::setValuesByGlobalId not defined for symbolic matrices" );

    if ( d_is_empty ) {
        return;
    }

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::setValuesByGlobalID not implemented for device memory" );

    const auto start = d_row_starts[local_row];
    auto end         = d_row_starts[local_row + 1];

    if ( d_is_diag ) {
        for ( size_t icol = 0; icol < num_cols; ++icol ) {
            for ( lidx_t j = start; j < end; ++j ) {
                if ( d_first_col + d_cols_loc[j] == static_cast<gidx_t>( cols[icol] ) ) {
                    d_coeffs[j] = vals[icol];
                    break;
                }
            }
        }
    } else {
        for ( size_t icol = 0; icol < num_cols; ++icol ) {
            for ( lidx_t j = start; j < end; ++j ) {
                if ( d_cols_unq[d_cols_loc[j]] == static_cast<gidx_t>( cols[icol] ) ) {
                    d_coeffs[j] = vals[icol];
                    break;
                }
            }
        }
    }
}

template<typename Config>
size_t CSRLocalMatrixData<Config>::numberColumnIDs( size_t local_row ) const
{
    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::numberColumnIDs not implemented for device memory" );
    AMP_INSIST( d_row_starts,
                "CSRLocalMatrixData::numberColumnIDs nnz layout must be initialized" );
    const auto start = d_row_starts[local_row];
    const auto end   = d_row_starts[local_row + 1];
    return end - start;
}

template<typename Config>
std::vector<size_t> CSRLocalMatrixData<Config>::getColumnIDs( const size_t local_row ) const
{
    PROFILE( "CSRLocalMatrixData::getColumnIDs" );

    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return std::vector<size_t>();
    }

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::getColumnIDs not implemented for device memory" );

    AMP_INSIST( d_cols_loc && d_row_starts,
                "CSRLocalMatrixData::getColumnIDs nnz layout must be initialized" );

    const auto start = d_row_starts[local_row];
    const auto end   = d_row_starts[local_row + 1];
    std::vector<size_t> cols( end - start, 0 );

    // don't store global ids, need to generate on the fly
    if ( d_is_diag ) {
        const auto first_col = d_first_col;
        std::transform( &d_cols_loc[start],
                        &d_cols_loc[end],
                        cols.begin(),
                        [&]( lidx_t col ) -> size_t { return col + first_col; } );
    } else {
        const auto cols_unq = d_cols_unq;
        std::transform( &d_cols_loc[start],
                        &d_cols_loc[end],
                        cols.begin(),
                        [&]( lidx_t col ) -> size_t { return cols_unq[col]; } );
    }

    return cols;
}

/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
template<typename Config>
void CSRLocalMatrixData<Config>::registerChildObjects( AMP::IO::RestartManager * ) const
{
}

template<typename Config>
void CSRLocalMatrixData<Config>::writeRestart( int64_t fid ) const
{
    IO::writeHDF5( fid, "memory_location", static_cast<signed char>( d_memory_location ) );
    IO::writeHDF5( fid, "first_row", d_first_row );
    IO::writeHDF5( fid, "last_row", d_last_row );
    IO::writeHDF5( fid, "first_col", d_first_col );
    IO::writeHDF5( fid, "last_col", d_last_col );

    IO::writeHDF5( fid, "is_diag", d_is_diag );
    IO::writeHDF5( fid, "is_empty", d_is_empty );
    IO::writeHDF5( fid, "is_symbolic", d_is_symbolic );

    IO::writeHDF5( fid, "num_rows", d_num_rows );
    IO::writeHDF5( fid, "nnz", d_nnz );
    IO::writeHDF5( fid, "ncols_unq", d_ncols_unq );

    AMP::Array<lidx_t> row_starts;
    AMP::Array<gidx_t> cols;
    AMP::Array<gidx_t> cols_unq;
    AMP::Array<lidx_t> cols_loc;
    AMP::Array<scalar_t> coeffs;

    if ( d_memory_location <= AMP::Utilities::MemoryType::host ) {

        row_starts.viewRaw( d_num_rows + 1, d_row_starts.get() );

        if ( d_ncols_unq > 0 && !d_is_diag )
            cols_unq.viewRaw( d_ncols_unq, d_cols_unq.get() );

        if ( d_nnz > 0 )
            cols_loc.viewRaw( d_nnz, d_cols_loc.get() );

        if ( d_nnz > 0 && !d_is_symbolic )
            coeffs.viewRaw( d_nnz, d_coeffs.get() );

    } else {

        row_starts.resize( d_num_rows + 1 );
        AMP::Utilities::Algorithms::copy_n( row_starts.data(),
                                            AMP::Utilities::MemoryType::host,
                                            d_row_starts.get(),
                                            Config::mem_loc,
                                            d_num_rows + 1 );

        if ( d_ncols_unq > 0 && !d_is_diag ) {
            cols_unq.resize( d_ncols_unq );
            AMP::Utilities::Algorithms::copy_n( cols_unq.data(),
                                                AMP::Utilities::MemoryType::host,
                                                d_cols_unq.get(),
                                                Config::mem_loc,
                                                d_ncols_unq );
        }

        if ( d_nnz > 0 ) {
            cols_loc.resize( d_nnz );
            AMP::Utilities::Algorithms::copy_n( cols_loc.data(),
                                                AMP::Utilities::MemoryType::host,
                                                d_cols_loc.get(),
                                                Config::mem_loc,
                                                d_nnz );
        }

        if ( d_nnz > 0 && !d_is_symbolic ) {
            coeffs.resize( d_nnz );
            AMP::Utilities::Algorithms::copy_n( coeffs.data(),
                                                AMP::Utilities::MemoryType::host,
                                                d_coeffs.get(),
                                                Config::mem_loc,
                                                d_nnz );
        }
    }

    if ( d_num_rows > 0 ) {
        AMP_INSIST( row_starts.data(), "CSRLocalMatrixData::writeRestart: bad row starts" );
        IO::writeHDF5( fid, "row_starts", row_starts );
    }
    if ( d_ncols_unq > 0 && !d_is_diag ) {
        AMP_INSIST( cols_unq.data(), "CSRLocalMatrixData::writeRestart: bad cols unq" );
        IO::writeHDF5( fid, "cols_unq", cols_unq );
    }
    if ( d_nnz > 0 ) {
        AMP_INSIST( cols_loc.data(), "CSRLocalMatrixData::writeRestart: bad cols loc" );
        IO::writeHDF5( fid, "cols_loc", cols_loc );
    }
    if ( d_nnz > 0 && !d_is_symbolic ) {
        AMP_INSIST( coeffs.data(), "CSRLocalMatrixData::writeRestart: bad coeffs" );
        IO::writeHDF5( fid, "coeffs", coeffs );
    }
}

template<typename Config>
CSRLocalMatrixData<Config>::CSRLocalMatrixData( int64_t fid, AMP::IO::RestartManager * )
{
    signed char memory_location;
    IO::readHDF5( fid, "memory_location", memory_location );
    AMP_ASSERT( d_memory_location == static_cast<AMP::Utilities::MemoryType>( memory_location ) );

    IO::readHDF5( fid, "first_row", d_first_row );
    IO::readHDF5( fid, "last_row", d_last_row );
    IO::readHDF5( fid, "first_col", d_first_col );
    IO::readHDF5( fid, "last_col", d_last_col );

    IO::readHDF5( fid, "is_diag", d_is_diag );
    IO::readHDF5( fid, "is_empty", d_is_empty );
    IO::readHDF5( fid, "is_symbolic", d_is_symbolic );

    IO::readHDF5( fid, "num_rows", d_num_rows );
    IO::readHDF5( fid, "nnz", d_nnz );
    IO::readHDF5( fid, "ncols_unq", d_ncols_unq );

    AMP::Array<lidx_t> row_starts;
    AMP::Array<gidx_t> cols_unq;
    AMP::Array<lidx_t> cols_loc;
    AMP::Array<scalar_t> coeffs;

    IO::readHDF5( fid, "row_starts", row_starts );
    IO::readHDF5( fid, "cols_unq", cols_unq );
    IO::readHDF5( fid, "cols_loc", cols_loc );

    if ( d_num_rows > 0 ) {
        d_row_starts = makeLidxArray( d_num_rows + 1 );
        AMP::Utilities::Algorithms::copy_n( d_row_starts.get(),
                                            Config::mem_loc,
                                            row_starts.data(),
                                            AMP::Utilities::MemoryType::host,
                                            d_num_rows + 1 );
    }

    if ( d_ncols_unq > 0 && !d_is_diag ) {
        d_cols_unq = makeGidxArray( d_ncols_unq );
        AMP::Utilities::Algorithms::copy_n( d_cols_unq.get(),
                                            Config::mem_loc,
                                            cols_unq.data(),
                                            AMP::Utilities::MemoryType::host,
                                            d_ncols_unq );
    }

    if ( d_nnz > 0 ) {
        d_cols_loc = makeLidxArray( d_nnz );
        AMP::Utilities::Algorithms::copy_n( d_cols_loc.get(),
                                            Config::mem_loc,
                                            cols_loc.data(),
                                            AMP::Utilities::MemoryType::host,
                                            d_nnz );
    }

    if ( d_nnz && ( !d_is_symbolic ) ) {
        IO::readHDF5( fid, "coeffs", coeffs );
        d_coeffs = makeScalarArray( d_nnz );
        AMP::Utilities::Algorithms::copy_n( d_coeffs.get(),
                                            Config::mem_loc,
                                            coeffs.data(),
                                            AMP::Utilities::MemoryType::host,
                                            d_nnz );
    }
}

} // namespace AMP::LinearAlgebra

#endif
