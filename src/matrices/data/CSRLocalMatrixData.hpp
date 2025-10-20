#ifndef included_AMP_CSRLocalMatrixData_hpp
#define included_AMP_CSRLocalMatrixData_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/AMPCSRMatrixParameters.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/RawCSRMatrixParameters.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixDataHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Algorithms.h"
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

template<typename size_type, class data_allocator>
std::shared_ptr<typename data_allocator::value_type[]> sharedArrayBuilder( size_type N,
                                                                           data_allocator &alloc )
{
    AMP_DEBUG_ASSERT( std::is_integral_v<size_type> );
    return std::shared_ptr<typename data_allocator::value_type[]>(
        alloc.allocate( N ), [N, &alloc]( auto p ) -> void { alloc.deallocate( p, N ); } );
}

template<typename data_type>
std::shared_ptr<data_type[]> sharedArrayWrapper( data_type *raw_array )
{
    return std::shared_ptr<data_type[]>( raw_array, []( auto p ) -> void { (void) p; } );
}

template<typename Config>
CSRLocalMatrixData<Config>::CSRLocalMatrixData( std::shared_ptr<MatrixParametersBase> params,
                                                AMP::Utilities::MemoryType memory_location,
                                                typename Config::gidx_t first_row,
                                                typename Config::gidx_t last_row,
                                                typename Config::gidx_t first_col,
                                                typename Config::gidx_t last_col,
                                                bool is_diag,
                                                bool is_symbolic )
    : d_memory_location( memory_location ),
      d_first_row( first_row ),
      d_last_row( last_row ),
      d_first_col( first_col ),
      d_last_col( last_col ),
      d_is_diag( is_diag ),
      d_is_symbolic( is_symbolic ),
      d_num_rows( last_row - first_row )
{
    AMPManager::incrementResource( "CSRLocalMatrixData" );

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

        if ( blParams.d_row_starts == nullptr ) {
            d_is_empty = true;
            return;
        }

        // count nnz and decide if block is empty
        d_nnz = blParams.d_row_starts[d_num_rows];

        if ( d_nnz == 0 ) {
            d_is_empty = true;
            return;
        }
        d_is_empty = false;

        // Wrap raw pointers from blParams to match internal
        // shared_ptr<T[]> type
        d_row_starts = sharedArrayWrapper( blParams.d_row_starts );
        d_cols       = sharedArrayWrapper( blParams.d_cols );
        d_coeffs     = sharedArrayWrapper( blParams.d_coeffs );
    } else if ( matParams ) {
        // can always allocate row starts without external information
        d_row_starts = sharedArrayBuilder( d_num_rows + 1, d_lidxAllocator );
        AMP::Utilities::Algorithms<lidx_t>::fill_n( d_row_starts.get(), d_num_rows + 1, 0 );

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
        d_cols = sharedArrayBuilder( d_nnz, d_gidxAllocator );
        if ( !d_is_symbolic ) {
            d_coeffs = sharedArrayBuilder( d_nnz, d_scalarAllocator );
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
        d_row_starts = sharedArrayBuilder( d_num_rows + 1, d_lidxAllocator );
        AMP::Utilities::Algorithms<lidx_t>::fill_n( d_row_starts.get(), d_num_rows + 1, 0 );
        return;
    }

    // fill in local column indices
    globalToLocalColumns();
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
    auto block         = ( *blocks.begin() ).second;
    const auto mem_loc = block->d_memory_location;
    AMP_INSIST( mem_loc < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::ConcatHorizontal not implemented on device yet" );
    const auto first_row = block->d_first_row;
    const auto last_row  = block->d_last_row;
    const auto nrows     = static_cast<lidx_t>( last_row - first_row );
    const auto first_col = block->d_first_col;
    const auto last_col  = block->d_last_col;
    std::vector<lidx_t> row_nnz( nrows, 0 );
    for ( auto it : blocks ) {
        block = it.second;
        if ( block->isEmpty() ) {
            continue;
        }
        AMP_INSIST( first_row == block->d_first_row && last_row == block->d_last_row &&
                        first_col == block->d_first_col && last_col == block->d_last_col,
                    "Blocks to concatenate must have compatible layouts" );
        AMP_INSIST( block->d_cols.get(), "Blocks to concatenate must have global columns" );
        AMP_INSIST( mem_loc == block->d_memory_location,
                    "Blocks to concatenate must be in same memory space" );
        AMP_INSIST( !block->d_is_symbolic, "Blocks to concatenate can't be symbolic" );
        for ( lidx_t row = 0; row < nrows; ++row ) {
            row_nnz[row] += ( block->d_row_starts[row + 1] - block->d_row_starts[row] );
        }
    }

    // Create empty matrix and trigger allocations to match
    auto concat_matrix = std::make_shared<CSRLocalMatrixData<Config>>(
        params, mem_loc, first_row, last_row, first_col, last_col, false );
    concat_matrix->setNNZ( row_nnz );

    // set row_nnz back to zeros to use as counters while appending entries
    AMP::Utilities::Algorithms<lidx_t>::fill_n( row_nnz.data(), nrows, 0 );

    // loop back over blocks and write into new matrix
    for ( auto it : blocks ) {
        block = it.second;
        if ( block->isEmpty() ) {
            continue;
        }
        for ( lidx_t row = 0; row < nrows; ++row ) {
            const auto rs = concat_matrix->d_row_starts[row];
            for ( auto n = block->d_row_starts[row]; n < block->d_row_starts[row + 1]; ++n ) {
                const auto ctr                    = row_nnz[row];
                concat_matrix->d_cols[rs + ctr]   = block->d_cols[n];
                concat_matrix->d_coeffs[rs + ctr] = block->d_coeffs[n];
                row_nnz[row]++;
            }
        }
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
    auto block         = ( *blocks.begin() ).second;
    const auto mem_loc = block->d_memory_location;
    lidx_t num_rows    = 0;
    bool all_empty     = block->isEmpty();
    for ( auto it : blocks ) {
        block = it.second;
        AMP_DEBUG_INSIST( mem_loc == block->d_memory_location,
                          "Blocks to concatenate must be in same memory space" );
        AMP_INSIST( !block->d_is_symbolic, "Blocks to concatenate can't be symbolic" );
        num_rows += block->d_num_rows;
        all_empty = all_empty && block->isEmpty();
    }

    // extreme edge case where every requested row happened to be empty
    if ( all_empty ) {
        return nullptr;
    }

    // create output matrix
    auto concat_matrix = std::make_shared<CSRLocalMatrixData<Config>>(
        params, mem_loc, 0, num_rows, first_col, last_col, is_diag );
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
    d_cols_loc = sharedArrayBuilder( d_nnz, d_lidxAllocator );

    if ( d_is_diag ) {
        CSRMatrixDataHelpers<Config>::GlobalToLocalDiag(
            d_cols.get(), d_nnz, d_first_col, d_cols_loc.get() );
    } else {
        // for offd setup column map as part of the process

        // first make a copy of the global columns and sort them
        // as a whole. This is different from the sortColumns call
        // that acts within a row. This jumbles all rows together.
        auto cols_tmp = sharedArrayBuilder( d_nnz, d_gidxAllocator );
        AMP::Utilities::Algorithms<gidx_t>::copy_n( d_cols.get(), d_nnz, cols_tmp.get() );
        AMP::Utilities::Algorithms<gidx_t>::sort( cols_tmp.get(), d_nnz );

        // make sorted entries unique and copy
        d_ncols_unq = static_cast<lidx_t>(
            AMP::Utilities::Algorithms<gidx_t>::unique( cols_tmp.get(), d_nnz ) );
        d_cols_unq = sharedArrayBuilder( d_ncols_unq, d_gidxAllocator );
        AMP::Utilities::Algorithms<gidx_t>::copy_n( cols_tmp.get(), d_ncols_unq, d_cols_unq.get() );
        cols_tmp.reset();

        CSRMatrixDataHelpers<Config>::GlobalToLocalOffd(
            d_cols.get(), d_nnz, d_cols_unq.get(), d_ncols_unq, d_cols_loc.get() );
    }

    // free global cols as they should not be used from here on out
    d_cols.reset();
}

template<typename Config>
typename Config::gidx_t
CSRLocalMatrixData<Config>::localToGlobal( const typename Config::lidx_t loc_id ) const
{
    if ( d_is_diag ) {
        return static_cast<typename Config::gidx_t>( loc_id ) + d_first_col;
    } else {
        return d_cols_unq[loc_id];
    }
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
    auto outData = std::make_shared<outdata_t>( nullptr,
                                                d_memory_location,
                                                d_first_row,
                                                d_last_row,
                                                d_first_col,
                                                d_last_col,
                                                d_is_diag,
                                                is_symbolic );

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
    using outdata_t = CSRLocalMatrixData<ConfigOut>;

    using out_alloc_t      = typename outdata_t::allocator_type;
    using out_lidx_alloc_t = typename std::allocator_traits<out_alloc_t>::template rebind_alloc<
        typename ConfigOut::lidx_t>;
    using out_gidx_alloc_t = typename std::allocator_traits<out_alloc_t>::template rebind_alloc<
        typename ConfigOut::gidx_t>;
    using out_scalar_alloc_t = typename std::allocator_traits<out_alloc_t>::template rebind_alloc<
        typename ConfigOut::scalar_t>;
    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::migrate not implemented for symbolic matrices" );

    auto memloc  = AMP::Utilities::getAllocatorMemoryType<out_alloc_t>();
    auto outData = std::make_shared<outdata_t>(
        nullptr, memloc, d_first_row, d_last_row, d_first_col, d_last_col, d_is_diag );

    outData->d_is_empty = d_is_empty;
    outData->d_nnz      = static_cast<typename outdata_t::lidx_t>( d_nnz );

    outData->d_cols     = nullptr;
    outData->d_cols_loc = nullptr;
    outData->d_coeffs   = nullptr;

    if ( !d_is_empty ) {
        out_lidx_alloc_t lidx_alloc;
        out_gidx_alloc_t gidx_alloc;
        out_scalar_alloc_t scalar_alloc;

        outData->d_cols_loc = sharedArrayBuilder( d_nnz, lidx_alloc );
        outData->d_coeffs   = sharedArrayBuilder( d_nnz, scalar_alloc );

        /****************************************************
         * Potential performance improvement:
         * If config::alloc == configout::alloc  &&
         *   config::lidx_t == configout::lidx_t &&
         *   config::gidx_t == configout::gidx_t
         * then it's not required to create copies of the index arrays pointers would suffice
         ****************************************************/

        AMP::Utilities::copy( d_num_rows + 1, d_row_starts.get(), outData->d_row_starts.get() );
        AMP::Utilities::copy( d_nnz, d_cols_loc.get(), outData->d_cols_loc.get() );
        AMP::Utilities::copy( d_nnz, d_coeffs.get(), outData->d_coeffs.get() );

        if ( d_cols.get() != nullptr ) {
            outData->d_cols = sharedArrayBuilder( d_nnz, gidx_alloc );
            AMP::Utilities::copy( d_nnz, d_cols.get(), outData->d_cols.get() );
        }
        if ( d_cols_unq.get() != nullptr ) {
            outData->d_ncols_unq = d_ncols_unq;
            outData->d_cols_unq  = sharedArrayBuilder( d_ncols_unq, gidx_alloc );
            AMP::Utilities::copy( d_ncols_unq, d_cols_unq.get(), outData->d_cols_unq.get() );
        }
    }

    return outData;
}

template<typename Config>
std::shared_ptr<CSRLocalMatrixData<Config>>
CSRLocalMatrixData<Config>::transpose( std::shared_ptr<MatrixParametersBase> params ) const
{
    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::transpose not implemented for symbolic matrices" );
    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::transpose not implemented for device memory" );

    // create new data, note swapped rows and cols
    auto transposeData = std::make_shared<CSRLocalMatrixData>(
        params, d_memory_location, d_first_col, d_last_col, d_first_row, d_last_row, d_is_diag );

    // handle rare edge case of empty diagonal block
    if ( d_is_empty ) {
        return transposeData;
    }

    auto trans_row = [is_diag   = d_is_diag,
                      first_col = d_first_col,
                      cols      = d_cols,
                      cols_loc  = d_cols_loc,
                      cols_unq  = d_cols_unq]( const lidx_t c ) -> lidx_t {
        gidx_t col_g = 0;
        if ( cols.get() ) {
            col_g = cols[c];
        } else if ( is_diag ) {
            return cols_loc[c];
        } else {
            col_g = cols_unq[cols_loc[c]];
        }
        return col_g - first_col;
    };

    // count nnz per column and store in transpose's rowstarts array
    for ( lidx_t row = 0; row < d_num_rows; ++row ) {
        for ( lidx_t c = d_row_starts[row]; c < d_row_starts[row + 1]; ++c ) {
            const auto t_row = trans_row( c );
            transposeData->d_row_starts[t_row]++;
        }
    }

    transposeData->setNNZ( true );

    // count nnz per column again and append into each row of transpose
    // create temporary vector of counters to hold position in each row
    std::vector<lidx_t> row_ctr( transposeData->d_num_rows, 0 );
    for ( lidx_t row = 0; row < d_num_rows; ++row ) {
        for ( lidx_t c = d_row_starts[row]; c < d_row_starts[row + 1]; ++c ) {
            const auto t_row = trans_row( c );
            const auto pos   = transposeData->d_row_starts[t_row] + row_ctr[t_row];
            // local transpose only fills global cols and coeffs
            // caller responsible for creation of local columns if desired
            transposeData->d_cols[pos]   = static_cast<gidx_t>( row ) + d_first_row;
            transposeData->d_coeffs[pos] = d_coeffs[c];
            row_ctr[t_row]++;
        }
    }

    return transposeData;
}

template<typename Config>
void CSRLocalMatrixData<Config>::setNNZ( lidx_t tot_nnz )
{
    d_nnz = tot_nnz;

    if ( d_nnz == 0 ) {
        d_is_empty = true;
        // nothing to do, block stays empty
        return;
    }

    // allocate and fill remaining arrays
    d_is_empty = false;
    d_cols     = sharedArrayBuilder( d_nnz, d_gidxAllocator );
    d_cols_loc = sharedArrayBuilder( d_nnz, d_lidxAllocator );
    if ( !d_is_symbolic ) {
        d_coeffs = sharedArrayBuilder( d_nnz, d_scalarAllocator );
    }

    AMP::Utilities::Algorithms<gidx_t>::fill_n( d_cols.get(), d_nnz, 0 );
    AMP::Utilities::Algorithms<lidx_t>::fill_n( d_cols_loc.get(), d_nnz, 0 );
    if ( !d_is_symbolic ) {
        AMP::Utilities::Algorithms<scalar_t>::fill_n( d_coeffs.get(), d_nnz, 0.0 );
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::setNNZ( bool do_accum )
{
    if ( do_accum ) {
        AMP::Utilities::Algorithms<lidx_t>::exclusive_scan(
            d_row_starts.get(), d_num_rows + 1, d_row_starts.get(), 0 );
    }

    if ( AMP::Utilities::getMemoryType( d_row_starts.get() ) ==
         AMP::Utilities::MemoryType::device ) {
        const lidx_t *ptr_loc = d_row_starts.get() + d_num_rows;
        AMP::Utilities::Algorithms<lidx_t>::copy_n( ptr_loc, 1, &d_nnz );
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
    d_cols     = sharedArrayBuilder( d_nnz, d_gidxAllocator );
    d_cols_loc = sharedArrayBuilder( d_nnz, d_lidxAllocator );
    if ( !d_is_symbolic ) {
        d_coeffs = sharedArrayBuilder( d_nnz, d_scalarAllocator );
    }

    AMP::Utilities::Algorithms<gidx_t>::fill_n( d_cols.get(), d_nnz, 0 );
    AMP::Utilities::Algorithms<lidx_t>::fill_n( d_cols_loc.get(), d_nnz, 0 );
    if ( !d_is_symbolic ) {
        AMP::Utilities::Algorithms<scalar_t>::fill_n( d_coeffs.get(), d_nnz, 0.0 );
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::setNNZ( const std::vector<lidx_t> &nnz )
{
    // copy passed nnz vector into row_starts and call internal setNNZ
    AMP::Utilities::Algorithms<lidx_t>::copy_n( nnz.data(), d_num_rows, d_row_starts.get() );
    setNNZ( true );
}

template<typename Config>
void CSRLocalMatrixData<Config>::removeRange( const scalar_t bnd_lo, const scalar_t bnd_up )
{
    AMP_INSIST( !d_is_symbolic,
                "CSRLocalMatrixData::removeRange not defined for symbolic matrices" );

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::removeRange not implemented on device yet" );
    if ( d_is_empty ) {
        return;
    }

    // count coeffs that lie within range and zero them along the way
    lidx_t num_delete = 0;
    std::vector<lidx_t> delete_per_row( d_num_rows, 0 );
    for ( lidx_t row = 0; row < d_num_rows; ++row ) {
        for ( lidx_t c = d_row_starts[row]; c < d_row_starts[row + 1]; ++c ) {
            if ( bnd_lo < d_coeffs[c] && d_coeffs[c] < bnd_up ) {
                delete_per_row[row]++;
                num_delete++;
            }
        }
    }

    // if none to delete then done
    if ( num_delete == 0 ) {
        return;
    }

    // if all entries will be deleted throw a warning and set the matrix
    // as empty
    if ( d_nnz == num_delete ) {
        AMP::Utilities::Algorithms<lidx_t>::fill_n( d_row_starts.get(), d_num_rows + 1, 0 );
        d_cols.reset();
        d_cols_unq.reset();
        d_cols_loc.reset();
        d_coeffs.reset();
        d_nnz       = 0;
        d_ncols_unq = 0;
        d_is_empty  = true;
        AMP_WARNING( "CSRLocalMatrixData::removeRange deleting all entries" );
        return;
    }

    // allocate space for new data fields and copy over parts to keep
    const lidx_t old_nnz = d_nnz;
    d_nnz -= num_delete;
    auto new_row_starts = sharedArrayBuilder( d_num_rows + 1, d_lidxAllocator );
    auto new_coeffs     = sharedArrayBuilder( d_nnz, d_scalarAllocator );
    std::shared_ptr<lidx_t[]> new_cols_loc;
    std::shared_ptr<gidx_t[]> new_cols;
    if ( d_is_diag ) {
        new_cols_loc = sharedArrayBuilder( d_nnz, d_lidxAllocator );
    } else {
        new_cols = sharedArrayBuilder( d_nnz, d_gidxAllocator );
    }

    // new row starts is old minus running total of deleted entries
    lidx_t run_ndel   = 0;
    new_row_starts[0] = 0;
    for ( lidx_t row = 1; row <= d_num_rows; ++row ) {
        run_ndel += delete_per_row[row - 1];
        new_row_starts[row] = d_row_starts[row] - run_ndel;
    }

    // coeffs is a masked copy
    // cols_loc is masked copy if this is diag block, otherwise
    // build cols from cols_unq and call globalToLocal
    lidx_t nctr = 0;
    for ( lidx_t n = 0; n < old_nnz; ++n ) {
        if ( bnd_lo < d_coeffs[n] && d_coeffs[n] < bnd_up ) {
            continue;
        }
        new_coeffs[nctr] = d_coeffs[n];
        if ( d_is_diag ) {
            new_cols_loc[nctr] = d_cols_loc[n];
        } else {
            new_cols[nctr] = d_cols_unq[d_cols_loc[n]];
        }
        ++nctr;
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
    std::cout << ( d_is_diag ? "  diag block:" : "  offd block:" ) << std::endl;
    if ( d_is_empty ) {
        std::cout << "    EMPTY" << std::endl;
        return;
    }
    std::cout << "    first | last row: " << d_first_row << " | " << d_last_row << std::endl;
    std::cout << "    first | last col: " << d_first_col << " | " << d_last_col << std::endl;

    if ( d_cols.get() ) {
        std::cout << "    min | max col: "
                  << AMP::Utilities::Algorithms<gidx_t>::min_element( d_cols.get(), d_nnz ) << " | "
                  << AMP::Utilities::Algorithms<gidx_t>::max_element( d_cols.get(), d_nnz )
                  << std::endl;
    }

    std::cout << "    num unique: " << d_ncols_unq << std::endl;
    scalar_t avg_nnz = static_cast<scalar_t>( d_nnz ) / static_cast<scalar_t>( d_num_rows );
    std::cout << "    avg nnz per row: " << avg_nnz << std::endl;
    std::cout << "    tot nnz: " << d_nnz << std::endl;
    if ( verbose && d_memory_location < AMP::Utilities::MemoryType::device ) {
        std::cout << "    row 0: ";
        for ( auto n = d_row_starts[0]; n < d_row_starts[10]; ++n ) {
            if ( d_coeffs.get() && ( d_coeffs[n] != 0 || show_zeros ) ) {
                std::cout << "("
                          << ( d_cols.get() ? static_cast<long long>( d_cols[n] ) :
                                              static_cast<long long>( d_cols_loc[n] ) )
                          << "," << d_coeffs[n] << "), ";
            } else if ( show_zeros ) {
                std::cout << "("
                          << ( d_cols.get() ? static_cast<long long>( d_cols[n] ) :
                                              static_cast<long long>( d_cols_loc[n] ) )
                          << ",--), ";
            }
        }
        std::cout << "\n    row last: ";
        for ( auto n = d_row_starts[d_num_rows - 1]; n < d_row_starts[d_num_rows]; ++n ) {
            if ( d_coeffs.get() && ( d_coeffs[n] != 0 || show_zeros ) ) {
                std::cout << "("
                          << ( d_cols.get() ? static_cast<long long>( d_cols[n] ) :
                                              static_cast<long long>( d_cols_loc[n] ) )
                          << "," << d_coeffs[n] << "), ";
            } else if ( show_zeros ) {
                std::cout << "("
                          << ( d_cols.get() ? static_cast<long long>( d_cols[n] ) :
                                              static_cast<long long>( d_cols_loc[n] ) )
                          << ",--), ";
            }
        }
        if ( d_ncols_unq > 0 && d_ncols_unq < 200 ) {
            std::cout << "\n    column map: ";
            for ( auto n = 0; n < d_ncols_unq; ++n ) {
                std::cout << "[" << n << "|" << d_cols_unq[n] << "], ";
            }
        }
    } else if ( verbose ) {
        AMP_INSIST( !d_is_symbolic,
                    "CSRLocalMatrixData::printStats not implemented for symbolic device matrices" );

        // copy row pointers back to host
        std::vector<lidx_t> rs_h( d_num_rows + 1, 0 );
        AMP::Utilities::copy( d_num_rows + 1, d_row_starts.get(), rs_h.data() );

        // if first row is non-empty copy it back to host
        // need to check if cols or cols_loc should be used
        if ( rs_h[1] > rs_h[0] ) {
            const auto fr_len = rs_h[1] - rs_h[0];
            std::vector<scalar_t> fr_coeffs( fr_len, 0.0 );
            AMP::Utilities::copy( fr_len, d_coeffs.get(), fr_coeffs.data() );
            if ( d_cols.get() ) {
                std::vector<gidx_t> fr_cols( fr_len, 0 );
                AMP::Utilities::copy( fr_len, d_cols.get(), fr_cols.data() );
                std::cout << "    row 0: ";
                for ( lidx_t n = 0; n < fr_len; ++n ) {
                    if ( fr_coeffs[n] != 0 || show_zeros ) {
                        std::cout << "(" << fr_cols[n] << "," << fr_coeffs[n] << "), ";
                    }
                }
                std::cout << std::endl;
            } else {
                std::vector<lidx_t> fr_cols( fr_len, 0 );
                AMP::Utilities::copy( fr_len, d_cols_loc.get(), fr_cols.data() );
                std::cout << "    row 0: ";
                for ( lidx_t n = 0; n < fr_len; ++n ) {
                    if ( fr_coeffs[n] != 0 || show_zeros ) {
                        std::cout << "(" << fr_cols[n] << "," << fr_coeffs[n] << "), ";
                    }
                }
                std::cout << std::endl;
            }
        }

        // same as before but for the last row
        if ( rs_h[d_num_rows] > rs_h[d_num_rows - 1] ) {
            const auto lr_len = rs_h[d_num_rows] - rs_h[d_num_rows - 1];
            std::vector<scalar_t> lr_coeffs( lr_len, 0.0 );
            AMP::Utilities::copy( lr_len, d_coeffs.get() + rs_h[d_num_rows - 1], lr_coeffs.data() );
            if ( d_cols.get() ) {
                std::vector<gidx_t> lr_cols( lr_len, 0 );
                AMP::Utilities::copy( lr_len, d_cols.get() + rs_h[d_num_rows - 1], lr_cols.data() );
                std::cout << "    row last: ";
                for ( lidx_t n = 0; n < lr_len; ++n ) {
                    if ( lr_coeffs[n] != 0 || show_zeros ) {
                        std::cout << "(" << lr_cols[n] << "," << lr_coeffs[n] << "), ";
                    }
                }
            } else {
                std::vector<lidx_t> lr_cols( lr_len, 0 );
                AMP::Utilities::copy(
                    lr_len, d_cols_loc.get() + rs_h[d_num_rows - 1], lr_cols.data() );
                std::cout << "    row last: ";
                for ( lidx_t n = 0; n < lr_len; ++n ) {
                    if ( lr_coeffs[n] != 0 || show_zeros ) {
                        std::cout << "(" << lr_cols[n] << "," << lr_coeffs[n] << "), ";
                    }
                }
            }
        }

        // copy down column map and print it
        if ( d_ncols_unq > 0 && d_ncols_unq < 200 ) {
            std::vector<gidx_t> colmap_h( d_ncols_unq, 0 );
            AMP::Utilities::copy( d_ncols_unq, d_cols_unq.get(), colmap_h.data() );
            std::cout << "\n    column map: ";
            for ( auto n = 0; n < d_ncols_unq; ++n ) {
                std::cout << "[" << n << "|" << colmap_h[n] << "], ";
            }
        }
    }
    std::cout << std::endl << std::endl;
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

    const bool have_loc = ( d_cols_loc.get() != nullptr );
    const bool have_gbl = ( d_cols.get() != nullptr );

    // migrate fields to host if needed
    std::vector<lidx_t> row_starts_h, cols_loc_h;
    std::vector<gidx_t> cols_h, cols_unq_h;
    std::vector<scalar_t> coeffs_h;
    lidx_t *row_starts = nullptr, *cols_loc = nullptr;
    gidx_t *cols = nullptr, *cols_unq = nullptr;
    scalar_t *coeffs = nullptr;
    if ( d_memory_location < AMP::Utilities::MemoryType::device ) {
        row_starts = d_row_starts.get();
        cols_loc   = d_cols_loc.get();
        cols       = d_cols.get();
        cols_unq   = d_cols_unq.get();
        coeffs     = d_coeffs.get();
    } else {
        row_starts_h.resize( d_num_rows + 1, 0 );
        AMP::Utilities::copy( d_num_rows + 1, d_row_starts.get(), row_starts_h.data() );
        row_starts = row_starts_h.data();
        if ( have_gbl ) {
            cols_h.resize( d_nnz, 0 );
            AMP::Utilities::copy( d_nnz, d_cols.get(), cols_h.data() );
            cols = cols_h.data();
        } else if ( !d_is_diag ) {
            cols_unq_h.resize( d_ncols_unq, 0 );
            AMP::Utilities::copy( d_ncols_unq, d_cols_unq.get(), cols_unq_h.data() );
            cols_unq = cols_unq_h.data();
        }
        if ( have_loc ) {
            cols_loc_h.resize( d_nnz, 0 );
            AMP::Utilities::copy( d_nnz, d_cols_loc.get(), cols_loc_h.data() );
            cols_loc = cols_loc_h.data();
        }
        coeffs_h.resize( d_nnz, 0 );
        AMP::Utilities::copy( d_nnz, d_coeffs.get(), coeffs_h.data() );
        coeffs = coeffs_h.data();
    }

    // print all unique columns
    if ( cols_unq ) {
        std::cout << "Unique cols: ";
        for ( lidx_t n = 0; n < d_ncols_unq; ++n ) {
            std::cout << "[" << n << "|" << cols_unq[n] << "] ";
        }
        std::cout << std::endl << std::endl;
    }

    // print all global columns and values row-by-row
    for ( lidx_t row = 0; row < d_num_rows; ++row ) {
        // skip empty rows to avoid a bunch of blank newlines
        if ( row_starts[row] < row_starts[row + 1] ) {
            std::cout << "Row " << row << ": ";
            for ( lidx_t c = row_starts[row]; c < row_starts[row + 1]; ++c ) {
                lidx_t cl = have_loc ? cols_loc[c] : -1;
                gidx_t cg;
                if ( d_is_diag ) {
                    cg = have_gbl ? cols[c] : d_first_col + static_cast<gidx_t>( cols_loc[c] );
                } else {
                    cg = have_gbl ? cols[c] : cols_unq[cols_loc[c]];
                }
                std::cout << "[" << cl << "|" << cg << "|" << coeffs[c] << "] ";
            }
            std::cout << std::endl;
        }
    }
}

template<typename Config>
void CSRLocalMatrixData<Config>::getRowByGlobalID( const size_t local_row,
                                                   std::vector<size_t> &cols,
                                                   std::vector<double> &values ) const
{
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
                                                      size_t *cols,
                                                      scalar_t *values ) const
{
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
std::vector<size_t> CSRLocalMatrixData<Config>::getColumnIDs( const size_t local_row ) const
{
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

} // namespace AMP::LinearAlgebra

#endif
