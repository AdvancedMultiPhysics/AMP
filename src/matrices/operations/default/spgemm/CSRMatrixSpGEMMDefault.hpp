#include "AMP/matrices/operations/default/spgemm/CSRMatrixSpGEMMDefault.h"
#include "AMP/utils/UtilityMacros.h"

#include "ProfilerApp.h"

#ifndef CSRSPGEMM_REPORT_SPACC_STATS
    #define CSRSPGEMM_REPORT_SPACC_STATS 0
#endif

namespace AMP::LinearAlgebra {

template<typename Config>
void CSRMatrixSpGEMMDefault<Config>::multiplyLocal( std::shared_ptr<localmatrixdata_t> A_data,
                                                    std::shared_ptr<localmatrixdata_t> B_data,
                                                    std::shared_ptr<localmatrixdata_t> C_data )
{
    if ( C_data->isDiag() ) {
        multiplyBlock<Mode::SYMBOLIC, BlockType::DIAG>( A_data, B_data, C_data );
        multiplyBlock<Mode::NUMERIC, BlockType::DIAG>( A_data, B_data, C_data );
    } else {
        multiplyBlock<Mode::SYMBOLIC, BlockType::OFFD>( A_data, B_data, C_data );
        multiplyBlock<Mode::NUMERIC, BlockType::OFFD>( A_data, B_data, C_data );
    }
}

template<typename Config>
template<typename CSRMatrixSpGEMMDefault<Config>::Mode mode_t,
         typename CSRMatrixSpGEMMDefault<Config>::BlockType block_t>
void CSRMatrixSpGEMMDefault<Config>::multiplyBlock( std::shared_ptr<localmatrixdata_t> A_data,
                                                    std::shared_ptr<localmatrixdata_t> B_data,
                                                    std::shared_ptr<localmatrixdata_t> C_data )
{
    using acc_t = typename std::conditional<block_t == BlockType::DIAG,
                                            DenseAccumulator<gidx_t>,
                                            SparseAccumulator<gidx_t>>::type;

    AMP_DEBUG_ASSERT( A_data != nullptr );
    AMP_DEBUG_ASSERT( B_data != nullptr );
    AMP_DEBUG_ASSERT( C_data != nullptr );

    if ( A_data->isEmpty() || B_data->isEmpty() ) {
        return;
    }

    const bool is_diag = block_t == BlockType::DIAG;

    // all fields from blocks involved
    lidx_t *A_rs = nullptr, *A_cols_loc = nullptr;
    gidx_t *A_cols     = nullptr;
    scalar_t *A_coeffs = nullptr;

    lidx_t *B_rs = nullptr, *B_cols_loc = nullptr;
    gidx_t *B_cols     = nullptr;
    scalar_t *B_coeffs = nullptr;

    lidx_t *C_rs = nullptr, *C_cols_loc = nullptr;
    gidx_t *C_cols     = nullptr;
    scalar_t *C_coeffs = nullptr;

    // Extract available fields
    std::tie( A_rs, A_cols, A_cols_loc, A_coeffs ) = A_data->getDataFields();
    std::tie( B_rs, B_cols, B_cols_loc, B_coeffs ) = B_data->getDataFields();
    std::tie( C_rs, C_cols, C_cols_loc, C_coeffs ) = C_data->getDataFields();

    AMP_ASSERT( A_cols_loc != nullptr );
    if constexpr ( is_diag ) {
        AMP_ASSERT( B_cols_loc != nullptr ); // dense needs local cols
    }
    AMP_ASSERT( B_cols != nullptr || B_cols_loc != nullptr ); // otherwise just need one of them

    auto B_colmap = B_data->getColumnMap();
    if ( !is_diag && B_cols == nullptr ) {
        AMP_ASSERT( B_colmap != nullptr );
    }
    [[maybe_unused]] const auto B_nnz = B_data->numberOfNonZeros();
    const auto first_col              = B_data->beginCol();
    const auto A_nrows                = A_data->numLocalRows();

    // DenseAcc's act on assembled blocks that may have global columns removed
    // set up conversion for that case
    DISABLE_WARNINGS
    auto B_to_global =
        [is_diag, B_cols, B_cols_loc, first_col, B_colmap]( const lidx_t k ) -> gidx_t {
        if ( B_cols != nullptr ) {
            return B_cols[k];
        }
        return is_diag ? first_col + B_cols_loc[k] : B_colmap[B_cols_loc[k]];
    };
    ENABLE_WARNINGS

    // Create accumulator with appropriate capacity
    const lidx_t acc_cap = is_diag ? B_data->numLocalColumns() : SPACC_SIZE;
    acc_t acc( acc_cap, first_col );

    // Finally, after all the setup do the actual computation
    if constexpr ( mode_t == Mode::SYMBOLIC ) {
        PROFILE( is_diag ? "CSRMatrixSpGEMMDefault::multiply (symbolic -- C_diag)" :
                           "CSRMatrixSpGEMMDefault::multiply (symbolic -- C_offd)" );
        // If this is a symbolic call just count NZ and write to
        // rs field in C
        for ( lidx_t row = 0; row < A_nrows; ++row ) {
            // get rows in B block from the A_diag column indices
            for ( lidx_t j = A_rs[row]; j < A_rs[row + 1]; ++j ) {
                const auto Acl = A_cols_loc[j];
                AMP_DEBUG_ASSERT( Acl <= B_data->numLocalRows() );
                // then row of C is union of those B row nz patterns
                for ( lidx_t k = B_rs[Acl]; k < B_rs[Acl + 1]; ++k ) {
                    AMP_DEBUG_ASSERT( k < B_nnz );
                    const auto gbl = B_to_global( k );
                    if ( is_diag ) {
                        AMP_DEBUG_ASSERT( B_data->beginCol() <= gbl && gbl < B_data->endCol() );
                    } else {
                        AMP_DEBUG_ASSERT( B_data->beginCol() > gbl || gbl >= B_data->endCol() );
                    }
                    acc.insert_or_append( gbl );
                }
            }
            // write out row length and clear accumulator
            C_rs[row] += acc.num_inserted;
            acc.clear();
        }
        C_data->setNNZ( true );
#if CSRSPGEMM_REPORT_SPACC_STATS
        if ( !use_dense && ( acc.total_collisions > 0 || acc.total_grows > 0 ) ) {
            AMP::pout << "\nSparseAcc stats:\n"
                      << "  Insertions: " << acc.total_inserted << "\n"
                      << "  Collisions: " << acc.total_collisions << "\n"
                      << "      Probes: " << acc.total_probe_steps << "\n"
                      << "      Clears: " << acc.total_clears << "\n"
                      << "       Grows: " << acc.total_grows << "\n"
                      << std::endl;
        }
#endif
    } else {
        PROFILE( is_diag ? "CSRMatrixSpGEMMDefault::multiply (numeric -- C_diag)" :
                           "CSRMatrixSpGEMMDefault::multiply (numeric -- C_offd)" );
        // Otherwise, for numeric call write directly into C by
        // passing pointers into cols and coeffs fields as workspace
        // for the accumulator
        for ( lidx_t row = 0; row < A_nrows; ++row ) {
            auto col_space = &C_cols[C_rs[row]];
            auto val_space = &C_coeffs[C_rs[row]];
            // get rows in B block from the A column indices
            for ( lidx_t j = A_rs[row]; j < A_rs[row + 1]; ++j ) {
                const auto Acl  = A_cols_loc[j];
                const auto Aval = A_coeffs[j];
                // then row of C is union of those B row nz patterns
                for ( lidx_t k = B_rs[Acl]; k < B_rs[Acl + 1]; ++k ) {
                    const auto gbl = B_to_global( k );
                    acc.insert_or_append( gbl, Aval * B_coeffs[k], col_space, val_space );
                }
            }
            // Clear accumulator to prepare for next row
            acc.clear();
        }
    }
}

#if 0
template<typename Config>
void CSRMatrixSpGEMMDefault<Config>::mergeDiag( std::shared_ptr<localmatrixdata_t> inL,
                                                std::shared_ptr<localmatrixdata_t> inR,
                                                std::shared_ptr<localmatrixdata_t> out )
{
    PROFILE( "CSRMatrixSpGEMMDefault::mergeDiag" );

    // handle special case where either (or both) inputs are empty/null
    if ( inL.get() == nullptr && inR.get() == nullptr ) {
        return;
    }
    if ( inR.get() == nullptr || inR->isEmpty() ) {
        out->swapDataFields( *inL );
        return;
    }
    if ( inL.get() == nullptr || inL->isEmpty() ) {
        out->swapDataFields( *inR );
        return;
    }

    const auto first_col = out->beginCol();

    // pull out fields from blocks to merge and row pointers from out
    lidx_t *inL_rs = nullptr, *inR_rs = nullptr, *out_rs = nullptr;
    lidx_t *inL_cols_loc = nullptr, *inR_cols_loc = nullptr;
    gidx_t *inL_cols = nullptr, *inR_cols = nullptr;
    scalar_t *inL_coeffs = nullptr, *inR_coeffs = nullptr;

    std::tie( inL_rs, inL_cols, inL_cols_loc, inL_coeffs ) = inL->getDataFields();
    std::tie( inR_rs, inR_cols, inR_cols_loc, inR_coeffs ) = inR->getDataFields();
    out_rs                                                 = out->getRowStarts();
    const auto num_rows                                    = out->numLocalRows();

    // Create allocator with space for out operations
    const auto acc_cap = out->numLocalColumns();
    DenseAccumulator<gidx_t> acc( acc_cap, first_col );

    // loop over all rows and count unique NZ positions in each
    for ( lidx_t row = 0; row < num_rows; ++row ) {
        // Add inL row to accumulator
        for ( lidx_t j = inL_rs[row]; j < inL_rs[row + 1]; ++j ) {
            acc.insert_or_append( inL_cols[j] );
        }
        // Add inR row to accumulator
        for ( lidx_t j = inR_rs[row]; j < inR_rs[row + 1]; ++j ) {
            acc.insert_or_append( inR_cols[j] );
        }
        // write out row length and clear accumulator
        out_rs[row] += acc.num_inserted;
        acc.clear();
    }

    // allocate space in matrix
    out->setNNZ( true );

    // pull result fields out
    lidx_t *out_cols_loc;
    gidx_t *out_cols;
    scalar_t *out_coeffs;
    std::tie( out_rs, out_cols, out_cols_loc, out_coeffs ) = out->getDataFields();

    // loop over all rows again and write columns/coeffs into allocated matrix
    for ( lidx_t row = 0; row < num_rows; ++row ) {
        auto cols = &out_cols[out_rs[row]];
        auto vals = &out_coeffs[out_rs[row]];
        // Add inL row to accumulator
        for ( lidx_t j = inL_rs[row]; j < inL_rs[row + 1]; ++j ) {
            acc.insert_or_append( inL_cols[j], inL_coeffs[j], cols, vals );
        }
        // Add inR row to accumulator
        for ( lidx_t j = inR_rs[row]; j < inR_rs[row + 1]; ++j ) {
            acc.insert_or_append( inR_cols[j], inR_coeffs[j], cols, vals );
        }
        // clear accumulator
        acc.clear();
    }
}

template<typename Config>
void CSRMatrixSpGEMMDefault<Config>::mergeOffd( std::shared_ptr<localmatrixdata_t> inL,
                                                std::shared_ptr<localmatrixdata_t> inR,
                                                std::shared_ptr<localmatrixdata_t> out )
{
    PROFILE( "CSRMatrixSpGEMMDefault::mergeOffd" );

    // handle special case where either (or both) inputs are empty/null
    if ( inL.get() == nullptr && inR.get() == nullptr ) {
        return;
    }
    if ( inR.get() == nullptr || inR->isEmpty() ) {
        out->swapDataFields( *inL );
        return;
    }
    if ( inL.get() == nullptr || inL->isEmpty() ) {
        out->swapDataFields( *inR );
        return;
    }

    // pull out fields from blocks to merge and row pointers from out
    lidx_t *inL_rs, *inR_rs, *out_rs;
    lidx_t *inL_cols_loc, *inR_cols_loc;
    gidx_t *inL_cols, *inR_cols;
    scalar_t *inL_coeffs, *inR_coeffs;

    std::tie( inL_rs, inL_cols, inL_cols_loc, inL_coeffs ) = inL->getDataFields();
    std::tie( inR_rs, inR_cols, inR_cols_loc, inR_coeffs ) = inR->getDataFields();
    out_rs                                                 = out->getRowStarts();
    const auto num_rows                                    = out->numLocalRows();

    // Create allocator with space for out operations
    SparseAccumulator<gidx_t> acc( SPACC_SIZE, 0 );

    // loop over all rows and count unique NZ positions in each
    for ( lidx_t row = 0; row < num_rows; ++row ) {
        // Add inL row to accumulator
        for ( lidx_t j = inL_rs[row]; j < inL_rs[row + 1]; ++j ) {
            acc.insert_or_append( inL_cols[j] );
        }
        // Add inR row to accumulator
        for ( lidx_t j = inR_rs[row]; j < inR_rs[row + 1]; ++j ) {
            acc.insert_or_append( inR_cols[j] );
        }
        // write out row length and clear accumulator
        out_rs[row] += acc.num_inserted;
        acc.clear();
    }

    // allocate space in matrix
    out->setNNZ( true );

    // report accumulator stats if useful
    #if CSRSPGEMM_REPORT_SPACC_STATS
    if ( acc.total_collisions > 0 || acc.total_grows > 0 ) {
        AMP::pout << "\nSparseAcc stats:\n"
                  << "  Insertions: " << acc.total_inserted << "\n"
                  << "  Collisions: " << acc.total_collisions << "\n"
                  << "      Probes: " << acc.total_probe_steps << "\n"
                  << "      Clears: " << acc.total_clears << "\n"
                  << "       Grows: " << acc.total_grows << "\n"
                  << std::endl;
    }
    #endif
    // pull result fields out
    lidx_t *out_cols_loc;
    gidx_t *out_cols;
    scalar_t *out_coeffs;
    std::tie( out_rs, out_cols, out_cols_loc, out_coeffs ) = out->getDataFields();

    // loop over all rows again and write columns/coeffs into allocated matrix
    for ( lidx_t row = 0; row < num_rows; ++row ) {
        auto cols = &out_cols[out_rs[row]];
        auto vals = &out_coeffs[out_rs[row]];
        // Add inL row to accumulator
        for ( lidx_t j = inL_rs[row]; j < inL_rs[row + 1]; ++j ) {
            acc.insert_or_append( inL_cols[j], inL_coeffs[j], cols, vals );
        }
        // Add inR row to accumulator
        for ( lidx_t j = inR_rs[row]; j < inR_rs[row + 1]; ++j ) {
            acc.insert_or_append( inR_cols[j], inR_coeffs[j], cols, vals );
        }
        // clear accumulator
        acc.clear();
    }
}
#endif

template<typename Config>
template<typename col_t>
typename Config::lidx_t
CSRMatrixSpGEMMDefault<Config>::DenseAccumulator<col_t>::contains( col_t col_idx ) const
{
    const auto loc = IsGlobal ? static_cast<lidx_t>( col_idx - offset ) : col_idx;
    return flags[loc];
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::DenseAccumulator<col_t>::set_flag( col_t col_idx,
                                                                        typename Config::lidx_t k )
{
    const auto loc      = IsGlobal ? static_cast<lidx_t>( col_idx - offset ) : col_idx;
    const auto old_flag = flags[loc];
    flags[loc]          = k;
    if ( old_flag == -1 ) {
        if ( num_inserted == static_cast<lidx_t>( flag_inv.size() ) ) {
            flag_inv.push_back( loc );
        } else {
            flag_inv[num_inserted] = loc;
        }
        ++num_inserted;
    }
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::DenseAccumulator<col_t>::insert_or_append( col_t col_idx )
{
    const auto loc = IsGlobal ? static_cast<lidx_t>( col_idx - offset ) : col_idx;
    const auto k   = flags[loc];
    if ( k == -1 ) {
        flags[loc] = num_inserted;
        if ( num_inserted == static_cast<lidx_t>( flag_inv.size() ) ) {
            flag_inv.push_back( loc );
            cols.push_back( col_idx );
        } else {
            flag_inv[num_inserted] = loc;
            cols[num_inserted]     = col_idx;
        }
        ++num_inserted;
    }
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::DenseAccumulator<col_t>::insert_or_append(
    col_t col_idx,
    typename Config::scalar_t val,
    col_t *col_space,
    typename Config::scalar_t *val_space )
{
    using lidx_t = typename Config::lidx_t;

    const auto loc = IsGlobal ? static_cast<lidx_t>( col_idx - offset ) : col_idx;
    const auto k   = flags[loc];
    if ( k == -1 ) {
        flags[loc] = num_inserted;
        if ( num_inserted == static_cast<lidx_t>( flag_inv.size() ) ) {
            flag_inv.push_back( loc );
        } else {
            flag_inv[num_inserted] = loc;
        }
        col_space[num_inserted] = col_idx;
        val_space[num_inserted] = val;
        ++num_inserted;
    } else {
        val_space[k] += val;
    }
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::DenseAccumulator<col_t>::clear()
{
    for ( lidx_t n = 0; n < num_inserted; ++n ) {
        flags[flag_inv[n]] = -1;
    }
    num_inserted = 0;
}

template<typename Config>
template<typename col_t>
uint16_t CSRMatrixSpGEMMDefault<Config>::SparseAccumulator<col_t>::hash( col_t col_idx ) const
{
    const uint16_t c0 = ( 506999 * col_idx ) & 0xFFFF;
    const uint16_t c1 = ( col_idx >> 16 ) & 0xFFFF;
    return ( c0 ^ c1 ) % capacity;
}

template<typename Config>
template<typename col_t>
typename Config::lidx_t
CSRMatrixSpGEMMDefault<Config>::SparseAccumulator<col_t>::contains( col_t col_idx ) const
{
    auto pos = hash( col_idx ), flag = flags[pos];
    if ( flag == 0xFFFF ) {
        // Location is empty, certainly not present
        return -1;
    } else {
        // location occupied, linear probe to empty or col_idx found
        do {
            if ( cols[flag] == col_idx ) {
                return static_cast<lidx_t>( flag );
            }
            pos  = ( pos + 1 ) % capacity;
            flag = flags[pos];
        } while ( flag != 0xFFFF );
    }
    // col_idx never found, is not contained
    return -1;
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator<col_t>::set_flag( col_t col_idx,
                                                                         typename Config::lidx_t k )
{
    auto pos = hash( col_idx ), flag = flags[pos];
    if ( flag == 0xFFFF ) {
        flags[pos] = k;
        if ( cols.size() <= num_inserted ) {
            AMP_DEBUG_ASSERT( num_inserted == k );
            cols.push_back( col_idx );
        } else {
            cols[k] = col_idx;
        }
        ++num_inserted;
    } else {
        do {
            AMP_DEBUG_ASSERT( cols[flag] != col_idx );
            pos  = ( pos + 1 ) % capacity;
            flag = flags[pos];
        } while ( flag != 0xFFFF );
        flags[pos] = k;
        if ( cols.size() <= num_inserted ) {
            AMP_DEBUG_ASSERT( num_inserted == k );
            cols.push_back( col_idx );
        } else {
            cols[k] = col_idx;
        }
        ++num_inserted;
    }
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator<col_t>::insert_or_append( col_t col_idx )
{
    if ( num_inserted == capacity ) {
        grow( cols.data() );
    }

    auto pos = hash( col_idx ), flag = flags[pos];
    if ( flag == 0xFFFF ) {
        // Location is empty, append
        flags[pos] = num_inserted;
        if ( cols.size() <= num_inserted ) {
            cols.push_back( col_idx );
        } else {
            cols[num_inserted] = col_idx;
        }
        ++num_inserted;
#if CSRSPGEMM_REPORT_SPACC_STATS
        ++total_inserted;
#endif
    } else {
        // location occupied, linear probe to empty or col_idx found
#if CSRSPGEMM_REPORT_SPACC_STATS
        if ( cols[flag] != col_idx ) {
            ++total_collisions;
        }
#endif
        do {
            if ( cols[flag] == col_idx ) {
                return;
            }
            pos  = ( pos + 1 ) % capacity;
            flag = flags[pos];
#if CSRSPGEMM_REPORT_SPACC_STATS
            ++total_probe_steps;
#endif
        } while ( flag != 0xFFFF );
        // col_idx never found, have empty slot
        flags[pos] = num_inserted;
        if ( cols.size() <= num_inserted ) {
            cols.push_back( col_idx );
        } else {
            cols[num_inserted] = col_idx;
        }
        ++num_inserted;
#if CSRSPGEMM_REPORT_SPACC_STATS
        ++total_inserted;
#endif
    }
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator<col_t>::insert_or_append(
    col_t col_idx,
    typename Config::scalar_t val,
    col_t *col_space,
    typename Config::scalar_t *val_space )
{
    if ( num_inserted == capacity ) {
        grow( col_space );
    }

    auto pos = hash( col_idx ), flag = flags[pos];
    if ( flag == 0xFFFF ) {
        // Location is empty, append
        flags[pos]              = num_inserted;
        col_space[num_inserted] = col_idx;
        val_space[num_inserted] = val;
        ++num_inserted;
    } else {
        // location occupied, linear probe to empty or col_idx found
        do {
            if ( col_space[flag] == col_idx ) {
                val_space[flag] += val;
                return;
            }
            pos  = ( pos + 1 ) % capacity;
            flag = flags[pos];
        } while ( flag != 0xFFFF );
        // col_idx never found, have empty slot
        flags[pos]              = num_inserted;
        col_space[num_inserted] = col_idx;
        val_space[num_inserted] = val;
        ++num_inserted;
    }
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator<col_t>::grow( col_t *col_space )
{
#if CSRSPGEMM_REPORT_SPACC_STATS
    ++total_grows;
#endif

    uint16_t old_capacity = capacity;
    capacity *= 2;
    std::vector<uint16_t> old_flags( capacity, 0xFFFF );
    flags.swap( old_flags );
    for ( uint16_t n = 0; n < old_capacity; ++n ) {
        // insert all existing global local pairs into new flags vector
        // do it inline since cols/col_space should not be touched,
        // and nor should the num_inserted value
        const auto loc     = old_flags[n];
        const auto col_idx = col_space[loc];
        auto pos = hash( col_idx ), flag = flags[pos];
        if ( flag == 0xFFFF ) {
            // Location is empty, set and carry on
            flags[pos] = loc;
        } else {
            // location occupied, linear probe to empty
            // Note: must find empty since grow is guaranteed to work with set
            // of unique columns
            do {
                pos  = ( pos + 1 ) % capacity;
                flag = flags[pos];
            } while ( flag != 0xFFFF );
            flags[pos] = loc;
        }
    }
}

template<typename Config>
template<typename col_t>
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator<col_t>::clear()
{
#if CSRSPGEMM_REPORT_SPACC_STATS
    total_clears++;
#endif
    num_inserted = 0;
    std::fill( flags.begin(), flags.end(), 0xFFFF );
}

} // namespace AMP::LinearAlgebra

#ifdef CSRSPGEMM_REPORT_SPACC_STATS
    #undef CSRSPGEMM_REPORT_SPACC_STATS
#endif
