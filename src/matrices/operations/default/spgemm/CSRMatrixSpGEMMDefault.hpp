#include "AMP/matrices/operations/default/spgemm/CSRMatrixSpGEMMDefault.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/UtilityMacros.h"

#include <algorithm>
#include <tuple>

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
    multiplyPhase<Mode::SYMBOLIC, BlockType::DIAG>( A_data, B_data, C_data );
    multiplyPhase<Mode::NUMERIC, BlockType::DIAG>( A_data, B_data, C_data );

    // Convert the local indices to globals to make merges easier
    lidx_t *C_rs = nullptr, *C_cols_loc = nullptr;
    gidx_t *C_cols                                 = nullptr;
    scalar_t *C_coeffs                             = nullptr;
    const lidx_t C_nnz                             = C_data->numberOfNonZeros();
    std::tie( C_rs, C_cols, C_cols_loc, C_coeffs ) = C_data->getDataFields();
    if ( C_data->isDiag() ) {
        const auto first_col = C_data->beginCol();
        std::transform(
            C_cols_loc, C_cols_loc + C_nnz, C_cols, [first_col]( const lidx_t lc ) -> gidx_t {
                return static_cast<gidx_t>( lc ) + first_col;
            } );
    } else {
        const auto colmap = B_data->getColumnMap();
        std::transform( C_cols_loc,
                        C_cols_loc + C_nnz,
                        C_cols,
                        [colmap]( const lidx_t lc ) -> gidx_t { return colmap[lc]; } );
    }
}

template<typename Config>
template<typename CSRMatrixSpGEMMDefault<Config>::Mode mode_t,
         typename CSRMatrixSpGEMMDefault<Config>::BlockType block_t>
void CSRMatrixSpGEMMDefault<Config>::multiplyPhase( std::shared_ptr<localmatrixdata_t> A_data,
                                                    std::shared_ptr<localmatrixdata_t> B_data,
                                                    std::shared_ptr<localmatrixdata_t> C_data )
{
    using acc_t = typename std::
        conditional<block_t == BlockType::DIAG, DenseAccumulator, SparseAccumulator>::type;

    AMP_DEBUG_ASSERT( A_data != nullptr );
    AMP_DEBUG_ASSERT( B_data != nullptr );
    AMP_DEBUG_ASSERT( C_data != nullptr );

    if ( A_data->isEmpty() || B_data->isEmpty() ) {
        return;
    }

    // shapes of A and B
    const auto A_nrows = A_data->numLocalRows();
    const auto B_ncols = B_data->numUniqueColumns();

    // all fields from blocks involved
    lidx_t *A_rs = nullptr, *A_cols_loc = nullptr;
    scalar_t *A_coeffs = nullptr;

    lidx_t *B_rs = nullptr, *B_cols_loc = nullptr;
    scalar_t *B_coeffs = nullptr;

    // Extract available fields
    std::tie( A_rs, std::ignore, A_cols_loc, A_coeffs ) = A_data->getDataFields();
    std::tie( B_rs, std::ignore, B_cols_loc, B_coeffs ) = B_data->getDataFields();

    // Create accumulator with appropriate capacity
    const lidx_t acc_cap = block_t == BlockType::DIAG ? B_ncols : SPACC_SIZE;
    acc_t acc( acc_cap );

    // Finally, after all the setup do the actual computation
    if constexpr ( mode_t == Mode::SYMBOLIC ) {
        PROFILE( block_t == BlockType::DIAG ?
                     "CSRMatrixSpGEMMDefault::multiply (symbolic -- C_diag)" :
                     "CSRMatrixSpGEMMDefault::multiply (symbolic -- C_offd)" );
        lidx_t *C_rs = C_data->getRowStarts();
        // If this is a symbolic call just count NZ and write to
        // rs field in C
        for ( lidx_t row = 0; row < A_nrows; ++row ) {
            // get rows in B block from the A_diag column indices
            for ( lidx_t j = A_rs[row]; j < A_rs[row + 1]; ++j ) {
                const auto Acl = A_cols_loc[j];
                AMP_DEBUG_ASSERT( Acl <= B_data->numLocalRows() );
                // then row of C is union of those B row nz patterns
                for ( lidx_t k = B_rs[Acl]; k < B_rs[Acl + 1]; ++k ) {
                    acc.insert_or_append( B_cols_loc[k] );
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
        PROFILE( block_t == BlockType::DIAG ?
                     "CSRMatrixSpGEMMDefault::multiply (numeric -- C_diag)" :
                     "CSRMatrixSpGEMMDefault::multiply (numeric -- C_offd)" );
        lidx_t *C_rs = nullptr, *C_cols_loc = nullptr;
        scalar_t *C_coeffs                                  = nullptr;
        std::tie( C_rs, std::ignore, C_cols_loc, C_coeffs ) = C_data->getDataFields();
        // Otherwise, for numeric call write directly into C by
        // passing pointers into cols and coeffs fields as workspace
        // for the accumulator
        for ( lidx_t row = 0; row < A_nrows; ++row ) {
            auto col_space = &C_cols_loc[C_rs[row]];
            auto val_space = &C_coeffs[C_rs[row]];
            // get rows in B block from the A column indices
            for ( lidx_t j = A_rs[row]; j < A_rs[row + 1]; ++j ) {
                const auto Acl  = A_cols_loc[j];
                const auto Aval = A_coeffs[j];
                // then row of C is union of those B row nz patterns
                for ( lidx_t k = B_rs[Acl]; k < B_rs[Acl + 1]; ++k ) {
                    acc.insert_or_append( B_cols_loc[k], Aval * B_coeffs[k], col_space, val_space );
                }
            }
            // Clear accumulator to prepare for next row
            acc.clear();
        }
    }
}

template<typename Config>
typename Config::lidx_t
CSRMatrixSpGEMMDefault<Config>::DenseAccumulator::contains( typename Config::lidx_t col_idx ) const
{
    return flags[col_idx];
}

template<typename Config>
void CSRMatrixSpGEMMDefault<Config>::DenseAccumulator::set_flag( typename Config::lidx_t col_idx,
                                                                 typename Config::lidx_t k )
{
    const auto old_flag = flags[col_idx];
    flags[col_idx]      = k;
    if ( old_flag == -1 ) {
        if ( num_inserted == static_cast<lidx_t>( flag_inv.size() ) ) {
            flag_inv.push_back( col_idx );
        } else {
            flag_inv[num_inserted] = col_idx;
        }
        ++num_inserted;
    }
}

template<typename Config>
void CSRMatrixSpGEMMDefault<Config>::DenseAccumulator::insert_or_append(
    typename Config::lidx_t col_idx )
{
    const auto k = flags[col_idx];
    if ( k == -1 ) {
        flags[col_idx] = num_inserted;
        if ( num_inserted == static_cast<lidx_t>( flag_inv.size() ) ) {
            flag_inv.push_back( col_idx );
        } else {
            flag_inv[num_inserted] = col_idx;
        }
        ++num_inserted;
    }
}

template<typename Config>
void CSRMatrixSpGEMMDefault<Config>::DenseAccumulator::insert_or_append(
    typename Config::lidx_t col_idx,
    typename Config::scalar_t val,
    typename Config::lidx_t *col_space,
    typename Config::scalar_t *val_space )
{
    using lidx_t = typename Config::lidx_t;

    const auto k = flags[col_idx];
    if ( k == -1 ) {
        flags[col_idx] = num_inserted;
        if ( num_inserted == static_cast<lidx_t>( flag_inv.size() ) ) {
            flag_inv.push_back( col_idx );
        } else {
            flag_inv[num_inserted] = col_idx;
        }
        col_space[num_inserted] = col_idx;
        val_space[num_inserted] = val;
        ++num_inserted;
    } else {
        val_space[k] += val;
    }
}

template<typename Config>
void CSRMatrixSpGEMMDefault<Config>::DenseAccumulator::clear()
{
    for ( lidx_t n = 0; n < num_inserted; ++n ) {
        flags[flag_inv[n]] = -1;
    }
    num_inserted = 0;
}

template<typename Config>
uint16_t
CSRMatrixSpGEMMDefault<Config>::SparseAccumulator::hash( typename Config::lidx_t col_idx ) const
{
    const uint16_t c0 = ( 506999 * col_idx ) & 0xFFFF;
    const uint16_t c1 = ( col_idx >> 16 ) & 0xFFFF;
    return ( c0 ^ c1 ) % capacity;
}

template<typename Config>
typename Config::lidx_t
CSRMatrixSpGEMMDefault<Config>::SparseAccumulator::contains( typename Config::lidx_t col_idx ) const
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
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator::set_flag( typename Config::lidx_t col_idx,
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
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator::insert_or_append(
    typename Config::lidx_t col_idx )
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
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator::insert_or_append(
    typename Config::lidx_t col_idx,
    typename Config::scalar_t val,
    typename Config::lidx_t *col_space,
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
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator::grow( typename Config::lidx_t *col_space )
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
void CSRMatrixSpGEMMDefault<Config>::SparseAccumulator::clear()
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
