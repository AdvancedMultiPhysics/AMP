#ifndef included_AMP_AMG_STATS_hpp
#define included_AMP_AMG_STATS_hpp

#include "AMP/solvers/amg/Stats.h"

#include <array>
#include <iomanip>
#include <string>

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/matrices/data/CSRMatrixData.h"

#if defined( AMP_USE_HYPRE )
    #include "AMP/solvers/hypre/HypreSolver.h"
    #include <_hypre_parcsr_ls.h>
#endif

namespace AMP::Solver::AMG {

template<class T>
struct column {
    template<class F>
    column( std::string h, std::vector<T> d, F &&f )
        : header{ std::move( h ) }, data{ std::move( d ) }, repr( std::forward<F>( f ) )
    {
    }
    column( std::string h, std::vector<T> d )
        : column( std::move( h ), std::move( d ), []( T val ) { return std::to_string( val ); } )
    {
    }
    std::string header;
    std::vector<T> data;
    std::function<std::string( T )> repr;
};
template<class T, class F>
column( std::string, std::vector<T>, F && ) -> column<T>;
template<class T>
column( std::string, std::vector<T> ) -> column<T>;

template<class... T>
void write_columns( std::ostream &os, column<T> &&...cols )
{
    std::array<std::size_t, sizeof...( cols )> clen{ std::max(
        cols.header.length(),
        cols.repr( *std::max_element( cols.data.begin(),
                                      cols.data.end(),
                                      [&]( auto a, auto b ) {
                                          return cols.repr( a ).length() < cols.repr( b ).length();
                                      } ) )
            .length() )... };

    auto write_space = [&]( std::size_t len ) {
        for ( std::size_t i = 0; i < len; ++i )
            os << ' ';
    };

    { // write headers
        auto len_it = clen.begin();
        (
            [&]( const std::string &hdr ) {
                auto len = *len_it++;
                auto num = len - hdr.length();
                write_space( num / 2 );
                os << hdr;
                write_space( num / 2 );
                write_space( num % 2 );
                write_space( 1 );
            }( cols.header ),
            ... );
    }

    os << '\n';

    // write rows
    for ( std::size_t i = 0; i < std::min( { cols.data.size()... } ); ++i ) {
        auto len_it = clen.begin();
        (
            [&]( std::string &&rep ) {
                write_space( *len_it++ - rep.length() );
                os << rep;
                write_space( 1 );
            }( cols.repr( cols.data[i] ) ),
            ... );
        os << '\n';
    }
}

template<class Config>
std::size_t get_nnz( const LinearAlgebra::CSRMatrix<Config> &A )
{
    using csr_data_t = LinearAlgebra::CSRMatrixData<Config>;
    const auto &comm = A.getComm();
    const auto &data = *( std::dynamic_pointer_cast<const csr_data_t>( A.getMatrixData() ) );
    auto nnz         = data.numberOfNonZerosDiag() + data.numberOfNonZerosOffDiag();
    return comm.sumReduce( nnz );
}


template<class Config>
std::size_t get_nrows( const LinearAlgebra::CSRMatrix<Config> &A )
{
    return A.numGlobalRows();
}


template<class Config>
int get_nprocs( const LinearAlgebra::CSRMatrix<Config> &A )
{
    return A.getComm().getSize();
}


template<class Config>
std::pair<size_t, size_t> get_local_nrows( const LinearAlgebra::CSRMatrix<Config> &A )
{
    const auto &comm = A.getComm();
    auto nrows       = A.numLocalRows();
    return { comm.maxReduce( nrows ), comm.minReduce( nrows ) };
}


template<class L>
HierarchyStats collect_statistics( const std::string &amg_name,
                                   const std::vector<L> &ml,
                                   const SolverStrategy &cg_solver )
{
    HierarchyStats stats;

    for ( auto &level : ml ) {
        LinearAlgebra::csrVisit( level.A->getMatrix(), [&]( auto A ) {
            auto [max_local, min_local] = get_local_nrows( *A );
            stats.levels.push_back( { amg_name,
                                      get_nprocs( *A ),
                                      get_nrows( *A ),
                                      get_nnz( *A ),
                                      max_local,
                                      min_local } );
        } );
    }

    stats.levels.back().solver_name = cg_solver.type();

#if defined( AMP_USE_HYPRE )
    const auto *hsolver = dynamic_cast<const HypreSolver *>( &cg_solver );
    if ( hsolver ) {
        auto *hypre_solver           = const_cast<HypreSolver *>( hsolver )->getHYPRESolver();
        hypre_ParAMGData *amg_data   = (hypre_ParAMGData *) hypre_solver;
        auto num_levels              = hypre_ParAMGDataNumLevels( amg_data );
        hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray( amg_data );

        for ( int lvl = 1; lvl < num_levels; ++lvl ) {
            AMP_MPI comm( hypre_ParCSRMatrixComm( A_array[lvl] ) );
            std::size_t nrows_local = hypre_ParCSRMatrixNumRows( A_array[lvl] );
            stats.levels.push_back(
                { cg_solver.type(),
                  comm.getSize(),
                  static_cast<std::size_t>( hypre_ParCSRMatrixGlobalNumRows( A_array[lvl] ) ),
                  static_cast<std::size_t>( hypre_ParCSRMatrixNumNonzeros( A_array[lvl] ) ),
                  comm.maxReduce( nrows_local ),
                  comm.minReduce( nrows_local ) } );
        }
    }
#endif

    using stats_level = HierarchyStats::level_type;
    auto total_nrows  = std::accumulate(
        stats.levels.begin(), stats.levels.end(), 0.f, []( float cur, const stats_level &lvl ) {
            return cur + lvl.nrows;
        } );

    stats.operator_complexity = std::accumulate(
        stats.levels.begin(),
        stats.levels.end(),
        0.f,
        [fine_nnz = static_cast<float>( stats.levels[0].nnz )](
            float cur, const stats_level &lvl ) { return cur + lvl.nnz / fine_nnz; } );

    stats.grid_complexity = total_nrows / stats.levels[0].nrows;

    return stats;
}

template<class L>
void print_summary( const std::string &amg_name,
                    const std::vector<L> &ml,
                    const SolverStrategy &cg_solver )
{
    auto stats = collect_statistics( amg_name, ml, cg_solver );

    AMP::pout << "Number of levels: " << stats.levels.size() << '\n';
    if ( ml.size() == 0 )
        return;
    AMP::pout << "Operator complexity: " << std::setprecision( 3 ) << stats.operator_complexity
              << '\n';
    AMP::pout << "Grid complexity: " << std::setprecision( 3 ) << stats.grid_complexity << '\n';

    auto make_vec = [&]( auto &&f ) {
        std::vector<std::decay_t<decltype( f( stats.levels[0] ) )>> vec;
        for ( const auto &lvl : stats.levels ) {
            vec.push_back( f( lvl ) );
        }
        return vec;
    };
    column lvl_col{ "level", [nlvl = stats.levels.size()]() {
                       std::vector<int> levels( nlvl );
                       std::iota( levels.begin(), levels.end(), 0 );
                       return levels;
                   }() };
    using stats_level = HierarchyStats::level_type;
    column type_col{ "type",
                     make_vec( []( const stats_level &lvl ) { return lvl.solver_name; } ),
                     []( const std::string &val ) { return val; } };

    auto maxmin_repr = []( const std::pair<size_t, size_t> &mm ) {
        std::stringstream ss;
        ss << '(' << mm.first << ' ' << mm.second << ')';
        return ss.str();
    };
    write_columns(
        AMP::pout,
        std::move( lvl_col ),
        std::move( type_col ),
        column{ "nprocs", make_vec( []( const stats_level &lvl ) { return lvl.comm_size; } ) },
        column{ "nrows", make_vec( []( const stats_level &lvl ) { return lvl.nrows; } ) },
        column{ "nonzeros",
                make_vec( []( const stats_level &lvl ) { return lvl.nnz; } ),
                [&]( std::size_t level_nnz ) {
                    std::stringstream ss;
                    ss << level_nnz << " [" << std::setprecision( 2 ) << [&]() {
                        // normalize to avoid overflow
                        auto nrm        = static_cast<float>( stats.levels[0].nnz );
                        float total_nrm = 0;
                        for ( const auto &lvl : stats.levels )
                            total_nrm += lvl.nnz / nrm;
                        auto level_nrm = level_nnz / nrm;
                        return level_nrm / total_nrm * 100;
                    }() << "%]";
                    return ss.str();
                } },
        column{ "nrows local",
                make_vec( []( const stats_level &lvl ) -> std::pair<std::size_t, std::size_t> {
                    return { lvl.max_local_rows, lvl.min_local_rows };
                } ),
                maxmin_repr } );
}

} // namespace AMP::Solver::AMG

#endif
