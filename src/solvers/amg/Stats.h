#ifndef included_AMP_AMG_STATS
#define included_AMP_AMG_STATS

#include <array>
#include <fstream>
#include <iomanip>
#include <string>
#include <tuple>
#include <vector>

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Cycle.h"

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
std::size_t get_nprocs( const LinearAlgebra::CSRMatrix<Config> &A )
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


inline void print_summary( std::vector<KCycleLevel> &ml, SolverStrategy &cg_solver )
{
    std::vector<size_t> nrows, nnz, nprocs;
    std::vector<std::pair<size_t, size_t>> nrows_local;

    for ( auto &level : ml ) {
        LinearAlgebra::csrVisit( level.A->getMatrix(), [&]( auto A ) {
            nnz.push_back( get_nnz( *A ) );
            nrows.push_back( get_nrows( *A ) );
            nprocs.push_back( get_nprocs( *A ) );
            nrows_local.push_back( get_local_nrows( *A ) );
        } );
    }

#if defined( AMP_USE_HYPRE )
    auto hsolver = dynamic_cast<HypreSolver *>( &cg_solver );
    if ( hsolver ) {
        auto hypre_solver            = hsolver->getHYPRESolver();
        hypre_ParAMGData *amg_data   = (hypre_ParAMGData *) hypre_solver;
        auto num_levels              = hypre_ParAMGDataNumLevels( amg_data );
        hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray( amg_data );

        for ( int lvl = 1; lvl < num_levels; ++lvl ) {
            AMP_MPI comm( hypre_ParCSRMatrixComm( A_array[lvl] ) );
            nnz.push_back( hypre_ParCSRMatrixNumNonzeros( A_array[lvl] ) );
            nrows.push_back( hypre_ParCSRMatrixGlobalNumRows( A_array[lvl] ) );
            nprocs.push_back( comm.getSize() );
            std::size_t my_nrows_local = hypre_ParCSRMatrixNumRows( A_array[lvl] );
            nrows_local.push_back(
                { comm.maxReduce( my_nrows_local ), comm.minReduce( my_nrows_local ) } );
        }
    }
#endif

    auto total_nnz   = std::accumulate( nnz.begin(), nnz.end(), 0 );
    auto total_nrows = std::accumulate( nrows.begin(), nrows.end(), 0 );

    AMP::pout << "Number of levels: " << nrows.size() << '\n';
    AMP::pout << "Operator complexity: " << std::setprecision( 3 )
              << static_cast<float>( total_nnz ) / static_cast<float>( nnz[0] ) << '\n';
    AMP::pout << "Grid complexity: " << std::setprecision( 3 )
              << static_cast<float>( total_nrows ) / static_cast<float>( nrows[0] ) << '\n';

    column lvl_col{ "level", [nlvl = nrows.size()]() {
                       std::vector<int> levels( nlvl );
                       std::iota( levels.begin(), levels.end(), 0 );
                       return levels;
                   }() };

    column type_col{ "type",
                     [&, nlvl = nrows.size()]() {
                         std::vector<std::string> types;
                         for ( std::size_t i = 0; i < nlvl; ++i )
                             types.push_back( ( i < ml.size() - 1 ) ? "UA AMG" : "BoomerAMG" );
                         return types;
                     }(),
                     []( std::string val ) { return val; } };

    auto maxmin_repr = []( const std::pair<size_t, size_t> &mm ) {
        std::stringstream ss;
        ss << '(' << mm.first << ' ' << mm.second << ')';
        return ss.str();
    };
    write_columns( AMP::pout,
                   std::move( lvl_col ),
                   std::move( type_col ),
                   column{ "nprocs", nprocs },
                   column{ "nrows", nrows },
                   column{ "nonzeros",
                           nnz,
                           [total_nnz]( std::size_t level_nnz ) {
                               std::stringstream ss;
                               ss << level_nnz << " [" << std::setprecision( 2 )
                                  << static_cast<float>( level_nnz ) /
                                         static_cast<float>( total_nnz ) * 100
                                  << "%]";
                               return ss.str();
                           } },
                   column{ "nrows local", nrows_local, maxmin_repr } );
}


} // namespace AMP::Solver::AMG

#endif
