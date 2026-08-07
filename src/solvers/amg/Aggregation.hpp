#ifndef included_AMP_AMG_Aggregation_hpp
#define included_AMP_AMG_Aggregation_hpp

#include <algorithm>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Aggregation.h"
#include "AMP/solvers/amg/Strength.hpp"
#include "AMP/vectors/VectorBuilder.h"

namespace AMP::Solver::AMG {

template<class Allocator>
void insist_cpu_only_aggregation( const char *routine )
{
    const auto mem_loc = AMP::Utilities::getAllocatorMemoryType<Allocator>();
    AMP_INSIST( mem_loc < AMP::Utilities::MemoryType::device,
                std::string{ routine } + " does not support device memory" );
}

template<class View>
void insist_cpu_only_aggregation_view( const char *routine )
{
    insist_cpu_only_aggregation<typename View::allocator_type>( routine );
}


template<class View>
struct aggregate_type {
    using value_type     = typename View::lidx_t;
    using alloc_t        = typename View::allocator_type;
    using allocator_type = rebind_alloc<alloc_t, value_type>;
    template<class T>
    using vector_type = std::vector<T, rebind_alloc<alloc_t, T>>;

    aggregate_type()
    {
        insist_cpu_only_aggregation<alloc_t>( "AMG::aggregate_type" );
        rowptr.push_back( 0 );
    }
    vector_type<value_type> colind;
    vector_type<typename decltype( colind )::size_type> rowptr;

    template<class It>
    constexpr void add_row( It first, It last )
    {
        for ( ; first != last; ++first )
            colind.push_back( *first );
        rowptr.push_back( colind.size() );
    }
    constexpr void add_row( std::initializer_list<value_type> l ) { add_row( l.begin(), l.end() ); }

    constexpr auto operator[]( value_type i )
    {
        return span<value_type>{ colind.data() + rowptr[i], rowptr[i + 1] - rowptr[i] };
    }
    constexpr auto operator[]( value_type i ) const
    {
        return span<const value_type>{ colind.data() + rowptr[i], rowptr[i + 1] - rowptr[i] };
    }

    constexpr std::size_t size() const { return rowptr.size() - 1; }
};

template<class Config>
using aggregateT_type =
    std::vector<typename Config::lidx_t,
                rebind_alloc<typename Config::allocator_type, typename Config::lidx_t>>;

template<class Mat>
struct unmarked_list {
    unmarked_list( csr_view<Mat> mat, bool checkdd, float threshold = 5. )
        : A( mat ), checkdd( checkdd ), dd_threshold( threshold )
    {
    }

    using lidx_t   = typename csr_view<Mat>::lidx_t;
    using scalar_t = typename csr_view<Mat>::scalar_t;
    constexpr bool operator()( lidx_t i ) const
    {
        if ( !checkdd )
            return true;

        auto get_row = [=]( auto csr_ptrs, bool has_data = true ) {
            auto [rowptr, colind, values] = csr_ptrs;
            if ( !has_data )
                return decltype( values ){};
            return values.subspan( rowptr[i], rowptr[i + 1] - rowptr[i] );
        };
        auto diag_row        = get_row( A.diag() );
        auto offd_row        = get_row( A.offd(), A.has_offd() );
        auto diag_val        = diag_row.front();
        auto local_offd_vals = diag_row.subspan( 1 );

        auto acc = []( auto a, auto b ) { return a + std::abs( b ); };
        auto offd_sum =
            std::accumulate( local_offd_vals.begin(), local_offd_vals.end(), scalar_t{}, acc ) +
            std::accumulate( offd_row.begin(), offd_row.end(), scalar_t{}, acc );

        return !( diag_val > dd_threshold * offd_sum );
    }

private:
    csr_view<Mat> A;
    bool checkdd;
    float dd_threshold;
};
template<class Mat>
unmarked_list( csr_view<Mat>, float ) -> unmarked_list<Mat>;

template<class Mat>
struct prospect {
    using strength_type = Strength<Mat>;
    using lidx_t        = typename strength_type::lidx_t;
    prospect( const Strength<Mat> &soc, const unmarked_list<Mat> &initial_unmarked )
        : node_prio( soc.numLocalRows() ), chosen( soc.numLocalRows(), false )
    {
        PROFILE( "AMG::prospect" );
        std::vector<lidx_t> priority( soc.numLocalRows(), 0 );
        for ( size_t i = 0; i < priority.size(); ++i ) {
            soc.do_strong( i, [&]( lidx_t col ) {
                if ( col != static_cast<lidx_t>( i ) && initial_unmarked( col ) )
                    ++priority[col];
            } );
        }

        for ( size_t i = 0; i < priority.size(); ++i ) {
            if ( initial_unmarked( i ) )
                node_prio[i] = prio_node.insert( { priority[i], i } );
            else
                chosen[i] = true;
        }
    }

    lidx_t pop()
    {
        auto selected   = prio_node.cbegin();
        const auto node = selected->second;
        chosen[node]    = true;
        prio_node.erase( selected );
        return node;
    }

    void remove( lidx_t e )
    {
        chosen[e] = true;
        prio_node.erase( node_prio[e] );
    }

    void decrement_priority( lidx_t e )
    {
        if ( !chosen[e] ) {
            auto old     = node_prio[e];
            node_prio[e] = prio_node.insert( { old->first - 1, e } );
            prio_node.erase( old );
        }
    }

    bool empty() const { return prio_node.empty(); }

    using mm_type = std::multimap<lidx_t, lidx_t>;
    mm_type prio_node;
    std::vector<typename mm_type::iterator> node_prio;
    std::vector<bool> chosen;
};
template<class M>
prospect( const Strength<M> & ) -> prospect<M>;

template<class T, class Mat>
auto find_pair( T r, const prospect<Mat> &unmarked, const Strength<Mat> &S )
{
    using scalar_t = typename csr_view<Mat>::scalar_t;
    using lidx_t   = typename csr_view<Mat>::lidx_t;

    struct {
        std::optional<lidx_t> colid;
        scalar_t value = std::numeric_limits<scalar_t>::max();
    } cur;

    S.do_strong_val( r, [&]( lidx_t col, scalar_t val ) {
        if ( !unmarked.chosen[col] && val < cur.value ) {
            cur.value = val;
            cur.colid = col;
        }
    } );

    return cur.colid;
}

template<class Mat>
aggregate_type<csr_view<Mat>> pairwise_aggregate( csr_view<Mat> A,
                                                  const PairwiseCoarsenSettings &settings )
{
    PROFILE( "AMG::pairwise_aggregate" );
    insist_cpu_only_aggregation_view<csr_view<Mat>>( "AMG::pairwise_aggregate" );
    aggregate_type<csr_view<Mat>> aggregates;
    aggregates.rowptr.reserve( A.numLocalRows() / 2 + 1 );
    aggregates.colind.reserve( A.numLocalRows() );
    using lidx_t = typename csr_view<Mat>::lidx_t;

    auto S = compute_soc<classical_strength<norm::min>>( A, nullptr, settings.strength_threshold );

    prospect unmarked( S, unmarked_list( A, settings.checkdd ) );

    auto update_priorities = [&]( lidx_t k ) {
        S.do_strong( k, [&]( lidx_t col ) { unmarked.decrement_priority( col ); } );
    };
    while ( !unmarked.empty() ) {
        auto selected   = unmarked.pop();
        auto maybe_pair = find_pair( selected, unmarked, S );
        if ( maybe_pair.has_value() ) {
            auto pair = maybe_pair.value();
            unmarked.remove( pair );
            update_priorities( pair );
            aggregates.add_row( { selected, pair } );
        } else {
            aggregates.add_row( { selected } );
        }
        update_priorities( selected );
    }

    return aggregates;
}

template<class T1, class T2>
auto agg_union( const T1 &agg1, const T2 &agg2 )
{
    T1 agg;
    agg.rowptr.reserve( agg2.size() + 1 );
    agg.colind.reserve( agg1.colind.size() );

    for ( std::size_t i = 0; i < agg2.size(); ++i ) {
        for ( auto e2 : agg2[i] ) {
            for ( auto e1 : agg1[e2] ) {
                agg.colind.push_back( e1 );
            }
        }
        agg.rowptr.push_back( agg.colind.size() );
    }

    return agg;
}


template<class T, class I>
auto transpose_aggregates( const T &agg, I fine_size )
{
    using lidx_t  = typename T::value_type;
    using alloc_t = typename T::allocator_type;
    std::vector<lidx_t, alloc_t> aggt( fine_size, AggregationFlags::ineligible );
    for ( std::size_t i = 0; i < agg.size(); ++i ) {
        for ( auto fi : agg[i] ) {
            aggt[fi] = i;
        }
    }

    return aggt;
}

template<class T, class A>
std::vector<size_t> argsort( const std::vector<T, A> &array )
{
    std::vector<size_t> indices( array.size() );
    std::iota( indices.begin(), indices.end(), 0 );
    std::sort( indices.begin(), indices.end(), [&array]( size_t left, size_t right ) -> bool {
        // sort indices according to corresponding array element
        return array[left] < array[right];
    } );

    return indices;
}

template<class Mat>
auto create_aux( csr_view<Mat> A, const aggregate_type<csr_view<Mat>> &agg )
{
    PROFILE( "AMG::create_aux" );
    using csr_policy = typename csr_view<Mat>::csr_policy;
    par_csr<csr_policy> aux;
    using lidx_t   = typename csr_view<Mat>::lidx_t;
    using scalar_t = typename csr_view<Mat>::scalar_t;

    auto aggt = transpose_aggregates( agg, A.numLocalRows() );

    auto collapse = [&]( auto src, auto &dst, auto &&cmap ) {
        auto [rowptr, colind, values] = src;
        dst.rowptr.resize( agg.size() + 1 );
        [&]( auto &...v ) { ( v.reserve( agg.size() ), ... ); }( dst.colind, dst.values );
        for ( size_t rc = 0; rc < agg.size(); ++rc ) {
            std::vector<lidx_t> agg_indices;
            std::vector<scalar_t> agg_values;
            [&]( auto &...v ) { ( v.reserve( agg[rc].size() ), ... ); }( agg_indices, agg_values );
            for ( auto r : agg[rc] ) {
                for ( auto off = rowptr[r]; off < rowptr[r + 1]; ++off ) {
                    auto ind = cmap( colind[off] );
                    if ( ind != AggregationFlags::ineligible ) { // check if col is in an aggregate
                        agg_indices.push_back( ind );
                        agg_values.push_back( values[off] );
                    }
                }
            }

            auto ind = argsort( agg_indices );
            for ( std::size_t i = 0; i < ind.size(); ++i ) {
                auto cur_val = agg_values[ind[i]];
                auto cur_ind = agg_indices[ind[i]];
                while ( i < ind.size() - 1 && agg_indices[ind[i + 1]] == cur_ind )
                    cur_val += agg_values[ind[++i]];
                dst.colind.push_back( cur_ind );
                dst.values.push_back( cur_val );
            }
            dst.rowptr[rc + 1] = dst.colind.size();
        }
    };
    collapse( A.diag(), aux.diag(), [&]( lidx_t col ) { return aggt[col]; } );
    if ( A.has_offd() ) {
        collapse( A.offd(),
                  aux.offd(),
                  // avoid communication by treating offd as distinct aggregates
                  []( lidx_t col ) { return col; } );
    }

    return csr_view( aux );
}


template<class Config>
auto coarsen_matrix( const LinearAlgebra::CSRMatrix<Config> &fine_matrix,
                     const aggregate_type<csr_view<LinearAlgebra::CSRMatrix<Config>>> &aggregates,
                     const aggregateT_type<Config> &aggregatesT )
{
    PROFILE( "AMG::coarsen_matrix" );
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    csr_view fine( fine_matrix );
    const auto &fine_data = fine.data();

    auto &comm = fine_data.getRightCommList()->getComm();

    coarse_matrix<Config> coarse_mat;
    coarse_mat.comm      = comm;
    coarse_mat.left_var  = fine_matrix.getMatrixData()->getRightVariable();
    coarse_mat.right_var = fine_matrix.getMatrixData()->getLeftVariable();
    auto size_rowptr     = [&]( auto &m ) { m.rowptr.resize( aggregates.size() + 1 ); };
    size_rowptr( coarse_mat.store.diag() );
    if ( fine.has_offd() )
        size_rowptr( coarse_mat.store.offd() );

    // using gidx_t = typename Config::gidx_t;
    using gidx_t = double;

    // maps local column ids from fine matrix to global column ids in coarse matrix
    struct {
        std::vector<gidx_t> diag, offd;
    } aggt;
    auto aggtvec =
        [&]() { // make vector for aggt storage and communication in (global) coarse indexing
            auto vec =
                LinearAlgebra::createSimpleVector<gidx_t,
                                                  LinearAlgebra::VectorOperationsDefault<gidx_t>,
                                                  LinearAlgebra::VectorDataDefault<gidx_t>>(
                    std::make_shared<LinearAlgebra::Variable>( "aggregates" ),
                    fine_data.getRightDOFManager(),
                    fine_data.getRightCommList() );
            aggt.diag.resize( vec->getLocalSize() );
            using ext_t = typename coarse_matrix<Config>::gidx_t;
            auto local_offset =
                static_cast<ext_t>( comm.sumScan( aggregates.size() ) - aggregates.size() );
            coarse_mat.diag_extents = { local_offset,
                                        local_offset + static_cast<ext_t>( aggregates.size() ) };
            const auto ineligible   = static_cast<lidx_t>( AggregationFlags::ineligible );
            for ( std::size_t i = 0; i < aggt.diag.size(); ++i ) {
                const auto aggregate = aggregatesT[i];
                aggt.diag[i]         = ( aggregate == ineligible ) ?
                                           static_cast<gidx_t>( ineligible ) :
                                           static_cast<gidx_t>( aggregate + local_offset );
            }


            vec->putRawData( aggt.diag.data() );
            vec->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );

            auto &offd_mat = *( fine_data.getOffdMatrix() );
            auto nghosts   = offd_mat.numUniqueColumns();
            aggt.offd.resize( nghosts );
            if constexpr ( std::is_same_v<size_t, typename Config::gidx_t> ) {
                size_t *colmap = offd_mat.getColumnMap();
                vec->getGhostValuesByGlobalID( nghosts, colmap, aggt.offd.data() );
            } else {
                std::vector<size_t> colmap;
                offd_mat.getColumnMap( colmap );
                vec->getGhostValuesByGlobalID( nghosts, colmap.data(), aggt.offd.data() );
            }

            return vec;
        }();


    auto collapse = [&]( auto &cmat,
                         auto fine_ptrs,
                         const std::vector<gidx_t> &aggregates_transpose ) {
        auto [rowptr, colind, values] = fine_ptrs;
        for ( size_t rc = 0; rc < aggregates.size(); ++rc ) {
            // coarse (global) column index -> aggregated value (nnz in this coarse row)
            std::vector<lidx_t> agg_indices;
            std::vector<scalar_t> agg_values;
            [&]( auto &...v ) { ( v.reserve( aggregates[rc].size() ), ... ); }( agg_indices,
                                                                                agg_values );
            for ( auto r : aggregates[rc] ) {
                for ( auto off = rowptr[r]; off < rowptr[r + 1]; ++off ) {
                    auto ind = aggregates_transpose[colind[off]];
                    if ( ind != static_cast<gidx_t>(
                                    AggregationFlags::ineligible ) ) { // check if in an aggregate
                        agg_indices.push_back( ind );
                        agg_values.push_back( values[off] );
                    }
                }
            }

            auto ind = argsort( agg_indices );
            for ( std::size_t i = 0; i < ind.size(); ++i ) {
                auto cur_val = agg_values[ind[i]];
                auto cur_ind = agg_indices[ind[i]];
                while ( i < ind.size() - 1 && agg_indices[ind[i + 1]] == cur_ind )
                    cur_val += agg_values[ind[++i]];
                cmat.colind.push_back( cur_ind );
                cmat.values.push_back( cur_val );
            }
            cmat.rowptr[rc + 1] = cmat.colind.size();
        }
    };

    collapse( coarse_mat.store.diag(), fine.diag(), aggt.diag );
    if ( fine.has_offd() ) {
        collapse( coarse_mat.store.offd(), fine.offd(), aggt.offd );
    }

    return coarse_mat;
}


template<class Config>
auto make_coarse_operator( const coarse_matrix<Config> &mat )
{
    auto params    = std::make_shared<coarse_operator_parameters<Config>>();
    params->d_db   = std::make_shared<AMP::Database>();
    params->matrix = mat;
    return std::make_shared<coarse_operator<Config>>( params );
}

template<class Config>
struct UAIntergridParams : AMP::Operator::OperatorParameters {
    enum class intergrid_type { interpolation, restriction };
    UAIntergridParams() : AMP::Operator::OperatorParameters( nullptr ) {}
    std::shared_ptr<const aggregateT_type<Config>> aggregatesT;
    std::shared_ptr<const LinearAlgebra::Vector> input_vector;
    std::shared_ptr<const LinearAlgebra::Vector> output_vector;
    intergrid_type transfer_type = intergrid_type::interpolation;
};

template<class Config>
struct AggregateInjection : AMP::Operator::Operator {
    using lidx_t         = typename Config::lidx_t;
    using scalar_t       = typename Config::scalar_t;
    using aggT_type      = aggregateT_type<Config>;
    using intergrid_type = typename UAIntergridParams<Config>::intergrid_type;

    explicit AggregateInjection( std::shared_ptr<const AMP::Operator::OperatorParameters> iparams )
        : AMP::Operator::Operator( iparams )
    {
        auto params = std::dynamic_pointer_cast<const UAIntergridParams<Config>>( iparams );
        AMP_DEBUG_ASSERT( params );
        d_aggregatesT = params->aggregatesT;
        d_input_vec   = params->input_vector;
        d_output_vec  = params->output_vector;
        transfer_type = params->transfer_type;
    }

    [[nodiscard]] std::string type() const override { return "AggregateInjection"; }

    auto T() const
    {
        auto params           = std::make_shared<UAIntergridParams<Config>>();
        params->d_db          = std::make_shared<AMP::Database>();
        params->aggregatesT   = d_aggregatesT;
        params->input_vector  = d_output_vec;
        params->output_vector = d_input_vec;
        params->transfer_type = ( transfer_type == intergrid_type::interpolation ) ?
                                    intergrid_type::restriction :
                                    intergrid_type::interpolation;
        return std::make_shared<AggregateInjection<Config>>( params );
    }

    std::shared_ptr<LinearAlgebra::Vector> createInputVector() const override
    {
        return d_input_vec ? d_input_vec->clone() : nullptr;
    }

    std::shared_ptr<LinearAlgebra::Vector> createOutputVector() const override
    {
        return d_output_vec ? d_output_vec->clone() : nullptr;
    }

    std::shared_ptr<LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_input_vec ? d_input_vec->getVariable() : nullptr;
    }

    std::shared_ptr<LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_output_vec ? d_output_vec->getVariable() : nullptr;
    }

    void apply( std::shared_ptr<const LinearAlgebra::Vector> xvec,
                std::shared_ptr<LinearAlgebra::Vector> yvec ) override
    {
        const auto &aggt = aggregateT();
        auto x           = xvec->getVectorData()->getRawDataBlock<scalar_t>( 0 );
        auto y           = yvec->getVectorData()->getRawDataBlock<scalar_t>( 0 );

        switch ( transfer_type ) {
        case intergrid_type::interpolation: {
            AMP_DEBUG_ASSERT( yvec->getLocalSize() == aggt.size() );

            for ( size_t i = 0; i < yvec->getLocalSize(); ++i ) {
                auto col = aggt[i];
                if ( col != AggregationFlags::ineligible )
                    y[i] = x[col];
            }
            break;
        }
        case intergrid_type::restriction: {
            AMP_DEBUG_ASSERT( xvec->getLocalSize() == aggt.size() );

            yvec->setToScalar( 0 );
            for ( size_t i = 0; i < xvec->getLocalSize(); ++i ) {
                auto col = aggt[i];
                if ( col != AggregationFlags::ineligible )
                    y[col] += x[i];
            }
        }
        };
        yvec->setUpdateStatus( LinearAlgebra::UpdateState::LOCAL_CHANGED );
    }

    const aggT_type &aggregateT() const { return *d_aggregatesT; }

protected:
    std::shared_ptr<const aggT_type> d_aggregatesT;
    std::shared_ptr<const LinearAlgebra::Vector> d_input_vec;
    std::shared_ptr<const LinearAlgebra::Vector> d_output_vec;
    intergrid_type transfer_type;
};

template<class Config>
auto make_ua_intergrid( aggregateT_type<Config> &&aggT,
                        std::shared_ptr<const LinearAlgebra::Vector> input_vec,
                        std::shared_ptr<const LinearAlgebra::Vector> output_vec )
{
    auto aggT_ptr = std::make_shared<const aggregateT_type<Config>>( std::move( aggT ) );

    auto params           = std::make_shared<UAIntergridParams<Config>>();
    params->d_db          = std::make_shared<AMP::Database>();
    params->aggregatesT   = aggT_ptr;
    params->input_vector  = std::move( input_vec );
    params->output_vector = std::move( output_vec );
    return std::make_shared<AggregateInjection<Config>>( params );
}

template<class Fine>
auto pairwise_aggregation( csr_view<Fine> A, const PairwiseCoarsenSettings &settings )
{
    PROFILE( "AMG::pairwise_aggregation" );
    insist_cpu_only_aggregation_view<csr_view<Fine>>( "AMG::pairwise_aggregation" );
    PairwiseCoarsenSettings settings_later_passes = settings;
    settings_later_passes.checkdd                 = false;
    if ( settings.pairwise_passes == 2 ) {
        auto agg1 = pairwise_aggregate( A, settings );
        auto aux  = create_aux( A, agg1 );
        auto agg2 = pairwise_aggregate( aux, settings_later_passes );
        return agg_union( agg1, agg2 );
    } else if ( settings.pairwise_passes == 3 ) {
        auto agg1 = pairwise_aggregate( A, settings );
        auto aux  = create_aux( A, agg1 );
        auto agg2 = pairwise_aggregate( aux, settings_later_passes );
        auto aux1 = create_aux( aux, agg2 );
        auto agg3 = pairwise_aggregate( aux1, settings_later_passes );

        return agg_union( agg_union( agg1, agg2 ), agg3 );
    } else {
        AMP_ERROR( "Pairwise Aggregation: Invalid number of pairwise passes" );
    }
}


template<class Config>
coarse_ops_type pairwise_coarsen( const LinearAlgebra::CSRMatrix<Config> &fine,
                                  const PairwiseCoarsenSettings &settings )
{
    insist_cpu_only_aggregation<typename Config::allocator_type>( "AMG::pairwise_coarsen" );
    AMP_INSIST( settings.pairwise_passes == 2 || settings.pairwise_passes == 3,
                "Pairwise Aggregation: invalid number of passes" );

    auto aggregates  = pairwise_aggregation( csr_view( fine ), settings );
    auto aggregatesT = transpose_aggregates( aggregates, fine.numLocalRows() );
    auto matrix      = coarsen_matrix( fine, aggregates, aggregatesT );
    auto Ac          = make_coarse_operator( matrix );
    auto P           = make_ua_intergrid<Config>(
        std::move( aggregatesT ), Ac->createInputVector(), fine.createInputVector() );

    return { P->T(), Ac, P };
}

template<class Config>
coarse_ops_type aggregator_coarsen( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> fine,
                                    Aggregator &aggregator )
{
    // avoid extra work if aggregator is pairwise.
    auto pairwise_aggregator = dynamic_cast<PairwiseAggregator *>( &aggregator );
    if ( pairwise_aggregator ) {
        return pairwise_coarsen( *fine, pairwise_aggregator->settings() );
    }

    aggregateT_type<Config> aggregatesT( fine->numLocalRows(), 0 );
    auto num_agg    = aggregator.assignLocalAggregates( fine, aggregatesT.data() );
    auto aggregates = []( auto &aggt, const auto num_agg ) {
        aggregate_type<csr_view<LinearAlgebra::CSRMatrix<Config>>> agg;
        agg.rowptr.reserve( num_agg + 1 );
        agg.colind.reserve( aggt.size() );
        auto inds = argsort( aggt );
        auto curr = aggt[inds[0]];
        for ( std::size_t i = 0; i < aggt.size(); ++i ) {
            auto row = inds[i];
            auto col = aggt[row];
            if ( col != curr ) {
                agg.rowptr.push_back( agg.colind.size() );
                curr = col;
            }
            agg.colind.push_back( row );
        }
        if ( static_cast<int>( agg.rowptr.size() ) < num_agg + 1 )
            agg.rowptr.push_back( agg.colind.size() );

        assert( static_cast<int>( agg.rowptr.size() ) == num_agg + 1 );

        return agg;
    }( aggregatesT, num_agg );

    auto matrix = coarsen_matrix( *fine, aggregates, aggregatesT );
    auto Ac     = make_coarse_operator( matrix );
    auto P      = make_ua_intergrid<Config>(
        std::move( aggregatesT ), Ac->createInputVector(), fine->createInputVector() );

    return { P->T(), Ac, P };
}


int PairwiseAggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A,
                                               int *agg_ids )
{
    return LinearAlgebra::csrVisit( A, [this, agg_ids]( auto csr_ptr ) {
        return this->assignLocalAggregates( csr_ptr, agg_ids );
    } );
}

template<class Config>
int PairwiseAggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                               int *agg_ids )
{
    insist_cpu_only_aggregation<typename Config::allocator_type>(
        "AMG::PairwiseAggregator::assignLocalAggregates" );
    auto aggregates  = pairwise_aggregation( csr_view( *A ), d_settings );
    auto aggregatesT = transpose_aggregates( aggregates, A->numLocalRows() );

    std::copy( aggregatesT.begin(), aggregatesT.end(), agg_ids );

    return aggregates.size();
}

} // namespace AMP::Solver::AMG
#endif
