#ifndef included_AMP_AMG_Relaxation_hpp
#define included_AMP_AMG_Relaxation_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Relaxation.h"
#include "AMP/utils/Constants.h"
#include "AMP/utils/memory.h"

#include <cmath>

namespace AMP::Solver::AMG {

Relaxation::Relaxation( std::shared_ptr<const SolverStrategyParameters> params )
    : SolverStrategy( params )
{
    getFromInput( params->d_db );
}

void Relaxation::getFromInput( std::shared_ptr<AMP::Database> db )
{
    d_num_sweeps = db->getWithDefault<size_t>( "num_sweeps", 1 );

    auto sweep_type = db->getWithDefault<std::string>( "sweep_type", "symmetric" );
    if ( sweep_type == "forward" )
        d_sweep = Relaxation::Sweep::forward;
    else if ( sweep_type == "backward" )
        d_sweep = Relaxation::Sweep::backward;
    else if ( sweep_type == "symmetric" )
        d_sweep = Relaxation::Sweep::symmetric;
    else {
        AMP_ERROR( "Relaxation: invalid sweep type (" + sweep_type + ")" );
    }
}

HybridGS::HybridGS( std::shared_ptr<const SolverStrategyParameters> iparams )
    : Relaxation( iparams ), d_ghost_vals( nullptr ), d_num_ghosts( 0 )
{
}

HybridGS::~HybridGS() { deallocateGhosts(); }

void HybridGS::deallocateGhosts()
{
    if ( d_ghost_vals != nullptr ) {
        auto mem_loc = AMP::Utilities::getMemoryType( d_ghost_vals );
        if ( mem_loc == AMP::Utilities::MemoryType::host ) {
            AMP::HostAllocator<std::byte> byteAlloc;
            byteAlloc.deallocate( d_ghost_vals, d_num_ghost_bytes );
        } else if ( mem_loc == AMP::Utilities::MemoryType::managed ) {
#ifdef AMP_USE_DEVICE
            AMP::ManagedAllocator<std::byte> byteAlloc;
            byteAlloc.deallocate( d_ghost_vals, d_num_ghost_bytes );
#else
            AMP_ERROR( "Non-host pointer on host only build" );
#endif
        } else if ( mem_loc == AMP::Utilities::MemoryType::device ) {
#ifdef AMP_USE_DEVICE
            AMP::DeviceAllocator<std::byte> byteAlloc;
            byteAlloc.deallocate( d_ghost_vals, d_num_ghost_bytes );
#else
            AMP_ERROR( "Non-host pointer on host only build" );
#endif
        } else {
            AMP_ERROR( "Unrecognized memory type" );
        }
        d_ghost_vals      = nullptr;
        d_num_ghosts      = 0;
        d_num_ghost_bytes = 0;
    }
}

void HybridGS::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    deallocateGhosts();
    d_pOperator = op;
    auto lin_op = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );
    AMP_DEBUG_INSIST( lin_op, "HybridGS: operator must be linear" );
    auto mat = lin_op->getMatrix();
    AMP_DEBUG_INSIST( mat, "HybridGS: matrix cannot be NULL" );
    d_matrix = mat;
}

void HybridGS::apply( std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x )
{
    LinearAlgebra::csrVisit( d_matrix, [=]( auto csr_ptr ) { relax( *csr_ptr, *b, *x ); } );
}

template<typename Config>
void HybridGS::relax( LinearAlgebra::CSRMatrix<Config> &A,
                      const LinearAlgebra::Vector &b,
                      LinearAlgebra::Vector &x )
{
    for ( size_t i = 0; i < d_num_sweeps; ++i ) {
        switch ( d_sweep ) {
        case Sweep::forward:
            sweep<Config>( Direction::forward, A, b, x );
            break;
        case Sweep::backward:
            sweep<Config>( Direction::backward, A, b, x );
            break;
        case Sweep::symmetric:
            sweep<Config>( Direction::forward, A, b, x );
            sweep<Config>( Direction::backward, A, b, x );
            break;
        }
    }

    x.makeConsistent();
}

template<typename Config>
void HybridGS::sweep( const Relaxation::Direction relax_dir,
                      LinearAlgebra::CSRMatrix<Config> &A,
                      const LinearAlgebra::Vector &bvec,
                      LinearAlgebra::Vector &xvec )
{
    using gidx_t       = typename Config::gidx_t;
    using lidx_t       = typename Config::lidx_t;
    using scalar_t     = typename Config::scalar_t;
    using allocator_t  = typename Config::allocator_type;
    using matrixdata_t = LinearAlgebra::CSRMatrixData<Config>;

    using scalarAllocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<scalar_t>;

    auto x = xvec.getVectorData()->getRawDataBlock<scalar_t>( 0 );
    auto b = bvec.getVectorData()->getRawDataBlock<scalar_t>( 0 );

    auto A_data = std::dynamic_pointer_cast<matrixdata_t>( A.getMatrixData() );
    auto A_diag = A_data->getDiagMatrix();
    auto A_offd = A_data->getOffdMatrix();

    const auto num_rows = A_data->numLocalRows();

    lidx_t *Ad_rs = nullptr, *Ao_rs = nullptr;
    lidx_t *Ad_cols_loc = nullptr, *Ao_cols_loc = nullptr;
    gidx_t *Ad_cols = nullptr, *Ao_cols = nullptr;
    scalar_t *Ad_coeffs = nullptr, *Ao_coeffs = nullptr;

    std::tie( Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs ) = A_diag->getDataFields();

    scalar_t *ghosts = nullptr;
    scalarAllocator_t scalarAlloc;
    if ( !A_offd->isEmpty() ) {
        std::tie( Ao_rs, Ao_cols, Ao_cols_loc, Ao_coeffs ) = A_offd->getDataFields();

        const auto num_ghosts = static_cast<size_t>( A_offd->numUniqueColumns() );
        if ( d_num_ghosts != num_ghosts ) {
            if ( d_ghost_vals != nullptr ) {
                ghosts = reinterpret_cast<scalar_t *>( d_ghost_vals );
                scalarAlloc.deallocate( ghosts, d_num_ghosts );
                d_ghost_vals = nullptr;
            }
            d_num_ghosts      = num_ghosts;
            d_num_ghost_bytes = d_num_ghosts * sizeof( scalar_t );
            d_ghost_vals = reinterpret_cast<std::byte *>( scalarAlloc.allocate( d_num_ghosts ) );
        }

        ghosts = reinterpret_cast<scalar_t *>( d_ghost_vals );

        if constexpr ( std::is_same_v<size_t, gidx_t> ) {
            // column map can be passed to get ghosts function directly
            size_t *Ao_colmap = A_offd->getColumnMap();
            xvec.getGhostValuesByGlobalID( num_ghosts, Ao_colmap, ghosts );
        } else {
            // type mismatch, need to copy/cast into temporary vector
            std::vector<size_t> Ao_colmap;
            A_offd->getColumnMap( Ao_colmap );
            xvec.getGhostValuesByGlobalID( num_ghosts, Ao_colmap.data(), ghosts );
        }
    }

    auto row_sum =
        []( lidx_t *rowptr, lidx_t *colind, scalar_t *coeffs, auto &xvals, bool skip = false ) {
            return [=, &xvals]( lidx_t r ) {
                scalar_t rsum = 0;
                if ( rowptr == nullptr ) {
                    return rsum;
                }
                for ( auto off = rowptr[r] + skip; off < rowptr[r + 1]; ++off ) {
                    rsum += xvals[colind[off]] * coeffs[off];
                }
                return rsum;
            };
        };

    auto diag_sum = row_sum( Ad_rs, Ad_cols_loc, Ad_coeffs, x, true ); // skip diagonal value
    auto offd_sum = row_sum( Ao_rs, Ao_cols_loc, Ao_coeffs, ghosts );

    auto update = [&]( lidx_t row ) {
        auto diag = Ad_coeffs[Ad_rs[row]];
        auto dinv = 1.0 / diag;
        auto rsum = diag_sum( row ) + offd_sum( row );
        x[row]    = dinv * ( b[row] - rsum );
    };

    switch ( relax_dir ) {
    case Relaxation::Direction::forward:
        for ( size_t r = 0; r < num_rows; ++r )
            update( r );
        break;
    case Relaxation::Direction::backward:
        for ( size_t r = num_rows; r-- > 0; )
            update( r );
        break;
    }

    xvec.setUpdateStatus( LinearAlgebra::UpdateState::LOCAL_CHANGED );
}

JacobiL1::JacobiL1( std::shared_ptr<const SolverStrategyParameters> iparams )
    : Relaxation( iparams )
{
    d_spec_lower = d_db->getWithDefault<float>( "spec_lower", 0.25 );
    AMP_DEBUG_INSIST( d_spec_lower >= 0.0 && d_spec_lower < 1.0,
                      "JacobiL1: Invalid damping range, need a in [0,1)" );
}

void JacobiL1::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    d_pOperator = op;
    auto lin_op = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );
    AMP_DEBUG_INSIST( lin_op, "JacobiL1: operator must be linear" );
    auto mat = lin_op->getMatrix();
    AMP_DEBUG_INSIST( mat, "JacobiL1: matrix cannot be NULL" );
    d_matrix = mat;
}

void JacobiL1::apply( std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x )
{
    LinearAlgebra::csrVisit( d_matrix, [=]( auto csr_ptr ) { relax( csr_ptr, b, x ); } );
}

template<typename Config>
void JacobiL1::relax( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                      std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x )
{
    using scalar_t = typename Config::scalar_t;

    // Application of Jacobi L1 is x += omega * Dinv * r
    // where r is (b - A * x), D is sum of absolute values
    // in each row of A, and omega is weight determined from
    // Chebyshev iteration knowing that we damp in range [a,1]

    const scalar_t pi = static_cast<scalar_t>( AMP::Constants::pi );
    const scalar_t ma = 1.0 - d_spec_lower, pa = 1.0 + d_spec_lower; // make a adjustable later

    // storage for r
    auto r = x->clone();

    // Get D as absolute row sums of A
    auto D = A->getRowSumsAbsolute();

    for ( size_t i = 0; i < d_num_sweeps; ++i ) {
        // find omega
        const scalar_t om = ( ma * std::cos( pi * static_cast<scalar_t>( 2 * i - 1 ) /
                                             static_cast<scalar_t>( d_num_sweeps ) ) +
                              pa ) /
                            2.0;
        // update residual
        A->mult( x, r );
        r->subtract( *b, *r );
        // scale by Dinv
        r->divide( *r, *D );
        // update solution
        x->axpby( om, 1.0, *r );
        x->makeConsistent();
    }
}

} // namespace AMP::Solver::AMG

#endif
