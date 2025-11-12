#ifndef included_AMP_AMG_Relaxation_hpp
#define included_AMP_AMG_Relaxation_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Relaxation.h"
#include "AMP/utils/Constants.h"
#include "AMP/utils/Memory.h"

#include <cmath>
#include <numeric>

#define AMP_AMG_RELAXATION_PROFILE

namespace AMP::Solver::AMG {

Relaxation::Relaxation( std::shared_ptr<const SolverStrategyParameters> params,
                        const std::string &name_,
                        const std::string &short_name_ )
    : SolverStrategy( params ),
      name( name_ ),
      short_name( short_name_ ),
      d_caller_lvl( std::numeric_limits<size_t>::max() )
{
    AMP_ASSERT( params );
    getFromInput( params->d_db );
    need_norms =
        d_iMaxIterations > 1 && ( d_dAbsoluteTolerance > 0.0 || d_dRelativeTolerance > 0.0 );
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

    // These are predominantly used as preconditioners/smoothers
    // so should default to zero tolerances and 1 iteration
    d_iMaxIterations     = db->getWithDefault<int>( "max_iterations", 1 );
    d_dAbsoluteTolerance = db->getWithDefault<double>( "absolute_tolerance", 0.0 );
    d_dRelativeTolerance = db->getWithDefault<double>( "relative_tolerance", 0.0 );
}

void Relaxation::apply( std::shared_ptr<const LinearAlgebra::Vector> b,
                        std::shared_ptr<LinearAlgebra::Vector> x )
{
    PROFILE( "Relaxation::apply" );

    // initialize, trivial if acting as a
    // preconditioner
    auto r             = need_norms ? b->clone() : nullptr;
    d_dInitialResidual = 0.0;
    if ( need_norms ) {
        const auto b_norm = static_cast<double>( b->L2Norm() );

        // Zero rhs implies zero solution, bail out early
        if ( b_norm == 0.0 ) {
            x->zero();
            d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
            d_dInitialResidual  = 0.0;
            d_dResidualNorm     = 0.0;
            if ( d_iDebugPrintInfoLevel > 0 ) {
                AMP::pout << name << "::apply: solution is zero" << std::endl;
            }
        }

        if ( d_bUseZeroInitialGuess ) {
            x->zero();
            d_dInitialResidual = b_norm;
        } else {
            d_pOperator->residual( b, x, r );
            d_dInitialResidual = static_cast<double>( r->L2Norm() );
        }

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << name << "::apply: initial L2Norm of solution vector: " << x->L2Norm()
                      << std::endl;
            AMP::pout << name << "::apply: initial L2Norm of rhs vector: " << b_norm << std::endl;
            AMP::pout << name << "::apply: initial L2Norm of residual: " << d_dInitialResidual
                      << std::endl;
        }
        if ( checkStoppingCriteria( d_dInitialResidual ) ) {
            if ( d_iDebugPrintInfoLevel > 0 ) {
                AMP::pout << name << "::apply: initial residual below tolerance" << std::endl;
            }
            return;
        }
    }
    auto current_res = static_cast<double>( d_dInitialResidual );

    // apply solver for needed number of iterations
    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {
        relax_visit( b, x );

        if ( need_norms ) {
            d_pOperator->residual( b, x, r );
            current_res = static_cast<double>( r->L2Norm() );

            if ( d_iDebugPrintInfoLevel > 1 ) {
                AMP::pout << short_name << ": iteration " << d_iNumberIterations << ", residual "
                          << current_res << std::endl;
            }

            if ( checkStoppingCriteria( current_res ) ) {
                break;
            }
        }
    }

    // Store final residual norm and update convergence flags
    // if this is acting as a solver and not a preconditioner
    if ( need_norms ) {
        d_dResidualNorm = current_res;
        checkStoppingCriteria( current_res );

        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << name << "::apply: final L2Norm of solution: " << x->L2Norm() << std::endl;
            AMP::pout << name << "::apply: final L2Norm of residual: " << current_res << std::endl;
            AMP::pout << name << "::apply: iterations: " << d_iNumberIterations << std::endl;
            AMP::pout << name << "::apply: convergence reason: "
                      << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
        }
    }
}

HybridGS::HybridGS( std::shared_ptr<const SolverStrategyParameters> iparams )
    : Relaxation( iparams, "HybridGS", "HGS" ), d_ghost_vals( nullptr ), d_num_ghosts( 0 )
{
    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
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
    // verify this is actually a CSRMatrix
    const auto mode = mat->mode();
    if ( mode == std::numeric_limits<std::uint16_t>::max() ) {
        AMP::pout << "Expected a CSRMatrix but received a matrix of type: " << mat->type()
                  << std::endl;
        AMP_ERROR( "HybridGS::registerOperator: Must pass in linear operator in CSRMatrix format" );
    }
}

void HybridGS::relax_visit( std::shared_ptr<const LinearAlgebra::Vector> b,
                            std::shared_ptr<LinearAlgebra::Vector> x )
{
    LinearAlgebra::csrVisit( d_matrix,
                             [this, b, x]( auto csr_ptr ) { this->relax( csr_ptr, b, x ); } );
}

#if 1
template<typename Config>
void HybridGS::relax( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                      std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x )
{
    auto run = [&]() {
        auto comp = [&]() {
            for ( size_t i = 0; i < d_num_sweeps; ++i ) {
                switch ( d_sweep ) {
                case Sweep::forward:
                    sweep<Config>( Direction::forward, *A, *b, *x );
                    break;
                case Sweep::backward:
                    sweep<Config>( Direction::backward, *A, *b, *x );
                    break;
                case Sweep::symmetric:
                    sweep<Config>( Direction::forward, *A, *b, *x );
                    sweep<Config>( Direction::backward, *A, *b, *x );
                    break;
                }
            }
        };

        if ( d_caller_lvl == 0 ) {
            {
                PROFILE( "HGS-relax-comp-0" );
                comp();
            }
            {
                PROFILE( "HGS-relax-comm-0" );
                x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
            }
        } else if ( d_caller_lvl == 1 ) {
            {
                PROFILE( "HGS-relax-comp-1" );
                comp();
            }
            {
                PROFILE( "HGS-relax-comm-1" );
                x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
            }
        } else if ( d_caller_lvl == 2 ) {
            {
                PROFILE( "HGS-relax-comp-2" );
                comp();
            }
            {
                PROFILE( "HGS-relax-comm-2" );
                x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
            }
        } else if ( d_caller_lvl == 3 ) {
            {
                PROFILE( "HGS-relax-comp-3" );
                comp();
            }
            {
                PROFILE( "HGS-relax-comm-3" );
                x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
            }
        } else if ( d_caller_lvl == 4 ) {
            {
                PROFILE( "HGS-relax-comp-4" );
                comp();
            }
            {
                PROFILE( "HGS-relax-comm-4" );
                x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
            }
        } else {
            {
                PROFILE( "HGS-relax-comp-5+" );
                comp();
            }
            {
                PROFILE( "HGS-relax-comm-5+" );
                x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
            }
        }
    };

    if ( d_caller_lvl == 0 ) {
        PROFILE( "HGS-relax-0" );
        run();
    } else if ( d_caller_lvl == 1 ) {
        PROFILE( "HGS-relax-1" );
        run();
    } else if ( d_caller_lvl == 2 ) {
        PROFILE( "HGS-relax-2" );
        run();
    } else if ( d_caller_lvl == 3 ) {
        PROFILE( "HGS-relax-3" );
        run();
    } else if ( d_caller_lvl == 4 ) {
        PROFILE( "HGS-relax-4" );
        run();
    } else {
        PROFILE( "HGS-relax-5+" );
        run();
    }
}

#else

template<typename Config>
void HybridGS::relax( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                      std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x )
{
    #ifdef AMP_AMG_RELAXATION_PROFILE
    PROFILE( "HGS-relax" );
    #endif
    {
    #ifdef AMP_AMG_RELAXATION_PROFILE
        PROFILE( "HGS-relax-comp" );
    #endif
        for ( size_t i = 0; i < d_num_sweeps; ++i ) {
            switch ( d_sweep ) {
            case Sweep::forward:
                sweep<Config>( Direction::forward, *A, *b, *x );
                break;
            case Sweep::backward:
                sweep<Config>( Direction::backward, *A, *b, *x );
                break;
            case Sweep::symmetric:
                sweep<Config>( Direction::forward, *A, *b, *x );
                sweep<Config>( Direction::backward, *A, *b, *x );
                break;
            }
        }
    }

    {
    #ifdef AMP_AMG_RELAXATION_PROFILE
        PROFILE( "HGS-relax-comm" );
    #endif
        x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
}
#endif

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
        if ( Ad_rs[row] == Ad_rs[row + 1] ) {
            // row is empty, skip it
            return;
        }
        auto diag = Ad_coeffs[Ad_rs[row]];
        auto dinv = 1.0 / diag;
        if ( std::isinf( dinv ) ) {
            return;
        }
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

JacobiL1::JacobiL1( std::shared_ptr<const SolverStrategyParameters> params )
    : Relaxation( params, "JacobiL1", "JL1" )
{
    d_spec_lower = d_db->getWithDefault<float>( "spec_lower", 0.8f );
    AMP_DEBUG_INSIST( d_spec_lower >= 0.0 && d_spec_lower < 1.0,
                      "JacobiL1: Invalid damping range, need a in [0,1)" );
    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

void JacobiL1::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    d_pOperator = op;
    auto lin_op = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );
    AMP_DEBUG_INSIST( lin_op, "JacobiL1: operator must be linear" );
    auto mat = lin_op->getMatrix();
    AMP_DEBUG_INSIST( mat, "JacobiL1: matrix cannot be NULL" );
    d_matrix = mat;
    // verify this is actually a CSRMatrix
    const auto mode = mat->mode();
    if ( mode == std::numeric_limits<std::uint16_t>::max() ) {
        AMP::pout << "Expected a CSRMatrix but received a matrix of type: " << mat->type()
                  << std::endl;
        AMP_ERROR( "HybridGS::registerOperator: Must pass in linear operator in CSRMatrix format" );
    }
    // Get D as absolute row sums of A
    std::shared_ptr<LinearAlgebra::Vector> D;
    LinearAlgebra::csrVisit( d_matrix,
                             [&D]( auto csr_ptr ) { D = csr_ptr->getRowSumsAbsolute( D, true ); } );
    d_diag.swap( D );
}

void JacobiL1::relax_visit( std::shared_ptr<const LinearAlgebra::Vector> b,
                            std::shared_ptr<LinearAlgebra::Vector> x )
{
    LinearAlgebra::csrVisit( d_matrix,
                             [this, b, x]( auto csr_ptr ) { this->relax( csr_ptr, b, x ); } );
}

#if 1
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
    const scalar_t ma = 1.0 - d_spec_lower, pa = 1.0 + d_spec_lower;

    // storage for r
    auto r = x->clone();

    auto run = [&]() {
        for ( size_t i = 0; i < d_num_sweeps; ++i ) {
            // find omega
            const auto irat =
                static_cast<scalar_t>( 2 * i - 1 ) / static_cast<scalar_t>( d_num_sweeps );
            const scalar_t om = 0.5 * ( ma * std::cos( pi * irat ) + pa );
            // update residual
            A->mult( x, r );
            r->subtract( *b, *r );
            // scale by Dinv
            r->divide( *r, *d_diag );
            // update solution
            x->axpby( om, 1.0, *r );
            x->makeConsistent();
        }
    };

    if ( d_caller_lvl == 0 ) {
        PROFILE( "JL1-relax-0" );
        run();
    } else if ( d_caller_lvl == 1 ) {
        PROFILE( "JL1-relax-1" );
        run();
    } else if ( d_caller_lvl == 2 ) {
        PROFILE( "JL1-relax-2" );
        run();
    } else if ( d_caller_lvl == 3 ) {
        PROFILE( "JL1-relax-3" );
        run();
    } else if ( d_caller_lvl == 4 ) {
        PROFILE( "JL1-relax-4" );
        run();
    } else {
        PROFILE( "JL1-relax-5+" );
        run();
    }
}

#else

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
    const scalar_t ma = 1.0 - d_spec_lower, pa = 1.0 + d_spec_lower;

    // storage for r
    auto r = x->clone();

    for ( size_t i = 0; i < d_num_sweeps; ++i ) {
        // find omega
        const scalar_t om = 0.5 * ( ma * std::cos( pi * static_cast<scalar_t>( 2 * i - 1 ) /
                                                   static_cast<scalar_t>( d_num_sweeps ) ) +
                                    pa );
        // update residual
        A->mult( x, r );
        r->subtract( *b, *r );
        // scale by Dinv
        r->divide( *r, *d_diag );
        // update solution
        x->axpby( om, 1.0, *r );
        x->makeConsistent();
    }
}
#endif

} // namespace AMP::Solver::AMG

#endif
