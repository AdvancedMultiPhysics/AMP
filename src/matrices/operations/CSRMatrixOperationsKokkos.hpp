#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/CSRMatrixOperationsKokkos.h"
#include "AMP/utils/Utilities.h"
#include "AMP/AMP_TPLs.h"

#include <algorithm>

#include "ProfilerApp.h"

#if defined(AMP_USE_KOKKOS) || defined(AMP_USE_TRILINOS_KOKKOS)

#include "Kokkos_Core.hpp"

namespace AMP::LinearAlgebra {

template<typename Policy>
auto wrapCSRDiagDataKokkos( CSRMatrixData<Policy> *csrData )
{
  using lidx_t   = typename Policy::lidx_t;
  using gidx_t   = typename Policy::gidx_t;
  using scalar_t = typename Policy::scalar_t;

  const lidx_t nrows = static_cast<lidx_t>( csrData->numLocalRows() );
  const lidx_t nnz_tot = csrData->numberOfNonZerosDiag();
  auto [nnz, cols, cols_loc, coeffs] = csrData->getCSRDiagData();
  auto *rowstarts = csrData->getDiagRowStarts();

  // coeffs not marked const so that setScalar and similar will work
  return std::make_tuple( Kokkos::View<const lidx_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( nnz, nrows ),
			  Kokkos::View<const gidx_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( cols, nnz_tot ),
			  Kokkos::View<const lidx_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( cols_loc, nnz_tot ),
			  Kokkos::View<scalar_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( coeffs, nnz_tot ),
			  Kokkos::View<const lidx_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( rowstarts, nrows ) );
}
  
template<typename Policy>
auto wrapCSROffDiagDataKokkos( CSRMatrixData<Policy> *csrData )
{
  using lidx_t   = typename Policy::lidx_t;
  using gidx_t   = typename Policy::gidx_t;
  using scalar_t = typename Policy::scalar_t;

  const lidx_t nrows = static_cast<lidx_t>( csrData->numLocalRows() );
  const lidx_t nnz_tot = csrData->numberOfNonZerosOffDiag();
  auto [nnz, cols, cols_loc, coeffs] = csrData->getCSROffDiagData();
  auto *rowstarts = csrData->getOffDiagRowStarts();

  return std::make_tuple( Kokkos::View<const lidx_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( nnz, nrows ),
			  Kokkos::View<const gidx_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( cols, nnz_tot ),
			  Kokkos::View<const lidx_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( cols_loc, nnz_tot ),
			  Kokkos::View<scalar_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( coeffs, nnz_tot ),
			  Kokkos::View<const lidx_t*,Kokkos::LayoutRight,Kokkos::SharedSpace>( rowstarts, nrows ) );
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::mult( std::shared_ptr<const Vector> in,
							 MatrixData const &A,
							 std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsKokkos::mult" );
    AMP_DEBUG_ASSERT( in && out );

    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto csrData = getCSRMatrixData<Policy>( const_cast<MatrixData &>( A ) );
    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    const auto nCols = static_cast<lidx_t>( csrData->numLocalColumns() );

    auto inData                 = in->getVectorData();
    const auto &ghosts          = inData->getGhosts();
    const auto nGhosts          = static_cast<lidx_t>( ghosts.size() );
    auto outData                = out->getVectorData();

    AMP_INSIST( csrData->getMemoryLocation() != AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsKokkos is not implemented for device memory" );

    AMP_INSIST(
        1 == inData->numberOfDataBlocks(),
        "CSRMatrixOperationsKokkos::mult only implemented for vectors with one data block" );

    AMP_INSIST(
        ghosts.size() == inData->getGhostSize(),
        "CSRMatrixOperationsKokkos::mult only implemented for vectors with accessible ghosts" );
    
    // Wrap in/out data into Kokkos Views
    Kokkos::View<const scalar_t*,
		 Kokkos::LayoutRight,
		 Kokkos::SharedSpace,
		 Kokkos::MemoryTraits<Kokkos::RandomAccess>> inDataBlock( inData->getRawDataBlock<scalar_t>( 0 ), nCols );
    Kokkos::View<scalar_t*,
		 Kokkos::LayoutRight,
		 Kokkos::SharedSpace> outDataBlock( outData->getRawDataBlock<scalar_t>( 0 ), nRows );
    Kokkos::View<const scalar_t*,
		 Kokkos::LayoutRight,
		 Kokkos::SharedSpace,
		 Kokkos::MemoryTraits<Kokkos::RandomAccess>> ghostDataBlock( ghosts.data(), nGhosts );

    {
	// lambda capture of structured bindings throws warning on c++17
	// unpack the tuple of views manually
	auto vtpl = wrapCSRDiagDataKokkos( csrData );
	auto nnz_d       = std::get<0>( vtpl );
	auto cols_loc_d  = std::get<2>( vtpl );
	auto coeffs_d    = std::get<3>( vtpl );
	auto rowstarts_d = std::get<4>( vtpl );

	Kokkos::parallel_for( Kokkos::TeamPolicy<ExecSpace, Kokkos::IndexType<lidx_t>>(d_exec_space, nRows, 1, 64),
			      KOKKOS_LAMBDA (const typename Kokkos::TeamPolicy<ExecSpace>::member_type &tm) {
				lidx_t row = tm.league_rank();
				const auto nC = nnz_d( row );
				const auto rs = rowstarts_d( row );
				scalar_t sum = 0.0;
				Kokkos::parallel_reduce( Kokkos::TeamVectorRange(tm,nC),
							 [=] (lidx_t &c, scalar_t &lsum) {
							   const auto cl = cols_loc_d( rs + c );
							   lsum += coeffs_d( rs + c ) * inDataBlock( cl );
							 }, sum);
				outDataBlock(row) = sum;
			      } );
    }

    if ( csrData->hasOffDiag() ) {
	// lambda capture of structured bindings throws warning on c++17
	// unpack the tuple of views manually
	auto vtpl = wrapCSROffDiagDataKokkos( csrData );
	auto nnz_od       = std::get<0>( vtpl );
	auto cols_loc_od  = std::get<2>( vtpl );
	auto coeffs_od    = std::get<3>( vtpl );
	auto rowstarts_od = std::get<4>( vtpl );

	Kokkos::parallel_for( "CSRMatrixOperationsKokkos::mult (ghost)",
			      Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows),
			      KOKKOS_LAMBDA ( const lidx_t row ) {
				const auto nC = nnz_od( row );
				const auto rs = rowstarts_od( row );
				
				for ( lidx_t c = 0; c < nC; ++c ) {
				  const auto cl = cols_loc_od( rs + c );
				  outDataBlock( row ) += coeffs_od( rs + c ) * ghostDataBlock( cl );
				}
			      } );
    }

    d_exec_space.fence();
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::multTranspose( std::shared_ptr<const Vector> in,
								  MatrixData const &A,
								  std::shared_ptr<Vector> out)
{
    PROFILE( "CSRMatrixOperationsKokkos::multTranspose" );
    
    // this is not meant to be an optimized version. It is provided for completeness
    AMP_DEBUG_ASSERT( in && out );

    out->zero();

    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto csrData = getCSRMatrixData<Policy>( const_cast<MatrixData &>( A ) );
    
    AMP_INSIST( csrData->getMemoryLocation() != AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsKokkos is not implemented for device memory" );

    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    const auto nCols = static_cast<lidx_t>( csrData->numLocalColumns() );
    auto inData = in->getVectorData();
    auto outData = out->getVectorData();
    
    // Wrap in/out data into Kokkos Views
    Kokkos::View<const scalar_t*,
		 Kokkos::LayoutRight,
		 Kokkos::SharedSpace> inDB( inData->getRawDataBlock<scalar_t>( 0 ), nRows );

    {
	// lambda capture of structured bindings throws warning on c++17
	// unpack the tuple of views manually
	auto vtpl = wrapCSRDiagDataKokkos( csrData );
	auto nnz       = std::get<0>( vtpl );
	auto cols_loc  = std::get<2>( vtpl );
	auto coeffs    = std::get<3>( vtpl );
	auto rowstarts = std::get<4>( vtpl );

	// Make temporary views for output columns and values
	auto outDB = Kokkos::View<scalar_t*,
				  Kokkos::LayoutRight,
				  Kokkos::SharedSpace,
				  Kokkos::MemoryTraits<Kokkos::Atomic>>( outData->getRawDataBlock<scalar_t>( 0 ),
									 nCols);
	Kokkos::deep_copy( d_exec_space, outDB, 0.0 );

	Kokkos::parallel_for( Kokkos::TeamPolicy<ExecSpace, Kokkos::IndexType<lidx_t>>(d_exec_space, nRows, 1, 64),
			      KOKKOS_LAMBDA (const typename Kokkos::TeamPolicy<ExecSpace>::member_type &tm) {
				lidx_t row = tm.league_rank();
				const auto n = nnz( row );
				const auto rs = rowstarts( row );
				auto val = inDB( n );
				Kokkos::parallel_for( Kokkos::TeamVectorRange(tm,n),
						      [=] (const lidx_t &j) {
							const auto cl = cols_loc( rs + j );
							outDB( cl ) += val * coeffs( rs + j );
						      });
			      } );
    }

    if ( csrData->hasOffDiag() ) {
	// lambda capture of structured bindings throws warning on c++17
	// unpack the tuple of views manually
	auto vtpl = wrapCSROffDiagDataKokkos( csrData );
	auto nnz       = std::get<0>( vtpl );
	auto cols      = std::get<1>( vtpl );
	auto cols_loc  = std::get<2>( vtpl );
	auto coeffs    = std::get<3>( vtpl );
	auto rowstarts = std::get<4>( vtpl );

	// get diag map but leave in std::vector
	// it is not needed inside compute kernel this time
	std::vector<size_t> rcols;
	csrData->getOffDiagColumnMap( rcols );
	const auto num_unq = rcols.size();
	
	// Make temporary view for output values
	auto vvals = Kokkos::View<scalar_t*,
				  Kokkos::LayoutRight,
				  Kokkos::SharedSpace,
				  Kokkos::MemoryTraits<Kokkos::Atomic>>( "vvals", num_unq);
	Kokkos::deep_copy( d_exec_space, vvals, 0.0 );

	Kokkos::parallel_for( "CSRMatrixOperationsKokkos::multTranspose (od)",
			      Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows ),
			      KOKKOS_LAMBDA ( const lidx_t n ) {
				auto rs = rowstarts( n );
				auto val = inDB( n );
				for ( lidx_t j = 0; j < nnz( n ); ++j ) {
				  auto cl = cols_loc( rs + j );
				  vvals( cl ) += val * coeffs( rs + j );
				}
			      } );
	
	// Need to fence before sending values off to be written out
	d_exec_space.fence();

	// copy rcols and vvals into std::vectors and write out
	out->addValuesByGlobalID( num_unq, rcols.data(), vvals.data() );
    } else {
        d_exec_space.fence(); // still finish with a fence if no offd term present
    }
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::scale( AMP::Scalar alpha_in, MatrixData &A )
{
    using lidx_t = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;
    
    auto csrData = getCSRMatrixData<Policy>( const_cast<MatrixData &>( A ) );
    
    AMP_INSIST( csrData->getMemoryLocation() != AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsKokkos is not implemented for device memory" );

    // lambda capture of structured bindings throws warning on c++17
    // unpack the tuple manually
    auto coeffs_d = std::get<3>( wrapCSRDiagDataKokkos( csrData ) );

    const auto tnnz_d = csrData->numberOfNonZerosDiag();
    auto alpha = static_cast<scalar_t>( alpha_in );
    
    Kokkos::parallel_for( "CSRMatrixOperationsKokkos::scale (d)",
			  Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, tnnz_d),
			  KOKKOS_LAMBDA ( const lidx_t n ) {
			    coeffs_d( n ) *= alpha;
			  } );

    if ( csrData->hasOffDiag() ) {
    auto coeffs_od = std::get<3>( wrapCSROffDiagDataKokkos( csrData ) );

    const auto tnnz_od = csrData->numberOfNonZerosOffDiag();
    
    Kokkos::parallel_for( "CSRMatrixOperationsKokkos::scale (od)",
			  Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, tnnz_od),
			  KOKKOS_LAMBDA ( const lidx_t n ) {
			    coeffs_od( n ) *= alpha;
			  } );
    }
    
    d_exec_space.fence();
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::matMultiply( MatrixData const &,
								MatrixData const &,
								MatrixData & )
{
    AMP_WARNING( "SpGEMM for CSRMatrixOperationsKokkos not implemented" );
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::axpy( AMP::Scalar alpha_in,
                                               const MatrixData &X,
                                               MatrixData &Y )
{
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    const auto csrDataX = getCSRMatrixData<Policy>( const_cast<MatrixData &>( X ) );
    const auto csrDataY = getCSRMatrixData<Policy>( const_cast<MatrixData &>( Y ) );
    
    AMP_INSIST( csrDataX->getMemoryLocation() != AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsKokkos is not implemented for device memory" );
    AMP_INSIST( csrDataY->getMemoryLocation() != AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsKokkos is not implemented for device memory" );
    AMP_INSIST( csrDataX->getMemoryLocation() == csrDataY->getMemoryLocation(),
                "CSRMatrixOperationsKokkos::axpy X and Y must be in same memory space" );

    auto alpha = static_cast<scalar_t>( alpha_in );
    
    {
	// lambda capture of structured bindings throws warning on c++17
	// unpack the tuple of views manually
	auto coeffsX_d = std::get<3>( wrapCSRDiagDataKokkos( csrDataX ) );
	auto coeffsY_d = std::get<3>( wrapCSRDiagDataKokkos( csrDataY ) );

    const auto tnnz_d = csrDataX->numberOfNonZerosDiag();
    Kokkos::parallel_for( "CSRMatrixOperationsKokkos::axpy (d)",
			  Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, tnnz_d),
			  KOKKOS_LAMBDA ( const gidx_t n ) {
			    coeffsY_d( n ) += alpha * coeffsX_d( n );
			  } );
    }
    
    if ( csrDataX->hasOffDiag() ) {
      const auto tnnz_od = csrDataX->numberOfNonZerosDiag();
	auto coeffsX_od = std::get<3>( wrapCSROffDiagDataKokkos( csrDataX ) );
	auto coeffsY_od = std::get<3>( wrapCSROffDiagDataKokkos( csrDataY ) );
      
      Kokkos::parallel_for( "CSRMatrixOperationsKokkos::axpy (od)",
			    Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, tnnz_od),
			    KOKKOS_LAMBDA ( const gidx_t n ) {
			      coeffsY_od( n ) += alpha * coeffsX_od( n );
			    } );
    }
    
    d_exec_space.fence();
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::setScalar( AMP::Scalar alpha_in, MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;

    auto alpha = static_cast<scalar_t>( alpha_in );
    auto csrData = getCSRMatrixData<Policy>( const_cast<MatrixData &>( A ) );
    
    AMP_INSIST( csrData->getMemoryLocation() != AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsKokkos is not implemented for device memory" );

    {
        // lambda capture of structured bindings throws warning on c++17
        // unpack the tuple manually
        auto coeffs_d = std::get<3>( wrapCSRDiagDataKokkos( csrData ) );
	Kokkos::deep_copy( d_exec_space, coeffs_d, alpha );
    }
    
    if ( csrData->hasOffDiag() ) {
        auto coeffs_od = std::get<3>( wrapCSROffDiagDataKokkos( csrData ) );
	Kokkos::deep_copy( d_exec_space, coeffs_od, alpha );
    }
    
    d_exec_space.fence();
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::zero( MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;
    setScalar( static_cast<scalar_t>( 0.0 ), A );
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::setDiagonal( std::shared_ptr<const Vector> in,
								MatrixData &A )
{
    using lidx_t = typename Policy::lidx_t;
    using gidx_t = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;
    
    auto csrData = getCSRMatrixData<Policy>( const_cast<MatrixData &>( A ) );
    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    auto beginRow = csrData->beginRow();
    
    AMP_INSIST( csrData->getMemoryLocation() != AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsKokkos is not implemented for device memory" );

    // lambda capture of structured bindings throws warning on c++17
    // unpack the tuple of views manually
    auto vtpl = wrapCSRDiagDataKokkos( csrData );
    auto nnz_d       = std::get<0>( vtpl );
    auto cols_d      = std::get<1>( vtpl );
    auto coeffs_d    = std::get<3>( vtpl );
    auto rowstarts_d = std::get<4>( vtpl );

    Kokkos::View<const scalar_t*,Kokkos::LayoutRight> vvals( in->getRawDataBlock<scalar_t>(), nRows );

    Kokkos::parallel_for( "CSRMatrixOperationsKokkos::setDiagonal",
			  Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows),
			  KOKKOS_LAMBDA ( const lidx_t row ) {
			    const auto nC = nnz_d( row );
			    const auto rs = rowstarts_d( row );
			    
			    for ( lidx_t c = 0; c < nC; ++c ) {
			      if ( cols_d( rs + c ) == static_cast<gidx_t>( row + beginRow ) ) {
				coeffs_d( rs + c ) = vvals( row );
				break;
			      }
			    }
			  } );
    
    d_exec_space.fence();
}

template<typename Policy, class ExecSpace>
void CSRMatrixOperationsKokkos<Policy, ExecSpace>::setIdentity( MatrixData &A )
{
    using lidx_t = typename Policy::lidx_t;
    using gidx_t = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;
    
    zero( A );
    
    auto csrData = getCSRMatrixData<Policy>( const_cast<MatrixData &>( A ) );
    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    auto beginRow = csrData->beginRow();

    // lambda capture of structured bindings throws warning on c++17
    // unpack the tuple of views manually
    auto vtpl = wrapCSRDiagDataKokkos( csrData );
    auto nnz_d       = std::get<0>( vtpl );
    auto cols_d      = std::get<1>( vtpl );
    auto coeffs_d    = std::get<3>( vtpl );
    auto rowstarts_d = std::get<4>( vtpl );

    Kokkos::parallel_for( "CSRMatrixOperationsKokkos::setIdentity",
			  Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows),
			   KOKKOS_LAMBDA ( const lidx_t row ) {
			    const auto nC = nnz_d( row );
			    const auto rs = rowstarts_d( row );
			    
			    for ( lidx_t c = 0; c < nC; ++c ) {
			      if ( cols_d( rs + c ) == static_cast<gidx_t>( row + beginRow ) ) {
				coeffs_d( rs + c ) = static_cast<scalar_t>( 1.0 );
				break;
			      }
			    }
			  } );
    
    d_exec_space.fence();
}

template<typename Policy, class ExecSpace>
AMP::Scalar CSRMatrixOperationsKokkos<Policy, ExecSpace>::L1Norm( MatrixData const & ) const
{
    AMP_ERROR( "Not implemented" );
}

} // namespace AMP::LinearAlgebra

#endif // close check for Kokkos being defined
