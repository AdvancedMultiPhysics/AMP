#include "AMP/vectors/trilinos/tpetra/TpetraHelpers.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVectorData.h"


namespace AMP::LinearAlgebra {


static inline double *getBufferPtr( std::shared_ptr<VectorData> buf )
{
    size_t N_blocks = buf->numberOfDataBlocks();
    if ( N_blocks == 0 )
        return nullptr;
    if ( N_blocks > 1 )
        AMP_ERROR( "More than 1 data block detected" );
    return buf->getRawDataBlock<double>( 0 );
}


/********************************************************
 * Get a Tpetra vector from an AMP vector               *
 ********************************************************/
Teuchos::RCP<Tpetra::Vector<>> getTpetra( std::shared_ptr<Vector> vec )
{
#ifdef AMP_USE_MPI
    const auto &mpiComm = vec->getComm().getCommunicator();
    auto comm           = Teuchos::rcp( new Teuchos::MpiComm<int>( mpiComm ) );
#else
    auto comm = Tpetra::getDefaultComm();
#endif

    const auto localSize = vec->getLocalSize();
    auto map = Teuchos::rcp( new Tpetra::Map<>( vec->getGlobalSize(), localSize, 0, comm ) );
    auto ptr = getBufferPtr( vec->getVectorData() );

    using HostSpace   = Kokkos::DefaultHostExecutionSpace::memory_space;
    using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

    const decltype( localSize ) ncols = 1;

    using MDViewH = Kokkos::
        View<double **, Kokkos::LayoutLeft, HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using MDViewD = Kokkos::
        View<double **, Kokkos::LayoutLeft, DeviceSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    MDViewH hview_unmanaged( ptr, localSize, ncols );

    MDViewD dview_unmanaged( ptr, localSize, ncols );

    using DualViewType = Tpetra::Vector<>::dual_view_type;
    DualViewType dv( hview_unmanaged, dview_unmanaged );

    auto vec2 = Teuchos::rcp( new Tpetra::Vector<>( map, dv ) );

    return vec2;
}

} // namespace AMP::LinearAlgebra
