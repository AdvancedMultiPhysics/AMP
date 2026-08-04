#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/operations/ManagedVectorOperations.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVector.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVectorData.h"


namespace AMP::LinearAlgebra {

template<typename ST, typename LO, typename GO, typename NT>
static inline Teuchos::RCP<Tpetra::Vector<ST, LO, GO, NT>>
wrapAMPVec( std::shared_ptr<Vector> vec, const Teuchos::RCP<const Tpetra::Map<LO, GO, NT>> &map )
{
    auto vecData = vec->getVectorData();

    const auto N_blocks = vecData->numberOfDataBlocks();
    AMP_INSIST(
        N_blocks == 1,
        "TpetraVector: Only single block vectors allowed when wrapping native AMP vectors" );

    // copy cast not implemented yet, verify scalar types match
    if constexpr ( std::is_same_v<ST, float> ) {
        AMP_INSIST( vecData->getType( 0 ) == AMP::getTypeID<float>(),
                    "TpetraVector: scalar types must match when wrapping native AMP vectors" );
    } else if constexpr ( std::is_same_v<ST, double> ) {
        AMP_INSIST( vecData->getType( 0 ) == AMP::getTypeID<double>(),
                    "TpetraVector: scalar types must match when wrapping native AMP vectors" );
    } else {
        // unreachable by static_assert in class decl
    }

    // check if this is already a Tpetra vector
    auto tpetraData = std::dynamic_pointer_cast<TpetraVectorData<>>( vecData );

    if ( tpetraData ) {
        return tpetraData->getTpetraVector();
    } else {
        // view space matched to NT
        using DualViewType = typename Tpetra::Vector<ST, LO, GO, NT>::dual_view_type;
        using ViewType     = typename DualViewType::t_dev;

        // memory space migration not yet supported verify that input data is compatible with NT
        // managed memory (dev_acc true, all_dev false) skips both checks
        const auto [dev_acc, all_dev] =
            AMP::Utilities::memoryLocationsDeviceAccessible( vecData->getMemoryLocation() );
        if ( all_dev ) {
            // strictly device accessible
            constexpr bool valid =
                Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace,
                                           typename ViewType::memory_space>::accessible;
            AMP_INSIST(
                valid,
                "TpetraVector: Must share memory accessibility when wrapping native AMP vectors" );
        } else if ( !dev_acc ) {
            // not device accessible at all
            constexpr bool valid =
                Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace,
                                           typename ViewType::memory_space>::accessible;
            AMP_INSIST(
                valid,
                "TpetraVector: Must share memory accessibility when wrapping native AMP vectors" );
        }

        // get pointer to data buffer
        auto ptr = vecData->getRawDataBlock<ST>( 0 );

        // wrap into view, then into dual_view with duplicates
        // this gets us a shallow copy while still satisfying the Tpetra interface
        const LO localSize = vec->getLocalSize();
        AMP_ASSERT( localSize == static_cast<LO>( map->getLocalNumElements() ) );
        ViewType wrap_buf( ptr, localSize );
        DualViewType dv( wrap_buf, wrap_buf );

        return Teuchos::rcp( new Tpetra::Vector<ST, LO, GO, NT>( map, dv ) );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
static inline Teuchos::RCP<const Tpetra::Map<LO, GO, NT>>
mapAMPVec( std::shared_ptr<const Vector> vec )
{
    // create a map based on vec then call other wrapper with it
#ifdef AMP_USE_MPI
    const auto &mpiComm = vec->getComm().getCommunicator();
    auto comm           = Teuchos::rcp( new Teuchos::MpiComm<int>( mpiComm ) );
#else
    auto comm = Tpetra::getDefaultComm();
#endif
    const auto localSize = vec->getLocalSize();
    return Teuchos::rcp( new Tpetra::Map<LO, GO, NT>( vec->getGlobalSize(), localSize, 0, comm ) );
}

/********************************************************
 * Constructor                                           *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
TpetraVector<ST, LO, GO, NT>::TpetraVector( std::shared_ptr<Vector> vec )
    : d_tpetra( wrapAMPVec<ST, LO, GO, NT>( vec, mapAMPVec<ST, LO, GO, NT>( vec ) ) ), d_AMP( vec )
{
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraVector<ST, LO, GO, NT>::TpetraVector( std::shared_ptr<Vector> vec,
                                            const Teuchos::RCP<const Tpetra::Map<LO, GO, NT>> &map )
    : d_tpetra( wrapAMPVec<ST, LO, GO, NT>( vec, map ) ), d_AMP( vec )
{
}

/********************************************************
 * View                                                  *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<TpetraVector<ST, LO, GO, NT>>
TpetraVector<ST, LO, GO, NT>::view( Vector::shared_ptr inVector,
                                    const Teuchos::RCP<const Tpetra::Map<LO, GO, NT>> &map )
{
    AMP_INSIST( inVector->numberOfDataBlocks() == 1,
                "Tpetra does not support more than 1 data block" );
    // Check if we have an existing view
    if ( std::dynamic_pointer_cast<TpetraVector<ST, LO, GO, NT>>( inVector ) )
        return std::dynamic_pointer_cast<TpetraVector<ST, LO, GO, NT>>( inVector );
    if ( std::dynamic_pointer_cast<MultiVector>( inVector ) ) {
        auto multivec = std::dynamic_pointer_cast<MultiVector>( inVector );
        if ( multivec->getNumberOfSubvectors() == 1 ) {
            return view( multivec->getVector( 0 ), map );
        } else {
            AMP_ERROR( "View of multi-block MultiVector is not supported yet" );
        }
    }
    // Check if we are dealing with a managed vector
    auto managedData = std::dynamic_pointer_cast<ManagedVectorData>( inVector->getVectorData() );
    if ( managedData ) {
        auto root = managedData->getVectorEngine();
        return view( root, map );
    }
    // Create the view
    std::shared_ptr<TpetraVector<ST, LO, GO, NT>> ptr(
        new TpetraVector<ST, LO, GO, NT>( inVector, map ) );
    inVector->registerView( ptr );
    return ptr;
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<TpetraVector<ST, LO, GO, NT>>
TpetraVector<ST, LO, GO, NT>::view( Vector::shared_ptr inVector )
{
    return TpetraVector<ST, LO, GO, NT>::view( inVector, mapAMPVec<ST, LO, GO, NT>( inVector ) );
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<const TpetraVector<ST, LO, GO, NT>>
TpetraVector<ST, LO, GO, NT>::constView( Vector::const_shared_ptr inVector,
                                         const Teuchos::RCP<const Tpetra::Map<LO, GO, NT>> &map )
{
    return view( std::const_pointer_cast<Vector>( inVector ), map );
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<const TpetraVector<ST, LO, GO, NT>>
TpetraVector<ST, LO, GO, NT>::constView( Vector::const_shared_ptr inVector )
{
    return view( std::const_pointer_cast<Vector>( inVector ),
                 mapAMPVec<ST, LO, GO, NT>( inVector ) );
}

} // namespace AMP::LinearAlgebra
