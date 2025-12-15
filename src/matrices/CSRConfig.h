#ifndef included_AMP_CSRConfig
#define included_AMP_CSRConfig

#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/utils/Memory.h"

#ifdef AMP_USE_HYPRE
    #include "HYPRE_utilities.h"
#endif

#include <cstdint>
#include <limits>
#include <tuple>


namespace AMP::LinearAlgebra {

// labels for components of a CSR configuration
enum class alloc : uint8_t { host, device, managed };
enum class index : uint8_t { i32, i64, ill };
enum class scalar : uint8_t { f32, f64, fld };

// Helpers to recover types and other info from these labels
template<alloc>
struct alloc_info;
template<>
struct alloc_info<alloc::host> {
    using type                = HostAllocator<void>;
    static constexpr alloc id = alloc::host;
    static const char *name() { return "host"; }
};
#ifdef AMP_USE_DEVICE
template<>
struct alloc_info<alloc::device> {
    using type                = DeviceAllocator<void>;
    static constexpr alloc id = alloc::device;
    static const char *name() { return "device"; }
};
template<>
struct alloc_info<alloc::managed> {
    using type                = ManagedAllocator<void>;
    static constexpr alloc id = alloc::managed;
    static const char *name() { return "managed"; }
};
#endif

template<index>
struct index_info;
template<>
struct index_info<index::i32> {
    using type                = int;
    static constexpr index id = index::i32;
    static const char *name() { return "int"; }
};
template<>
struct index_info<index::i64> {
    using type                = size_t;
    static constexpr index id = index::i64;
    static const char *name() { return "size_t"; }
};

template<>
struct index_info<index::ill> {
    using type                = long long int;
    static constexpr index id = index::ill;
    static const char *name() { return "long long"; }
};

template<scalar>
struct scalar_info;
template<>
struct scalar_info<scalar::f32> {
    using type                 = float;
    static constexpr scalar id = scalar::f32;
    static const char *name() { return "single precision"; }
};
template<>
struct scalar_info<scalar::f64> {
    using type                 = double;
    static constexpr scalar id = scalar::f64;
    static const char *name() { return "double precision"; }
};

template<>
struct scalar_info<scalar::fld> {
    using type                 = long double;
    static constexpr scalar id = scalar::fld;
    static const char *name() { return "long double"; }
};

/*
   Describes the layout of a CSR mode (uint16_t).

   Component types define shift and mask which indicate the bits
   of the overall CSR mode integer given to each component.
 */
struct csr_mode_layout {
    using I = std::uint16_t;
    struct alloc {
        static constexpr I shift = 0U;
        static constexpr I mask  = 0xfU;
        using type               = LinearAlgebra::alloc;
    };
    struct lidx {
        static constexpr I shift = alloc::shift + 4;
        static constexpr I mask  = 0xfU << shift;
        using type               = index;
    };
    struct gidx {
        static constexpr I shift = lidx::shift + 4;
        static constexpr I mask  = 0xfU << shift;
        using type               = index;
    };
    struct scalar {
        static constexpr I shift = gidx::shift + 4;
        static constexpr I mask  = 0xfU << shift;
        using type               = LinearAlgebra::scalar;
    };
    using components = std::tuple<alloc, lidx, gidx, scalar>;
    template<I index>
    using element = std::tuple_element_t<index, components>;
};

// Helper type to assemble the overall CSR mode integer from component enums.
template<alloc Alloc, index local, index global, scalar Scalar>
struct make_mode {
    using I = std::uint16_t;
    template<auto... vs>
    struct assemble {
        template<I... Is>
        static constexpr I get( std::integer_sequence<I, Is...> )
        {
            return ( ( static_cast<I>( vs ) << csr_mode_layout::element<Is>::shift ) | ... );
        }
        static constexpr auto value = get( std::make_integer_sequence<I, sizeof...( vs )>{} );
    };
    static constexpr I value = assemble<Alloc, local, global, Scalar>::value;
};
template<alloc a, index l, index g, scalar s>
constexpr std::uint16_t make_mode_v = make_mode<a, l, g, s>::value;

#define AMP_GEN_MODE( i1_name, i1_id, i2_name, i2_id )                               \
    h##i1_name##i2_name##f = make_mode_v<alloc::host, i1_id, i2_id, scalar::f32>,    \
    h##i1_name##i2_name##d = make_mode_v<alloc::host, i1_id, i2_id, scalar::f64>,    \
    d##i1_name##i2_name##f = make_mode_v<alloc::device, i1_id, i2_id, scalar::f32>,  \
    d##i1_name##i2_name##d = make_mode_v<alloc::device, i1_id, i2_id, scalar::f64>,  \
    m##i1_name##i2_name##f = make_mode_v<alloc::managed, i1_id, i2_id, scalar::f32>, \
    m##i1_name##i2_name##d = make_mode_v<alloc::managed, i1_id, i2_id, scalar::f64>
// CSR mode labels for a CSR configurations
enum class csr_mode : std::uint16_t {
    AMP_GEN_MODE( i, index::i32, I, index::i64 ),
    AMP_GEN_MODE( i, index::i32, i, index::i32 ),
    AMP_GEN_MODE( i, index::i32, l, index::ill ),
    AMP_GEN_MODE( l, index::ill, l, index::ill ),
    other = std::numeric_limits<std::uint16_t>::max()
};

#define EMPTY()
#define DEFER( X ) X EMPTY()
#define EXPAND( X ) X

#if defined( AMP_USE_HYPRE )
    #include "CSRConfigHypre.h"
#else
    #define CSR_CONFIG_FORALL_HYPRE_HOST( INST )
    #ifdef AMP_USE_DEVICE
        #define CSR_CONFIG_FORALL_HYPRE_DEVICE( INST )
    #endif
    #define CSR_CONFIG_CC_FORALL_HYPRE( INST )
    #define CSR_CONFIG_CC_FORALL0_HYPRE( F, G )
#endif

#define CSR_CONFIG_FORALL_HOST( INST ) \
    INST( csr_mode::hiIf )             \
    INST( csr_mode::hiId )             \
    CSR_CONFIG_FORALL_HYPRE_HOST( INST )
#define CSR_CONFIG_CC_FORALL_HOST( F, G ) \
    F( G, csr_mode::hiIf )                \
    F( G, csr_mode::hiId )

#ifdef AMP_USE_DEVICE
    #define CSR_CONFIG_FORALL_DEVICE( INST ) \
        INST( csr_mode::diIf )               \
        INST( csr_mode::diId )               \
        INST( csr_mode::miIf )               \
        INST( csr_mode::miId )               \
        CSR_CONFIG_FORALL_HYPRE_DEVICE( INST )

    #define CSR_CONFIG_CC_FORALL_DEVICE( F, G ) \
        F( G, csr_mode::diIf )                  \
        F( G, csr_mode::diId )                  \
        F( G, csr_mode::miIf )                  \
        F( G, csr_mode::miId )
#else
    #define CSR_CONFIG_FORALL_DEVICE( INST )
    #define CSR_CONFIG_CC_FORALL_DEVICE( F, G )
#endif

#define CSR_CONFIG_FORALL( INST )  \
    CSR_CONFIG_FORALL_HOST( INST ) \
    CSR_CONFIG_FORALL_DEVICE( INST )

#define CSR_CONFIG_CC_FORALL0( F, G ) \
    CSR_CONFIG_CC_FORALL_HOST( F, G ) \
    CSR_CONFIG_CC_FORALL_DEVICE( F, G )
#define CSR_CONFIG_CC_FORALL1() CSR_CONFIG_CC_FORALL0
#define CSR_CONFIG_CC_FORALL2( F, G ) DEFER( CSR_CONFIG_CC_FORALL1 )()( F, G )
#define CSR_CONFIG_CC_FORALL( INST )                               \
    EXPAND( CSR_CONFIG_CC_FORALL0( CSR_CONFIG_CC_FORALL2, INST ) ) \
    CSR_CONFIG_CC_FORALL_HYPRE( INST )


/******************************************************************************
 * Helpers for defining the instantiations for the rutines supporting migration
 * of csr matrices between host / managed / device memories, between global
 * indexing using ill / i64, and between float and double scalar types.
 *
 * There are two main macros supposed to be used externally:
 * \brief instantiate migrate routines templated againt both in and out
 *        csr_config's
 * \param[in] INST macro with config_in and config_out arguments
 * \detail As oposed to CSR_CONFIG_CC_FORALL, the permutations include hypre
 *         and AMP configurations
 * CSR_INOUT_CONFIG_MIGRATE ( INST )
 *
 * There are several inner helper macros with the following naming convention:
 * 1.- OUTLIST_(HOST/DEVICE/HYPRE_HOST/HYPRE_DEVICE)**
 *     Perform a list of calls to the subsequent macros for
 *     host/device/hypre_host/hypre_device apropriate csr configurations.
 * 2.- CSR_INOUT_CONFIG_MIGRATE_LOOP
 *     Given an in csr_config and the INST macro it calls through the list of
 *     of host/device/hypre_host/hypre_device CSR_CONFIG_CC* macros which
 *     define the out csr_config.
 *
 * These instantiations add the possibility to migrate between different
 * global index types, not needed (for the moment) in CSR_CONFIG_CC_FORALL
 ******************************************************************************/

#define CSR_INOUT_CONFIG_MIGRATE_LOOP( mode_in, INST ) \
    CSR_CONFIG_CC_FORALL0( INST, mode_in )             \
    CSR_CONFIG_CC_FORALL0_HYPRE( INST, mode_in )

#define CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HOST( INST )     \
    CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::hiIf, INST ) \
    CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::hiId, INST )

#if defined( AMP_USE_DEVICE )
    #define CSR_INOUT_CONFIG_MIGRATE_OUTLIST_DEVICE( INST )   \
        CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::miIf, INST ) \
        CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::miId, INST ) \
        CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::diIf, INST ) \
        CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::diId, INST )
#else
    #define CSR_INOUT_CONFIG_MIGRATE_OUTLIST_DEVICE( INST )
#endif

#ifndef AMP_USE_HYPRE
    #define CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HYPRE_HOST( INST )
    #define CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HYPRE_DEVICE( INST )
#endif

#define CSR_INOUT_CONFIG_MIGRATE( INST )                \
    CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HOST( INST )       \
    CSR_INOUT_CONFIG_MIGRATE_OUTLIST_DEVICE( INST )     \
    CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HYPRE_HOST( INST ) \
    CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HYPRE_DEVICE( INST )


template<alloc Alloc, index LocalInd, index GlobalInd, scalar Scalar>
struct CSRConfig {
    static constexpr alloc allocator  = Alloc;
    static constexpr index lidx       = LocalInd;
    static constexpr index gidx       = GlobalInd;
    static constexpr scalar scalar_id = Scalar;

    static constexpr csr_mode mode =
        static_cast<csr_mode>( make_mode_v<allocator, lidx, gidx, scalar_id> );

    using lidx_t         = typename index_info<lidx>::type;
    using gidx_t         = typename index_info<gidx>::type;
    using scalar_t       = typename scalar_info<scalar_id>::type;
    using allocator_type = typename alloc_info<Alloc>::type;

    template<alloc new_alloc>
    struct set_alloc {
        using type = CSRConfig<new_alloc, LocalInd, GlobalInd, Scalar>;
    };
    template<alloc new_alloc>
    using set_alloc_t = typename set_alloc<new_alloc>::type;

    template<scalar new_scalar>
    struct set_scalar {
        using type = CSRConfig<Alloc, lidx, gidx, new_scalar>;
    };
    template<scalar new_scalar>
    using set_scalar_t = typename set_scalar<new_scalar>::type;

    template<index new_lidx>
    using set_lidx_t = CSRConfig<Alloc, new_lidx, gidx, scalar_id>;

    static void print()
    {
        AMP::pout << "Local Index: " << index_info<lidx>::name() << '\n'
                  << "Global Index: " << index_info<gidx>::name() << '\n'
                  << "Scalar: " << scalar_info<scalar_id>::name() << std::endl;
    }
};

// Helpers to statically determine a component mode from a csr_mode
template<csr_mode mode, class L>
struct get_mode_comp {
    static constexpr auto value = static_cast<typename L::type>(
        ( static_cast<std::uint16_t>( mode ) & L::mask ) >> L::shift );
};

template<csr_mode mode>
constexpr alloc get_alloc_v = get_mode_comp<mode, csr_mode_layout::alloc>::value;

template<csr_mode mode>
constexpr index get_lidx_v = get_mode_comp<mode, csr_mode_layout::lidx>::value;

template<csr_mode mode>
constexpr index get_gidx_v = get_mode_comp<mode, csr_mode_layout::gidx>::value;

template<csr_mode mode>
constexpr scalar get_scalar_v = get_mode_comp<mode, csr_mode_layout::scalar>::value;

// Helpers to dynamically determine a component mode from a csr_mode
template<class L>
auto get_from_mode( csr_mode mode )
{
    return static_cast<typename L::type>( ( static_cast<std::uint16_t>( mode ) & L::mask ) >>
                                          L::shift );
}
inline auto get_alloc( csr_mode mode ) { return get_from_mode<csr_mode_layout::alloc>( mode ); }
inline auto get_lidx( csr_mode mode ) { return get_from_mode<csr_mode_layout::lidx>( mode ); }
inline auto get_gidx( csr_mode mode ) { return get_from_mode<csr_mode_layout::gidx>( mode ); }
inline auto get_scalar( csr_mode mode ) { return get_from_mode<csr_mode_layout::scalar>( mode ); }

template<csr_mode mode>
struct config_mode {
    using type =
        CSRConfig<get_alloc_v<mode>, get_lidx_v<mode>, get_gidx_v<mode>, get_scalar_v<mode>>;
};
template<csr_mode mode>
using config_mode_t = typename config_mode<mode>::type;

namespace detail {
template<class... Cs>
struct config_list {
};
using built_configs_extra_first = config_list<config_mode_t<csr_mode::hiIf>
#define X( C ) , config_mode_t<C>
                                                  CSR_CONFIG_FORALL( X )
#undef X
                                              >;

template<class T>
struct ignore_first;
template<class Ignore, class... Cs>
struct ignore_first<config_list<Ignore, Cs...>> {
    using type = config_list<Cs...>;
};
using built_configs = typename ignore_first<built_configs_extra_first>::type;
} // namespace detail
using built_configs = detail::built_configs;

namespace detail {
template<class C, class L>
struct contains;
template<class C, class... Cs>
struct contains<C, config_list<Cs...>> {
    static constexpr bool value = ( std::is_same_v<C, Cs> || ... );
};
} // namespace detail

template<class C>
constexpr bool is_config_built = detail::contains<C, built_configs>::value;

inline bool is_built( csr_mode mode )
{
    bool built = false;
    switch ( mode ) {
#define X( MODE ) \
case MODE:        \
    built = true; \
    break;
        CSR_CONFIG_FORALL( X )
#undef X
    default:
        built = false;
    }
    return built;
}

#if defined( AMP_USE_HYPRE )
template<alloc Alloc>
using DefaultCSRConfig     = HypreConfig<Alloc>;
using DefaultHostCSRConfig = DefaultCSRConfig<alloc::host>;
#else
template<alloc Alloc>
using DefaultCSRConfig     = CSRConfig<Alloc, index::i32, index::i64, scalar::f64>;
using DefaultHostCSRConfig = DefaultCSRConfig<alloc::host>;
#endif

} // namespace AMP::LinearAlgebra
#endif
