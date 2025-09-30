#include "AMP/AMP_TPLs.h"


#if defined( HYPRE_SINGLE )
inline constexpr scalar hypre_real = scalar::f32;
#elif defined( HYPRE_LONG_DOUBLE )
inline constexpr scalar hypre_real = scalar::fld;
#else
inline constexpr scalar hypre_real = scalar::f64;
#endif // FP precision

#if defined( HYPRE_BIGINT )
inline constexpr index hypre_small = index::ill;
inline constexpr index hypre_big   = index::ill;
    #define CSR_CONFIG_FORALL_HYPRE_HOST( INST ) \
        INST( csr_mode::hllf )                   \
        INST( csr_mode::hlld )
    #define CSR_CONFIG_CC_FORALL_HYPRE_HOST( F, G ) \
        F( G, csr_mode::hllf )                      \
        F( G, csr_mode::hlld )
#elif defined( HYPRE_MIXEDINT )
inline constexpr index hypre_small = index::i32;
inline constexpr index hypre_big   = index::ill;
    #define CSR_CONFIG_FORALL_HYPRE_HOST( INST ) \
        INST( csr_mode::hilf )                   \
        INST( csr_mode::hild )
    #define CSR_CONFIG_CC_FORALL_HYPRE_HOST( F, G ) \
        F( G, csr_mode::hilf )                      \
        F( G, csr_mode::hild )
#else
inline constexpr index hypre_small = index::i32;
inline constexpr index hypre_big   = index::i32;
    #define CSR_CONFIG_FORALL_HYPRE_HOST( INST ) \
        INST( csr_mode::hiif )                   \
        INST( csr_mode::hiid )
    #define CSR_CONFIG_CC_FORALL_HYPRE_HOST( F, G ) \
        F( G, csr_mode::hiif )                      \
        F( G, csr_mode::hiid )
#endif // Integer config

#ifdef AMP_USE_DEVICE
    #if defined( HYPRE_BIGINT )
        #define CSR_CONFIG_FORALL_HYPRE_DEVICE( INST ) \
            INST( csr_mode::dllf )                     \
            INST( csr_mode::dlld )                     \
            INST( csr_mode::mllf )                     \
            INST( csr_mode::mlld )
        #define CSR_CONFIG_CC_FORALL_HYPRE_DEVICE( F, G ) \
            F( G, csr_mode::dllf )                        \
            F( G, csr_mode::dlld )                        \
            F( G, csr_mode::mllf )                        \
            F( G, csr_mode::mlld )
    #elif defined( HYPRE_MIXEDINT )
        #define CSR_CONFIG_FORALL_HYPRE_DEVICE( INST ) \
            INST( csr_mode::dilf )                     \
            INST( csr_mode::dild )                     \
            INST( csr_mode::milf )                     \
            INST( csr_mode::mild )
        #define CSR_CONFIG_CC_FORALL_HYPRE_DEVICE( F, G ) \
            F( G, csr_mode::dilf )                        \
            F( G, csr_mode::dild )                        \
            F( G, csr_mode::milf )                        \
            F( G, csr_mode::mild )
    #else
        #define CSR_CONFIG_FORALL_HYPRE_DEVICE( INST ) \
            INST( csr_mode::diif )                     \
            INST( csr_mode::diid )                     \
            INST( csr_mode::miif )                     \
            INST( csr_mode::miid )
        #define CSR_CONFIG_CC_FORALL_HYPRE_DEVICE( F, G ) \
            F( G, csr_mode::diif )                        \
            F( G, csr_mode::diid )                        \
            F( G, csr_mode::miif )                        \
            F( G, csr_mode::miid )
    #endif // Integer config
#else
    #define CSR_CONFIG_FORALL_HYPRE_DEVICE( INST )
    #define CSR_CONFIG_CC_FORALL_HYPRE_DEVICE( F, G )
#endif

#define CSR_CONFIG_CC_FORALL0_HYPRE( F, G ) \
    CSR_CONFIG_CC_FORALL_HYPRE_HOST( F, G ) \
    CSR_CONFIG_CC_FORALL_HYPRE_DEVICE( F, G )
#define CSR_CONFIG_CC_FORALL1_HYPRE() CSR_CONFIG_CC_FORALL0_HYPRE
#define CSR_CONFIG_CC_FORALL2_HYPRE( F, G ) DEFER( CSR_CONFIG_CC_FORALL1_HYPRE )()( F, G )
#define CSR_CONFIG_CC_FORALL_HYPRE( INST ) \
    EXPAND( CSR_CONFIG_CC_FORALL0_HYPRE( CSR_CONFIG_CC_FORALL2_HYPRE, INST ) )


#define CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HYPRE_HOST( INST ) \
    CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::hilf, INST )   \
    CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::hild, INST )
#if defined( AMP_USE_DEVICE )
    #define CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HYPRE_DEVICE( INST ) \
        CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::milf, INST )     \
        CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::mild, INST )     \
        CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::dilf, INST )     \
        CSR_INOUT_CONFIG_MIGRATE_LOOP( csr_mode::dild, INST )
#else
    #define CSR_INOUT_CONFIG_MIGRATE_OUTLIST_HYPRE_DEVICE( INST )
#endif

template<alloc, index, index, scalar>
struct CSRConfig;
template<alloc Alloc>
using HypreConfig = CSRConfig<Alloc, hypre_small, hypre_big, hypre_real>;
