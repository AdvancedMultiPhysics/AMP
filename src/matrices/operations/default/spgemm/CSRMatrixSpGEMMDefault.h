#ifndef included_AMP_CSRMatrixSpGEMMDefault
#define included_AMP_CSRMatrixSpGEMMDefault

#include "AMP/matrices/data/CSRMatrixCommunicator.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/spgemm/CSRMatrixSpGEMMCommon.h"

#include <memory>
#include <vector>

namespace AMP::LinearAlgebra {

template<typename Config>
class CSRMatrixSpGEMMDefault : public CSRMatrixSpGEMMCommon<Config>
{
public:
    using allocator_type    = typename Config::allocator_type;
    using matrixdata_t      = CSRMatrixData<Config>;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;
    using lidx_t            = typename Config::lidx_t;
    using gidx_t            = typename Config::gidx_t;
    using scalar_t          = typename Config::scalar_t;

    CSRMatrixSpGEMMDefault() = default;
    CSRMatrixSpGEMMDefault( std::shared_ptr<matrixdata_t> A_,
                            std::shared_ptr<matrixdata_t> B_,
                            std::shared_ptr<matrixdata_t> C_ )
        : CSRMatrixSpGEMMCommon<Config>( A_, B_, C_ )
    {
    }

    ~CSRMatrixSpGEMMDefault() = default;

    virtual void multiplyLocal( std::shared_ptr<localmatrixdata_t> A_data,
                                std::shared_ptr<localmatrixdata_t> B_data,
                                std::shared_ptr<localmatrixdata_t> C_data ) override;

    enum class Mode { SYMBOLIC, NUMERIC };
    enum class BlockType { DIAG, OFFD };

    template<Mode mode_t, BlockType block_t>
    void multiplyBlock( std::shared_ptr<localmatrixdata_t> A_data,
                        std::shared_ptr<localmatrixdata_t> B_data,
                        std::shared_ptr<localmatrixdata_t> C_data );

protected:
    // default starting size for sparse accumulators
    static constexpr lidx_t SPACC_SIZE = 256;

    // Internal row accumlator classes
    template<typename col_t>
    struct DenseAccumulator {
        DenseAccumulator( int capacity_, gidx_t offset_ )
            : capacity( capacity_ ),
              offset( offset_ ),
              num_inserted( 0 ),
              total_inserted( 0 ),
              total_collisions( 0 ),
              total_probe_steps( 0 ),
              total_clears( 0 ),
              total_grows( 0 ),
              flags( capacity, -1 )
        {
            static_assert( std::is_same_v<col_t, gidx_t> || std::is_same_v<col_t, lidx_t> );
        }

        void insert_or_append( col_t col_idx );
        void insert_or_append( col_t col_idx, scalar_t val, col_t *col_space, scalar_t *val_space );
        void clear();
        lidx_t contains( col_t col_idx ) const;
        void set_flag( col_t col_idx, lidx_t k );

        static constexpr bool IsGlobal = std::is_same_v<gidx_t, col_t>;

        const lidx_t capacity;
        const col_t offset;
        lidx_t num_inserted;
        size_t total_inserted;
        size_t total_collisions;
        size_t total_probe_steps;
        size_t total_clears;
        size_t total_grows;
        std::vector<lidx_t> flags;
        std::vector<lidx_t> flag_inv;
        std::vector<col_t> cols;
    };

    template<typename col_t>
    struct SparseAccumulator {
        SparseAccumulator( int capacity_, gidx_t offset_ )
            : capacity( capacity_ ),
              offset( offset_ ),
              num_inserted( 0 ),
              total_inserted( 0 ),
              total_collisions( 0 ),
              total_probe_steps( 0 ),
              total_clears( 0 ),
              total_grows( 0 ),
              flags( capacity, 0xFFFF )
        {
            AMP_DEBUG_ASSERT( capacity > 1 );
            static_assert( std::is_same_v<col_t, gidx_t> || std::is_same_v<col_t, lidx_t> );
        }

        uint16_t hash( col_t col_idx ) const;
        void insert_or_append( col_t col_idx );
        void insert_or_append( col_t col_idx, scalar_t val, col_t *col_space, scalar_t *val_space );
        void clear();
        lidx_t contains( col_t col_idx ) const;
        void set_flag( col_t col_idx, lidx_t k );

        static constexpr bool IsGlobal = std::is_same_v<gidx_t, col_t>;

        uint16_t capacity;
        const gidx_t offset;
        uint16_t num_inserted;
        size_t total_inserted;
        size_t total_collisions;
        size_t total_probe_steps;
        size_t total_clears;
        size_t total_grows;
        std::vector<uint16_t> flags;
        std::vector<col_t> cols;

    private:
        void grow( col_t *col_space );
    };
};

} // namespace AMP::LinearAlgebra

#endif
