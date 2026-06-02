#ifndef included_AMP_CSRMatrixSpGEMMKokkos
#define included_AMP_CSRMatrixSpGEMMKokkos

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/data/CSRMatrixCommunicator.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/spgemm/CSRMatrixSpGEMMCommon.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Memory.h"

#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_StaticCrsGraph.hpp>
#include <KokkosSparse_spgemm_numeric.hpp>
#include <KokkosSparse_spgemm_symbolic.hpp>
#include <Kokkos_Core.hpp>

#include <memory>
#include <tuple>

namespace AMP::LinearAlgebra {

template<typename Config, class ExecSpace>
class CSRMatrixSpGEMMKokkos : public CSRMatrixSpGEMMCommon<Config>
{
public:
    using allocator_type    = typename Config::allocator_type;
    using matrixdata_t      = CSRMatrixData<Config>;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;
    using lidx_t            = typename Config::lidx_t;
    using gidx_t            = typename Config::gidx_t;
    using scalar_t          = typename Config::scalar_t;
    using ViewSpace         = typename ExecSpace::memory_space;

    using handle_t = typename KokkosKernels::Experimental::
        KokkosKernelsHandle<lidx_t, lidx_t, scalar_t, ExecSpace, ViewSpace, ViewSpace>;
    using device_t = typename Kokkos::Device<ExecSpace, ViewSpace>;
    using matrix_t = typename KokkosSparse::
        CrsMatrix<scalar_t, lidx_t, device_t, Kokkos::MemoryTraits<Kokkos::Unmanaged>, lidx_t>;
    using graph_t   = typename matrix_t::staticcrsgraph_type;
    using rowmap_t  = typename graph_t::row_map_type::non_const_type;
    using entries_t = typename graph_t::entries_type::non_const_type;
    using values_t  = typename matrix_t::values_type::non_const_type;

    CSRMatrixSpGEMMKokkos() = default;
    CSRMatrixSpGEMMKokkos( std::shared_ptr<matrixdata_t> A_,
                           std::shared_ptr<matrixdata_t> B_,
                           std::shared_ptr<matrixdata_t> C_ )
        : CSRMatrixSpGEMMCommon<Config>( A_, B_, C_ )
    {
    }

    ~CSRMatrixSpGEMMKokkos() = default;

    virtual void multiplyLocal( std::shared_ptr<localmatrixdata_t> A_data,
                                std::shared_ptr<localmatrixdata_t> B_data,
                                std::shared_ptr<localmatrixdata_t> C_data ) override;

    static std::tuple<rowmap_t, entries_t, values_t>
    wrapDataFields( std::shared_ptr<localmatrixdata_t> mat )
    {
        auto [rs, cols, cols_loc, coeffs] = mat->getDataFields();
        const auto nrows                  = mat->numLocalRows();
        const auto nnz                    = mat->numberOfNonZeros();
        rowmap_t rm( rs, nrows + 1 );
        entries_t ent( cols_loc, nnz );
        values_t val( coeffs, nnz );
        return std::make_tuple( rm, ent, val );
    }
};

} // namespace AMP::LinearAlgebra

#endif
