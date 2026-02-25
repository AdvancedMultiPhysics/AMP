#ifndef included_AMP_SpGEMM_Hip
#define included_AMP_SpGEMM_Hip

#include "AMP/utils/hip/Helper_Hip.h"

#include <rocsparse/rocsparse.h>

#include <cstdint>
#include <type_traits>

namespace AMP::LinearAlgebra {

// This class wraps the operations in the rocsparse-spgemm example found at
// https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html#rocsparse-spgemm
// while maintaining the rocsparse handle and internal temp buffer allocation.
// Allocations for the result matrix are done by the caller of this class, not internally here.
template<typename rowidx_t, typename colidx_t, typename scalar_t>
class VendorSpGEMM
{
    // Only signed int 32's and 64's are supported for index types
    // Only floats and doubles supported for value types
    static_assert( std::is_same_v<rowidx_t, int> || std::is_same_v<rowidx_t, long long> );
    static_assert( std::is_same_v<colidx_t, int> || std::is_same_v<colidx_t, long long> );
    static_assert( std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double> );


    // 64 bit row pointers with 32 bit column indices is invalid
    // TODO: cusparse does *not* allow mixed types for these currently,
    //       and for consistency we disallow it here in the rocsparse side too
    //       add this assert and remove the prior one if that ever changes
    // static_assert( std::is_same_v<rowidx_t, int> ||
    //                (std::is_same_v<rowidx_t, long long> && std::is_same_v<colidx_t, long long>)
    //                );
    static_assert( std::is_same_v<rowidx_t, colidx_t> );

public:
    VendorSpGEMM( const int64_t M_,
                  const int64_t N_,
                  const int64_t K_,
                  const int64_t A_nnz,
                  rowidx_t *A_rs,
                  colidx_t *A_cols,
                  scalar_t *A_vals,
                  const int64_t B_nnz,
                  rowidx_t *B_rs,
                  colidx_t *B_cols,
                  scalar_t *B_vals,
                  rowidx_t *C_rs );

    ~VendorSpGEMM();

    int64_t getCnnz();

    void compute( rowidx_t *C_rs, colidx_t *C_cols, scalar_t *C_vals );

private:
    const int64_t M;
    const int64_t N;
    const int64_t K;

    scalar_t alpha;
    scalar_t beta;

    rocsparse_indextype itype;
    rocsparse_indextype jtype;
    rocsparse_datatype ttype;

    rocsparse_handle handle;

    rocsparse_spmat_descr matA;
    rocsparse_spmat_descr matB;
    rocsparse_spmat_descr matC;
    rocsparse_spmat_descr matD;

    size_t buffer_size;
    void *temp_buffer;
};

} // namespace AMP::LinearAlgebra

#endif
