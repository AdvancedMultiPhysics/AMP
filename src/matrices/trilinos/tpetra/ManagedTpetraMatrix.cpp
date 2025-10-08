#include "AMP/matrices/trilinos/tpetra/ManagedTpetraMatrix.hpp"

namespace AMP::LinearAlgebra {

#if defined( HAVE_TPETRA_INST_DOUBLE )
using ST = double;
#elif defined( HAVE_TPETRA_INST_FLOAT )
using ST = float;
#elif defined( HAVE_TPETRA_INST_LONG_DOUBLE )
using ST = long double;
#else
    #error "Tpetra not configured for given scalar type"
#endif

#if defined( HAVE_TPETRA_INST_INT_INT )
using LO = int32_t;
using GO = int32_t;
#elif defined( HAVE_TPETRA_INST_INT_UNSIGNED )
using LO = int32_t;
using GO = uint32_t;
#elif defined( HAVE_TPETRA_INST_INT_LONG )
using LO = int32_t;
using GO = long;
#elif defined( HAVE_TPETRA_INST_INT_LONG_LONG )
using LO = int32_t;
using GO = long long;
#else
    #error "Tpetra not configured for given local and global ordinal types"
#endif

template class ManagedTpetraMatrix<ST, LO, GO>;

} // namespace AMP::LinearAlgebra
