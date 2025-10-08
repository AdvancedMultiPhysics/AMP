#ifndef included_TpetraDefaults_H
#define included_TpetraDefaults_H

#include <Tpetra_Core.hpp>

#if defined( HAVE_TPETRA_INST_DOUBLE )
using Tpetra_ST = double;
#elif defined( HAVE_TPETRA_INST_FLOAT )
using Tpetra_ST = float;
#elif defined( HAVE_TPETRA_INST_LONG_DOUBLE )
using Tpetra_ST = long double;
#else
    #error "Tpetra not configured for given scalar type"
#endif

#if defined( HAVE_TPETRA_INST_INT_INT )
using Tpetra_LO = int32_t;
using Tpetra_GO = int32_t;
#elif defined( HAVE_TPETRA_INST_INT_UNSIGNED )
using Tpetra_LO = int32_t;
using Tpetra_GO = uint32_t;
#elif defined( HAVE_TPETRA_INST_INT_LONG )
using Tpetra_LO = int32_t;
using Tpetra_GO = long;
#elif defined( HAVE_TPETRA_INST_INT_LONG_LONG )
using Tpetra_LO = int32_t;
using Tpetra_GO = long long;
#else
    #error "Tpetra not configured for given local and global ordinal types"
#endif

#endif
