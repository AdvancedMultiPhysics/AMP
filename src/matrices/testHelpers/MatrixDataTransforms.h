#ifndef included_MatrixDataTransforms_h
#define included_MatrixDataTransforms_h

#include "AMP/matrices/Matrix.h"

#include <numeric>
#include <type_traits>

namespace AMP::LinearAlgebra {

template<typename Policy>
void transformDofToCSR( std::shared_ptr<Matrix> matrix,
                        typename Policy::gidx_t &startRow,
                        typename Policy::gidx_t &endRow,
                        typename Policy::gidx_t &startCol,
                        typename Policy::gidx_t &endCol,
                        std::vector<typename Policy::lidx_t> &rs_d,
                        std::vector<typename Policy::gidx_t> &cols_d,
                        std::vector<typename Policy::scalar_t> &coeffs_d,
                        std::vector<typename Policy::lidx_t> &rs_od,
                        std::vector<typename Policy::gidx_t> &cols_od,
                        std::vector<typename Policy::scalar_t> &coeffs_od )
{
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    AMP_ASSERT( matrix );

    auto lDOF = matrix->getLeftDOFManager();
    auto rDOF = matrix->getRightDOFManager();

    startRow = static_cast<gidx_t>( rDOF->beginDOF() );
    endRow   = static_cast<gidx_t>( rDOF->endDOF() );
    startCol = static_cast<gidx_t>( lDOF->beginDOF() );
    endCol   = static_cast<gidx_t>( lDOF->endDOF() );

    // prime rowstart vectors with first zero entry
    rs_d.push_back( 0 );
    rs_od.push_back( 0 );

    // loop over rows and examine structure
    for ( auto row = startRow; row < endRow; ++row ) {

        std::vector<size_t> rcols;
        std::vector<scalar_t> rvals;

        matrix->getRowByGlobalID( row, rcols, rvals );

        // Loop over columns and insert into on and off diagonal blocks
        lidx_t nnzd = 0, nnzod = 0;
        for ( size_t i = 0; i < rcols.size(); ++i ) {
            const auto col = static_cast<gidx_t>( rcols[i] );
            if ( startCol <= col && col < endCol ) {
                ++nnzd;
                cols_d.push_back( col );
                coeffs_d.push_back( static_cast<scalar_t>( rvals[i] ) );
            } else {
                ++nnzod;
                cols_od.push_back( col );
                coeffs_od.push_back( static_cast<scalar_t>( rvals[i] ) );
            }
        }
        const lidx_t prev_rs_d  = rs_d.back();
        const lidx_t prev_rs_od = rs_od.back();
        rs_d.push_back( prev_rs_d + nnzd );
        rs_od.push_back( prev_rs_od + nnzod );
    }
}
} // namespace AMP::LinearAlgebra

#endif
