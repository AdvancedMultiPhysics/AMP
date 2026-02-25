#include "AMP/matrices/operations/default/MatrixOperationsDefault.h"
#include "AMP/matrices/data/MatrixData.h"

#include <algorithm>

namespace AMP::LinearAlgebra {

void MatrixOperationsDefault::matMatMult( MatrixData const &, MatrixData const &, MatrixData & )
{
    AMP_ERROR( "matMatMult not implemented for matrices of different types" );
}

void MatrixOperationsDefault::axpy( AMP::Scalar alpha_in, const MatrixData &X, MatrixData &Y )
{
    AMP_ASSERT( X.beginRow() == Y.beginRow() && X.endRow() == Y.endRow() );
    AMP_ASSERT( X.beginCol() == Y.beginCol() && X.endCol() == Y.endCol() );
    AMP_ASSERT( X.numLocalRows() == Y.numLocalRows() );
    AMP_ASSERT( X.numGlobalRows() == Y.numGlobalRows() );
    AMP_ASSERT( X.numLocalColumns() == Y.numLocalColumns() );
    AMP_ASSERT( X.numGlobalColumns() == Y.numGlobalColumns() );

    auto alpha = static_cast<double>( alpha_in );
    std::vector<size_t> xcols;
    std::vector<double> xvals;
    auto beginRow = X.beginRow();
    auto endRow   = X.endRow();
    for ( auto i = beginRow; i < endRow; ++i ) {
        X.getRowByGlobalID( i, xcols, xvals );
        std::transform(
            xvals.begin(), xvals.end(), xvals.begin(), [&alpha]( auto &c ) { return c * alpha; } );
        Y.addValuesByGlobalID( 1, xcols.size(), &i, xcols.data(), xvals.data() );
    }
}

void MatrixOperationsDefault::copy( const MatrixData &X, MatrixData &Y )
{
    AMP_ASSERT( X.beginRow() == Y.beginRow() && X.endRow() == Y.endRow() );
    AMP_ASSERT( X.beginCol() == Y.beginCol() && X.endCol() == Y.endCol() );
    AMP_ASSERT( X.numLocalRows() == Y.numLocalRows() );
    AMP_ASSERT( X.numGlobalRows() == Y.numGlobalRows() );
    AMP_ASSERT( X.numLocalColumns() == Y.numLocalColumns() );
    AMP_ASSERT( X.numGlobalColumns() == Y.numGlobalColumns() );

    std::vector<size_t> xcols;
    std::vector<double> xvals;
    auto beginRow = X.beginRow();
    auto endRow   = X.endRow();
    for ( auto i = beginRow; i < endRow; ++i ) {
        X.getRowByGlobalID( i, xcols, xvals );
        Y.setValuesByGlobalID( 1, xcols.size(), &i, xcols.data(), xvals.data() );
    }
}

} // namespace AMP::LinearAlgebra
