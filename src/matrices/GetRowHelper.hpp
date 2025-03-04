#ifndef included_AMP_Matrix_GetRowHelper_hpp
#define included_AMP_Matrix_GetRowHelper_hpp

#include "AMP/matrices/GetRowHelper.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * Get rows / size of rows                                       *
 ****************************************************************/
template<class INT>
void GetRowHelper::NNZ( size_t row, INT &N_local, INT &N_remote ) const
{
    auto N   = NNZ( row );
    N_local  = N[0];
    N_remote = N[1];
}
template<class INT>
void GetRowHelper::getRow( INT row, INT *local, INT *remote ) const
{
    auto [N_local, N_remote] = NNZ( row );
    auto [p_local, p_remote] = getRow2( row );
    if ( local ) {
        for ( size_t i = 0; i < N_local; i++ )
            local[i] = static_cast<INT>( p_local[i] );
    }
    if ( remote ) {
        for ( size_t i = 0; i < N_remote; i++ )
            remote[i] = static_cast<INT>( p_remote[i] );
    }
}


} // namespace AMP::LinearAlgebra

#endif
