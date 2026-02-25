#ifndef included_MatrixData_HPP_
#define included_MatrixData_HPP_

#include "AMP/utils/typeid.h"

namespace AMP::LinearAlgebra {

template<typename TYPE>
void MatrixData::addValuesByGlobalID(
    size_t num_rows, size_t num_cols, const size_t *rows, const size_t *cols, const TYPE *values )
{
    constexpr auto type = getTypeID<TYPE>();
    addValuesByGlobalID( num_rows, num_cols, rows, cols, values, type );
}
template<typename TYPE>
void MatrixData::setValuesByGlobalID(
    size_t num_rows, size_t num_cols, const size_t *rows, const size_t *cols, const TYPE *values )
{
    constexpr auto type = getTypeID<TYPE>();
    setValuesByGlobalID( num_rows, num_cols, rows, cols, values, type );
}
template<typename TYPE>
void MatrixData::getValuesByGlobalID(
    size_t num_rows, size_t num_cols, const size_t *rows, const size_t *cols, TYPE *values ) const
{
    constexpr auto type = getTypeID<TYPE>();
    getValuesByGlobalID( num_rows, num_cols, rows, cols, values, type );
}

} // namespace AMP::LinearAlgebra
#endif
