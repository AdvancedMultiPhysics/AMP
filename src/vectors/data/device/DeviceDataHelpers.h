#ifndef included_AMP_DeviceDataHelpers_h
#define included_AMP_DeviceDataHelpers_h


namespace AMP {
namespace LinearAlgebra {


/**
 * \brief  A default set of helper functions for vector operations
 * \details DataHelpers impliments a default set of
 *    vector operations on the GPU.
 */
template<typename STYPE, typename DTYPE = STYPE>
class DeviceDataHelpers
{
public:
    //  functions that operate on VectorData

    static bool containsIndex( const size_t N, const size_t *indices, const size_t i );

    static bool
    allGhostIndices( const size_t N, const size_t *indices, const size_t start, const size_t end );

    static void setValuesByIndex( size_t N, const size_t *indices, const STYPE *src, DTYPE *dst );

    static void addValuesByIndex( size_t N, const size_t *indices, const STYPE *src, DTYPE *dst );

    static void getValuesByIndex( size_t N, const size_t *indices, const STYPE *src, DTYPE *dst );

    static void setGhostValuesByGlobalID( const size_t gsize,
                                          const size_t *globalIDs,
                                          const size_t N,
                                          const size_t *ndx,
                                          const STYPE *src,
                                          const size_t dst_size,
                                          DTYPE *dst );

    static void addGhostValuesByGlobalID( const size_t gsize,
                                          const size_t *globalIDs,
                                          const size_t N,
                                          const size_t *ndx,
                                          const STYPE *src,
                                          const size_t dst_size,
                                          DTYPE *dst );

    static void getGhostValuesByGlobalID( const size_t gsize,
                                          const size_t *globalIDs,
                                          const size_t N,
                                          const size_t *ndx,
                                          const size_t src_size,
                                          const STYPE *src1,
                                          const STYPE *src2,
                                          DTYPE *dst );

    static void getGhostAddValuesByGlobalID( const size_t gsize,
                                             const size_t *globalIDs,
                                             const size_t N,
                                             const size_t *ndx,
                                             const size_t src_size,
                                             const STYPE *src,
                                             DTYPE *dst );
};

} // namespace LinearAlgebra
} // namespace AMP


#endif
