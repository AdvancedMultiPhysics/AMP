#ifndef included_AMP_Matrix_GetRowHelper
#define included_AMP_Matrix_GetRowHelper

#include "AMP/discretization/DOF_Manager.h"

#include <array>
#include <memory>
#include <vector>


namespace AMP::LinearAlgebra {


class GetRowHelper final
{
public:
    /** \brief Construct GetRowHelper
     * \details  This will construct the GetRowHelper from the left and right DOFManager
     * \param[in]  leftDOF      The left DOFManager
     * \param[in]  rightDOF     The right DOFManager
     */
    GetRowHelper( std::shared_ptr<const AMP::Discretization::DOFManager> leftDOF,
                  std::shared_ptr<const AMP::Discretization::DOFManager> rightDOF );

    //! Destructor
    ~GetRowHelper();

    // Copy/assignment operators
    GetRowHelper()                                  = default;
    GetRowHelper( GetRowHelper && )                 = default;
    GetRowHelper( const GetRowHelper & )            = delete;
    GetRowHelper &operator=( GetRowHelper && )      = default;
    GetRowHelper &operator=( const GetRowHelper & ) = delete;

    //! Release all internal storage
    void deallocate();

    /** \brief  Get the number of non-zeros
     * \details  This will return the number of non-zeros for the row as [local,remote]
     * \param[in]  row          The row
     */
    std::array<size_t, 2> NNZ( size_t row ) const;

    /** \brief  Get the number of non-zeros
     * \details  This will return the number of non-zeros for the row
     * \param[in]  row          The row
     * \param[out] N_local      The number of local non-zeros
     * \param[out] N_remote     The number of remote non-zeros
     */
    template<class INT>
    void NNZ( size_t row, INT &N_local, INT &N_remote ) const;

    /** \brief  Get the row
     * \details  This will return the local and remote non-zero entries for the row
     * \param[in]  row          The row of interest
     * \param[out] local        The local non-zero entries (may be null)
     * \param[out] remote       The remote non-zero entries (may be null)
     */
    template<class INT>
    void getRow( size_t row, INT *local, INT *remote ) const;

    const size_t *getLocals() const { return d_local; }

    const size_t *getRemotes() const { return d_remote; }

private: // Private routines
    std::array<size_t *, 2> getRow2( size_t row ) const;
    void reserve( size_t N );


private: // Member data
    bool d_hasFields             = true;
    std::array<size_t, 2> *d_NNZ = nullptr;
    size_t *d_local              = nullptr;
    size_t *d_remote             = nullptr;
    size_t *d_localOffset        = nullptr;
    size_t *d_remoteOffset       = nullptr;
    size_t d_size[2]             = { 0, 0 };
    size_t d_capacity[2]         = { 0, 0 };
    std::shared_ptr<const AMP::Discretization::DOFManager> d_leftDOF;
    std::shared_ptr<const AMP::Discretization::DOFManager> d_rightDOF;
};


} // namespace AMP::LinearAlgebra

#include "AMP/matrices/GetRowHelper.hpp"

#endif
