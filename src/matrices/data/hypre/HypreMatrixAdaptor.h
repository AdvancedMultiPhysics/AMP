#ifndef INCLUDED_HYPRE_MATRIX_ADAPTOR_H
#define INCLUDED_HYPRE_MATRIX_ADAPTOR_H

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/data/MatrixData.h"

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"

#include <vector>

namespace AMP::LinearAlgebra {

/** \class HypreMatrixAdaptor
 * \brief  Wrapper class for constructing a hypre IJ matrix for use with solvers
 * \details It is initialized based on CSR data.
 */
class HypreMatrixAdaptor
{

public:
    HypreMatrixAdaptor() = delete;
    ~HypreMatrixAdaptor();

    //!! Main constructor using a MatrixData object
    HypreMatrixAdaptor( std::shared_ptr<MatrixData> );

    //! Returns the handle to the HYPRE IJMatrix
    HYPRE_IJMatrix getHypreMatrix( void ) { return d_matrix; }

private:
    //! Main internal routine for initializing the matrix
    template<class Config>
    void initializeHypreMatrix( std::shared_ptr<CSRMatrixData<Config>> csrData );

    //! hypre IJ matrix that this class wraps
    HYPRE_IJMatrix d_matrix;

    //! Storage for migrated matrix data
    std::shared_ptr<MatrixData> d_csrdata_migrated;
};

} // namespace AMP::LinearAlgebra

#endif // INCLUDED_HYPRE_MATRIX_ADAPTOR_H
