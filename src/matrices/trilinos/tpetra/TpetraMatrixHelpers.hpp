#include "AMP/matrices/trilinos/tpetra/TpetraMatrixHelpers.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/trilinos/tpetra/TpetraMatrixData.h"
#include "AMP/matrices/trilinos/tpetra/ManagedTpetraMatrix.h"

#include <algorithm>
#include <functional>

namespace AMP::LinearAlgebra {

/********************************************************
 * Get an Tpetra matrix from an AMP matrix              *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<ManagedTpetraMatrix<ST, LO, GO, NT>> getTpetraMatrix( std::shared_ptr<Matrix> mat )
{
    AMP_ASSERT( mat );
    if ( mat->type() == "ManagedTpetraMatrix" ) {
        return std::dynamic_pointer_cast<ManagedTpetraMatrix<ST, LO, GO, NT>>( mat );
    } else {
        // Wrap the input matrix's getRowByGlobalID function into a new getRow function
        // This is necessary in the event that the DOFManagers of the input matrix are
        // of the base type (e.g. getElement is not defined)
        auto getRow = [mat]( size_t row ) -> std::vector<size_t> {
            std::vector<size_t> cols;
            std::vector<double> vals;
            mat->getRowByGlobalID( row, cols, vals );
            return cols;
        };

        // This approach of making a whole new TpetraMatrix is inefficient
        // -> should consider deprecating, but likely can't if ML still used...
        auto matParams =
            std::make_shared<MatrixParameters>( mat->getLeftDOFManager(),
                                                mat->getRightDOFManager(),
                                                mat->getComm(),
                                                mat->getMatrixData()->getLeftVariable(),
                                                mat->getMatrixData()->getRightVariable(),
                                                getRow );
        auto tpetraMat = std::make_shared<ManagedTpetraMatrix<ST, LO, GO, NT>>( matParams );
        tpetraMat->copy( mat );
        return tpetraMat;
    }
}


} // namespace AMP::LinearAlgebra
