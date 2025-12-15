#include "AMP/matrices/trilinos/EpetraMatrixHelpers.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/trilinos/EpetraMatrixData.h"
#include "AMP/matrices/trilinos/ManagedEpetraMatrix.h"

#include <algorithm>
#include <functional>

namespace AMP::LinearAlgebra {

/********************************************************
 * Get an Epetra matrix from an AMP matrix              *
 ********************************************************/
std::shared_ptr<ManagedEpetraMatrix> getEpetraMatrix( std::shared_ptr<Matrix> mat )
{
    AMP_ASSERT( mat );
    if ( mat->type() == "ManagedEpetraMatrix" ) {
        return std::dynamic_pointer_cast<ManagedEpetraMatrix>( mat );
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

        // This approach of making a whole new EpetraMatrix is inefficient
        // -> should consider deprecating, but likely can't if ML still used...
        auto matParams =
            std::make_shared<MatrixParameters>( mat->getLeftDOFManager(),
                                                mat->getRightDOFManager(),
                                                mat->getComm(),
                                                mat->getMatrixData()->getLeftVariable(),
                                                mat->getMatrixData()->getRightVariable(),
                                                getRow );
        auto epetraMat = std::make_shared<ManagedEpetraMatrix>( matParams );
        epetraMat->copy( mat );
        return epetraMat;
    }
}


} // namespace AMP::LinearAlgebra
