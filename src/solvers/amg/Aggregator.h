#ifndef included_AMP_Aggregator_H_
#define included_AMP_Aggregator_H_

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixData.h"

#include <memory>

namespace AMP::Solver::AMG {

// Base class for all aggregators
struct Aggregator {
    Aggregator() : d_weak_thresh( 50.0 ) {}
    Aggregator( float wt_ ) : d_weak_thresh( wt_ ) {}

    virtual ~Aggregator() {}

    // This function must be supplied by each specific aggregator implementation
    virtual int assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A, int *agg_ids ) = 0;

    // Invoke aggregator and return in the form of a tentative prolongator
    std::shared_ptr<LinearAlgebra::Matrix>
    getAggregateMatrix( std::shared_ptr<LinearAlgebra::Matrix> A,
                        std::shared_ptr<LinearAlgebra::MatrixParameters> matParams = {} );

    // Produce non-type erased matrix for above tentative prolongator
    template<typename Config>
    std::shared_ptr<LinearAlgebra::Matrix>
    getAggregateMatrix( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                        std::shared_ptr<LinearAlgebra::MatrixParameters> matParams = {} );

    const float d_weak_thresh;
};

} // namespace AMP::Solver::AMG

#endif
