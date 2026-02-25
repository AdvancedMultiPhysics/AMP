
#ifndef included_AMP_WeldOperator
#define included_AMP_WeldOperator

#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"

#include <memory>

#include "AMP/operators/Operator.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include <string>

#ifdef DEBUG_CHECK_ASSERTIONS

#endif

namespace AMP::Operator {

class WeldOperator : public Operator
{

public:
    WeldOperator( std::shared_ptr<const OperatorParameters> params ) : Operator( params ) {}

    virtual ~WeldOperator() {}

    std::string type() const override { return "WeldOperator"; }

    virtual void reset( std::shared_ptr<const OperatorParameters> params ) { (void) params; }

    virtual void apply( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                        AMP::LinearAlgebra::Vector::const_shared_ptr u,
                        AMP::LinearAlgebra::Vector::shared_ptr r,
                        const double a = -1.0,
                        const double b = 1.0 )
    {
        auto inVec = u->subsetVectorForVariable( d_inpVar );

        auto dof_map = d_MeshAdapter->getDOFMap( d_inpVar );

        auto bnd     = d_MeshAdapter->beginOwnedBoundary( d_inputBoundaryId );
        auto end_bnd = d_MeshAdapter->endOwnedBoundary( d_inputBoundaryId );

        double val = 0.;
        int cnt    = 0;
        for ( ; bnd != end_bnd; ++bnd ) {
            std::vector<unsigned int> bndGlobalIds;
            std::vector<unsigned int> singleton( 1 );
            singleton[0] = 2;
            dof_map->getDOFs( *bnd, bndGlobalIds, singleton );
            val = inVec->getLocalValueByGlobalID( bndGlobalIds[0] );
            cnt++;
        } // end for bnd
        AMP_ASSERT( ( cnt == 0 ) or ( cnt == 1 ) );
        // d_comm is constructed so that rank 0 has the input boundary
        //          MPI_Bcast(&val, 1, MPI_DOUBLE, 0, d_comm);
        val = d_comm.bcast( val, 0 );
        d_outVec->setToScalar( val );
    }

    unsigned int d_inputBoundaryId;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVar;
    AMP::LinearAlgebra::Vector::shared_ptr d_outVec;
    AMP::AMP_MPI d_comm;
};
} // namespace AMP::Operator

#endif
