#ifndef included_AMP_TpetraVectorFactor
#define included_AMP_TpetraVectorFactor


#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/testHelpers/VectorTests.h"

#include "Tpetra_Vector_decl.hpp"
/// \cond UNDOCUMENTED


namespace AMP::LinearAlgebra {


template<typename ST = double,
         typename LO = int32_t,
         typename GO = long long,
         typename NT = Tpetra::Vector<>::node_type>
class NativeTpetraFactory : public VectorFactory
{
public:
    NativeTpetraFactory() {}
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        const int nLocal = 210;
        AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
        const int start   = nLocal * globalComm.getRank();
        const int nGlobal = nLocal * globalComm.getSize();
        auto commList     = std::make_shared<CommunicationList>( nLocal, globalComm );
        auto dofManager   = std::make_shared<AMP::Discretization::DOFManager>( nLocal, globalComm );
        auto buffer =
            std::make_shared<AMP::LinearAlgebra::VectorDataDefault<ST>>( start, nLocal, nGlobal );
        auto vec = createTpetraVector( commList, dofManager, buffer );
        return vec;
    }
    std::string name() const override { return "NativeTpetraFactory"; }
};


} // namespace AMP::LinearAlgebra

/// \endcond

#endif
