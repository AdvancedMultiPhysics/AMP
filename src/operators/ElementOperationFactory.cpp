#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Utilities.h"

#ifdef AMP_USE_LIBMESH
    #include "AMP/operators/diffusion/DiffusionElement.h"
    #include "AMP/operators/diffusion/DiffusionLinearElement.h"
    #include "AMP/operators/diffusion/DiffusionNonlinearElement.h"
    #include "AMP/operators/libmesh/MassLinearElement.h"
    #include "AMP/operators/libmesh/SourceNonlinearElement.h"
    #include "AMP/operators/mechanics/MechanicsElement.h"
    #include "AMP/operators/mechanics/MechanicsLinearElement.h"
    #include "AMP/operators/mechanics/MechanicsLinearUpdatedLagrangianElement.h"
    #include "AMP/operators/mechanics/MechanicsNonlinearElement.h"
    #include "AMP/operators/mechanics/MechanicsNonlinearUpdatedLagrangianElement.h"
#endif


#define resetElementOperation( NAME )                 \
    do {                                              \
        if ( name == #NAME )                          \
            retElementOp.reset( new NAME( params ) ); \
    } while ( 0 )


namespace AMP::Operator {


std::shared_ptr<ElementOperation>
ElementOperationFactory::createElementOperation( std::shared_ptr<Database> elementOperationDb )
{
    std::shared_ptr<ElementOperation> retElementOp;
    std::shared_ptr<ElementOperationParameters> params;

    AMP_INSIST( elementOperationDb,
                "ElementOperationFactory::createElementOperation:: NULL Database object input" );

    [[maybe_unused]] auto name = elementOperationDb->getString( "name" );

    params.reset( new ElementOperationParameters( elementOperationDb ) );

#ifdef AMP_USE_LIBMESH
    resetElementOperation( MechanicsLinearElement );
    resetElementOperation( MechanicsNonlinearElement );
    resetElementOperation( MechanicsLinearUpdatedLagrangianElement );
    resetElementOperation( MechanicsNonlinearUpdatedLagrangianElement );
    resetElementOperation( DiffusionLinearElement );
    resetElementOperation( DiffusionNonlinearElement );
    resetElementOperation( MassLinearElement );
    resetElementOperation( SourceNonlinearElement );
#endif

    return retElementOp;
}


} // namespace AMP::Operator
