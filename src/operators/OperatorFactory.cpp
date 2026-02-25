#include "AMP/operators/OperatorFactory.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/CoupledOperator.h"
#include "AMP/operators/IdentityOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/ParameterFactory.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/boundary/MassMatrixCorrection.h"
#ifdef AMP_USE_LIBMESH
    #include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
    #include "AMP/operators/boundary/libmesh/PressureBoundaryOperator.h"
    #include "AMP/operators/boundary/libmesh/RobinMatrixCorrection.h"
    #include "AMP/operators/boundary/libmesh/RobinVectorCorrection.h"
    #include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
    #include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
    #include "AMP/operators/diffusion/FickSoretNonlinearFEOperator.h"
    #include "AMP/operators/libmesh/MassLinearFEOperator.h"
    #include "AMP/operators/libmesh/VolumeIntegralOperator.h"
    #include "AMP/operators/map/libmesh/MapSurface.h"
    #include "AMP/operators/mechanics/MechanicsConstants.h"
    #include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
    #include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
    #include "AMP/operators/subchannel/FlowFrapconJacobian.h"
    #include "AMP/operators/subchannel/FlowFrapconOperator.h"
    #include "AMP/operators/subchannel/SubchannelFourEqLinearOperator.h"
    #include "AMP/operators/subchannel/SubchannelFourEqNonlinearOperator.h"
    #include "AMP/operators/subchannel/SubchannelTwoEqLinearOperator.h"
    #include "AMP/operators/subchannel/SubchannelTwoEqNonlinearOperator.h"
#endif


namespace AMP::Operator {


// Macros to register an operator
#define REGISTER_OPERATOR( OP, NAME )                                      \
    d_factories[NAME] = []( std::shared_ptr<OperatorParameters> params ) { \
        return std::make_unique<OP>( params );                             \
    }
#ifdef AMP_USE_LIBMESH
    #define REGISTER_OPERATOR_LIBMESH( OP, NAME ) REGISTER_OPERATOR( OP, NAME )
#else
    #define REGISTER_OPERATOR_LIBMESH( OP, NAME )                       \
        d_factories[NAME] = []( std::shared_ptr<OperatorParameters> ) { \
            AMP_ERROR( std::string( NAME ) + " requires libMesh" );     \
            return nullptr;                                             \
        }
#endif


// Create the operator
std::unique_ptr<Operator> OperatorFactory::create( std::shared_ptr<OperatorParameters> params )
{
    AMP_ASSERT( params );
    auto db = params->d_db;
    AMP_ASSERT( db );
    auto objectName = db->getString( "name" );
    return create( objectName, params );
}


} // namespace AMP::Operator


// Register operators
template<>
void AMP::FactoryStrategy<AMP::Operator::Operator,
                          std::shared_ptr<AMP::Operator::OperatorParameters>>::registerDefault()
{
    using namespace AMP::Operator;
    REGISTER_OPERATOR( IdentityOperator, "IdentityOperator" );
    REGISTER_OPERATOR( CoupledOperator, "CoupledOperator" );
    REGISTER_OPERATOR( ColumnOperator, "ColumnOperator" );
    REGISTER_OPERATOR( LinearBVPOperator, "LinearBVPOperator" );
    REGISTER_OPERATOR( NonlinearBVPOperator, "NonlinearBVPOperator" );
    REGISTER_OPERATOR( ColumnBoundaryOperator, "ColumnBoundaryOperator" );
    REGISTER_OPERATOR( DirichletMatrixCorrection, "DirichletMatrixCorrection" );
    REGISTER_OPERATOR( DirichletVectorCorrection, "DirichletVectorCorrection" );
    REGISTER_OPERATOR( NeutronicsRhs, "NeutronicsRhsOperator" );
    REGISTER_OPERATOR_LIBMESH( MapSurface, "MapSurface" );
    REGISTER_OPERATOR_LIBMESH( VolumeIntegralOperator, "VolumeIntegralOperator" );
    REGISTER_OPERATOR_LIBMESH( MassLinearFEOperator, "MassLinearFEOperator" );
    REGISTER_OPERATOR_LIBMESH( DiffusionLinearFEOperator, "DiffusionLinearFEOperator" );
    REGISTER_OPERATOR_LIBMESH( DiffusionNonlinearFEOperator, "DiffusionNonlinearFEOperator" );
    REGISTER_OPERATOR_LIBMESH( MechanicsLinearFEOperator, "MechanicsLinearFEOperator" );
    REGISTER_OPERATOR_LIBMESH( MechanicsNonlinearFEOperator, "MechanicsNonlinearFEOperator" );
    REGISTER_OPERATOR_LIBMESH( RobinMatrixCorrection, "RobinMatrixCorrection" );
    REGISTER_OPERATOR_LIBMESH( RobinVectorCorrection, "RobinVectorCorrection" );
    REGISTER_OPERATOR_LIBMESH( FlowFrapconJacobian, "FlowFrapconJacobian" );
    REGISTER_OPERATOR_LIBMESH( FlowFrapconOperator, "FlowFrapconOperator" );
    REGISTER_OPERATOR_LIBMESH( SubchannelTwoEqLinearOperator, "SubchannelTwoEqLinearOperator" );
    REGISTER_OPERATOR_LIBMESH( SubchannelFourEqLinearOperator, "SubchannelFourEqLinearOperator" );
    REGISTER_OPERATOR_LIBMESH( SubchannelTwoEqNonlinearOperator,
                               "SubchannelTwoEqNonlinearOperator" );
    REGISTER_OPERATOR_LIBMESH( SubchannelFourEqNonlinearOperator,
                               "SubchannelFourEqNonlinearOperator" );
    REGISTER_OPERATOR_LIBMESH( PressureBoundaryOperator, "PressureBoundaryOperator" );
    REGISTER_OPERATOR_LIBMESH( MassMatrixCorrection, "MassMatrixCorrection" );
    REGISTER_OPERATOR_LIBMESH( NeumannVectorCorrection, "NeumannVectorCorrection" );
    REGISTER_OPERATOR_LIBMESH( FickSoretNonlinearFEOperator, "FickSoretNonlinearFEOperator" );
}
