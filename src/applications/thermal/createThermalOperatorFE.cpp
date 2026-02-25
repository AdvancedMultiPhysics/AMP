#include "AMP/AMP_TPLs.h"
#include "AMP/applications/thermal/SolveThermal.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/CoupledOperator.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/operators/VectorCopyOperator.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/map/AsyncMapColumnOperator.h"
#include "AMP/operators/map/NodeToNodeMap.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/NonlinearKrylovAccelerator.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"


#ifdef AMP_USE_LIBMESH

    #include "AMP/operators/boundary/libmesh/RobinMatrixCorrection.h"
    #include "AMP/operators/boundary/libmesh/RobinVectorCorrection.h"
    #include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
    #include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
    #include "AMP/operators/libmesh/VolumeIntegralOperator.h"


namespace AMP::applications {


/***************************************************************************
 * Integrate a nodal or gauss point source vector to a nodal source vector  *
 ***************************************************************************/
std::shared_ptr<AMP::LinearAlgebra::Vector>
integrateSourceVector( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                       std::shared_ptr<const AMP::LinearAlgebra::Vector> src,
                       std::string srcName,
                       std::string dstName )
{
    // Get the Input/Output variable names
    if ( srcName.empty() )
        srcName = src->getVariable()->getName();
    if ( dstName.empty() )
        dstName = srcName;

    // Get the integration type
    std::string integrationType;
    auto srcDOFs = src->subsetVectorForVariable( srcName )->getDOFManager();
    auto type    = srcDOFs->getElement( 0 )->elementType();
    if ( type == AMP::Mesh::GeomType::Vertex )
        integrationType = "NodalScalar";
    else if ( type == AMP::Mesh::GeomType::Cell )
        integrationType = "IntegrationPointScalar";

    // Create the element operator
    auto sourceDB = std::make_shared<AMP::Database>( "source" );
    sourceDB->putScalar( "name", "SourceNonlinearElement" );
    sourceDB->putScalar( "FE_ORDER", "FIRST" );
    sourceDB->putScalar( "FE_FAMILY", "LAGRANGE" );
    sourceDB->putScalar( "QRULE_TYPE", "QGAUSS" );
    sourceDB->putScalar( "QRULE_ORDER", "DEFAULT" );
    auto elemOp = AMP::Operator::ElementOperationFactory::createElementOperation( sourceDB );

    // Create the VolumeIntegralOperator
    auto activeDB = AMP::Database::create( "ActiveVariable_0", srcName );
    auto volumeDB = std::make_shared<AMP::Database>( "volume" );
    volumeDB->putScalar( "InputVariableType", integrationType );
    volumeDB->putScalar( "Number_Active_Variables", 1 );
    volumeDB->putScalar( "Number_Auxillary_Variables", 0 );
    volumeDB->putScalar( "ConstantSource", false );
    volumeDB->putScalar( "OutputVariable", dstName );
    volumeDB->putDatabase( "ActiveInputVariables", std::move( activeDB ) );
    auto volumeParams =
        std::make_shared<AMP::Operator::VolumeIntegralOperatorParameters>( volumeDB );
    volumeParams->d_elemOp = elemOp;
    volumeParams->d_Mesh   = mesh;
    auto volumeOp = std::make_shared<AMP::Operator::VolumeIntegralOperator>( volumeParams );

    // Apply
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto dstVar = std::make_shared<AMP::LinearAlgebra::Variable>( dstName );
    auto dst    = AMP::LinearAlgebra::createVector( DOFs, dstVar );
    dst->zero();
    volumeOp->apply( src, dst );

    return dst;
}


/***************************************************************************
 * Create a diffusion element                                               *
 ***************************************************************************/
void addDiffusionElementDB( AMP::Database &main_db, const std::string &name )
{
    auto db = main_db.createAddDatabase( "DiffusionElement" );
    db->putScalar( "name", name );
    db->putScalar( "TransportAtGaussPoints", true );
    db->putScalar( "FE_ORDER", "FIRST" );
    db->putScalar( "FE_FAMILY", "LAGRANGE" );
    db->putScalar( "QRULE_TYPE", "QGAUSS" );
    db->putScalar( "QRULE_ORDER", "DEFAULT" );
}


/***************************************************************************
 * Copy entries from one database to another                                *
 ***************************************************************************/
void setBoundaryIds( AMP::Database &dst,
                     const std::vector<int> &ids,
                     const std::vector<double> &values,
                     bool isCoupled = false )
{
    AMP_ASSERT( ids.size() == values.size() );
    dst.putScalar( "number_of_ids", ids.size() );
    for ( size_t i = 0; i < ids.size(); i++ ) {
        dst.putScalar( AMP::Utilities::stringf( "id_%i", i ), ids[i] );
        dst.putScalar( AMP::Utilities::stringf( "number_of_dofs_%i", i ), 1 );
        dst.putScalar( AMP::Utilities::stringf( "dof_%i_0", i ), 0 );
        dst.putScalar( AMP::Utilities::stringf( "value_%i_0", i ), values[i] );
        dst.putScalar( AMP::Utilities::stringf( "IsCoupledBoundary_%i", i ), isCoupled );
    }
}


/***************************************************************************
 * Create the non-linear operators                                          *
 ***************************************************************************/
static std::shared_ptr<AMP::Operator::NonlinearBVPOperator>
createOperators( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                 std::shared_ptr<AMP::LinearAlgebra::Vector> solVec,
                 const std::string &InputVariable,
                 const std::string &OutputVariable,
                 const std::string &material,
                 const std::vector<int> &boundaryIds,
                 const std::vector<double> &boundaryValues,
                 const std::vector<double> &boundaryImpedance,
                 const std::vector<int> &mapBoundaryIds,
                 std::shared_ptr<AMP::LinearAlgebra::Vector> mapVec )
{
    using namespace AMP::Operator;

    // Subset the temperature vector
    auto tempVec = solVec->select( AMP::LinearAlgebra::VS_Mesh( mesh ) );

    // Create the local model for thermal diffusion
    auto DiffusionTransportModelDB = std::make_unique<AMP::Database>( "DiffusionTransportModel" );
    DiffusionTransportModelDB->putScalar( "name", "DiffusionTransportModel" );
    DiffusionTransportModelDB->putScalar( "Material", material );
    DiffusionTransportModelDB->putScalar( "Property", "Thermal Conductivity" );

    // Create the non-linear diffusion operator
    auto NonlinearDiffusionDB = std::make_shared<AMP::Database>( "DiffusionNonlinearFEOperator" );
    NonlinearDiffusionDB->putDatabase( "ActiveInputVariables",
                                       AMP::Database::create( "temperature", InputVariable ) );
    NonlinearDiffusionDB->putDatabase( "LocalModel", std::move( DiffusionTransportModelDB ) );
    NonlinearDiffusionDB->putScalar( "name", "DiffusionNonlinearFEOperator" );
    NonlinearDiffusionDB->putScalar( "OutputVariable", OutputVariable );
    NonlinearDiffusionDB->putScalar( "PrincipalVariable", "temperature" );
    addDiffusionElementDB( *NonlinearDiffusionDB, "DiffusionNonlinearElement" );
    auto nonlinearDiffusion =
        std::dynamic_pointer_cast<DiffusionNonlinearFEOperator>( OperatorBuilder::createOperator(
            mesh, "DiffusionNonlinearFEOperator", NonlinearDiffusionDB ) );
    nonlinearDiffusion->setVector( "temperature", tempVec );

    // Create the column boundary operators
    auto emptyDB             = std::make_shared<AMP::Database>( "empty" );
    auto columnParams        = std::make_shared<OperatorParameters>( emptyDB );
    columnParams->d_Mesh     = mesh;
    auto nonlinearBoundaryOp = std::make_shared<ColumnBoundaryOperator>( columnParams );

    // Create the fixed boundary operators
    AMP_ASSERT( boundaryIds.size() == boundaryValues.size() );
    if ( boundaryImpedance.empty() ) {
        auto DirichletDB = std::make_shared<AMP::Database>( "" );
        DirichletDB->putScalar( "skip_params", false );
        DirichletDB->putScalar( "setResidual", true );
        DirichletDB->putScalar( "isAttachedToVolumeOperator", true );
        setBoundaryIds( *DirichletDB, boundaryIds, boundaryValues );
        auto DirichletParams = std::make_shared<DirichletVectorCorrectionParameters>( DirichletDB );
        DirichletParams->d_variable = nonlinearDiffusion->getOutputVariable();
        DirichletParams->d_Mesh     = mesh;
        auto DirichletOp = std::make_shared<DirichletVectorCorrection>( DirichletParams );
        nonlinearBoundaryOp->append( DirichletOp );
    } else {
        AMP_ASSERT( boundaryIds.size() == boundaryImpedance.size() );
        auto boundaryVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "temperature" );
        auto boundaryDOFs = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
        for ( size_t i = 0; i < boundaryIds.size(); i++ ) {
            auto RobinDB = std::make_shared<AMP::Database>( "" );
            RobinDB->putScalar( "FE_ORDER", "FIRST" );
            RobinDB->putScalar( "FE_FAMILY", "LAGRANGE" );
            RobinDB->putScalar( "QRULE_TYPE", "QGAUSS" );
            RobinDB->putScalar( "QRULE_ORDER", "DEFAULT" );
            RobinDB->putScalar( "alpha", 1.0 );
            RobinDB->putScalar( "beta", boundaryImpedance[i] );
            RobinDB->putScalar( "gamma", boundaryImpedance[i] );
            RobinDB->putScalar( "skip_params", false );
            RobinDB->putScalar( "IsFluxGaussPtVector", false );
            setBoundaryIds( *RobinDB, { boundaryIds[i] }, { boundaryValues[i] }, true );
            auto RobinParams = std::make_shared<NeumannVectorCorrectionParameters>( RobinDB );
            RobinParams->d_variable = nonlinearDiffusion->getOutputVariable();
            RobinParams->d_Mesh     = mesh;
            auto RobinOp            = std::make_shared<RobinVectorCorrection>( RobinParams );
            auto boundaryTempVec    = AMP::LinearAlgebra::createVector( boundaryDOFs, boundaryVar );
            boundaryTempVec->setToScalar( boundaryValues[i] );
            RobinOp->setVariableFlux( boundaryTempVec );
            nonlinearBoundaryOp->append( RobinOp );
        }
    }

    // Create the map boundary operators
    if ( !mapBoundaryIds.empty() ) {
        std::vector<double> mapBoundaryValues( mapBoundaryIds.size(), 0 );
        auto RobinDB = std::make_shared<AMP::Database>( "" );
        RobinDB->putScalar( "FE_ORDER", "FIRST" );
        RobinDB->putScalar( "FE_FAMILY", "LAGRANGE" );
        RobinDB->putScalar( "QRULE_TYPE", "QGAUSS" );
        RobinDB->putScalar( "QRULE_ORDER", "DEFAULT" );
        RobinDB->putScalar( "alpha", 1.0 );
        RobinDB->putScalar( "beta", 1000.0 );
        RobinDB->putScalar( "gamma", 1000.0 );
        RobinDB->putScalar( "skip_params", false );
        RobinDB->putScalar( "IsFluxGaussPtVector", false );
        setBoundaryIds( *RobinDB, mapBoundaryIds, mapBoundaryValues, true );
        auto RobinVectorParams = std::make_shared<NeumannVectorCorrectionParameters>( RobinDB );
        RobinVectorParams->d_variable = nonlinearDiffusion->getOutputVariable();
        RobinVectorParams->d_Mesh     = mesh;
        auto RobinVectorOp = std::make_shared<RobinVectorCorrection>( RobinVectorParams );
        RobinVectorOp->setVariableFlux( mapVec );
        nonlinearBoundaryOp->append( RobinVectorOp );
    }

    // Create the non-linear BVP operator
    auto nonlinearBVPOperatorParams = std::make_shared<BVPOperatorParameters>( emptyDB );
    nonlinearBVPOperatorParams->d_volumeOperator   = nonlinearDiffusion;
    nonlinearBVPOperatorParams->d_boundaryOperator = nonlinearBoundaryOp;
    auto nonlinearThermalOperator =
        std::make_shared<NonlinearBVPOperator>( nonlinearBVPOperatorParams );

    return nonlinearThermalOperator;
}


/***************************************************************************
 * Create the non-linear thermal operator and preconditioner                *
 ***************************************************************************/
std::tuple<std::shared_ptr<AMP::Operator::Operator>, std::shared_ptr<AMP::LinearAlgebra::Vector>>
createThermalOperatorsFE( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                          std::shared_ptr<const AMP::Database> input_db )
{
    std::string InputVariable  = "temperature";
    std::string OutputVariable = "temperature";

    // Check the inputs
    AMP_ASSERT( input_db );

    // create solution, rhs, and residual vectors
    auto outputVar = std::make_shared<AMP::LinearAlgebra::Variable>( OutputVariable );
    auto DOFs      = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto solVec = AMP::LinearAlgebra::createVector( DOFs, outputVar );

    // Create the column operator/solver
    auto nonlinearColumnOperator = std::make_shared<AMP::Operator::ColumnOperator>();

    // Get global boundary properties
    auto globalMaterial = input_db->getWithDefault<std::string>( "material", "" );
    auto globalIds      = input_db->getWithDefault<std::vector<int>>( "boundary_ids", {} );
    auto globalValues = input_db->getWithDefault<std::vector<double>>( "boundary_values", {}, "K" );
    auto globalImpedance =
        input_db->getWithDefault<std::vector<double>>( "boundary_impedance", {}, "W/(cm^2*K)" );

    // Create the map copy operator
    auto copyOp_db = std::make_shared<AMP::Database>( "CopyOperator" );
    auto vecCopyOperatorParams =
        std::make_shared<AMP::Operator::VectorCopyOperatorParameters>( copyOp_db );
    vecCopyOperatorParams->d_copyVariable = outputVar;
    vecCopyOperatorParams->d_copyVector   = solVec->clone();
    vecCopyOperatorParams->d_Mesh         = mesh;
    vecCopyOperatorParams->d_copyVector->zero();
    auto thermalCopyOperator =
        std::make_shared<AMP::Operator::VectorCopyOperator>( vecCopyOperatorParams );

    // Create the maps
    std::shared_ptr<AMP::Operator::AsyncMapColumnOperator> maps;
    std::map<std::string, std::set<int>> mapSurfaces;
    if ( input_db->keyExists( "maps" ) ) {
        auto map_db =
            std::shared_ptr<AMP::Database>( input_db->getDatabase( "maps" )->cloneDatabase() );
        map_db->putScalar( "DOFsPerObject", 1 );
        map_db->putScalar( "VariableName", OutputVariable );
        auto type = map_db->getString( "MapType" );
        if ( type == "NodeToNode" ) {
            maps = AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::NodeToNodeMap>(
                mesh, map_db );
            maps->setVector( vecCopyOperatorParams->d_copyVector );
            auto map_db2 = AMP::Operator::AsyncMapColumnOperator::createDatabases( map_db );
            for ( auto db : map_db2 ) {
                auto Mesh1    = db->getString( "Mesh1" );
                auto Mesh2    = db->getString( "Mesh2" );
                auto Surface1 = db->getScalar<int>( "Surface1" );
                auto Surface2 = db->getScalar<int>( "Surface2" );
                mapSurfaces[Mesh1].insert( Surface1 );
                mapSurfaces[Mesh2].insert( Surface2 );
            }
        } else {
            AMP_ERROR( "Unknown map type: " + type );
        }
    }

    // Loop through the individual domains and create the nonlinear operators
    double T0 = 0;
    for ( auto id : mesh->getBaseMeshIDs() ) {

        // Subset for the given domain
        auto mesh2 = mesh->Subset( id );

        // Get the boundary conditions
        auto db = input_db->getDatabase( mesh2->getName() );
        if ( !db )
            db = std::make_shared<AMP::Database>( mesh2->getName() );
        auto material = db->getWithDefault<std::string>( "material", globalMaterial );
        if ( material.empty() )
            material = mesh2->getName();
        auto boundaryIds       = globalIds;
        auto boundaryValues    = globalValues;
        auto boundaryImpedance = globalImpedance;
        if ( db->keyExists( "boundary_ids" ) ) {
            boundaryIds    = db->getWithDefault<std::vector<int>>( "boundary_ids", {} );
            boundaryValues = db->getWithDefault<std::vector<double>>( "boundary_values", {}, "K" );
            boundaryImpedance =
                db->getWithDefault<std::vector<double>>( "boundary_impedance", {}, "W/(cm^2*K)" );
        }
        if ( T0 == 0 && !boundaryValues.empty() )
            T0 = boundaryValues[0];

        // Create the input database
        std::vector<int> mapIds( mapSurfaces[mesh2->getName()].begin(),
                                 mapSurfaces[mesh2->getName()].end() );
        auto nonlinearOp = createOperators( mesh2,
                                            solVec,
                                            InputVariable,
                                            OutputVariable,
                                            material,
                                            boundaryIds,
                                            boundaryValues,
                                            boundaryImpedance,
                                            mapIds,
                                            vecCopyOperatorParams->d_copyVector );

        // Add the operators
        nonlinearColumnOperator->append( nonlinearOp );
    }
    if ( T0 == 0 )
        T0 = 300;
    solVec->setToScalar( T0 );
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    vecCopyOperatorParams->d_copyVector->setToScalar( T0 );

    // Create the coupled operator
    auto coupledOpDB = std::make_shared<AMP::Database>( "CoupledOperator" );
    auto coupledOpParams =
        std::make_shared<AMP::Operator::CoupledOperatorParameters>( coupledOpDB );
    coupledOpParams->d_CopyOperator = thermalCopyOperator;
    coupledOpParams->d_MapOperator  = maps;
    coupledOpParams->d_BVPOperator  = nonlinearColumnOperator;
    auto nonlinearCoupledOperator =
        std::make_shared<AMP::Operator::CoupledOperator>( coupledOpParams );

    // Return the operator and solution vector
    std::tuple<std::shared_ptr<AMP::Operator::Operator>,
               std::shared_ptr<AMP::LinearAlgebra::Vector>>
        rtn;
    std::get<0>( rtn ) = nonlinearCoupledOperator;
    std::get<1>( rtn ) = solVec;
    return rtn;
}


} // namespace AMP::applications


#else

std::shared_ptr<AMP::LinearAlgebra::Vector>
AMP::applications::integrateSourceVector( std::shared_ptr<AMP::Mesh::Mesh>,
                                          std::shared_ptr<const AMP::LinearAlgebra::Vector>,
                                          std::string,
                                          std::string )
{
    AMP_ERROR( "integrateSourceVector requires libmesh" );
}
std::tuple<std::shared_ptr<AMP::Operator::Operator>, std::shared_ptr<AMP::LinearAlgebra::Vector>>
AMP::applications::createThermalOperatorsFE( std::shared_ptr<AMP::Mesh::Mesh>,
                                             std::shared_ptr<const AMP::Database> )
{
    AMP_ERROR( "integrateSourceVector requires libmesh" );
}


#endif
