#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/operators/boundary/BoundaryOperatorParameters.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/utils/Utilities.h"
#ifdef AMP_USE_LIBMESH
    #include "AMP/operators/diffusion/FickSoretNonlinearFEOperator.h"
#endif

#include <string>

#include "ProfilerApp.h"


namespace AMP::Operator::OperatorBuilder {


static std::shared_ptr<BoundaryOperator>
createBoundaryOperator( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                        std::string boundaryOperatorName,
                        std::shared_ptr<AMP::Database> input_db,
                        AMP::Operator::Operator::shared_ptr volumeOperator );


/********************************************************
 * Create specific operators implementation              *
 ********************************************************/
#ifdef AMP_USE_LIBMESH
static std::shared_ptr<OperatorParameters>
createNonlinearFickSoretOperatorParameters( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                            std::string operatorName,
                                            std::shared_ptr<AMP::Database> input_db )
{
    AMP_INSIST( input_db, "NULL database object passed" );
    auto operator_db = input_db->getDatabase( operatorName );
    AMP_INSIST( operator_db, "operator database is null" );
    // operator names
    auto fickOperatorName  = operator_db->getString( "FickOperator" );
    auto soretOperatorName = operator_db->getString( "SoretOperator" );
    // Ensure consistency of operator memory locations
    std::shared_ptr<Database> db = input_db->cloneDatabase();
    AMP::Utilities::setNestedOperatorMemoryLocations(
        db, operatorName, { fickOperatorName, soretOperatorName } );
    auto fickOperator  = createOperator( mesh, fickOperatorName, db );
    auto soretOperator = createOperator( mesh, soretOperatorName, db );
    auto params        = std::make_shared<FickSoretNonlinearFEOperatorParameters>( db );
    params->d_Mesh     = mesh;
    params->d_FickOperator =
        std::dynamic_pointer_cast<DiffusionNonlinearFEOperator>( fickOperator );
    params->d_SoretOperator =
        std::dynamic_pointer_cast<DiffusionNonlinearFEOperator>( soretOperator );
    params->d_name = operatorName;
    return params;
}
#else
static std::shared_ptr<OperatorParameters> createNonlinearFickSoretOperatorParameters(
    std::shared_ptr<AMP::Mesh::Mesh>, std::string, std::shared_ptr<AMP::Database> )
{
    return nullptr;
}
#endif
static std::shared_ptr<OperatorParameters>
createLinearBVPOperatorParameters( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                   std::string operatorName,
                                   std::shared_ptr<AMP::Database> input_db )
{
    PROFILE( "createLinearBVPOperator" );
    AMP_INSIST( input_db, "NULL database object passed" );
    auto operator_db = input_db->getDatabase( operatorName );
    AMP_INSIST( operator_db, "NULL database object passed" );
    // names of internal operators
    auto volumeOperatorName   = operator_db->getString( "VolumeOperator" );
    auto boundaryOperatorName = operator_db->getString( "BoundaryOperator" );
    // Ensure consistency of operator memory locations
    std::shared_ptr<Database> db = input_db->cloneDatabase();
    AMP::Utilities::setNestedOperatorMemoryLocations(
        db, operatorName, { volumeOperatorName, boundaryOperatorName } );
    // create the volume operator
    auto volumeOperator = createOperator( mesh, volumeOperatorName, db );
    auto volumeLinearOp = std::dynamic_pointer_cast<LinearOperator>( volumeOperator );
    AMP_ASSERT( volumeLinearOp );
    // create the boundary operator
    auto boundaryOperator_db = db->getDatabase( boundaryOperatorName );
    AMP_INSIST( boundaryOperator_db, "NULL database object passed for boundary operator" );
    boundaryOperator_db->putScalar(
        "isAttachedToVolumeOperator", true, Units(), Database::Check::Overwrite );
    auto boundaryOperator =
        createBoundaryOperator( mesh, boundaryOperatorName, db, volumeLinearOp );
    auto bvpOperatorParams                = std::make_shared<BVPOperatorParameters>( operator_db );
    bvpOperatorParams->d_volumeOperator   = volumeOperator;
    bvpOperatorParams->d_boundaryOperator = boundaryOperator;
    return bvpOperatorParams;
}
static std::shared_ptr<OperatorParameters>
createNonlinearBVPOperatorParameters( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                      std::string operatorName,
                                      std::shared_ptr<AMP::Database> input_db )
{
    AMP_INSIST( input_db, "NULL database object passed" );
    auto operator_db = input_db->getDatabase( operatorName );
    AMP_INSIST( operator_db, "NULL database object passed" );
    // operator names
    auto volumeOperatorName   = operator_db->getString( "VolumeOperator" );
    auto boundaryOperatorName = operator_db->getString( "BoundaryOperator" );
    // Ensure consistency of operator memory locations
    std::shared_ptr<Database> db = input_db->cloneDatabase();
    AMP::Utilities::setNestedOperatorMemoryLocations(
        db, operatorName, { volumeOperatorName, boundaryOperatorName } );
    // create the volume operator
    auto volumeOperator = createOperator( mesh, volumeOperatorName, db );
    // create the boundary operator
    auto boundaryOperator_db = db->getDatabase( boundaryOperatorName );
    AMP_INSIST( boundaryOperator_db, "NULL database object passed for boundary operator" );
    boundaryOperator_db->putScalar(
        "isAttachedToVolumeOperator", true, Units(), Database::Check::Overwrite );
    auto boundaryOperator =
        createBoundaryOperator( mesh, boundaryOperatorName, db, volumeOperator );
    auto bvpOperatorParams                = std::make_shared<BVPOperatorParameters>( db );
    bvpOperatorParams->d_volumeOperator   = volumeOperator;
    bvpOperatorParams->d_boundaryOperator = boundaryOperator;
    return bvpOperatorParams;
}


/********************************************************
 * createColumnBoundaryOperator                          *
 ********************************************************/
static std::shared_ptr<BoundaryOperator>
createColumnBoundaryOperator( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                              std::string boundaryOperatorName,
                              std::shared_ptr<AMP::Database> input_db,
                              Operator::shared_ptr volumeOperator )
{
    AMP_INSIST( input_db, "NULL database object passed" );
    auto operator_db = input_db->getDatabase( boundaryOperatorName );
    AMP_INSIST( operator_db,
                "Error: createBoundaryOperator(): "
                "database object with given name not in database" );
    int N            = operator_db->getWithDefault<int>( "numberOfBoundaryOperators", 1 );
    auto boundaryOps = operator_db->getVector<std::string>( "boundaryOperators" );
    AMP_ASSERT( N == (int) boundaryOps.size() );
    // Ensure consistency of operator memory locations
    std::shared_ptr<Database> db = input_db->cloneDatabase();
    AMP::Utilities::setNestedOperatorMemoryLocations( db, boundaryOperatorName, boundaryOps );
    // Create the parameters and operators
    auto params                 = std::make_shared<OperatorParameters>( operator_db, mesh );
    auto columnBoundaryOperator = std::make_shared<ColumnBoundaryOperator>( params );
    for ( int i = 0; i < N; i++ ) {
        auto bcOperator = createBoundaryOperator( mesh, boundaryOps[i], db, volumeOperator );
        AMP_ASSERT( bcOperator );
        columnBoundaryOperator->append( bcOperator );
    }
    return columnBoundaryOperator;
}


/********************************************************
 * createOperator                                        *
 ********************************************************/
std::shared_ptr<Operator> createOperator( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                          const std::string &operatorName,
                                          std::shared_ptr<AMP::Database> input_db )
{
    PROFILE( "createOperator" );
    auto operator_db = input_db;
    if ( input_db->keyExists( operatorName ) )
        operator_db = input_db->getDatabase( operatorName );
    AMP_INSIST( operator_db, "operator database is null" );
    auto operatorType = operator_db->getString( "name" );
    auto params       = std::make_shared<OperatorParameters>( operator_db, mesh );
    if ( operatorType == "FickSoretNonlinearFEOperator" )
        params = createNonlinearFickSoretOperatorParameters( mesh, operatorName, input_db );
    if ( operatorType == "LinearBVPOperator" )
        params = createLinearBVPOperatorParameters( mesh, operatorName, input_db );
    if ( operatorType == "NonlinearBVPOperator" )
        params = createNonlinearBVPOperatorParameters( mesh, operatorName, input_db );
    if ( !params->d_db->keyExists( "name" ) )
        params->d_db->putScalar( "name", operatorType );
    return OperatorFactory::create( params );
}


/********************************************************
 * createBoundaryOperator                                *
 ********************************************************/
std::shared_ptr<BoundaryOperator> createBoundaryOperator( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                                          std::string boundaryOperatorName,
                                                          std::shared_ptr<AMP::Database> input_db,
                                                          Operator::shared_ptr volumeOperator )
{
    AMP_ASSERT( input_db );
    auto operator_db = input_db->getDatabase( boundaryOperatorName );
    AMP_INSIST( operator_db, "operator database is null" );
    auto boundaryType = operator_db->getString( "name" );
    if ( boundaryType == "ColumnBoundaryOperator" ) {
        // note that the global input database is passed here instead of the operator
        // database
        return createColumnBoundaryOperator( mesh, boundaryOperatorName, input_db, volumeOperator );
    } else if ( OperatorFactory::exists( boundaryType ) ) {
        // Use the OperatorFactory to create the operator
        auto params                  = std::make_shared<BoundaryOperatorParameters>( operator_db );
        params->d_Mesh               = mesh;
        params->d_volumeOperator     = volumeOperator;
        std::shared_ptr<Operator> op = OperatorFactory::create( params );
        auto op2                     = std::dynamic_pointer_cast<BoundaryOperator>( op );
        AMP_ASSERT( op2 );
        return op2;
    }
    return nullptr;
}


} // namespace AMP::Operator::OperatorBuilder
