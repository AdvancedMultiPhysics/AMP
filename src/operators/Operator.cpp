#include "AMP/operators/Operator.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorSelector.h"

#include "ProfilerApp.h"


namespace AMP::Operator {


int Operator::d_iInstance_id = 0;


Operator::Operator()
{
    d_iObject_id           = Operator::d_iInstance_id;
    d_iDebugPrintInfoLevel = 0;
    Operator::d_iInstance_id++;
    d_memory_location = AMP::Utilities::MemoryType::none;
}


Operator::Operator( std::shared_ptr<const OperatorParameters> params )
{
    AMP_INSIST( params, "NULL parameter" );

    d_Mesh                 = params->d_Mesh;
    d_iObject_id           = Operator::d_iInstance_id;
    d_iDebugPrintInfoLevel = 0;
    Operator::d_iInstance_id++;

    // try and keep the next call the last in the function
    // so as not to override any parameters set through it
    // by accident
    getFromInput( params->d_db );
}


void Operator::reset( std::shared_ptr<const OperatorParameters> params )
{
    // try and keep the next call the last in the function
    // so as not to override any parameters set through it
    // by accident
    if ( params )
        getFromInput( params->d_db );
}


void Operator::residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                         AMP::LinearAlgebra::Vector::const_shared_ptr u,
                         AMP::LinearAlgebra::Vector::shared_ptr r )
{
    AMP_INSIST( u, "NULL Solution Vector" );
    AMP_INSIST( r, "NULL Residual Vector" );
    AMP_ASSERT( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    apply( u, r );

    auto rInternal = subsetOutputVector( r );
    AMP_INSIST( ( rInternal ), "rInternal is NULL" );

    // the rhs can be NULL
    if ( f ) {
        auto fInternal = subsetOutputVector( f );
        rInternal->subtract( *fInternal, *rInternal );
    } else {
        rInternal->scale( -1.0 );
    }

    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


std::shared_ptr<OperatorParameters>
Operator::getParameters( const std::string &type,
                         std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                         std::shared_ptr<OperatorParameters> )
{
    if ( type == "Jacobian" ) {
        auto params = getJacobianParameters( u );
        if ( params ) {
            setMemoryAndBackendParameters( params->d_db );
        }
        return params;
    } else {
        // Derived class should implement this
        AMP_ERROR( "Unknown OperatorParameters type specified" );
    }
    return nullptr;
}


void Operator::getFromInput( std::shared_ptr<AMP::Database> db )
{
    AMP_INSIST( db, "NULL database" );

    d_iDebugPrintInfoLevel = db->getWithDefault<int>( "print_info_level", 0 );

    if ( d_memory_location == AMP::Utilities::MemoryType::none ) {
        auto memLoc       = db->getWithDefault<std::string>( "MemoryLocation", "host" );
        d_memory_location = AMP::Utilities::memoryLocationFromString( memLoc );
    }
    if ( d_backend == AMP::Utilities::Backend::none ) {
        if ( db->keyExists( "AccelerationBackend" ) ) {
            auto bcknd = db->getString( "AccelerationBackend" );
            d_backend  = AMP::Utilities::backendFromString( bcknd );
        } else {
            d_backend = AMP::Utilities::getDefaultBackend( d_memory_location );
        }
    }
}

void Operator::setMemoryAndBackendParameters( std::shared_ptr<AMP::Database> db )
{
    if ( d_memory_location != AMP::Utilities::MemoryType::none )
        db->putScalar<std::string>( "MemoryLocation",
                                    AMP::Utilities::getString( d_memory_location ) );
    if ( d_backend != AMP::Utilities::Backend::none )
        db->putScalar<std::string>( "AccelerationBackend", AMP::Utilities::getString( d_backend ) );
}

/********************************************************
 * Return the default VectorSelector                     *
 ********************************************************/
std::shared_ptr<AMP::LinearAlgebra::VectorSelector> Operator::selectOutputVector() const
{
    std::vector<std::shared_ptr<AMP::LinearAlgebra::VectorSelector>> selectors;
    if ( d_Mesh )
        selectors.push_back( std::make_shared<AMP::LinearAlgebra::VS_Mesh>( d_Mesh ) );
    auto var = getOutputVariable();
    if ( var )
        selectors.push_back( var->createVectorSelector() );
    return AMP::LinearAlgebra::VectorSelector::create( selectors );
}
std::shared_ptr<AMP::LinearAlgebra::VectorSelector> Operator::selectInputVector() const
{
    std::vector<std::shared_ptr<AMP::LinearAlgebra::VectorSelector>> selectors;
    if ( d_Mesh )
        selectors.push_back( std::make_shared<AMP::LinearAlgebra::VS_Mesh>( d_Mesh ) );
    auto var = getInputVariable();
    if ( var )
        selectors.push_back( var->createVectorSelector() );
    return AMP::LinearAlgebra::VectorSelector::create( selectors );
}


/********************************************************
 * Subset vectors                                        *
 ********************************************************/
std::shared_ptr<AMP::LinearAlgebra::Vector>
Operator::subsetInputVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec ) const
{
    PROFILE( "subsetInputVector", 1 );
    return vec->select( *selectInputVector() );
}
std::shared_ptr<const AMP::LinearAlgebra::Vector>
Operator::subsetInputVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> vec ) const
{
    PROFILE( "constSubsetInputVector", 1 );
    return vec->select( *selectInputVector() );
}
std::shared_ptr<AMP::LinearAlgebra::Vector>
Operator::subsetOutputVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec ) const
{
    PROFILE( "subsetOutputVector", 1 );
    return vec->select( *selectOutputVector() );
}
std::shared_ptr<const AMP::LinearAlgebra::Vector>
Operator::subsetOutputVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> vec ) const
{
    PROFILE( "constSubsetOutputVector", 1 );
    return vec->select( *selectOutputVector() );
}


/********************************************************
 * makeConsistent                                        *
 ********************************************************/
void Operator::makeConsistent( std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
{
    AMP_ASSERT( vec );
    vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

} // namespace AMP::Operator
