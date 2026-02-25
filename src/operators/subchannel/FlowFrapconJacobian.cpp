#include "AMP/operators/subchannel/FlowFrapconJacobian.h"
#include "AMP/vectors/VectorSelector.h"

#include <string>


namespace AMP::Operator {


FlowFrapconJacobian::FlowFrapconJacobian( std::shared_ptr<const OperatorParameters> params )
    : Operator( params ), dCp( 0 )
{
    AMP_ASSERT( params );
    auto inpVar = params->d_db->getString( "InputVariable" );
    d_inpVariable.reset( new AMP::LinearAlgebra::Variable( inpVar ) );

    auto outVar = params->d_db->getString( "OutputVariable" );
    d_outVariable.reset( new AMP::LinearAlgebra::Variable( outVar ) );

    d_SimpleVariable.reset( new AMP::LinearAlgebra::Variable( "FlowInternal" ) );

    reset( params );
}


std::shared_ptr<AMP::LinearAlgebra::Variable>
FlowFrapconJacobian::createInputVariable( const std::string &name, int )
{
    return d_inpVariable->clone( name );
}


std::shared_ptr<AMP::LinearAlgebra::Variable>
FlowFrapconJacobian::createOutputVariable( const std::string &name, int )
{
    return d_outVariable->clone( name );
}


std::shared_ptr<AMP::LinearAlgebra::Variable> FlowFrapconJacobian::getInputVariable() const
{
    return d_inpVariable;
}


std::shared_ptr<AMP::LinearAlgebra::Variable> FlowFrapconJacobian::getOutputVariable() const
{
    return d_outVariable;
}


void FlowFrapconJacobian::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );
    AMP_ASSERT( params->d_db );

    bool skipParams = params->d_db->getWithDefault<bool>( "skip_params", false );

    if ( !skipParams ) {
        d_numpoints = params->d_db->getScalar<int>( "numpoints" );
        d_De        = params->d_db->getScalar<double>( "Channel_Diameter" );
        Cp          = params->d_db->getScalar<double>( "Heat_Capacity" );
        d_G         = params->d_db->getScalar<double>( "Mass_Flux" );
        d_Tin       = params->d_db->getScalar<double>( "Temp_Inlet" );
        d_K         = params->d_db->getScalar<double>( "Conductivity" );
        d_Re        = params->d_db->getScalar<double>( "Reynolds" );
        d_Pr        = params->d_db->getScalar<double>( "Prandtl" );
    }

    auto myparams = std::dynamic_pointer_cast<const FlowFrapconJacobianParameters>( params );
    if ( myparams ) {
        if ( myparams->d_frozenSolution )
            d_frozenVec = myparams->d_frozenSolution;
    }
}


// This is an in-place apply
void FlowFrapconJacobian::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                 AMP::LinearAlgebra::Vector::shared_ptr r )
{

    AMP_INSIST( r, "NULL Residual Vector" );
    AMP_INSIST( u, "NULL Solution Vector" );

    std::vector<double> box = d_Mesh->getBoundingBox();
    const double min_z      = box[4];
    const double max_z      = box[5];
    const double del_z      = ( max_z - min_z ) / d_numpoints;

    auto flowInputVec = subsetInputVector( u );

    auto outputVec = subsetOutputVector( r );

    AMP_INSIST( ( d_frozenVec ), "Null Frozen Vector inside Jacobian" );

    if ( !zPoints.empty() ) {
        d_numpoints = zPoints.size();
    }

    zPoints.resize( d_numpoints );

    // set the inlet flow temperature value
    // flowInputVec->setValueByLocalID(0, 0.0);
    size_t idx       = 0;
    const double val = flowInputVec->getValueByLocalID( idx );
    outputVec->setValuesByLocalID( 1, &idx, &val );

    zPoints[0] = min_z;
    for ( int j = 1; j < d_numpoints; j++ ) {
        zPoints[j] = zPoints[j - 1] + del_z;
    }

    // Iterate through the flow boundary
    for ( size_t i = 1; i < (size_t) d_numpoints; i++ ) {

        double cur_node, next_node;

        cur_node  = zPoints[i - 1];
        next_node = zPoints[i];

        double Heff, he_z, dT_b_i, dT_b_im1, T_c_i, T_b_i;
        double JT_b = 0;

        T_c_i = d_cladVec->getValueByLocalID( i );
        T_b_i = d_frozenVec->getValueByLocalID( i );

        dT_b_i   = flowInputVec->getValueByLocalID( i );
        dT_b_im1 = flowInputVec->getValueByLocalID( i - 1 );

        Heff = ( 0.023 * d_K / d_De ) * std::pow( d_Re, 0.8 ) * std::pow( d_Pr, 0.4 );
        //                  Cp   = getHeatCapacity(T_b_i);
        //                  dCp  = getHeatCapacityGradient(T_b_i);
        dCp  = 0.0;
        he_z = next_node - cur_node;

        JT_b =
            dT_b_i - dT_b_im1 -
            ( ( 4 * Heff * ( T_c_i - T_b_i ) ) / ( Cp * Cp * d_G * d_De ) ) * he_z * dCp * dT_b_i +
            ( ( 4 * Heff ) / ( Cp * d_G * d_De ) ) * he_z * dT_b_i;

        outputVec->setValuesByLocalID( 1, &i, &JT_b );

    } // end for i
}


// Create the VectorSelector, the vectors are simple vectors and
//    we need to subset for the current comm instead of the mesh
std::shared_ptr<AMP::LinearAlgebra::VectorSelector> FlowFrapconJacobian::selectOutputVector() const
{
    std::vector<std::shared_ptr<AMP::LinearAlgebra::VectorSelector>> selectors;
    if ( d_Mesh )
        selectors.push_back( std::make_shared<AMP::LinearAlgebra::VS_Comm>( d_Mesh->getComm() ) );
    auto var = getInputVariable();
    if ( var )
        selectors.push_back( var->createVectorSelector() );
    return AMP::LinearAlgebra::VectorSelector::create( selectors );
}
std::shared_ptr<AMP::LinearAlgebra::VectorSelector> FlowFrapconJacobian::selectInputVector() const
{
    std::vector<std::shared_ptr<AMP::LinearAlgebra::VectorSelector>> selectors;
    if ( d_Mesh )
        selectors.push_back( std::make_shared<AMP::LinearAlgebra::VS_Comm>( d_Mesh->getComm() ) );
    auto var = getInputVariable();
    if ( var )
        selectors.push_back( var->createVectorSelector() );
    return AMP::LinearAlgebra::VectorSelector::create( selectors );
}


} // namespace AMP::Operator
