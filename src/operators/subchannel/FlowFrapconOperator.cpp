#include "AMP/operators/subchannel/FlowFrapconOperator.h"
#include "AMP/operators/subchannel/FlowFrapconJacobianParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/VectorSelector.h"

#include <string>


namespace AMP::Operator {


FlowFrapconOperator::FlowFrapconOperator( std::shared_ptr<const OperatorParameters> params )
    : Operator( params ), d_boundaryId( 0 )
{
    AMP_ASSERT( params );
    AMP_ASSERT( params->d_db );

    auto inpVar   = params->d_db->getString( "InputVariable" );
    auto outVar   = params->d_db->getString( "OutputVariable" );
    d_inpVariable = std::make_shared<AMP::LinearAlgebra::Variable>( inpVar );
    d_outVariable = std::make_shared<AMP::LinearAlgebra::Variable>( outVar );

    reset( params );
}

void FlowFrapconOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    AMP_ASSERT( params->d_db );

    d_numpoints = params->d_db->getScalar<int>( "numpoints" );
    d_De        = params->d_db->getScalar<double>( "Channel_Diameter" );
    Cp          = params->d_db->getScalar<double>( "Heat_Capacity" );
    d_G         = params->d_db->getScalar<double>( "Mass_Flux" );
    d_Tin       = params->d_db->getWithDefault<double>( "Temp_Inlet", 300. );
    d_K         = params->d_db->getScalar<double>( "Conductivity" );
    d_Re        = params->d_db->getScalar<double>( "Reynolds" );
    d_Pr        = params->d_db->getScalar<double>( "Prandtl" );
}


// This is an in-place apply
void FlowFrapconOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                 AMP::LinearAlgebra::Vector::shared_ptr r )
{
    AMP_INSIST( r, "NULL Residual Vector" );
    AMP_INSIST( u, "NULL Solution Vector" );

    if ( !zPoints.empty() )
        d_numpoints = zPoints.size();

    std::vector<double> box = d_Mesh->getBoundingBox();
    const double min_z      = box[4];
    const double max_z      = box[5];
    const double del_z      = ( max_z - min_z ) / d_numpoints;

    // Subset the vectors
    auto flowInputVec = subsetInputVector( u );
    auto outputVec    = subsetOutputVector( r );
    AMP_ASSERT( r );
    AMP_ASSERT( u );
    zPoints.resize( d_numpoints );

    // set the inlet flow temperature value
    double T1        = flowInputVec->getValueByLocalID( 0 );
    size_t idx       = 0;
    const double val = T1 - d_Tin;
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

        double Heff, he_z, T_b_i, T_b_im1, T_c_i;
        double R_b = 0;

        T_c_i   = d_cladVec->getValueByLocalID( i );
        T_b_i   = flowInputVec->getValueByLocalID( i );
        T_b_im1 = flowInputVec->getValueByLocalID( i - 1 );

        Heff = ( 0.023 * d_K / d_De ) * std::pow( d_Re, 0.8 ) * std::pow( d_Pr, 0.4 );
        //       Cp   = getHeatCapacity(T_b_i);
        he_z = next_node - cur_node;

        R_b = T_b_i - T_b_im1 - ( ( 4 * Heff * ( T_c_i - T_b_i ) ) / ( Cp * d_G * d_De ) ) * he_z;

        outputVec->setValuesByLocalID( 1, &i, &R_b );

    } // end for i
}


std::shared_ptr<OperatorParameters>
FlowFrapconOperator::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in )
{
    auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );

    Operator::setMemoryAndBackendParameters( tmp_db );
    tmp_db->putScalar( "name", "FlowFrapconJacobian" );
    tmp_db->putScalar( "InputVariable", d_inpVariable->getName() );
    tmp_db->putScalar( "OutputVariable", d_outVariable->getName() );
    tmp_db->putScalar( "numpoints", d_numpoints );
    tmp_db->putScalar( "Channel_Diameter", d_De );
    tmp_db->putScalar( "Mass_Flux", d_G );
    tmp_db->putScalar( "Heat_Capacity", Cp );
    tmp_db->putScalar( "Temp_Inlet", d_Tin );
    tmp_db->putScalar( "Conductivity", d_K );
    tmp_db->putScalar( "Reynolds", d_Re );
    tmp_db->putScalar( "Prandtl", d_Pr );

    auto outParams              = std::make_shared<FlowFrapconJacobianParameters>( tmp_db );
    auto u                      = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( u_in );
    outParams->d_frozenSolution = subsetInputVector( u );
    return outParams;
}


// Create the VectorSelector, the vectors are simple vectors and
//    we need to subset for the current comm instead of the mesh
std::shared_ptr<AMP::LinearAlgebra::VectorSelector> FlowFrapconOperator::selectOutputVector() const
{
    std::vector<std::shared_ptr<AMP::LinearAlgebra::VectorSelector>> selectors;
    if ( d_Mesh )
        selectors.push_back( std::make_shared<AMP::LinearAlgebra::VS_Comm>( d_Mesh->getComm() ) );
    auto var = getInputVariable();
    if ( var )
        selectors.push_back( var->createVectorSelector() );
    return AMP::LinearAlgebra::VectorSelector::create( selectors );
}
std::shared_ptr<AMP::LinearAlgebra::VectorSelector> FlowFrapconOperator::selectInputVector() const
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
