#include "AMP/operators/diffusion/FickSoretNonlinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperatorParameters.h"
#include "AMP/operators/diffusion/FickSoretNonlinearFEOperatorParameters.h"


namespace AMP::Operator {


FickSoretNonlinearFEOperator::FickSoretNonlinearFEOperator(
    std::shared_ptr<const OperatorParameters> params )
{
    auto fsParams =
        std::dynamic_pointer_cast<const FickSoretNonlinearFEOperatorParameters>( params );

    // get databases for each sub-operator
    auto db          = params->d_db;
    auto ficksoretDb = db->getDatabase( fsParams->d_name );
    auto fickDb      = db->getDatabase( ficksoretDb->getString( "FickOperator" ) );
    auto soretDb     = db->getDatabase( ficksoretDb->getString( "SoretOperator" ) );

    // verify operator and sub-operator names are correct
    auto name = ficksoretDb->getString( "name" );
    AMP_INSIST( name == "FickSoretNonlinearFEOperator",
                "incorrect nonlinear Fick-Soret operator name" );
    auto fickName  = fickDb->getString( "name" );
    auto soretName = soretDb->getString( "name" );

    AMP_INSIST( fickName == "DiffusionNonlinearFEOperator", "Fick operator has incorrect name" );
    AMP_INSIST( soretName == "DiffusionNonlinearFEOperator", "Soret operator has incorrect name" );

    // check transport models correct
    auto fickModelDb    = fickDb->getDatabase( "LocalModel" );
    auto soretModelDb   = soretDb->getDatabase( "LocalModel" );
    auto fickModelProp  = fickModelDb->getString( "Property" );
    auto soretModelProp = soretModelDb->getString( "Property" );
    AMP_INSIST( fickModelProp == "FickCoefficient", "FickCoefficient property was not specified" );
    AMP_INSIST( soretModelProp == "ThermalDiffusionCoefficient",
                "ThermalDiffusionCoefficient property was not specified" );

    // verify active variables are correct
    auto fickActiveDb  = fickDb->getDatabase( "ActiveInputVariables" );
    auto soretActiveDb = soretDb->getDatabase( "ActiveInputVariables" );
    bool fickActive    = fickActiveDb->keyExists( "concentration" );
    fickActive         = fickActive && fickActiveDb->keyExists( "temperature" );
    bool soretActive   = soretActiveDb->keyExists( "temperature" );
    soretActive        = soretActive && soretActiveDb->keyExists( "concentration" );
    AMP_INSIST( fickActive, "Fick operator must have concentration and temperature active" );
    AMP_INSIST( soretActive, "Soret operator must have concentration and temperature active" );
    auto fickTempName  = fickActiveDb->getString( "temperature" );
    auto fickConcName  = fickActiveDb->getString( "concentration" );
    auto soretTempName = soretActiveDb->getString( "temperature" );
    auto soretConcName = soretActiveDb->getString( "concentration" );
    AMP_INSIST(
        fickTempName == soretTempName && fickConcName == soretConcName,
        "Fick and Soret operators must agree on names of concentration and temperature variables" );

    // verify frozen variables are correct
    bool fickNotFrozen  = !fickActiveDb->getWithDefault<bool>( "Freezeconcentration", false );
    bool soretNotFrozen = !soretActiveDb->getWithDefault<bool>( "Freezeconcentration", false );
    AMP_INSIST( fickNotFrozen, "Fick operator must not have concentration frozen" );
    AMP_INSIST( soretNotFrozen, "Soret operator must not have concentration frozen" );
    fickNotFrozen  = !fickActiveDb->getWithDefault<bool>( "Freezetemperature", false );
    soretNotFrozen = !soretActiveDb->getWithDefault<bool>( "Freezetemperature", false );
    AMP_INSIST( fickNotFrozen == soretNotFrozen,
                "Fick and Soret operators must freeze temperature the same way" );

    // get output variable and make sure it is the same in the two operators
    AMP_INSIST( fickDb->keyExists( "OutputVariable" ), "Fick output variable not specified" );
    AMP_INSIST( soretDb->keyExists( "OutputVariable" ), "Soret output variable not specified" );
    std::string fickOut  = fickDb->getString( "OutputVariable" );
    std::string soretOut = soretDb->getString( "OutputVariable" );
    AMP_INSIST( fickOut == soretOut, "Fick and Soret output variables must be the same" );
    d_OutputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( fickOut );

    // get the switch to add the Soret term
    d_AddSoretTerm = ficksoretDb->getWithDefault<bool>( "AddSoretTerm", true );

    // set the member operators
    if ( fsParams->d_FickOperator ) {
        d_FickOperator = fsParams->d_FickOperator;
    } else {
        auto fickParams = std::make_shared<DiffusionNonlinearFEOperatorParameters>( fickDb );
        d_FickOperator.reset( new DiffusionNonlinearFEOperator( fickParams ) );
    }
    if ( fsParams->d_SoretOperator ) {
        d_SoretOperator = fsParams->d_SoretOperator;
    } else {
        auto soretParams = std::make_shared<DiffusionNonlinearFEOperatorParameters>( soretDb );
        d_SoretOperator.reset( new DiffusionNonlinearFEOperator( soretParams ) );
    }
}

void FickSoretNonlinearFEOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                          AMP::LinearAlgebra::Vector::shared_ptr r )
{
    // apply Soret operator and store in r
    if ( d_AddSoretTerm ) {
        d_SoretOperator->apply( u, r );
    }

    // apply Fick operator and store in r.
    d_FickOperator->apply( u, r );
}

void FickSoretNonlinearFEOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    auto fsParams =
        std::dynamic_pointer_cast<const FickSoretNonlinearFEOperatorParameters>( params );
    d_FickOperator->reset( fsParams->d_FickParameters );
    d_SoretOperator->reset( fsParams->d_SoretParameters );
}

} // namespace AMP::Operator
