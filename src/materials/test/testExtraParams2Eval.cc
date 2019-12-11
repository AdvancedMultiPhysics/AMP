
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include <memory>

#include "AMP/materials/Material.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string matname = "UO2_MSRZC_09";
    std::shared_ptr<AMP::Materials::Material> material =
        AMP::voodoo::Factory<AMP::Materials::Material>::instance().create( matname );

    std::map<std::string, std::shared_ptr<std::vector<double>>> inputMaterialParameters;

    std::string temperatureString = "temperature";
    std::string burnupString      = "burnup";
    std::string oxygenString      = "concentration";

    std::shared_ptr<std::vector<double>> tempVec( new std::vector<double> );
    std::shared_ptr<std::vector<double>> burnupVec( new std::vector<double> );
    std::shared_ptr<std::vector<double>> oxygenVec( new std::vector<double> );

    tempVec->push_back( 310.0 );
    burnupVec->push_back( 0.0 );
    oxygenVec->push_back( 0.0 );

    std::vector<double> YM( 1 );
    std::vector<double> PR( 1 );

    std::string ymString = "YoungsModulus";
    std::string prString = "PoissonRatio";

    for ( int i = 0; i < 2; i++ ) {
        if ( i == 0 ) {
            inputMaterialParameters.insert( std::make_pair( temperatureString, tempVec ) );
            inputMaterialParameters.insert( std::make_pair( oxygenString, oxygenVec ) );
            material->property( ymString )->evalv( YM, inputMaterialParameters );
            material->property( prString )->evalv( PR, inputMaterialParameters );
            std::cout << exeName << ": Passed if burnup is NOT used." << std::endl;
        } else {
            inputMaterialParameters.insert( std::make_pair( oxygenString, oxygenVec ) );
            inputMaterialParameters.insert( std::make_pair( burnupString, burnupVec ) );
            inputMaterialParameters.insert( std::make_pair( temperatureString, tempVec ) );
            material->property( ymString )->evalv( YM, inputMaterialParameters );
            material->property( prString )->evalv( PR, inputMaterialParameters );
            std::cout << exeName << ": Passed if burnup is used." << std::endl;
        }
    } // end for i

    ut->passes( exeName );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testExtraParams2Eval";

    myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
