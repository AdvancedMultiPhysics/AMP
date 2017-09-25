/*
 * testMaterial.cc
 *
 *  Created on: Feb 11, 2010
 *      Author: gad
 */

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <valarray>
#include <vector>
using std::cout;
using std::endl;
using std::exception;
using std::map;
using std::string;
using std::valarray;
using std::vector;

#include "materials/Material.h"
#include "materials/Property.h"
#include "materials/TensorProperty.h"
#include "materials/VectorProperty.h"
#include "utils/AMPManager.h"
#include "utils/AMP_MPI.h"
#include "utils/Factory.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "vectors/MultiVector.h"
#include "vectors/SimpleVector.h"

// Allow external materials to include additional headers in the test
// Note: this includes 1 additional include header that is passed from the command line:
//   Ex:  -D EXTRA_MATERIAL_HEADER='"materials/FuelMaterial.h"'
#ifdef EXTRA_MATERIAL_HEADER
#include EXTRA_MATERIAL_HEADER
#endif

static const size_t NSUCCESS = 12;
static const size_t NARGEVAL = 2;
static const size_t NVECTOR  = 6;
static const size_t NTENSOR  = 6;

class PropTestResult
{
public:
    PropTestResult()
        : range( false ),
          params( false ),
          unknown( false ),
          name( "none" ),
          isVector( false ),
          isTensor( false )
    {
        for ( auto &elem : success )
            elem = false;
        for ( auto &elem : nargeval )
            elem = false;
        for ( auto &elem : vector )
            elem = false;
        for ( auto &elem : tensor )
            elem = false;
    }
    bool range;
    bool success[NSUCCESS];
    bool params;
    bool nargeval[NARGEVAL];
    bool unknown;
    string name;
    bool vector[NVECTOR];
    bool isVector;
    bool tensor[NTENSOR];
    bool isTensor;
};

class MatTestResult
{
public:
    MatTestResult() : undefined( false ), unknown( false ), creationGood( false ), name( "none" ) {}
    bool undefined;
    bool unknown;
    bool creationGood;
    string name;
    vector<PropTestResult> propResults;
};

MatTestResult testMaterial( string &name )
{
    using namespace AMP::Materials;
    MatTestResult results;

    // create material object
    Material::shared_ptr mat;
    try {
        mat = AMP::voodoo::Factory<AMP::Materials::Material>::instance().create( name );
        results.creationGood = true;
    } catch ( std::exception ) {
        results.creationGood = false;
    } catch ( ... ) {
        results.unknown = true;
    }

    // check for undefined property
    try {
        mat->property( "RiDiCuLoUs#!$^&*Name" );
    } catch ( std::exception & ) {
        results.undefined = true;
    } catch ( ... ) {
        results.undefined = false;
        results.unknown   = true;
    }

    // test property evaluations
    vector<string> proplist( mat->list() );
    size_t nprop = proplist.size();
    results.propResults.resize( nprop );
    for ( size_t type = 0; type < proplist.size(); type++ ) {

        string propname                = proplist[type];
        results.propResults[type].name = propname;
        auto property                  = mat->property( propname );

        // test parameter get and set
        try {
            valarray<double> params = property->get_parameters();
            if ( params.size() > 0 ) {
                params *= 10.0;
                property->set_parameters( &params[0], params.size() );
                params /= 10.0;
                property->set_parameters( &params[0], params.size() );
                valarray<double> nparams = property->get_parameters();
                bool good = abs( nparams - params ).max() <= 1.e-10 * abs( params ).max();
                if ( good )
                    results.propResults[type].params = true;
                else
                    results.propResults[type].params = false;
            } else {
                results.propResults[type].params = true;
            }
        } catch ( std::exception & ) {
            results.propResults[type].params = false;
        } catch ( ... ) {
            results.propResults[type].params  = false;
            results.propResults[type].unknown = true;
        }

        // get argument info
        vector<string> argnames( property->get_arguments() );
        size_t nargs = property->get_number_arguments();

        // get min and max arg values
        vector<vector<double>> ranges = property->get_arg_ranges();
        const size_t npoints          = 10;
        vector<vector<double>> toosmall( nargs, vector<double>( npoints ) );
        vector<vector<double>> justright( nargs, vector<double>( npoints ) );
        vector<vector<double>> toobig( nargs, vector<double>( npoints ) );
        for ( size_t j = 0; j < npoints; j++ ) {
            for ( size_t i = 0; i < ranges.size(); i++ ) {
                if ( ranges[i][0] > 0. )
                    toosmall[i][j] = 0.;
                else if ( ranges[i][0] < 0. )
                    toosmall[i][j] = 2. * ranges[i][0];
                else
                    toosmall[i][j] = -1.;

                justright[i][j] = .5 * ( ranges[i][1] + ranges[i][0] );

                if ( ranges[i][1] > 0. )
                    toobig[i][j] = 2. * ranges[i][1];
                else if ( ranges[i][1] < 0. )
                    toobig[i][j] = 0.;
                else
                    toobig[i][j] = 1.;
            }
        }

        // set up AMP::SimpleVector versions of above
        std::vector<AMP::LinearAlgebra::Variable::shared_ptr> toosmallVar( nargs );
        std::vector<AMP::LinearAlgebra::Variable::shared_ptr> justrightVar( nargs );
        std::vector<AMP::LinearAlgebra::Variable::shared_ptr> toobigVar( nargs );
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> toosmallVec( nargs );
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> justrightVec( nargs );
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> toobigVec( nargs );
        for ( size_t i = 0; i < nargs; i++ ) {
            std::stringstream istr;
            istr << i;
            toosmallVar[i].reset( new AMP::LinearAlgebra::Variable( "toosmall" + istr.str() ) );
            justrightVar[i].reset( new AMP::LinearAlgebra::Variable( "justright" + istr.str() ) );
            toobigVar[i].reset( new AMP::LinearAlgebra::Variable( "toobig" + istr.str() ) );
            toosmallVec[i] =
                AMP::LinearAlgebra::SimpleVector<double>::create( npoints, toosmallVar[i] );
            justrightVec[i] =
                AMP::LinearAlgebra::SimpleVector<double>::create( npoints, justrightVar[i] );
            toobigVec[i] =
                AMP::LinearAlgebra::SimpleVector<double>::create( npoints, toobigVar[i] );
            for ( size_t j = 0; j < npoints; j++ ) {
                toosmallVec[i]->setValueByLocalID( j, toosmall[i][j] );
                justrightVec[i]->setValueByLocalID( j, justright[i][j] );
                toobigVec[i]->setValueByLocalID( j, toobig[i][j] );
            }
        }

        // set up std::vector arguments to evalv
        std::vector<double> value( npoints ), nominal;
        map<string, AMP::shared_ptr<vector<double>>> args;
        for ( size_t i = 0; i < nargs; i++ ) {
            args.insert(
                std::make_pair( argnames[i], AMP::make_shared<vector<double>>( justright[i] ) ) );
        }

        // set up AMP::Vector arguments to evalv
        AMP::LinearAlgebra::Variable::shared_ptr valueVar(
            new AMP::LinearAlgebra::Variable( "value" ) );
        AMP::LinearAlgebra::Vector::shared_ptr valueVec =
            AMP::LinearAlgebra::SimpleVector<double>::create( npoints, valueVar );
        AMP::LinearAlgebra::Variable::shared_ptr nominalVar(
            new AMP::LinearAlgebra::Variable( "nominal" ) );
        AMP::LinearAlgebra::Vector::shared_ptr nominalVec =
            AMP::LinearAlgebra::SimpleVector<double>::create( npoints, nominalVar );
        map<string, AMP::LinearAlgebra::Vector::shared_ptr> argsVec;
        for ( size_t i = 0; i < nargs; i++ ) {
            argsVec.insert( std::make_pair( argnames[i], justrightVec[i] ) );
        }

        // set up AMP::MultiVector arguments to evalv
        AMP::LinearAlgebra::Vector::shared_ptr argsMultiVecVec =
            AMP::LinearAlgebra::MultiVector::create( "argsMultiVec", AMP_COMM_SELF );
        AMP::shared_ptr<AMP::LinearAlgebra::MultiVector> argsMultiVec =
            AMP::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( argsMultiVecVec );
        for ( size_t i = 0; i < nargs; i++ ) {
            argsMultiVec->addVector( toosmallVec[i] );  // extra junk, should be ignored
            argsMultiVec->addVector( justrightVec[i] ); // paydirt
            argsMultiVec->addVector( toobigVec[i] );    // extra junk, should be ignored
        }
        std::map<std::string, std::string> xlator;
        int count = 0;
        for ( size_t i = 0; i < nargs; i++ ) {
            std::string name = justrightVar[i]->getName();
            xlator.insert( std::make_pair( argnames[i], name ) );
            count++;
        }
        AMP::LinearAlgebra::Variable::shared_ptr nominalMultiVar(
            new AMP::LinearAlgebra::Variable( "nominalMulti" ) );
        AMP::LinearAlgebra::Vector::shared_ptr nominalMultiVec =
            AMP::LinearAlgebra::SimpleVector<double>::create( npoints, nominalMultiVar );

        // test material range functions
        vector<double> range( 2 );
        bool good = true;
        good      = ( good && ( nargs == argnames.size() ) );
        for ( size_t i = 0; i < argnames.size(); i++ ) {
            range = property->get_arg_range( argnames[i] );
            good  = good && range[0] <= range[1];
            good  = good && ( range[0] == ranges[i][0] ) && ( range[1] == ranges[i][1] );
            good  = good && property->in_range( argnames[i], justright[i] );
            good  = good && !property->in_range( argnames[i], toosmall[i] );
            good  = good && !property->in_range( argnames[i], toobig[i] );
            good  = good && property->in_range( argnames[i], justright[i][0] );
            good  = good && !property->in_range( argnames[i], toosmall[i][0] );
            good  = good && !property->in_range( argnames[i], toobig[i][0] );

            good = good && property->in_range( argnames[i], *justrightVec[i] );
            good = good && !property->in_range( argnames[i], *toosmallVec[i] );
            good = good && !property->in_range( argnames[i], *toobigVec[i] );
        }
        if ( good )
            results.propResults[type].range = true;

        // test defaults get and set
        try {
            auto prop = property;
            vector<double> defin( nargs );
            for ( size_t i = 0; i < nargs; i++ )
                defin[i] = justright[i][0];
            prop->set_defaults( defin );
            vector<double> defaults( prop->get_defaults() );
            if ( defaults == defin )
                results.propResults[type].nargeval[0] = true;
            else
                results.propResults[type].nargeval[0] = false;
        } catch ( std::exception & ) {
            results.propResults[type].nargeval[0] = false;
        } catch ( ... ) {
            results.propResults[type].nargeval[0] = false;
            results.propResults[type].unknown     = true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        // Scalar Property
        /////////////////////////////////////////////////////////////////////////////////////////
        if ( property->isScalar() ) {

            // all in range, std::vector
            try {
                property->evalv( value, args );
                nominal                              = value;
                results.propResults[type].success[0] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[0] = false;
            } catch ( ... ) {
                results.propResults[type].success[0] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, std::vector
            if ( !args.empty() ) {
                try {
                    args.find( argnames[0] )->second->operator[]( 5 ) = toosmall[0][5];
                    property->evalv( value, args );
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                    results.propResults[type].success[1]              = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[1]              = true;
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                } catch ( ... ) {
                    results.propResults[type].success[1] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[1] = true;
            }

            // first out of range hi, std::vector
            if ( !args.empty() ) {
                try {
                    args.find( argnames[0] )->second->operator[]( 5 ) = toobig[0][5];
                    property->evalv( value, args );
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                    results.propResults[type].success[2]              = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[2]              = true;
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                } catch ( ... ) {
                    results.propResults[type].success[2] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[2] = true;
            }

            // all in range, AMP::Vector
            try {
                property->evalv( valueVec, argsVec );
                nominalVec->copyVector( valueVec );
                results.propResults[type].success[4] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[4] = false;
            } catch ( ... ) {
                results.propResults[type].success[4] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, AMP::Vector
            if ( nargs > 0 ) {
                try {
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, toosmall[0][5] );
                    property->evalv( valueVec, argsVec );
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[5] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[5] = true;
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[5] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[5] = true;
            }

            // first out of range hi, AMP::Vector
            if ( nargs > 0 ) {
                try {
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, toobig[0][5] );
                    property->evalv( valueVec, argsVec );
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[6] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[6] = true;
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[6] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[6] = true;
            }

            // test make_map, first without setting a translator or setting an empty translator
            std::map<std::string, AMP::LinearAlgebra::Vector::shared_ptr> testMap;
            std::map<std::string, std::string> currentXlator = property->get_translator();
            if ( !currentXlator.empty() ) {
                currentXlator.clear();
                property->set_translator( currentXlator );
            }
            bool xlateGood = false;
            if ( nargs > 0 ) {
                try {
                    testMap                              = property->make_map( argsMultiVec );
                    results.propResults[type].success[7] = false;
                } catch ( std::exception & ) {
                    xlateGood = true;
                } catch ( ... ) {
                    results.propResults[type].success[7] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                xlateGood = true;
            }
            property->set_translator( xlator );
            std::map<std::string, std::string> testXlatorGet = property->get_translator();
            if ( testXlatorGet == xlator && xlateGood ) {
                results.propResults[type].success[7] = true;
            }

            // test make_map, now with a translator
            try {
                testMap   = property->make_map( argsMultiVec );
                bool good = true;
                for ( size_t i = 0; i < nargs; i++ ) {
                    auto vec1It = testMap.find( argnames[i] );
                    auto vec2It = argsVec.find( argnames[i] );
                    bool goodIt = true;
                    if ( vec1It == testMap.end() ) {
                        goodIt = false; // make_map missed an argument
                    }
                    if ( vec2It == argsVec.end() ) {
                        AMP_INSIST( false, "argsVec composed incorrectly" );
                    }
                    if ( goodIt ) {
                        good = good && vec1It->second == vec2It->second;
                    }
                }
                if ( good )
                    results.propResults[type].success[8] = true;
                else
                    results.propResults[type].success[8] = false;
            } catch ( std::exception & ) {
                results.propResults[type].success[8] = false;
            } catch ( ... ) {
                results.propResults[type].success[8] = false;
                results.propResults[type].unknown    = true;
            }

            // all in range, AMP::MultiVector
            try {
                property->evalv( valueVec, argsMultiVec );
                nominalMultiVec->copyVector( valueVec );
                results.propResults[type].success[9] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[9] = false;
            } catch ( ... ) {
                results.propResults[type].success[9] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, AMP::MultiVector
            if ( nargs > 0 ) {
                try {
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, toosmall[0][5] );
                    property->evalv( valueVec, argsMultiVec );
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[10] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[10] = true;
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[10] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].success[10] = true;
            }

            // first out of range hi, AMP::MultiVector
            if ( nargs > 0 ) {
                try {
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, toobig[0][5] );
                    property->evalv( valueVec, argsMultiVec );
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[11] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[11] = true;
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[11] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].success[11] = true;
            }

            // check vector, Vector, MultiVector all agree
            good = true;
            if ( results.propResults[type].success[0] && results.propResults[type].success[4] &&
                 results.propResults[type].success[9] ) {
                for ( size_t i = 0; i < npoints; i++ ) {
                    double vstd      = nominal[i];
                    double vVec      = nominalVec->getValueByLocalID( i );
                    double vMultiVec = nominalMultiVec->getValueByLocalID( i );
                    good             = good && ( vstd == vVec && vVec == vMultiVec );
                }
                if ( good )
                    results.propResults[type].success[3] = true;
            } else {
                results.propResults[type].success[3] = false;
            }

            // set up reduced argument list
            map<string, AMP::shared_ptr<vector<double>>> argsm( args );
            if ( nargs > 0 ) {
                auto argend = argsm.end();
                --argend;
                argsm.erase( argend );
            }

            // check that evalv with fewer than normal number of arguments works
            if ( results.propResults[type].success[0] ) {
                try {
                    property->evalv( value, argsm );
                    results.propResults[type].nargeval[1] = true;
                } catch ( std::exception & ) {
                    results.propResults[type].nargeval[1] = false;
                } catch ( ... ) {
                    results.propResults[type].nargeval[1] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].nargeval[1] = false;
            }


            /////////////////////////////////////////////////////////////////////////////////////////////
            // Vector Property
            /////////////////////////////////////////////////////////////////////////////////////////////
        } else if ( property->isVector() ) {

            results.propResults[type].isVector = true;

            AMP::shared_ptr<AMP::Materials::VectorProperty<double>> vectorProperty =
                AMP::dynamic_pointer_cast<AMP::Materials::VectorProperty<double>>( property );

            // check that scalar nature is not signaled
            if ( vectorProperty->isScalar() ) {
                results.propResults[type].vector[2] = false;
            } else {
                results.propResults[type].vector[2] = true;
            }

            // check scalar evaluator for std::vector disabled
            try {
                vectorProperty->evalv( value, args );
                results.propResults[type].vector[3] = false;
            } catch ( std::exception & ) {
                results.propResults[type].vector[3] = true;
            } catch ( ... ) {
                results.propResults[type].vector[3] = false;
                results.propResults[type].unknown   = true;
            }

            // check scalar evaluator for AMP::Vector disabled
            try {
                vectorProperty->evalv( valueVec, argsVec );
                results.propResults[type].vector[4] = false;
            } catch ( std::exception & ) {
                results.propResults[type].vector[4] = true;
            } catch ( ... ) {
                results.propResults[type].vector[4] = false;
                results.propResults[type].unknown   = true;
            }

            // test make_map, first without setting a translator or setting an empty translator
            std::map<std::string, AMP::LinearAlgebra::Vector::shared_ptr> testMap;
            std::map<std::string, std::string> currentXlator = vectorProperty->get_translator();
            if ( !currentXlator.empty() ) {
                currentXlator.clear();
                vectorProperty->set_translator( currentXlator );
            }
            bool xlateGood = false;
            if ( nargs > 0 ) {
                try {
                    testMap                              = vectorProperty->make_map( argsMultiVec );
                    results.propResults[type].success[7] = false;
                } catch ( std::exception & ) {
                    xlateGood = true;
                } catch ( ... ) {
                    results.propResults[type].success[7] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                xlateGood = true;
            }
            vectorProperty->set_translator( xlator );
            std::map<std::string, std::string> testXlatorGet = vectorProperty->get_translator();
            if ( testXlatorGet == xlator && xlateGood ) {
                results.propResults[type].success[7] = true;
            }

            // test make_map, now with a translator
            try {
                testMap   = vectorProperty->make_map( argsMultiVec );
                bool good = true;
                for ( size_t i = 0; i < nargs; i++ ) {
                    auto vec1It = testMap.find( argnames[i] );
                    auto vec2It = argsVec.find( argnames[i] );
                    bool goodIt = true;
                    if ( vec1It == testMap.end() ) {
                        goodIt = false; // make_map missed an argument
                    }
                    if ( vec2It == argsVec.end() ) {
                        AMP_INSIST( false, "argsVec composed incorrectly" );
                    }
                    if ( goodIt ) {
                        good = good && vec1It->second == vec2It->second;
                    }
                }
                if ( good )
                    results.propResults[type].success[8] = true;
                else
                    results.propResults[type].success[8] = false;
            } catch ( std::exception & ) {
                results.propResults[type].success[8] = false;
            } catch ( ... ) {
                results.propResults[type].success[8] = false;
                results.propResults[type].unknown    = true;
            }

            // check scalar evaluator for AMP::MultiVector disabled
            try {
                vectorProperty->evalv( valueVec, argsMultiVec );
                results.propResults[type].vector[5] = false;
            } catch ( std::exception & ) {
                results.propResults[type].vector[5] = true;
            } catch ( ... ) {
                results.propResults[type].vector[5] = false;
                results.propResults[type].unknown   = true;
            }

            // prepare results vector, check for reasonable size info
            size_t nvec = 0;
            try {
                nvec                                = vectorProperty->get_dimension();
                results.propResults[type].vector[0] = true;
            } catch ( std::exception & ) {
                results.propResults[type].vector[0] = false;
            } catch ( ... ) {
                results.propResults[type].vector[0] = false;
                results.propResults[type].unknown   = true;
            }
            std::vector<AMP::shared_ptr<std::vector<double>>> stdEval( nvec );
            std::vector<AMP::shared_ptr<std::vector<double>>> nominalEval( nvec );
            for ( size_t i = 0; i < nvec; i++ ) {
                stdEval[i]     = AMP::make_shared<std::vector<double>>( npoints );
                nominalEval[i] = AMP::make_shared<std::vector<double>>( npoints );
            }

            // check that number of components is positive
            if ( results.propResults[type].vector[0] && nvec > 0 ) {
                results.propResults[type].vector[1] = true;
            } else {
                results.propResults[type].vector[1] = false;
            }

            // all in range, std::vector
            try {
                vectorProperty->evalv( stdEval, args );
                nominalEval                          = stdEval;
                results.propResults[type].success[0] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[0] = false;
            } catch ( ... ) {
                results.propResults[type].success[0] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, std::vector
            if ( !args.empty() ) {
                try {
                    args.find( argnames[0] )->second->operator[]( 5 ) = toosmall[0][5];
                    vectorProperty->evalv( stdEval, args );
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                    results.propResults[type].success[1]              = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[1]              = true;
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                } catch ( ... ) {
                    results.propResults[type].success[1] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[1] = true;
            }

            // first out of range hi, std::vector
            if ( !args.empty() ) {
                try {
                    args.find( argnames[0] )->second->operator[]( 5 ) = toobig[0][5];
                    vectorProperty->evalv( stdEval, args );
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                    results.propResults[type].success[2]              = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[2]              = true;
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                } catch ( ... ) {
                    results.propResults[type].success[2] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[2] = true;
            }

            // setup AMP::Vector evalv results
            std::vector<AMP::LinearAlgebra::Variable::shared_ptr> ampEvalVar( nvec );
            std::vector<AMP::LinearAlgebra::Vector::shared_ptr> ampEval( nvec );
            std::vector<AMP::LinearAlgebra::Variable::shared_ptr> nominalAmpEvalVar( nvec );
            std::vector<AMP::LinearAlgebra::Vector::shared_ptr> nominalAmpEval( nvec );
            std::vector<AMP::LinearAlgebra::Variable::shared_ptr> nominalMultiEvalVar( nvec );
            std::vector<AMP::LinearAlgebra::Vector::shared_ptr> nominalMultiEval( nvec );
            for ( size_t i = 0; i < nvec; i++ ) {
                std::stringstream istr;
                istr << i;
                ampEvalVar[i].reset( new AMP::LinearAlgebra::Variable( "ampEval" + istr.str() ) );
                ampEval[i] =
                    AMP::LinearAlgebra::SimpleVector<double>::create( npoints, ampEvalVar[i] );
                nominalAmpEvalVar[i].reset(
                    new AMP::LinearAlgebra::Variable( "nominalAmpEval" + istr.str() ) );
                nominalAmpEval[i] = AMP::LinearAlgebra::SimpleVector<double>::create(
                    npoints, nominalAmpEvalVar[i] );
                nominalMultiEvalVar[i].reset(
                    new AMP::LinearAlgebra::Variable( "nominalMultiEval" + istr.str() ) );
                nominalMultiEval[i] = AMP::LinearAlgebra::SimpleVector<double>::create(
                    npoints, nominalMultiEvalVar[i] );
            }

            // all in range, AMP::Vector
            try {
                vectorProperty->evalv( ampEval, argsVec );
                for ( size_t i = 0; i < nvec; i++ )
                    nominalAmpEval[i]->copyVector( ampEval[i] );
                results.propResults[type].success[4] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[4] = false;
            } catch ( ... ) {
                results.propResults[type].success[4] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, AMP::Vector
            if ( nargs > 0 ) {
                try {
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, toosmall[0][5] );
                    vectorProperty->evalv( ampEval, argsVec );
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[5] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[5] = true;
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[5] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[5] = true;
            }

            // first out of range hi, AMP::Vector
            if ( nargs > 0 ) {
                try {
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, toobig[0][5] );
                    vectorProperty->evalv( ampEval, argsVec );
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[6] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[6] = true;
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[6] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[6] = true;
            }

            // all in range, AMP::MultiVector
            try {
                vectorProperty->evalv( ampEval, argsMultiVec );
                for ( size_t i = 0; i < nvec; i++ )
                    nominalMultiEval[i]->copyVector( ampEval[i] );
                results.propResults[type].success[9] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[9] = false;
            } catch ( ... ) {
                results.propResults[type].success[9] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, AMP::MultiVector
            if ( nargs > 0 ) {
                try {
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, toosmall[0][5] );
                    vectorProperty->evalv( ampEval, argsMultiVec );
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[10] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[10] = true;
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[10] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].success[10] = true;
            }

            // first out of range hi, AMP::MultiVector
            if ( nargs > 0 ) {
                try {
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, toobig[0][5] );
                    vectorProperty->evalv( ampEval, argsMultiVec );
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[11] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[11] = true;
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[11] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].success[11] = true;
            }

            // check vector, Vector, MultiVector all agree
            good = true;
            if ( results.propResults[type].success[0] && results.propResults[type].success[4] &&
                 results.propResults[type].success[9] ) {
                for ( size_t j = 0; j < nvec; j++ ) {
                    for ( size_t i = 0; i < npoints; i++ ) {
                        double vstd      = ( *nominalEval[j] )[i];
                        double vVec      = nominalAmpEval[j]->getValueByLocalID( i );
                        double vMultiVec = nominalMultiEval[j]->getValueByLocalID( i );
                        good             = good && ( vstd == vVec && vVec == vMultiVec );
                    }
                }
                if ( good )
                    results.propResults[type].success[3] = true;
            } else {
                results.propResults[type].success[3] = false;
            }

            // set up reduced argument list
            map<string, AMP::shared_ptr<vector<double>>> argsm( args );
            if ( nargs > 0 ) {
                auto argend = argsm.end();
                --argend;
                argsm.erase( argend );
            }

            // check that evalv with fewer than normal number of arguments works
            if ( results.propResults[type].success[0] ) {
                try {
                    vectorProperty->evalv( stdEval, argsm );
                    results.propResults[type].nargeval[1] = true;
                } catch ( std::exception & ) {
                    results.propResults[type].nargeval[1] = false;
                } catch ( ... ) {
                    results.propResults[type].nargeval[1] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].nargeval[1] = false;
            }

            /////////////////////////////////////////////////////////////////////////////////////////////
            // Tensor Property
            /////////////////////////////////////////////////////////////////////////////////////////////
        } else if ( property->isTensor() ) {

            results.propResults[type].isTensor = true;

            AMP::shared_ptr<AMP::Materials::TensorProperty<double>> tensorProperty =
                AMP::dynamic_pointer_cast<AMP::Materials::TensorProperty<double>>( property );

            // check that scalar nature is not signaled
            if ( tensorProperty->isScalar() ) {
                results.propResults[type].tensor[2] = false;
            } else {
                results.propResults[type].tensor[2] = true;
            }

            // check scalar evaluator for std::vector disabled
            try {
                tensorProperty->evalv( value, args );
                results.propResults[type].tensor[3] = false;
            } catch ( std::exception & ) {
                results.propResults[type].tensor[3] = true;
            } catch ( ... ) {
                results.propResults[type].tensor[3] = false;
                results.propResults[type].unknown   = true;
            }

            // check scalar evaluator for AMP::Vector disabled
            try {
                tensorProperty->evalv( valueVec, argsVec );
                results.propResults[type].tensor[4] = false;
            } catch ( std::exception & ) {
                results.propResults[type].tensor[4] = true;
            } catch ( ... ) {
                results.propResults[type].tensor[4] = false;
                results.propResults[type].unknown   = true;
            }

            // test make_map, first without setting a translator or setting an empty translator
            std::map<std::string, AMP::LinearAlgebra::Vector::shared_ptr> testMap;
            std::map<std::string, std::string> currentXlator = tensorProperty->get_translator();
            if ( !currentXlator.empty() ) {
                currentXlator.clear();
                tensorProperty->set_translator( currentXlator );
            }
            bool xlateGood = false;
            if ( nargs > 0 ) {
                try {
                    testMap                              = tensorProperty->make_map( argsMultiVec );
                    results.propResults[type].success[7] = false;
                } catch ( std::exception & ) {
                    xlateGood = true;
                } catch ( ... ) {
                    results.propResults[type].success[7] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                xlateGood = true;
            }
            tensorProperty->set_translator( xlator );
            std::map<std::string, std::string> testXlatorGet = tensorProperty->get_translator();
            if ( testXlatorGet == xlator && xlateGood ) {
                results.propResults[type].success[7] = true;
            }

            // test make_map, now with a translator
            try {
                testMap   = tensorProperty->make_map( argsMultiVec );
                bool good = true;
                for ( size_t i = 0; i < nargs; i++ ) {
                    auto vec1It = testMap.find( argnames[i] );
                    auto vec2It = argsVec.find( argnames[i] );
                    bool goodIt = true;
                    if ( vec1It == testMap.end() ) {
                        goodIt = false; // make_map missed an argument
                    }
                    if ( vec2It == argsVec.end() ) {
                        AMP_INSIST( false, "argsVec composed incorrectly" );
                    }
                    if ( goodIt ) {
                        good = good && vec1It->second == vec2It->second;
                    }
                }
                if ( good )
                    results.propResults[type].success[8] = true;
                else
                    results.propResults[type].success[8] = false;
            } catch ( std::exception & ) {
                results.propResults[type].success[8] = false;
            } catch ( ... ) {
                results.propResults[type].success[8] = false;
                results.propResults[type].unknown    = true;
            }

            // check scalar evaluator for AMP::MultiVector disabled
            try {
                tensorProperty->evalv( valueVec, argsMultiVec );
                results.propResults[type].tensor[5] = false;
            } catch ( std::exception & ) {
                results.propResults[type].tensor[5] = true;
            } catch ( ... ) {
                results.propResults[type].tensor[5] = false;
                results.propResults[type].unknown   = true;
            }

            // prepare results vector, check for reasonable size info
            std::vector<size_t> nvecs( 2, 0U );
            try {
                nvecs                               = tensorProperty->get_dimensions();
                results.propResults[type].tensor[0] = true;
            } catch ( std::exception & ) {
                results.propResults[type].tensor[0] = false;
            } catch ( ... ) {
                results.propResults[type].tensor[0] = false;
                results.propResults[type].unknown   = true;
            }
            std::vector<std::vector<AMP::shared_ptr<std::vector<double>>>> stdEval(
                nvecs[0], std::vector<AMP::shared_ptr<std::vector<double>>>( nvecs[1] ) );
            std::vector<std::vector<AMP::shared_ptr<std::vector<double>>>> nominalEval(
                nvecs[0], std::vector<AMP::shared_ptr<std::vector<double>>>( nvecs[1] ) );
            for ( size_t i = 0; i < nvecs[0]; i++ )
                for ( size_t j = 0; j < nvecs[1]; j++ ) {
                    stdEval[i][j]     = AMP::make_shared<std::vector<double>>( npoints );
                    nominalEval[i][j] = AMP::make_shared<std::vector<double>>( npoints );
                }

            // check that number of components is positive
            if ( results.propResults[type].tensor[0] && nvecs[0] > 0 && nvecs[1] > 0 &&
                 nvecs.size() == 2 ) {
                results.propResults[type].tensor[1] = true;
            } else {
                results.propResults[type].tensor[1] = false;
            }

            // all in range, std::vector
            try {
                tensorProperty->evalv( stdEval, args );
                nominalEval                          = stdEval;
                results.propResults[type].success[0] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[0] = false;
            } catch ( ... ) {
                results.propResults[type].success[0] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, std::vector
            if ( !args.empty() ) {
                try {
                    args.find( argnames[0] )->second->operator[]( 5 ) = toosmall[0][5];
                    tensorProperty->evalv( stdEval, args );
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                    results.propResults[type].success[1]              = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[1]              = true;
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                } catch ( ... ) {
                    results.propResults[type].success[1] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[1] = true;
            }

            // first out of range hi, std::vector
            if ( !args.empty() ) {
                try {
                    args.find( argnames[0] )->second->operator[]( 5 ) = toobig[0][5];
                    tensorProperty->evalv( stdEval, args );
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                    results.propResults[type].success[2]              = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[2]              = true;
                    args.find( argnames[0] )->second->operator[]( 5 ) = justright[0][5];
                } catch ( ... ) {
                    results.propResults[type].success[2] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[2] = true;
            }

            // setup AMP::Vector evalv results
            std::vector<std::vector<AMP::LinearAlgebra::Variable::shared_ptr>> ampEvalVar(
                nvecs[0], std::vector<AMP::LinearAlgebra::Variable::shared_ptr>( nvecs[1] ) );
            std::vector<std::vector<AMP::LinearAlgebra::Vector::shared_ptr>> ampEval(
                nvecs[0], std::vector<AMP::LinearAlgebra::Vector::shared_ptr>( nvecs[1] ) );
            std::vector<std::vector<AMP::LinearAlgebra::Variable::shared_ptr>> nominalAmpEvalVar(
                nvecs[0], std::vector<AMP::LinearAlgebra::Variable::shared_ptr>( nvecs[1] ) );
            std::vector<std::vector<AMP::LinearAlgebra::Vector::shared_ptr>> nominalAmpEval(
                nvecs[0], std::vector<AMP::LinearAlgebra::Vector::shared_ptr>( nvecs[1] ) );
            std::vector<std::vector<AMP::LinearAlgebra::Variable::shared_ptr>> nominalMultiEvalVar(
                nvecs[0], std::vector<AMP::LinearAlgebra::Variable::shared_ptr>( nvecs[1] ) );
            std::vector<std::vector<AMP::LinearAlgebra::Vector::shared_ptr>> nominalMultiEval(
                nvecs[0], std::vector<AMP::LinearAlgebra::Vector::shared_ptr>( nvecs[1] ) );
            for ( size_t i = 0; i < nvecs[0]; i++ )
                for ( size_t j = 0; j < nvecs[1]; j++ ) {
                    std::stringstream istr;
                    istr << i;
                    ampEvalVar[i][j].reset(
                        new AMP::LinearAlgebra::Variable( "ampEval" + istr.str() ) );
                    ampEval[i][j] = AMP::LinearAlgebra::SimpleVector<double>::create(
                        npoints, ampEvalVar[i][j] );
                    nominalAmpEvalVar[i][j].reset(
                        new AMP::LinearAlgebra::Variable( "nominalAmpEval" + istr.str() ) );
                    nominalAmpEval[i][j] = AMP::LinearAlgebra::SimpleVector<double>::create(
                        npoints, nominalAmpEvalVar[i][j] );
                    nominalMultiEvalVar[i][j].reset(
                        new AMP::LinearAlgebra::Variable( "nominalMultiEval" + istr.str() ) );
                    nominalMultiEval[i][j] = AMP::LinearAlgebra::SimpleVector<double>::create(
                        npoints, nominalMultiEvalVar[i][j] );
                }

            // all in range, AMP::Vector
            try {
                tensorProperty->evalv( ampEval, argsVec );
                for ( size_t i = 0; i < nvecs[0]; i++ )
                    for ( size_t j = 0; j < nvecs[1]; j++ )
                        nominalAmpEval[i][j]->copyVector( ampEval[i][j] );
                results.propResults[type].success[4] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[4] = false;
            } catch ( ... ) {
                results.propResults[type].success[4] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, AMP::Vector
            if ( nargs > 0 ) {
                try {
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, toosmall[0][5] );
                    tensorProperty->evalv( ampEval, argsVec );
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[5] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[5] = true;
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[5] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[5] = true;
            }

            // first out of range hi, AMP::Vector
            if ( nargs > 0 ) {
                try {
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, toobig[0][5] );
                    tensorProperty->evalv( ampEval, argsVec );
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[6] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[6] = true;
                    argsVec.find( argnames[0] )->second->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[6] = false;
                    results.propResults[type].unknown    = true;
                }
            } else {
                results.propResults[type].success[6] = true;
            }

            // all in range, AMP::MultiVector
            try {
                tensorProperty->evalv( ampEval, argsMultiVec );
                for ( size_t i = 0; i < nvecs[0]; i++ )
                    for ( size_t j = 0; j < nvecs[1]; j++ )
                        nominalMultiEval[i][j]->copyVector( ampEval[i][j] );
                results.propResults[type].success[9] = true;
            } catch ( std::exception & ) {
                results.propResults[type].success[9] = false;
            } catch ( ... ) {
                results.propResults[type].success[9] = false;
                results.propResults[type].unknown    = true;
            }

            // first out of range low, AMP::MultiVector
            if ( nargs > 0 ) {
                try {
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, toosmall[0][5] );
                    tensorProperty->evalv( ampEval, argsMultiVec );
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[10] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[10] = true;
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[10] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].success[10] = true;
            }

            // first out of range hi, AMP::MultiVector
            if ( nargs > 0 ) {
                try {
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, toobig[0][5] );
                    tensorProperty->evalv( ampEval, argsMultiVec );
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                    results.propResults[type].success[11] = false;
                } catch ( std::exception & ) {
                    results.propResults[type].success[11] = true;
                    argsMultiVec->subsetVectorForVariable( justrightVec[0]->getVariable() )
                        ->setValueByLocalID( 5, justright[0][5] );
                } catch ( ... ) {
                    results.propResults[type].success[11] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].success[11] = true;
            }

            // check vector, Vector, MultiVector all agree
            good = true;
            if ( results.propResults[type].success[0] && results.propResults[type].success[4] &&
                 results.propResults[type].success[9] ) {
                for ( size_t k = 0; k < nvecs[0]; k++ ) {
                    for ( size_t j = 0; j < nvecs[1]; j++ ) {
                        for ( size_t i = 0; i < npoints; i++ ) {
                            double vstd      = ( *nominalEval[k][j] )[i];
                            double vVec      = nominalAmpEval[k][j]->getValueByLocalID( i );
                            double vMultiVec = nominalMultiEval[k][j]->getValueByLocalID( i );
                            good             = good && ( vstd == vVec && vVec == vMultiVec );
                        }
                    }
                }
                if ( good )
                    results.propResults[type].success[3] = true;
            } else {
                results.propResults[type].success[3] = false;
            }

            // set up reduced argument list
            map<string, AMP::shared_ptr<vector<double>>> argsm( args );
            if ( nargs > 0 ) {
                auto argend = argsm.end();
                --argend;
                argsm.erase( argend );
            }

            // check that evalv with fewer than normal number of arguments works
            if ( results.propResults[type].success[0] ) {
                try {
                    tensorProperty->evalv( stdEval, argsm );
                    results.propResults[type].nargeval[1] = true;
                } catch ( std::exception & ) {
                    results.propResults[type].nargeval[1] = false;
                } catch ( ... ) {
                    results.propResults[type].nargeval[1] = false;
                    results.propResults[type].unknown     = true;
                }
            } else {
                results.propResults[type].nargeval[1] = false;
            }
        }
    }

    return results;
}

string xlate( bool val )
{
    string y( "yes" ), n( "no " ), r;
    r = val ? y : n;
    return r;
}

int main( int argc, char **argv )
{
    AMP::AMPManagerProperties amprops;
    amprops.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, amprops );
    AMP::UnitTest ut;

    bool good = true;

    using namespace AMP::Materials;

    // test all materials and all properties and print report
    vector<string> matlist = AMP::voodoo::Factory<AMP::Materials::Material>::instance().getKeys();
    vector<MatTestResult> scoreCard;
    cout << "In the following output the labels have the following meaning:\n"
         << "creation:  yes = material was created successfully\n"
         << "undefined: undefined material create was correctly detected\n"
         << "range:     yes=material range functions were successfully tested\n"
         << "params:    yes=get and set property parameters successful\n"
         << "nevalv:     # errors reported/max #, for calls to evalv\n"
         << "nargsize:  # errors reported/max #, for incorrect number of arguments to evalv\n"
         << "unknown:   yes=an unknown error occurred during property tests\n\n\n";
    cout << "number of materials = " << matlist.size() << endl;
    cout << "materials = ";
    for ( auto &elem : matlist )
        cout << elem << " ";
    cout << endl;
    for ( auto &elem : matlist ) {
        MatTestResult score = testMaterial( elem );
        score.name          = elem;
        scoreCard.push_back( score );
        cout << "for material " << elem << ": ";
        cout << "creation=" << xlate( score.creationGood ) << " ";
        cout << "undefined=" << xlate( score.undefined ) << " ";
        cout << "unknown=" << xlate( score.unknown ) << " ";
        cout << endl;
        cout << "    property name                           range params nevalv nargsize "
                "unknown"
             << endl;
        for ( auto &_j : score.propResults ) {
            cout << "    ";
            unsigned int osize = cout.width();
            cout.width( 29 );
            cout << _j.name << " ";
            cout.width( osize );
            cout << "          ";
            cout << xlate( _j.range ) << "   ";
            cout << xlate( _j.params ) << "    ";
            unsigned int nsuccess = 0, nargeval = 0;
            for ( bool succes : _j.success )
                if ( succes )
                    nsuccess++;
            cout << nsuccess << "/" << NSUCCESS << "    ";
            for ( bool k : _j.nargeval )
                if ( k )
                    nargeval++;
            cout << nargeval << "/" << NARGEVAL << "      ";
            if ( _j.isVector ) {
                unsigned int nvector = 0;
                for ( bool k : _j.vector )
                    if ( k )
                        nvector++;
                cout << nvector << "/" << NVECTOR << "      ";
            }
            if ( _j.isTensor ) {
                unsigned int ntensor = 0;
                for ( bool k : _j.tensor )
                    if ( k )
                        ntensor++;
                cout << ntensor << "/" << NTENSOR << "      ";
            }
            cout << xlate( _j.unknown ) << "     ";
            cout << endl;
        }
        cout << endl << endl << endl;
    }

    size_t maxpassed = 0;
    for ( auto score : scoreCard ) {

        string msg = "material " + score.name + " ";
        if ( score.creationGood )
            ut.passes( msg + "created" );
        else
            ut.failure( msg + "created" );
        maxpassed += 1;
        for ( auto j = score.propResults.begin(); j != score.propResults.end(); ++j ) {
            msg = "material " + score.name + " property" + " " + j->name + " ";
            if ( j->params )
                ut.passes( msg + "get/set parameters" );
            else
                ut.failure( msg + "get/set parameters" );

            if ( !j->unknown )
                ut.passes( msg + "unknown error" );
            else
                ut.failure( msg + "unknown error" );

            if ( j->range )
                ut.passes( msg + "in_range std::vector" );
            else
                ut.failure( msg + "in_range std::vector" );

            if ( j->success[0] )
                ut.passes( msg + "evalv std::vector" );
            else
                ut.failure( msg + "evalv std::vector" );

            if ( j->success[1] )
                ut.passes( msg + "evalv std::vector out of range lo 1" );
            else
                ut.failure( msg + "evalv std::vector out of range lo 1" );

            if ( j->success[2] )
                ut.passes( msg + "evalv std::vector out of range hi 1" );
            else
                ut.failure( msg + "evalv std::vector out of range hi 1" );

            if ( j->success[4] )
                ut.passes( msg + "evalv AMP::Vector" );
            else
                ut.failure( msg + "evalv AMP::Vector" );

            if ( j->success[5] )
                ut.passes( msg + "evalv AMP::Vector out of range lo 1" );
            else
                ut.failure( msg + "evalv AMP::Vector out of range lo 1" );

            if ( j->success[6] )
                ut.passes( msg + "evalv AMP::Vector out of range hi 1" );
            else
                ut.failure( msg + "evalv AMP::Vector out of range hi 1" );

            if ( j->success[7] )
                ut.passes( msg + "AMP::Multivector translator" );
            else
                ut.failure( msg + "AMP::Multivector translator" );

            if ( j->success[8] )
                ut.passes( msg + "make_map" );
            else
                ut.failure( msg + "make_map" );

            if ( j->success[9] )
                ut.passes( msg + "evalv AMP::MultiVector" );
            else
                ut.failure( msg + "evalv AMP::MultiVector" );

            if ( j->success[10] )
                ut.passes( msg + "evalv AMP::MultiVector out of range lo 1" );
            else
                ut.failure( msg + "evalv AMP::MultiVector out of range lo 1" );

            if ( j->success[11] )
                ut.passes( msg + "evalv AMP::MultiVector out of range hi 1" );
            else
                ut.failure( msg + "evalv AMP::MultiVector out of range hi 1" );

            if ( j->success[3] )
                ut.passes( msg + "evalv agrees std::vector, AMP::Vector, AMP::MultiVector" );
            else
                ut.failure( msg + "evalv agrees std::vector, AMP::Vector, AMP::MultiVector" );

            if ( j->nargeval[0] )
                ut.passes( msg + "set/get defaults" );
            else
                ut.failure( msg + "set/get defaults" );

            if ( j->nargeval[1] )
                ut.passes( msg + "evalv with missing arguments" );
            else
                ut.failure( msg + "evalv with missing arguments" );
            maxpassed += 17;

            if ( j->isVector ) {
                if ( j->vector[0] )
                    ut.passes( msg + "get_dimension() ok" );
                else
                    ut.failure( msg + "get_dimension() ok" );

                if ( j->vector[1] )
                    ut.passes( msg + "number of components positive" );
                else
                    ut.failure( msg + "number of components positive" );

                if ( j->vector[2] )
                    ut.passes( msg + "not a scalar" );
                else
                    ut.failure( msg + "not a scalar" );

                if ( j->vector[3] )
                    ut.passes( msg + "scalar evaluator std::vector disabled" );
                else
                    ut.failure( msg + "scalar evaluator std::vector disabled" );

                if ( j->vector[4] )
                    ut.passes( msg + "scalar evaluator AMP::Vector disabled" );
                else
                    ut.failure( msg + "scalar evaluator AMP::Vector disabled" );

                if ( j->vector[5] )
                    ut.passes( msg + "scalar evaluator AMP::Multivector disabled" );
                else
                    ut.failure( msg + "scalar evaluator AMP::Multivector disabled" );

                maxpassed += 6;
            }
        }
    }
    cout << endl << endl << endl;

    // check that undefined material name is caught
    try {
        string name( "flubber" );
        Material::shared_ptr mat =
            AMP::voodoo::Factory<AMP::Materials::Material>::instance().create( name );
        if ( mat != nullptr )
            maxpassed += 1;
    } catch ( std::exception &err ) {
        string msg = err.what();
        bool check = ( msg == "Unregistered creator" );
        good       = good && check;
        if ( good )
            ut.passes( "detected undefined material" );
        else
            ut.failure( "did not detect undefined material" );
        maxpassed += 1;
    }

    cout << endl << endl << endl;
    cout << "number of tests passed = " << ut.NumPassLocal() << "/" << maxpassed << " possible"
         << endl;
    cout << endl << endl << endl;

    ut.report( 2 );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
