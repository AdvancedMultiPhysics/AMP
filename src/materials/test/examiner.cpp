/*
 * examiner.cc
 *
 *  Created on: Sep 17, 2010
 *      Author: gad
 */

#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
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

#include "AMP/materials/Material.h"
#include "AMP/materials/Property.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Factory.h"

// Allow external materials to include additional headers in the test
// Note: this includes 1 additional include header that is passed from the command line:
//   Ex:  -D EXTRA_MATERIAL_HEADER='"materials/FuelMaterial.h"'
#ifdef EXTRA_MATERIAL_HEADER
#include EXTRA_MATERIAL_HEADER
#endif

using namespace AMP;

/**
 * Examine the values put out by a material model.
 */

size_t nhelp     = 26;
string helpmsg[] = { "usage: examiner -h",
                     "	print this message",
                     "",
                     "usage: examines filename",
                     "	read contents of filename and print material property evaluations",
                     "",
                     "output is to stdout",
                     "",
                     "Input file has the form: (comments in parentheses)",
                     "",
                     "Material name",
                     "Property name",
                     "Count_NAME = value",
                     "Low_NAME = value",
                     "High_NAME = value",
                     "(where NAME is the name of a material argument)",
                     "(an alternate specification of evaluation grid is)",
                     "Grid_NAME = [value0 value1 ...]",
                     "Format = TSV | CSV | Mathematica",
                     "(format of the output)"
                     "",
                     "You can mix these two forms of grid specs.",
                     "If Count_* is specified and Low * and High * are not, then the whole "
                     "material argument range is used.",
                     "If Count_* is one, then the midpoint of the range is used.",
                     "TSV=tab-separated values",
                     "CSV=comma-separated values",
                     "" };

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );

    // help message
    if ( argc == 2 && string( argv[1] ) == "-h" ) {
        for ( size_t i = 0; i < nhelp; i++ )
            cout << helpmsg[i] << endl;
        return 1;
    }

    // input section
    string infile( "inputExaminer" );
    if ( argc > 1 ) {
        infile = string( argv[1] );
    }
    auto inDb = AMP::Database::parseInputFile( infile );

    string format;
    format = inDb->getWithDefault<std::string>( "Format", "TSV" );
    AMP_INSIST( format == "TSV" || format == "CSV" || format == "Mathematica",
                "invalid format specified" );

    AMP_INSIST( inDb->keyExists( "Material" ), "must specify material" );
    string matname = inDb->getString( "Material" );

    AMP_INSIST( inDb->keyExists( "Property" ), "must specify material property" );
    string propname = inDb->getString( "Property" );

    // Use the material factory to grab an instance of the material named by matname
    AMP::Materials::Material::shared_ptr material =
        AMP::voodoo::Factory<AMP::Materials::Material>::instance().create( matname );

    // get argument names and ranges
    vector<string> names          = material->property( propname )->get_arguments();
    vector<vector<double>> ranges = material->property( propname )->get_arg_ranges();
    size_t nargs                  = names.size();
    vector<size_t> narg( nargs );
    vector<double> lowarg( nargs ), hiarg( nargs );

    // Create a map that will hold the input variable name and a corresponding pointer to a vector
    // of input values
    std::map<std::string, std::shared_ptr<std::vector<double>>> argMap;
    for ( size_t i = 0; i < nargs; i++ ) {
        argMap.insert( std::make_pair( names[i], std::make_shared<std::vector<double>>( 1 ) ) );
    }

    // Fill in the argument value grid
    vector<vector<double>> args( nargs );
    for ( size_t iarg = 0; iarg < nargs; iarg++ ) {
        string keyNumber = string( "Count_" ) + names[iarg];
        if ( inDb->keyExists( keyNumber ) ) {
            narg[iarg] = inDb->getScalar<int>( keyNumber );
            AMP_INSIST( narg[iarg] >= 1, string( "must have" + keyNumber + " >= 1" ) );

            bool haveLow  = inDb->keyExists( string( "Low_" ) + names[iarg] );
            bool haveHi   = inDb->keyExists( string( "High_" ) + names[iarg] );
            bool haveBoth = haveLow && haveHi;
            if ( haveLow || haveHi )
                AMP_INSIST( haveBoth,
                            string( "must specify Low and High " ) + names[iarg] +
                                string( " together" ) );
            if ( haveBoth ) {
                lowarg[iarg] = inDb->getScalar<double>( string( "Low_" ) + names[iarg] );
                hiarg[iarg]  = inDb->getScalar<double>( string( "High_" ) + names[iarg] );
            } else {
                lowarg[iarg] = ranges[iarg][0];
                hiarg[iarg]  = ranges[iarg][1];
            }
            args[iarg].resize( narg[iarg] );
            if ( narg[iarg] == 1 ) {
                args[iarg][0] = .5 * ( lowarg[iarg] + hiarg[iarg] );
            } else {
                for ( size_t i = 0; i < narg[iarg]; i++ )
                    args[iarg][i] =
                        lowarg[iarg] + i * ( hiarg[iarg] - lowarg[iarg] ) / ( narg[iarg] - 1 );
            }
        } else {
            AMP_INSIST( inDb->keyExists( "Grid_" + names[iarg] ),
                        string( "must specify a Grid for " ) + names[iarg] );
            args[iarg] = inDb->getVector<double>( "Grid_" + names[iarg] );
            narg[iarg] = args[iarg].size();
        }
    }
    size_t nargTotal = 1;
    vector<size_t> argJump( nargs );
    argJump[nargs - 1] = 1;
    for ( size_t i = nargs - 1; i >= 1; i-- ) {
        argJump[i - 1] = argJump[i] * narg[i];
        nargTotal *= narg[i];
    }
    nargTotal *= narg[0];

    // output section, arbitrary dimension loop
    string separator;
    if ( format == "TSV" )
        separator = " ";
    if ( format == "CSV" )
        separator = ",";
    if ( format == "Mathematica" )
        separator = ",";
    if ( format == "Mathematica" ) {
        cout << "(* material = " << matname << ", property = " << propname << " *)" << endl;
        cout << "sizes={";
        for ( size_t i = 0; i < nargs; i++ ) {
            cout << narg[i];
            if ( i < nargs - 1 )
                cout << ",";
        }
        cout << "};" << endl << endl;
        cout << "values={" << endl;
    }
    for ( size_t m = 0; m < nargTotal; m++ ) {
        vector<size_t> indices( nargs );
        indices[0] = m / argJump[0];
        for ( size_t i = 1; i < nargs; i++ )
            indices[i] = ( m - indices[i - 1] * argJump[i - 1] ) / argJump[i];
        for ( size_t i = 0; i < nargs; i++ ) {
            ( *argMap[names[i]] )[0] = args[i][indices[i]];
        }
        std::vector<double> value( 1 );
        material->property( propname )->evalv( value, argMap );
        if ( format == "Mathematica" )
            cout << "{";
        for ( size_t i = 0; i < nargs; i++ )
            cout << args[i][indices[i]] << separator;
        cout << value[0];
        if ( format == "Mathematica" ) {
            cout << "}";
            if ( m < nargTotal - 1 )
                cout << ",";
            else
                cout << ";";
        }
        cout << endl;
    }
    if ( format == "Mathematica" )
        cout << "};" << endl;

    AMP::AMPManager::shutdown();
}