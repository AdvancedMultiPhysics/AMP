#include "AMP/materials/Material.h"
#include "AMP/materials/Property.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <valarray>
#include <vector>


// Allow external materials to include additional headers in the test
// Note: this includes 1 additional include header that is passed from the command line:
//   Ex:  -D EXTRA_MATERIAL_HEADER='"materials/FuelMaterial.h"'
#ifdef EXTRA_MATERIAL_HEADER
    #include EXTRA_MATERIAL_HEADER
#endif


// Helper functions
double maxabs( const std::vector<double> &x )
{
    double y = 0;
    for ( auto v : x )
        y = std::max( y, std::abs( v ) );
    return y;
}
static inline void
record( const std::string &prefix, bool pass, const std::string &test, AMP::UnitTest &ut )
{
    if ( pass )
        ut.passes( prefix + test );
    else
        ut.failure( prefix + test );
}


// Test getting a value out of range
static inline void set( std::vector<double> &vec, size_t index, double value )
{
    vec[index] = value;
}
static inline void set( AMP::LinearAlgebra::Vector &vec, size_t index, double value )
{
    vec.setValuesByLocalID( 1, &index, &value );
}
template<class VEC>
static inline VEC &find( std::map<std::string, VEC> &args, const std::string &var )
{
    return args.find( var )->second;
}
static inline AMP::LinearAlgebra::Vector::shared_ptr
find( std::shared_ptr<AMP::LinearAlgebra::MultiVector> &multivec,
      const std::shared_ptr<AMP::LinearAlgebra::Variable> &var )
{
    return multivec->subsetVectorForVariable( var );
}
template<class ARGS, class VAR, class VAL>
void testArgsRange( const std::shared_ptr<AMP::Materials::Property> &property,
                    ARGS &args,
                    const VAR &var,
                    VAL &value,
                    double outOfRange,
                    double justRight,
                    const std::string &msg,
                    AMP::UnitTest &ut )
{
    bool pass = true;
    auto vec  = find( args, var );
    try {
        set( *vec, 5, outOfRange );
        using Vector = AMP::LinearAlgebra::Vector;
        if constexpr ( std::is_same_v<VAL, Vector> ||
                       std::is_same_v<VAL, std::vector<std::shared_ptr<Vector>>> ||
                       std::is_same_v<VAL, AMP::Array<std::shared_ptr<Vector>>> )
            property->evalv( value, args );
        else
            property->evalv( value, {}, args );
        pass = false;
    } catch ( const std::exception & ) {
        // We caught a std::exception as expected
    } catch ( ... ) {
        pass = false;
    }
    set( *vec, 5, justRight );
    if ( pass )
        ut.passes( msg );
    else
        ut.failure( msg );
}


// Test a given material
void testMaterial( std::string &name, AMP::UnitTest &ut )
{
    // create material object
    std::string matPrefix = "material " + name + " ";
    std::shared_ptr<AMP::Materials::Material> mat;
    try {
        mat = AMP::Materials::getMaterial( name );
    } catch ( const std::exception &e ) {
        std::cerr << "Error creating material " << name << ":\n" << e.what();
    } catch ( ... ) {
        std::cerr << "Error creating material " << name << "\n";
    }
    record( matPrefix, mat != nullptr, "created", ut );
    if ( !mat )
        return;

    // check for undefined property
    record( matPrefix, !mat->property( "RiDiCuLoUs#!$^&*Name" ), "RiDiCuLoUs#!$^&*Name", ut );

    // test property evaluations
    auto proplist = mat->list();
    for ( size_t type = 0; type < proplist.size(); type++ ) {

        auto propname          = proplist[type];
        auto property          = mat->property( propname );
        std::string propPrefix = "material " + name + " property" + " " + propname + " ";

        // get argument info
        std::vector<std::string> argnames( property->get_arguments() );
        size_t nargs = property->get_number_arguments();

        // get min and max arg values
        auto ranges    = property->get_arg_ranges();
        size_t npoints = 10;
        std::vector<std::vector<double>> toosmall( nargs, std::vector<double>( npoints ) );
        std::vector<std::vector<double>> justright( nargs, std::vector<double>( npoints ) );
        std::vector<std::vector<double>> toobig( nargs, std::vector<double>( npoints ) );
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
        std::vector<std::string> justrightname( nargs );
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> toosmallVec( nargs );
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> justrightVec( nargs );
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> toobigVec( nargs );
        for ( size_t i = 0; i < nargs; i++ ) {
            std::string istr  = std::to_string( i );
            justrightname[i]  = "justright" + istr;
            auto toosmallVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "toosmall" + istr );
            auto justrightVar = std::make_shared<AMP::LinearAlgebra::Variable>( justrightname[i] );
            auto toobigVar = std::make_shared<AMP::LinearAlgebra::Variable>( "toobigVar" + istr );
            toosmallVec[i] = AMP::LinearAlgebra::createSimpleVector<double>( npoints, toosmallVar );
            justrightVec[i] =
                AMP::LinearAlgebra::createSimpleVector<double>( npoints, justrightVar );
            toobigVec[i] = AMP::LinearAlgebra::createSimpleVector<double>( npoints, toobigVar );
            for ( size_t j = 0; j < npoints; j++ ) {
                toosmallVec[i]->setValuesByLocalID( 1, &j, &toosmall[i][j] );
                justrightVec[i]->setValuesByLocalID( 1, &j, &justright[i][j] );
                toobigVec[i]->setValuesByLocalID( 1, &j, &toobig[i][j] );
            }
        }

        // set up std::vector arguments to evalv
        std::vector<double> value( npoints ), nominal( npoints );
        std::map<std::string, std::shared_ptr<std::vector<double>>> args;
        for ( size_t i = 0; i < nargs; i++ ) {
            args.insert( std::make_pair( argnames[i],
                                         std::make_shared<std::vector<double>>( justright[i] ) ) );
        }

        // set up AMP::Vector arguments to evalv
        auto valueVar   = std::make_shared<AMP::LinearAlgebra::Variable>( "value" );
        auto valueVec   = AMP::LinearAlgebra::createSimpleVector<double>( npoints, valueVar );
        auto nominalVar = std::make_shared<AMP::LinearAlgebra::Variable>( "nominal" );
        auto nominalVec = AMP::LinearAlgebra::createSimpleVector<double>( npoints, nominalVar );
        std::map<std::string, AMP::LinearAlgebra::Vector::shared_ptr> argsVec;
        for ( size_t i = 0; i < nargs; i++ ) {
            argsVec.insert( std::make_pair( argnames[i], justrightVec[i] ) );
        }

        // set up AMP::MultiVector arguments to evalv
        auto argsMultiVec =
            AMP::LinearAlgebra::MultiVector::create( "argsMultiVec", AMP_COMM_SELF );
        for ( size_t i = 0; i < nargs; i++ ) {
            argsMultiVec->addVector( toosmallVec[i] );  // extra junk, should be ignored
            argsMultiVec->addVector( justrightVec[i] ); // paydirt
            argsMultiVec->addVector( toobigVec[i] );    // extra junk, should be ignored
        }
        std::map<std::string, std::string> xlator;
        for ( size_t i = 0; i < nargs; i++ ) {
            std::string jname = justrightname[i];
            xlator.insert( std::make_pair( argnames[i], jname ) );
        }
        auto nominalMultiVar = std::make_shared<AMP::LinearAlgebra::Variable>( "nominalMulti" );
        auto nominalMultiVec =
            AMP::LinearAlgebra::createSimpleVector<double>( npoints, nominalMultiVar );

        // test material range functions
        bool pass = nargs == argnames.size();
        for ( size_t i = 0; i < argnames.size(); i++ ) {
            auto range = property->get_arg_range( argnames[i] );
            pass       = pass && range[0] <= range[1];
            pass       = pass && ( range[0] == ranges[i][0] ) && ( range[1] == ranges[i][1] );
            pass       = pass && property->in_range( argnames[i], justright[i] );
            pass       = pass && !property->in_range( argnames[i], toosmall[i] );
            pass       = pass && !property->in_range( argnames[i], toobig[i] );
            pass       = pass && property->in_range( argnames[i], justright[i][0] );
            pass       = pass && !property->in_range( argnames[i], toosmall[i][0] );
            pass       = pass && !property->in_range( argnames[i], toobig[i][0] );
            pass       = pass && property->in_range( argnames[i], *justrightVec[i] );
            pass       = pass && !property->in_range( argnames[i], *toosmallVec[i] );
            pass       = pass && !property->in_range( argnames[i], *toobigVec[i] );
        }
        record( propPrefix, pass, "in_range std::vector", ut );

        // test defaults get and set
        try {
            auto prop = property;
            std::vector<double> defin( nargs );
            for ( size_t i = 0; i < nargs; i++ )
                defin[i] = justright[i][0];
            prop->set_defaults( defin );
            std::vector<double> defaults( prop->get_defaults() );
            record( propPrefix, defaults == defin, "set/get defaults", ut );
        } catch ( ... ) {
            record( propPrefix, false, "set/get defaults", ut );
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        // Scalar Property
        /////////////////////////////////////////////////////////////////////////////////////////
        if ( property->isScalar() ) {

            // all in range, std::vector
            try {
                property->evalv( value, {}, args );
                nominal = value;
                record( propPrefix, true, "evalv std::vector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv std::vector", ut );
            }

            // first out of range low/hi, std::vector
            if ( !args.empty() ) {
                testArgsRange( property,
                               args,
                               argnames[0],
                               value,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv std::vector out of range lo 1",
                               ut );
                testArgsRange( property,
                               args,
                               argnames[0],
                               value,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv std::vector out of range hi 1",
                               ut );
            }

            // all in range, AMP::Vector
            try {
                property->evalv( *valueVec, argsVec );
                nominalVec->copyVector( valueVec );
                record( propPrefix, true, "evalv AMP::Vector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv AMP::Vector", ut );
            }

            // first out of range low/hi, AMP::Vector
            if ( nargs > 0 ) {
                testArgsRange( property,
                               argsVec,
                               argnames[0],
                               *valueVec,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::Vector out of range lo 1",
                               ut );
                testArgsRange( property,
                               argsVec,
                               argnames[0],
                               *valueVec,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::Vector out of range hi 1",
                               ut );
            }

            // test make_map
            try {
                auto testMap = property->make_map( argsMultiVec, xlator );
                bool good    = true;
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
                record( propPrefix, good, "make_map", ut );
            } catch ( ... ) {
                record( propPrefix, false, "make_map", ut );
            }

            // all in range, AMP::MultiVector
            try {
                property->evalv( *valueVec, argsMultiVec, xlator );
                nominalMultiVec->copyVector( valueVec );
                record( propPrefix, true, "evalv AMP::MultiVector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv AMP::MultiVector", ut );
            }

            // first out of range low/hi, AMP::MultiVector
            if ( nargs > 0 ) {
                testArgsRange( property,
                               argsMultiVec,
                               justrightVec[0]->getVariable(),
                               *valueVec,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::MultiVector out of range lo 1",
                               ut );
                testArgsRange( property,
                               argsMultiVec,
                               justrightVec[0]->getVariable(),
                               *valueVec,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::MultiVector out of range hi 1",
                               ut );
            }

            // check vector, Vector, MultiVector all agree
            pass = true;
            for ( size_t i = 0; i < npoints; i++ ) {
                double vstd      = nominal[i];
                double vVec      = nominalVec->getValueByLocalID( i );
                double vMultiVec = nominalMultiVec->getValueByLocalID( i );
                pass             = pass && ( vstd == vVec && vVec == vMultiVec );
            }
            record(
                propPrefix, pass, "evalv agrees std::vector, AMP::Vector, AMP::MultiVector", ut );

            // set up reduced argument list
            std::map<std::string, std::shared_ptr<std::vector<double>>> argsm( args );
            if ( nargs > 0 ) {
                auto argend = argsm.end();
                --argend;
                argsm.erase( argend );
            }

            // check that evalv with fewer than normal number of arguments works
            try {
                property->evalv( value, {}, argsm );
                record( propPrefix, true, "evalv with missing arguments", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv with missing arguments", ut );
            }


            /////////////////////////////////////////////////////////////////////////////////////////////
            // Vector Property
            /////////////////////////////////////////////////////////////////////////////////////////////
        } else if ( property->isVector() ) {

            // test make_map
            try {
                auto testMap = property->make_map( argsMultiVec, xlator );
                bool good    = true;
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
                record( propPrefix, good, "make_map", ut );
            } catch ( ... ) {
                record( propPrefix, false, "make_map", ut );
            }

            // prepare results vector, check for reasonable size info
            size_t nvec = 0;
            try {
                nvec = property->size().length();
                record( propPrefix, true, "size() ok", ut );
            } catch ( ... ) {
                record( propPrefix, false, "size() ok", ut );
            }
            std::vector<std::shared_ptr<std::vector<double>>> stdEval( nvec );
            std::vector<std::shared_ptr<std::vector<double>>> nominalEval( nvec );
            for ( size_t i = 0; i < nvec; i++ ) {
                stdEval[i]     = std::make_shared<std::vector<double>>( npoints );
                nominalEval[i] = std::make_shared<std::vector<double>>( npoints );
            }

            // check that number of components is positive
            if ( nvec > 0 ) {
                ut.passes( propPrefix + "number of components positive" );
            } else {
                ut.failure( propPrefix + "number of components positive" );
            }

            // all in range, std::vector
            try {
                property->evalv( stdEval, {}, args );
                nominalEval = stdEval;
                record( propPrefix, true, "evalv std::vector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv std::vector", ut );
            }

            // first out of range low/hi, std::vector
            if ( !args.empty() ) {
                testArgsRange( property,
                               args,
                               argnames[0],
                               stdEval,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv std::vector out of range lo 1",
                               ut );
                testArgsRange( property,
                               args,
                               argnames[0],
                               stdEval,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv std::vector out of range hi 1",
                               ut );
            }

            // setup AMP::Vector evalv results
            std::vector<AMP::LinearAlgebra::Vector::shared_ptr> ampEval( nvec );
            std::vector<AMP::LinearAlgebra::Vector::shared_ptr> nominalAmpEval( nvec );
            std::vector<AMP::LinearAlgebra::Vector::shared_ptr> nominalMultiEval( nvec );
            for ( size_t i = 0; i < nvec; i++ ) {
                auto istr     = std::to_string( i );
                auto evalVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "amp" + istr );
                auto nomVar   = std::make_shared<AMP::LinearAlgebra::Variable>( "nominal" + istr );
                auto multiVar = std::make_shared<AMP::LinearAlgebra::Variable>( "multivec" + istr );
                ampEval[i]    = AMP::LinearAlgebra::createSimpleVector<double>( npoints, evalVar );
                nominalAmpEval[i] =
                    AMP::LinearAlgebra::createSimpleVector<double>( npoints, nomVar );
                nominalMultiEval[i] =
                    AMP::LinearAlgebra::createSimpleVector<double>( npoints, multiVar );
            }

            // all in range, AMP::Vector
            try {
                property->evalv( ampEval, argsVec );
                for ( size_t i = 0; i < nvec; i++ )
                    nominalAmpEval[i]->copyVector( ampEval[i] );
                record( propPrefix, true, "evalv AMP::Vector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv AMP::Vector", ut );
            }

            // first out of range low/hi, AMP::Vector
            if ( nargs > 0 ) {
                testArgsRange( property,
                               argsVec,
                               argnames[0],
                               ampEval,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::Vector out of range lo 1",
                               ut );
                testArgsRange( property,
                               argsVec,
                               argnames[0],
                               ampEval,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::Vector out of range hi 1",
                               ut );
            }

            // all in range, AMP::MultiVector
            try {
                property->evalv( ampEval, argsMultiVec, xlator );
                for ( size_t i = 0; i < nvec; i++ )
                    nominalMultiEval[i]->copyVector( ampEval[i] );
                record( propPrefix, true, "evalv AMP::MultiVector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv AMP::MultiVector", ut );
            }

            // first out of range low/hi, AMP::MultiVector
            if ( nargs > 0 ) {
                testArgsRange( property,
                               argsMultiVec,
                               justrightVec[0]->getVariable(),
                               ampEval,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::MultiVector out of range lo 1",
                               ut );
                testArgsRange( property,
                               argsMultiVec,
                               justrightVec[0]->getVariable(),
                               ampEval,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::MultiVector out of range hi 1",
                               ut );
            }

            // check vector, Vector, MultiVector all agree
            pass = true;
            for ( size_t j = 0; j < nvec; j++ ) {
                for ( size_t i = 0; i < npoints; i++ ) {
                    double vstd      = ( *nominalEval[j] )[i];
                    double vVec      = nominalAmpEval[j]->getValueByLocalID( i );
                    double vMultiVec = nominalMultiEval[j]->getValueByLocalID( i );
                    pass             = pass && ( vstd == vVec && vVec == vMultiVec );
                }
            }
            record(
                propPrefix, pass, "evalv agrees std::vector, AMP::Vector, AMP::MultiVector", ut );

            // set up reduced argument list
            std::map<std::string, std::shared_ptr<std::vector<double>>> argsm( args );
            if ( nargs > 0 ) {
                auto argend = argsm.end();
                --argend;
                argsm.erase( argend );
            }

            // check that evalv with fewer than normal number of arguments works
            try {
                property->evalv( stdEval, {}, argsm );
                record( propPrefix, true, "evalv with missing arguments", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv with missing arguments", ut );
            }

            /////////////////////////////////////////////////////////////////////////////////////////////
            // Tensor Property
            /////////////////////////////////////////////////////////////////////////////////////////////
        } else if ( property->isTensor() ) {

            // test make_map
            try {
                auto testMap = property->make_map( argsMultiVec, xlator );
                bool good    = true;
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
                record( propPrefix, good, "make_map", ut );
            } catch ( ... ) {
                record( propPrefix, false, "make_map", ut );
            }

            // prepare results vector, check for reasonable size info
            AMP::ArraySize nvecs;
            try {
                nvecs = property->size();
                record( propPrefix, true, "size() ok", ut );
            } catch ( ... ) {
                record( propPrefix, false, "size() ok", ut );
            }
            AMP::Array<std::shared_ptr<std::vector<double>>> stdEval( nvecs[0], nvecs[1] );
            AMP::Array<std::shared_ptr<std::vector<double>>> nominalEval( nvecs[0], nvecs[1] );
            for ( size_t i = 0; i < nvecs[0]; i++ )
                for ( size_t j = 0; j < nvecs[1]; j++ ) {
                    stdEval( i, j )     = std::make_shared<std::vector<double>>( npoints );
                    nominalEval( i, j ) = std::make_shared<std::vector<double>>( npoints );
                }

            // check that number of components is positive
            if ( nvecs[0] > 0 && nvecs[1] > 0 && nvecs.size() == 2 ) {
                record( propPrefix, true, "number of components positive", ut );
            } else {
                record( propPrefix, false, "number of components positive", ut );
            }

            // all in range, std::vector
            try {
                property->evalv( stdEval, {}, args );
                nominalEval = stdEval;
                record( propPrefix, true, "evalv std::vector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv std::vector", ut );
            }

            // first out of range low/hi, std::vector
            if ( !args.empty() ) {
                testArgsRange( property,
                               args,
                               argnames[0],
                               stdEval,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv std::vector out of range lo 1",
                               ut );
                testArgsRange( property,
                               args,
                               argnames[0],
                               stdEval,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv std::vector out of range hi 1",
                               ut );
            }

            // setup AMP::Vector evalv results
            AMP::Array<AMP::LinearAlgebra::Vector::shared_ptr> ampEval( nvecs );
            AMP::Array<AMP::LinearAlgebra::Vector::shared_ptr> nominalAmpEval( nvecs );
            AMP::Array<AMP::LinearAlgebra::Vector::shared_ptr> nominalMultiEval( nvecs );
            for ( size_t i = 0; i < nvecs[0]; i++ ) {
                for ( size_t j = 0; j < nvecs[1]; j++ ) {
                    auto istr    = std::to_string( i );
                    auto evalVar = std::make_shared<AMP::LinearAlgebra::Variable>( "amp" + istr );
                    auto nomVar =
                        std::make_shared<AMP::LinearAlgebra::Variable>( "nominal" + istr );
                    auto multiVar =
                        std::make_shared<AMP::LinearAlgebra::Variable>( "multivec" + istr );
                    ampEval( i, j ) =
                        AMP::LinearAlgebra::createSimpleVector<double>( npoints, evalVar );
                    nominalAmpEval( i, j ) =
                        AMP::LinearAlgebra::createSimpleVector<double>( npoints, nomVar );
                    nominalMultiEval( i, j ) =
                        AMP::LinearAlgebra::createSimpleVector<double>( npoints, multiVar );
                }
            }

            // all in range, AMP::Vector
            try {
                property->evalv( ampEval, argsVec );
                for ( size_t i = 0; i < nvecs[0]; i++ )
                    for ( size_t j = 0; j < nvecs[1]; j++ )
                        nominalAmpEval( i, j )->copyVector( ampEval( i, j ) );
                record( propPrefix, true, "evalv AMP::Vector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv AMP::Vector", ut );
            }

            // first out of range low/hi, AMP::Vector
            if ( !args.empty() ) {
                testArgsRange( property,
                               argsVec,
                               argnames[0],
                               ampEval,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::Vector out of range lo 1",
                               ut );
                testArgsRange( property,
                               argsVec,
                               argnames[0],
                               ampEval,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::Vector out of range hi 1",
                               ut );
            }

            // all in range, AMP::MultiVector
            try {
                property->evalv( ampEval, argsMultiVec, xlator );
                for ( size_t i = 0; i < nvecs[0]; i++ )
                    for ( size_t j = 0; j < nvecs[1]; j++ )
                        nominalMultiEval( i, j )->copyVector( ampEval( i, j ) );
                record( propPrefix, true, "evalv AMP::MultiVector", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv AMP::MultiVector", ut );
            }

            // first out of range low/hi, AMP::MultiVector
            if ( nargs > 0 ) {
                testArgsRange( property,
                               argsMultiVec,
                               justrightVec[0]->getVariable(),
                               ampEval,
                               toosmall[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::MultiVector out of range lo 1",
                               ut );
                testArgsRange( property,
                               argsMultiVec,
                               justrightVec[0]->getVariable(),
                               ampEval,
                               toobig[0][5],
                               justright[0][5],
                               propPrefix + "evalv AMP::MultiVector out of range hi 1",
                               ut );
            }

            // check vector, Vector, MultiVector all agree
            pass = true;
            for ( size_t k = 0; k < nvecs[0]; k++ ) {
                for ( size_t j = 0; j < nvecs[1]; j++ ) {
                    for ( size_t i = 0; i < npoints; i++ ) {
                        double vstd      = ( *nominalEval( k, j ) )[i];
                        double vVec      = nominalAmpEval( k, j )->getValueByLocalID( i );
                        double vMultiVec = nominalMultiEval( k, j )->getValueByLocalID( i );
                        pass             = pass && ( vstd == vVec && vVec == vMultiVec );
                    }
                }
            }
            record(
                propPrefix, pass, "evalv agrees std::vector, AMP::Vector, AMP::MultiVector", ut );

            // set up reduced argument list
            std::map<std::string, std::shared_ptr<std::vector<double>>> argsm( args );
            if ( nargs > 0 ) {
                auto argend = argsm.end();
                --argend;
                argsm.erase( argend );
            }

            // check that evalv with fewer than normal number of arguments works
            try {
                property->evalv( stdEval, {}, argsm );
                record( propPrefix, true, "evalv with missing arguments", ut );
            } catch ( ... ) {
                record( propPrefix, false, "evalv with missing arguments", ut );
            }
        }
    }
}


// Register a dummy material created from a database
void registerDatabaseMaterial( [[maybe_unused]] AMP::UnitTest &ut )
{
    // Create the material
    auto fun = []() {
        const char databaseText[] = "a = 3.2            // constant\n"
                                    "b = 4.3 um         // constant (with units)\n"
                                    "c = @(x) 2*x;      // equation\n"
                                    "d = @(x) 2*x; cm   // equation with units\n"
                                    "e = 1,2, 3.0, 4,5  // vector\n";
        //"f {                // Database property\n"
        //"   value = 5.1\n"
        //"}\n";
        auto db = AMP::Database::createFromString( databaseText );
        return std::make_unique<AMP::Materials::DatabaseMaterial>( "databaseMaterial",
                                                                   std::move( db ) );
    };
    AMP::Materials::registerMaterial( "databaseMaterial", fun );
    // Run some basic checks
    auto mat = AMP::Materials::getMaterial( "databaseMaterial" );
}


int main( int argc, char **argv )
{
    AMP::AMPManagerProperties amprops;
    amprops.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, amprops );
    AMP::UnitTest ut;

    { // Limit scope

        // Register a database material
        registerDatabaseMaterial( ut );

        // test all materials and all properties and print report
        auto matlist = AMP::Materials::getMaterialList();
        std::cout << "number of materials = " << matlist.size() << std::endl;
        std::cout << "materials = ";
        for ( auto &elem : matlist )
            std::cout << elem << " ";
        std::cout << std::endl;
        for ( auto &elem : matlist )
            testMaterial( elem, ut );

        // check that undefined material name is caught
        try {
            auto mat = AMP::Materials::getMaterial( "flubber" );
            ut.failure( "Failed to catch unknown material" );
        } catch ( const StackTrace::abort_error &err ) {
            if ( err.message == "Unable to create object flubber" )
                ut.passes( "detected undefined material" );
            else
                ut.failure( "did not detect undefined material" );
        } catch ( const std::exception &err ) {
            ut.failure( "Caught unknown exception type" );
        }
    }

    ut.report();
    int num_failed = ut.NumFailGlobal();
    ut.reset();

    AMP::AMPManager::shutdown();
    return num_failed;
}
