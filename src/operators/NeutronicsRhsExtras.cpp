#include "AMP/operators/NeutronicsRhsExtras.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/NeutronicsRhsExtrasParameters.h"
#include "AMP/operators/Operator.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include <memory>

#include <cmath>
#include <vector>

namespace AMP::Operator {

/*
 ***********************************************************************************
 * Constructor for NeutronicsRhsExtras.  The constructor initializes the values   *
 * from the parameters.                                                           *
 ***********************************************************************************
 */
NeutronicsRhsExtras::NeutronicsRhsExtras(
    std::shared_ptr<NeutronicsRhsExtrasParameters> parameters )
    : Operator( parameters ),
      d_useFixedValue( false ),
      d_type( SourceType::Isotopes ),
      d_timeStep( 0 ),
      d_extrasId( 0 )
{
    AMP_ASSERT( parameters );
    d_Mesh              = parameters->d_Mesh;
    d_numExtras         = parameters->d_numExtras;
    d_extrasName        = parameters->d_extrasName;
    d_timeStepInSeconds = 0.;
    d_secondsPerDay     = 86400.;
    getFromInput( parameters->d_db );
    if ( !d_useFixedValue ) {
        int numValues;
        numValues = parameters->d_db->getScalar<int>( "numValues" );
        std::vector<double> tmp1( numValues, 0 );
        std::vector<std::vector<double>> tmp2( d_numTimeSteps, tmp1 );
        d_values.resize( d_numExtras, tmp2 );

        for ( int extras = 0; extras < d_numExtras; extras++ ) {
            for ( int t = 0; t < d_numTimeSteps; t++ ) {
                char key[100];
                snprintf( key, sizeof key, "%s_%d", d_extrasName[extras].c_str(), t );
                d_values[extras][t] = parameters->d_db->getVector<double>( key );
            }
        }
    }
}


/*
 *************************************************************************
 * Destructor.                                                           *
 *************************************************************************
 */
NeutronicsRhsExtras::~NeutronicsRhsExtras() = default;

/*
 *************************************************************************
 * If simulation is not from restart, read data from input database.     *
 * Otherwise, override restart values for a subset of the data members   *
 * with those found in input.                                            *
 *************************************************************************
 */
void NeutronicsRhsExtras::getFromInput( std::shared_ptr<AMP::Database> db )
{
    AMP_ASSERT( db );

    // define the source type and create the output variable.
    auto str = db->getWithDefault<std::string>( "type", "Power" );
    d_type   = str2id( str );

    auto outVarName = db->getWithDefault<std::string>( "OutputVariable", str );
    d_outputVariable.reset( new AMP::LinearAlgebra::Variable( outVarName ) );

    // number of time steps
    d_numTimeSteps = db->getWithDefault<int>( "numTimeSteps", 1 );
    AMP_ASSERT( d_numTimeSteps > 0 );
    d_timeStepsInDays.resize( d_numTimeSteps );
    d_fixedValues.resize( d_numTimeSteps );

    if ( db->keyExists( "numTimeSteps" ) ) {
        // time-step sizes
        if ( db->keyExists( "timeSteps" ) ) {
            d_timeStepsInDays = db->getVector<double>( "timeSteps" );
        } else {
            // default value is only valid if the default number of time steps is used.
            AMP_ASSERT( d_numTimeSteps == 1 );
            d_timeStepsInDays[0] = 100000000000.;
        }
        // default value is 1
    }

    // Power in Watts per gram
    d_useFixedValue = db->getWithDefault<bool>( "useFixedValue", true );
    if ( d_useFixedValue ) {
        if ( db->keyExists( "fixedValues" ) ) {
            d_fixedValues = db->getVector<double>( "fixedValues" );
        } else {
            // default value is only valid if the default number of time steps is used.
            AMP_ASSERT( d_numTimeSteps == 1 );
            d_fixedValues[0] = 1.;
        }
    }
}

/*
 *************************************************************************
 * Write out class version number and data members to database.          *
 *************************************************************************
 */
void NeutronicsRhsExtras::putToDatabase( std::shared_ptr<AMP::Database> db )
{
    AMP_ASSERT( !db.use_count() );
    db->putScalar( "numTimeSteps", d_numTimeSteps );
    db->putVector( "timeSteps", d_timeStepsInDays );
    db->putVector( "fixedValues", d_fixedValues );
}

/*
 *************************************************************************
 * Print class data members to given output stream.                      *
 *************************************************************************
 */
void NeutronicsRhsExtras::printClassData( std::ostream &os ) const
{
    os << "\nNeutronicsRhsExtras::printClassData..." << std::endl;
    os << "d_numTimeSteps = " << d_numTimeSteps << std::endl;
}

/*
 *************************************************************************
 * Reset the class.                                                      *
 *************************************************************************
 */
void NeutronicsRhsExtras::reset( std::shared_ptr<const OperatorParameters> parameters )
{

    AMP_ASSERT( parameters );
    auto params = std::dynamic_pointer_cast<const NeutronicsRhsExtrasParameters>( parameters );
    AMP_ASSERT( params );
    AMP_ASSERT( params->d_db );

    if ( !d_useFixedValue ) {
        int numValues;
        numValues = params->d_db->getScalar<int>( "numValues" );
        d_values.resize( d_numExtras,
                         std::vector<std::vector<double>>( d_numTimeSteps,
                                                           std::vector<double>( numValues, 0 ) ) );

        for ( int extras = 0; extras < d_numExtras; extras++ ) {
            for ( int t = 0; t < d_numTimeSteps; t++ ) {
                char key[100];
                snprintf( key, sizeof key, "%s_%d", d_extrasName[extras].c_str(), t );
                d_values[extras][t] = params->d_db->getVector<double>( key );
            }
        }
    }
}

/*
 *************************************************************************
 * Provide the Apply function
 *************************************************************************
 */
void NeutronicsRhsExtras::apply( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                 AMP::LinearAlgebra::Vector::shared_ptr r )
{
    auto rInternal = r->subsetVectorForVariable( d_outputVariable );

    // determine the present time and extras
    int this_step     = d_timeStep;
    int this_extrasId = d_extrasId;

    // compute power distribution
    if ( d_useFixedValue ) {
        double value = d_fixedValues[this_step];
        rInternal->setToScalar( value );
    } else {
        auto elem      = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 1 );
        auto end_elems = elem.begin();

        int DOFsPerElement = 8;
        int ghostWidth     = 1;
        bool split         = true;
        auto dof_map       = AMP::Discretization::simpleDOFManager::create(
            d_Mesh, AMP::Mesh::GeomType::Cell, ghostWidth, DOFsPerElement, split );

        int gp = 0;
        for ( ; elem != end_elems; ++elem ) {
            std::vector<size_t> gid;
            dof_map->getDOFs( elem->globalID(), gid );
            for ( unsigned int i = 0; i < gid.size(); gp++, i++ ) {
                rInternal->setValuesByGlobalID(
                    1, &gid[i], &d_values[this_extrasId][this_step][gp] );
            } // end for gauss-points
        } // end for elements
    }
}


NeutronicsRhsExtras::SourceType NeutronicsRhsExtras::str2id( const std::string &str )
{
    if ( str == "Isotopes" ) {
        return Isotopes;
    } else if ( str == "Elements" ) {
        return Elements;
    } else {
        std::string msg = "str2id could not find the right enumerated ID with string !" + str +
                          "!.  Options are: Isotopes and Elements";
        AMP_INSIST( false, msg );
    }
    return NUM_SOURCE_TYPES;
}


/*
 *************************************************************************
 * Set the time
 *************************************************************************
 */
void NeutronicsRhsExtras::setTimeInSeconds( double setSeconds )
{
    AMP_ASSERT( setSeconds >= 0. );
    // first assume time does not go backwards.
    int timeStep   = d_timeStep;
    double seconds = d_timeStepInSeconds;
    if ( setSeconds >= d_timeStepInSeconds ) {
        for ( int i = d_timeStep; i < d_numTimeSteps; i++ ) {
            seconds += d_timeStepsInDays[i] * d_secondsPerDay;
            if ( setSeconds > seconds ) {
                timeStep = i + 1;
            } else {
                AMP_ASSERT( timeStep < d_numTimeSteps );
                d_timeStep          = timeStep;
                d_timeStepInSeconds = seconds - d_timeStepsInDays[d_timeStep];
                return;
            }
        }
    } else {
        seconds  = 0.;
        timeStep = 0;
        for ( int i = 0; i < d_numTimeSteps; i++ ) {
            seconds += d_timeStepsInDays[i] * d_secondsPerDay;
            if ( setSeconds > seconds ) {
                timeStep = i + 1;
            } else {
                AMP_ASSERT( timeStep < d_numTimeSteps );
                d_timeStep          = timeStep;
                d_timeStepInSeconds = seconds - d_timeStepsInDays[d_timeStep] * d_secondsPerDay;
                return;
            }
        }
    }
    AMP_INSIST( false, "Could not find the appropriate time." );
}


} // namespace AMP::Operator
