#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/Operator.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/Vector.h"

#include <cmath>
#include <memory>
#include <vector>


namespace AMP::Operator {

/*
 *************************************************************************
 * Constructor for NeutronicsRhs.  The constructor initializes the       *
 * values from the parameters.                                           *
 *************************************************************************
 */
NeutronicsRhs::NeutronicsRhs( std::shared_ptr<OperatorParameters> parameters )
    : Operator( parameters ), d_useFixedValue( false ), d_type( SourceType::Power ), d_timeStep( 0 )
{
    AMP_ASSERT( parameters );
    d_Mesh              = parameters->d_Mesh;
    d_timeStepInSeconds = 0.;
    d_secondsPerDay     = 86400.;
    getFromInput( parameters->d_db );
    if ( !d_useFixedValue ) {
        int numValues;
        numValues = parameters->d_db->getScalar<int>( "numValues" );
        d_values.resize( d_numTimeSteps, std::vector( numValues, 0.0 ) );
        char key[100];
        for ( int step = 0; step < d_numTimeSteps; step++ ) {
            snprintf( key, sizeof key, "value_%d", step );
            AMP_INSIST( parameters->d_db->keyExists( key ), "Key is missing!" );
            d_values[step] = parameters->d_db->getVector<double>( key );
        }
    }
}

/*
 *************************************************************************
 * Destructor.                                                           *
 *************************************************************************
 */
NeutronicsRhs::~NeutronicsRhs() = default;

/*
 *************************************************************************
 * If simulation is not from restart, read data from input database.     *
 * Otherwise, override restart values for a subset of the data members   *
 * with those found in input.                                            *
 *************************************************************************
 */
void NeutronicsRhs::getFromInput( std::shared_ptr<AMP::Database> db )
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
void NeutronicsRhs::putToDatabase( std::shared_ptr<AMP::Database> db )
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
void NeutronicsRhs::printClassData( std::ostream &os ) const
{
    os << "\nNeutronicsRhs::printClassData..." << std::endl;
    os << "d_numTimeSteps = " << d_numTimeSteps << std::endl;
}

/*
 *************************************************************************
 * Reset the class.                                                      *
 *************************************************************************
 */
void NeutronicsRhs::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );
    AMP_ASSERT( params->d_db );
    d_db = params->d_db;
    getFromInput( params->d_db );

    if ( !d_useFixedValue ) {
        int numValues;
        numValues = params->d_db->getScalar<int>( "numValues" );
        d_values.resize( d_numTimeSteps, std::vector( numValues, 0.0 ) );
        char key[100];
        for ( int step = 0; step < d_numTimeSteps; step++ ) {
            snprintf( key, sizeof key, "value_%d", step );
            AMP_INSIST( params->d_db->keyExists( key ), "Key is missing!" );
            d_values[step] = params->d_db->getVector<double>( key );
            AMP_ASSERT( numValues == (int) d_values.size() );
        }
    }
}

/*
 *************************************************************************
 * Provide the Apply function
 *************************************************************************
 */
void NeutronicsRhs::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                           AMP::LinearAlgebra::Vector::shared_ptr r )
{
    (void) u;

    // subsetOutputVector is from Operator.h
    auto rInternal = this->subsetOutputVector( r );

    AMP_ASSERT( rInternal );

    // determine the present time
    int this_step = d_timeStep;

    // compute power distribution
    if ( d_useFixedValue ) {
        double value = d_fixedValues[this_step];
        rInternal->setToScalar( value );
    } else {
        rInternal->zero();
        int ghostWidth               = 0;
        AMP::Mesh::MeshIterator elem = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, ghostWidth );
        AMP::Mesh::MeshIterator end_elems = elem.end();

        unsigned int DOFsPerVolume = 8;
        bool split                 = true;
        std::shared_ptr<AMP::Discretization::DOFManager> dof_map =
            AMP::Discretization::simpleDOFManager::create(
                d_Mesh, AMP::Mesh::GeomType::Cell, ghostWidth, DOFsPerVolume, split );

        int gp = 0;
        std::vector<size_t> gid;
        AMP::pout << "The intial value is: " << rInternal->L2Norm() << std::endl;
        for ( ; elem != end_elems; ++elem ) {
            dof_map->getDOFs( elem->globalID(), gid );
            for ( unsigned int i = 0; i < DOFsPerVolume; gp++, i++ ) {
                rInternal->setValuesByGlobalID( 1, &gid[i], &d_values[this_step][gp] );
                /*          if( gp==0 ) {
                            if( (rInternal->max()>0) &&
                   (!AMP::Utilities::approx_equal(rInternal->max(),
                   rInternal->L2Norm(), 1e-4)) ) {
                              AMP::pout<<"The setValueByGlobalID function set this value twice
                   because it is confused
                   about multiple meshes with the same variable name"<<std::endl;
                              AMP::pout<<"max value is: "<<rInternal->max()<<std::endl;
                              AMP::pout<<"L2  value is: "<<rInternal->L2Norm()<<std::endl;
                              AMP_ERROR("There is a problem in NeutronicsRhs.");
                            }
                          }*/
            } // end for gauss-points
        }     // end for elements
        /*double nrm = rInternal->L2Norm();
        printf("%e\n",nrm);
        AMP_MPI(AMP_COMM_WORLD).barrier();
        AMP_ERROR("stop");*/
    }
    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


NeutronicsRhs::SourceType NeutronicsRhs::str2id( const std::string &str )
{
    if ( str == "Power" ) {
        return Power;
    } else if ( str == "Oxygen" ) {
        return Oxygen;
    } else if ( str == "Metal" ) {
        return Metal;
    } else if ( str == "FissionGas" ) {
        return FissionGas;
    } else if ( str == "Isotopes" ) {
        return Isotopes;
    } else {
        std::string msg = "str2id could not find the right enumerated ID with string !" + str +
                          "!.  Options are: Power, Oxygen, Metal, and FissionGas";
        AMP_INSIST( false, msg );
    }
    return NUM_SOURCE_TYPES;
}


void NeutronicsRhs::setOutputVariableName( const std::string &name, int varId )
{
    (void) varId;
    d_outputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( name );
}

std::shared_ptr<AMP::LinearAlgebra::Variable> NeutronicsRhs::getOutputVariable() const
{
    return d_outputVariable;
}


/*SP_HexGaussPointVariable NeutronicsRhs::createOutputVariable (const std::string & name, int varId
  = -1)
  {
  (void) varId;
  SP_HexGaussPointVariable var( new HexGaussPointVariable (name) );
  return var;
  }*/


/*
 *************************************************************************
 * Set the time
 *************************************************************************
 */
void NeutronicsRhs::setTimeInSeconds( double setSeconds )
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
void NeutronicsRhs::setTimeInDays( double days ) { setTimeInSeconds( days * d_secondsPerDay ); }


} // namespace AMP::Operator
