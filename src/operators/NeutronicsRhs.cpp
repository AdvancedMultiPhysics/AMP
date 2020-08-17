/* AMP Files */
#include "NeutronicsRhs.h"
#include "AMP/ampmesh/Mesh.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/Operator.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/Vector.h"
#include "NeutronicsRhsParameters.h"
#include <memory>

#include <cmath>
#include <vector>


namespace AMP {
namespace Operator {

/*
 *************************************************************************
 * Constructor for NeutronicsRhs.  The constructor initializes the values   *
 * from the parameters.                                                  *
 *************************************************************************
 */
NeutronicsRhs::NeutronicsRhs( SP_Parameters parameters ) : Operator( parameters ), d_timeStep( 0 )
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
            sprintf( key, "value_%d", step );
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
void NeutronicsRhs::getFromInput( SP_Database db )
{
    AMP_ASSERT( db );

    // define the source type and create the output variable.
    std::string str = db->getWithDefault<std::string>( "type", "Power" );
    d_type          = str2id( str );

    std::string outVarName = db->getWithDefault<std::string>( "OutputVariable", str );
    d_outputVariable.reset( new AMP::LinearAlgebra::Variable( outVarName ) );

    // number of time steps
    d_numTimeSteps = db->getWithDefault( "numTimeSteps", 1 );
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
    d_useFixedValue = db->getWithDefault( "useFixedValue", true );
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
void NeutronicsRhs::putToDatabase( SP_Database db )
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
void NeutronicsRhs::reset( const SP_OperatorParameters &parameters )
{

    AMP_ASSERT( parameters.get() != nullptr );
    d_db = parameters->d_db;
    SP_Parameters params =
        std::dynamic_pointer_cast<NeutronicsRhsParameters, OperatorParameters>( parameters );
    AMP_ASSERT( params.get() != nullptr );
    AMP_ASSERT( ( ( params->d_db ).get() ) != nullptr );
    getFromInput( params->d_db );

    if ( !d_useFixedValue ) {
        int numValues;
        numValues = params->d_db->getScalar<int>( "numValues" );
        d_values.resize( d_numTimeSteps, std::vector( numValues, 0.0 ) );
        char key[100];
        for ( int step = 0; step < d_numTimeSteps; step++ ) {
            sprintf( key, "value_%d", step );
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
    AMP::LinearAlgebra::Vector::shared_ptr rInternal = this->subsetOutputVector( r );

    AMP_ASSERT( rInternal != nullptr );

    // determine the present time
    int this_step = d_timeStep;

    // compute power distribution
    if ( d_useFixedValue ) {
        double value = d_fixedValues[this_step];
        rInternal->setToScalar( value );
    } else {
        rInternal->zero();
        int ghostWidth = 0;
        AMP::Mesh::MeshIterator elem =
            d_Mesh->getIterator( AMP::Mesh::GeomType::Volume, ghostWidth );
        AMP::Mesh::MeshIterator end_elems = elem.end();

        unsigned int DOFsPerVolume = 8;
        bool split                 = true;
        AMP::Discretization::DOFManager::shared_ptr dof_map =
            AMP::Discretization::simpleDOFManager::create(
                d_Mesh, AMP::Mesh::GeomType::Volume, ghostWidth, DOFsPerVolume, split );

        int gp = 0;
        std::vector<size_t> gid;
        AMP::pout << "The intial value is: " << rInternal->L2Norm() << std::endl;
        for ( ; elem != end_elems; ++elem ) {
            dof_map->getDOFs( elem->globalID(), gid );
            for ( unsigned int i = 0; i < DOFsPerVolume; gp++, i++ ) {
                rInternal->setValueByGlobalID( gid[i], d_values[this_step][gp] );
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
    rInternal->makeConsistent( AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_SET );
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

AMP::LinearAlgebra::Variable::shared_ptr NeutronicsRhs::getOutputVariable()
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
} // namespace Operator
} // namespace AMP