#ifndef included_AMP_NeutronicsRhs
#define included_AMP_NeutronicsRhs

#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/Variable.h"

#include <memory>
#include <vector>

namespace AMP::Operator {

//===========================================================================//
/*!
 * \class NeutronicsRhs
 * \brief A class for representing the neutronics source operator.
 */
//===========================================================================//

class NeutronicsRhs : public Operator
{
public:
    //! Neutronics Input Types
    enum SourceType { Power, Oxygen, Metal, FissionGas, Isotopes, NUM_SOURCE_TYPES };

private:
public:
    explicit NeutronicsRhs( std::shared_ptr<OperatorParameters> parameters );

    /**
     * Empty destructor for NeutronicsRhs
     */
    virtual ~NeutronicsRhs();

    //! Return the name of the operator
    std::string type() const override { return "NeutronicsRhs"; }

    /**
     * Print out all members of integrator instance to given output stream.
     */
    void printClassData( std::ostream &os ) const;

    /**
     * Write out state of object to given database.
     *
     * When assertion checking is active, the database pointer must be non-null.
     */
    void putToDatabase( std::shared_ptr<AMP::Database> db );

    /**
      The function that computes the residual.
     * @param u: multivector of the state.
     * @param f: The result of apply is f = A(u)
     */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    /**
      A function to reinitialize this object.
      */
    void reset( std::shared_ptr<const OperatorParameters> parameters ) override;

    // static SP_HexGaussPointVariable createOutputVariable (const std::string & name, int varId =
    // -1);

    void setOutputVariableName( const std::string &name, int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    void setTimeStep( int tStep ) { d_timeStep = tStep; }
    void setTimeInSeconds( double seconds );
    void setTimeInDays( double days );

protected:
    /*
     * Read input data from specified database and initialize class members.
     * If run is from restart, a subset of the restart values may be replaced
     * with those read from input.
     *
     * When assertion checking is active, the database pointer must be non-null.
     */
    void getFromInput( std::shared_ptr<AMP::Database> db );

    std::shared_ptr<AMP::Database> d_db;
    bool d_useFixedValue;
    int d_numTimeSteps;
    std::vector<double> d_timeStepsInDays;
    SourceType d_type;
    std::vector<double> d_fixedValues;
    int d_timeStep;
    double d_timeStepInSeconds;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outputVariable;
    std::vector<std::vector<double>> d_values;
    double d_secondsPerDay;
    SourceType str2id( const std::string &str );
};
} // namespace AMP::Operator


#endif
