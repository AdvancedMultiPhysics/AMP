#ifndef included_AMP_RobinPhysicsModel
#define included_AMP_RobinPhysicsModel

#include <string>
#include <vector>

#include "AMP/operators/ElementPhysicsModel.h"
#include "AMP/utils/UtilityMacros.h"
#include <memory>


namespace AMP::Operator {

typedef ElementPhysicsModelParameters RobinPhysicsModelParameters;

class RobinPhysicsModel : public ElementPhysicsModel
{
public:
    explicit RobinPhysicsModel( std::shared_ptr<const RobinPhysicsModelParameters> params )
        : ElementPhysicsModel( params ), d_numActiveVariables( 0 )
    {
        reset( params );
    }

    virtual ~RobinPhysicsModel() {}

    virtual void reset( std::shared_ptr<const RobinPhysicsModelParameters> params )
    {
        if ( params->d_db->keyExists( "Number_Active_Variables" ) ) {
            d_numActiveVariables = params->d_db->getScalar<int>( "Number_Active_Variables" );
        }
        std::shared_ptr<AMP::Database> activeDb;
        if ( params->d_db->keyExists( "ActiveInputVariables" ) ) {
            activeDb               = params->d_db->getDatabase( "ActiveInputVariables" );
            unsigned int numactive = activeDb->getAllKeys().size();
            if ( params->d_db->keyExists( "Number_Active_Variables" ) ) {
                AMP_INSIST( numactive == d_numActiveVariables,
                            "Number of active variables disagrees with Number_Active_Variables." );
            }
            d_numActiveVariables = numactive;
        }

        d_activeVariableNames.resize( d_numActiveVariables );
        for ( unsigned int var = 0; var < d_numActiveVariables; var++ ) {
            char key[100];
            snprintf( key, sizeof( key ), "ActiveVariable_%u", var );
            auto varName               = activeDb->getString( key );
            d_activeVariableNames[var] = varName;
        }
    }

    virtual void getConductance( std::vector<double> &beta,
                                 std::vector<double> &gamma,
                                 const std::vector<std::vector<double>> &arguments ) = 0;

    const auto &getVariableName() { return d_activeVariableNames; }

protected:
    unsigned int d_numActiveVariables;              // Number of Active Variables
    std::vector<std::string> d_activeVariableNames; // Active Variable names

private:
};
} // namespace AMP::Operator

#endif
