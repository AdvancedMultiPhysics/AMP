#ifndef included_AMP_MapOperator
#define included_AMP_MapOperator


#include "AMP/utils/Database.h"

#include <memory>

#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/map/MapOperatorParameters.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"

#include <string>


namespace AMP::Operator {


/**
 * Class MapOperator is the base class for various mapping alogorithms. This
 * class stores a pointer to the mapAdapter to which the solution has to be
 * mapped from Operator's mesh.
 */

class MapOperator : public Operator
{
public:
    /**
      Constructor calls the reset member which reads the information about the boundary.
      */
    explicit MapOperator( std::shared_ptr<const OperatorParameters> params ) : Operator( params )
    {
        auto myparams = std::dynamic_pointer_cast<const MapOperatorParameters>( params );
        d_boundaryId  = 0;
        reset( myparams );
    }

    /**
      Destructor
      */
    virtual ~MapOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "MapOperator"; }

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    virtual std::shared_ptr<AMP::LinearAlgebra::Variable> createInputVariable( const std::string &,
                                                                               int )
    {
        // Implemented in derived classes
        std::shared_ptr<AMP::LinearAlgebra::Variable> emptyPointer;
        return emptyPointer;
    }

    virtual std::shared_ptr<AMP::LinearAlgebra::Variable> createOutputVariable( const std::string &,
                                                                                int )
    {
        // Implemented in derived classes
        std::shared_ptr<AMP::LinearAlgebra::Variable> emptyPointer;
        return emptyPointer;
    }

    virtual void setInputVariableName( const std::string &, int )
    {
        // Implemented in derived classes.
    }

    virtual void setOutputVariableName( const std::string &, int )
    {
        // Implemented in derived classes.
    }

protected:
    unsigned int d_boundaryId;

    std::shared_ptr<AMP::Mesh::Mesh> d_MapMesh;

    // Communicator for the Map
    AMP_MPI d_MapComm;

private:
};
} // namespace AMP::Operator

#endif
