#ifndef included_AMP_Map3Dto1D
#define included_AMP_Map3Dto1D


#include "AMP/discretization/createLibmeshElements.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/map/MapOperator.h"
#include "AMP/operators/map/MapOperatorParameters.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include <memory>

#include <string>

namespace AMP::Operator {

/**
 *  Class Map3Dto1D is used to map the solution on the boundary of the 3D mesh to
 *  the 1D location by interpolation. The 3D mesh is the d_MeshAdapter from the
 *  Operator class.
 */
class Map3Dto1D : public MapOperator
{
public:
    //!  Constructor
    explicit Map3Dto1D( std::shared_ptr<const OperatorParameters> params );

    //!  Destructor
    virtual ~Map3Dto1D() {}

    //! Return the name of the operator
    std::string type() const override { return "Map3Dto1D"; }

    /**
      This function reads the entries of the database for the operator
      and can also be used to change the parameters if required.
     */
    void reset( std::shared_ptr<const OperatorParameters> ) override;

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    void apply_Gauss( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::LinearAlgebra::Vector::shared_ptr f );

    void apply_Nodal( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::LinearAlgebra::Vector::shared_ptr f );

    std::shared_ptr<AMP::LinearAlgebra::Variable> createInputVariable( const std::string &name,
                                                                       int = -1 ) override
    {
        return d_inpVariable->clone( name );
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> createOutputVariable( const std::string &name,
                                                                        int = -1 ) override
    {
        return d_outVariable->clone( name );
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_inpVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outVariable;
    }

    /**
      This member function is used to set the 1D Z locations std vector.
      */
    void setZLocations( const std::vector<double> &zloc ) { d_zLocations = zloc; }

    void makeZLocationsConsistent();

    void setVector( AMP::LinearAlgebra::Vector::shared_ptr vec )
    {
        outputVec = vec->subsetVectorForVariable( d_outVariable );
    }

protected:
    std::shared_ptr<AMP::LinearAlgebra::Vector> outputVec = nullptr;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVariable = nullptr;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable = nullptr;

    std::vector<double> d_zLocations; /**< std vector to store 1D z locations. */

    bool d_useGaussVec = false;

private:
    Discretization::createLibmeshElements libmeshElements;
};
} // namespace AMP::Operator

#endif
