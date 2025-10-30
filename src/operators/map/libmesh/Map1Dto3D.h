#ifndef included_AMP_Map1Dto3D
#define included_AMP_Map1Dto3D


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
  Class Map1Dto3D is used to map the 1D solution from simple vector
  onto the 2D Surface by injunction as the 1D location are constructed
  by the nodal points on the 2D surface. The 2D surface has to be on the
  d_MapAdapater from the MapOperator class.
 */
class Map1Dto3D : public MapOperator
{
public:
    //! Default Constructor
    explicit Map1Dto3D( std::shared_ptr<const OperatorParameters> params );

    //! De-constructor
    virtual ~Map1Dto3D() {}

    //! Return the name of the operator
    std::string type() const override { return "Map1Dto3D"; }

    /**
      This function reads the entries of the database for the operator
      and can also be used to change the parameters if required.
     */
    void reset( std::shared_ptr<const OperatorParameters> ) override;

    /**
      For this operator the apply function would map the solution by injunction from
      Simple Vector in u to NodalScalar Vector in r.
      @param [in]  u input vector.
      @param [out] r output vector.
      */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    void apply_Gauss( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::LinearAlgebra::Vector::shared_ptr f );

    void apply_Nodal( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::LinearAlgebra::Vector::shared_ptr f );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_inpVariable;
    }


    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outVariable;
    }


    //!  This function is used to compute 1D locations from the output vector.
    void computeZNodeLocations();
    void computeZGaussLocations();


    //!  This function returns the number of 1D locations.
    size_t getNumZlocations() { return d_zLocations.size(); }


    //!  This function returns the 1D locations stl vector.
    const auto &getZLocations() { return d_zLocations; }


    /**
      This function sets the 1D locations stl vector.
      Note: This routine will check that the values are unique and in sorted
      order.  It may modify the resulting internal vector accordingly.
     */
    void setZLocations( const std::vector<double> &z );


    void setVector( AMP::LinearAlgebra::Vector::shared_ptr vec );


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
