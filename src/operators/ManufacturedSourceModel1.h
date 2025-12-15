#ifndef included_AMP_ManufacturedSourceModel1
#define included_AMP_ManufacturedSourceModel1

#include "AMP/operators/ElementPhysicsModel.h"
#include <memory>


// LibMesh include
DISABLE_WARNINGS
#include "libmesh/point.h"
ENABLE_WARNINGS

#include <iostream>
#include <string>


namespace AMP::Operator {


typedef ElementPhysicsModelParameters ManufacturedSourceModel1Parameters;


class ManufacturedSourceModel1 : public ElementPhysicsModel
{
public:
    explicit ManufacturedSourceModel1(
        std::shared_ptr<const ManufacturedSourceModel1Parameters> params )
        : ElementPhysicsModel( params )
    {
        d_Dzero = 1.0;
        d_beta  = 1.0;
    }

    virtual ~ManufacturedSourceModel1() {}


    virtual void getManufacturedSource1( std::vector<double> &result,
                                         const std::vector<double> &T,
                                         const std::vector<libMesh::Point> &Coordinates )
    {
        AMP_ASSERT( ( Coordinates.size() == T.size() ) && ( T.size() == result.size() ) );

        for ( unsigned int qp = 0; qp < Coordinates.size(); qp++ ) {
            double x    = Coordinates[qp]( 0 );
            double y    = Coordinates[qp]( 1 );
            double z    = Coordinates[qp]( 2 );
            double r    = std::sqrt( x * x + y * y + z * z );
            double temp = d_beta * r * r * r + d_Dzero * exp( -r * T[qp] ) * ( 12 * r - 3 * r * r );
            result[qp]  = temp;
        }
    }

    // virtual void getManufacturedSource(std::vector<double> & result,
    // std::vector<std::vector<double> >& args);

protected:
private:
    double d_Dzero;
    double d_beta;
};


} // namespace AMP::Operator

#endif
