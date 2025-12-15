#ifndef included_AMP_ManufacturedSourceModel2
#define included_AMP_ManufacturedSourceModel2

#include "AMP/operators/ElementPhysicsModel.h"
#include <memory>


namespace AMP::Operator {


typedef ElementPhysicsModelParameters ManufacturedSourceModel2Parameters;


class ManufacturedSourceModel2 : public ElementPhysicsModel
{
public:
    explicit ManufacturedSourceModel2(
        std::shared_ptr<const ManufacturedSourceModel2Parameters> params )
        : ElementPhysicsModel( params )
    {
        d_Dzero = 1.0;
        d_beta  = 1.0;
    }

    virtual ~ManufacturedSourceModel2() {}

    virtual void getManufacturedSource2( std::vector<double> &result,
                                         const std::vector<double> &T,
                                         const std::vector<libMesh::Point> &Coordinates )
    {
        AMP_ASSERT( ( Coordinates.size() == T.size() ) && ( T.size() == result.size() ) );

        for ( unsigned int qp = 0; qp < Coordinates.size(); qp++ ) {
            double x = Coordinates[qp]( 0 );
            double y = Coordinates[qp]( 1 );
            double z = Coordinates[qp]( 2 );
            double r = std::sqrt( x * x + y * y + z * z );

            double temp = 12 * d_Dzero * r * r * r * r * r * exp( -r * T[qp] );

            result[qp] = temp;
        }
    }


protected:
private:
    double d_Dzero;
    double d_beta;
};


} // namespace AMP::Operator

#endif
