#ifndef included_AMP_Map3to1to3Parameters
#define included_AMP_Map3to1to3Parameters

#include "AMP/operators/map/AsyncMapOperatorParameters.h"


namespace AMP::Operator {


class Map3to1to3Parameters : public AsyncMapOperatorParameters
{
public:
    int d_MasterValue;
    int d_NumToSend;

    explicit Map3to1to3Parameters( std::shared_ptr<AMP::Database> db )
        : AsyncMapOperatorParameters( db ), d_MasterValue( 0 ), d_NumToSend( 0 )
    {
    }
};
} // namespace AMP::Operator

#endif
