#ifndef included_AMP_RandomSubsetVariable
#define included_AMP_RandomSubsetVariable

#include "RandomAccessIndexer.h"
#include "SubsetVariable.h"


namespace AMP {
namespace LinearAlgebra {

class RandomSubsetVariable : public SubsetVariable
{
private:
    VectorIndexer::shared_ptr d_Indexer;

public:
    RandomSubsetVariable( const std::string &name, VectorIndexer::shared_ptr i )
        : SubsetVariable( name ), d_Indexer( i )
    {
    }

    virtual VectorIndexer::shared_ptr getIndexer() { return d_Indexer; }
    virtual size_t DOFsPerObject() const { return 0; }
};
}
}

#endif
