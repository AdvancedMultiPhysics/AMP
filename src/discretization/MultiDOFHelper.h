#ifndef included_AMP_MultiDOFHelper
#define included_AMP_MultiDOFHelper

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/Array.h"

#include <vector>

namespace AMP::LinearAlgebra {
class VectorData;
}


namespace AMP::Discretization {

/**
 * \class multiDOFHelper
 * \brief A class to manage mapping multiple DOFs to global indicies
 * \details  This class provides mapping between local and global DOFs for multiple DOF managers
 */
class multiDOFHelper final
{
public:
    // Constructor
    multiDOFHelper( const DOFManager &manager );

    // Constructor
    multiDOFHelper( const AMP::LinearAlgebra::VectorData &data );

    // Constructor
    multiDOFHelper( const std::vector<std::shared_ptr<DOFManager>> &managers,
                    const AMP::AMP_MPI &comm );

    // Constructor
    multiDOFHelper( const std::vector<AMP::LinearAlgebra::VectorData *> &data,
                    const AMP::AMP_MPI &comm );

    //! Convert the local to global dof
    size_t subToGlobal( int manager, size_t dof ) const;

    //! Convert the local to global dof
    std::vector<size_t> getSubDOF( const int manager, const std::vector<size_t> &globalDOFs ) const;

    //! Convert the global to local dof
    std::pair<size_t, int> globalToSub( size_t dof ) const;

    //! Convert the global to local dof
    std::vector<size_t> getGlobalDOF( const int manager, const std::vector<size_t> &subDOFs ) const;

    // Get the number of local DOFs
    inline size_t numLocal() const { return d_local[d_rank]; }

    // Get the number of global DOFs
    inline size_t numGlobal() const { return d_begin.back() + d_local.back(); }

    // Get the first local dof
    inline size_t begin() const { return d_begin[d_rank]; }

    // Get one past the last local dof
    inline size_t end() const { return d_begin[d_rank] + d_local[d_rank]; }

    // Get the local size for each rank
    const std::vector<size_t> &getLocalSize() const { return d_local; }

public: // Default constructors
    multiDOFHelper()                    = default;
    multiDOFHelper( multiDOFHelper && ) = default;
    multiDOFHelper( const multiDOFHelper & );

    multiDOFHelper &operator=( multiDOFHelper && ) = default;
    multiDOFHelper &operator                       =( const multiDOFHelper                        &);

public: // HDF5 interfaces
    void writeHDF5( size_t ) const;
    multiDOFHelper( size_t );

private:
    void initialize( int rank, std::vector<size_t> &&data );
    void initialize( const AMP::AMP_MPI &comm, const AMP::Array<size_t> &data );

private:
    int d_rank = 0;
    std::vector<size_t> d_index;
    std::vector<size_t> d_local;
    std::vector<size_t> d_begin;
    std::vector<size_t> d_globalSize;
    AMP::Array<size_t> d_localSize;
    AMP::Array<size_t> d_localOffset;
    AMP::Array<size_t> d_globalOffset;
};


} // namespace AMP::Discretization

#endif
