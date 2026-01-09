#ifndef included_AMP_kdtree
#define included_AMP_kdtree

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


namespace AMP::Mesh {
template<class TYPE>
class MeshPoint; // Forward declare MeshPoint
} // namespace AMP::Mesh


namespace AMP {


/**
 * \class kdtree
 * \brief A class used to to perform kd-tree based operations
 *
 * \details  This class provides routines for creating and search a kd-tree.
 *  The kdtree allows for log(N) nearest neighbor search.  The memory requirements
 *  for this class are O(N) on the number of points.  The search performance for small
 *  dimension data is O(log(N)).  For high dimension data, the number of points
 *  should be N >> 2^d where d is the dimension.  Otherwise the performance will degrade
 *  approaching O(N).
 */
class kdtree
{
public:
    // Empty constructor
    kdtree() : d_dim( 0 ), d_N( 0 ), d_tree( nullptr ) {}

    /**
     * \brief   Default constructor
     * \details  This is the default constructor for creating the kdtree
     * \param[in] ndim  The number of physical dimensions
     * \param[in] N     The number of points in the tree
     * \param[in] x     The coordinates of each point in the tree (x[ndim][N])
     */
    kdtree( int ndim, size_t N, const double *const *x );

    /**
     * \brief   Default constructor
     * \details  This an alternative constructor for creating the kdtree
     * \param[in] x     The coordinates of each point in the tree
     */
    kdtree( const std::vector<AMP::Mesh::MeshPoint<double>> &x );

    //!  Destructor
    ~kdtree();

    //! Copy constructor
    kdtree( const kdtree & ) = delete;

    //! Move constructor
    kdtree( kdtree && );

    //! Assignment operator
    kdtree &operator=( const kdtree & ) = delete;

    //! Move operator
    kdtree &operator=( kdtree && );


    /**
     * \brief   Constructor for 2d
     * \details  This will create a kdtree in 2d space
     * \param[in] N     The number of points in the tree
     * \param[in] x     The x coordinates of each point
     * \param[in] y     The x coordinates of each point
     */
    static std::shared_ptr<kdtree> create2d( size_t N, const double *x, const double *y );

    /**
     * \brief   Constructor for 3d
     * \details  This will create a kdtree in 3d space
     * \param[in] N     The number of points in the tree
     * \param[in] x     The x coordinates of each point
     * \param[in] y     The y coordinates of each point
     * \param[in] z     The z coordinates of each point
     */
    static std::shared_ptr<kdtree>
    create3d( size_t N, const double *x, const double *y, const double *z );


    //! Function to return the bounding box for the tree
    std::vector<double> box() const;


    /**
     * \brief   Add a point
     * \details  This will add a point to the kdtree.
     *    Note that no rebalancing will be performed
     * \param[in] x     The coordinates of the point
     */
    void add( const double *x );

    /**
     * \brief   Search the tree for the nearest neighbor point
     * \details  This will return the index of the nearest neighbor in the tree
     * \param[in] p      The point to search
     * \return           The nearest point
     */
    AMP::Mesh::MeshPoint<double> find_nearest( const AMP::Mesh::MeshPoint<double> &p ) const;

    /**
     * \brief   Search the tree for the nearest neighbor point
     * \details  This will return the index of the nearest neighbor in the tree
     * \param[in] x      The coordinates of the point to search (NDIM)
     * \param[out] dist  Optional output array for the distance to the nearest neighbor (1)
     * \param[out] pos   Optional output array for the position of the nearest neighbor (NDIM)
     */
    size_t find_nearest( const double *x, double *dist = nullptr, double *pos = nullptr ) const;

    /**
     * \brief   Search the tree for the nearest neighbor point
     * \details  This will return the index of the nearest neighbor in the tree for each point
     * \param[in] N       The number of points we want to search
     * \param[in] x       The coordinates of the points to search ( NDIM x N )
     * \param[out] index  The index of the nearest neighbors (N)
     * \param[out] dist   Optional output array for the distance to the nearest neighbor (N)
     * \param[out] pos    Optional output array for the position of the nearest neighbor (NDIMxN)
     */
    void find_nearest( int N,
                       const double *x,
                       size_t *index,
                       double *dist = nullptr,
                       double *pos  = nullptr ) const;

    /**
     * \brief   Search the tree for the nearest neighbor point
     * \details  This will return the index of the nearest neighbor in the tree
     * \param[in] x       The x coordinate of the search point
     * \param[in] y       The y coordinate of the search point
     */
    size_t find_nearest2d( const double x, const double y ) const;

    /**
     * \brief   Search the tree for the nearest neighbor point
     * \details  This will return the index of the nearest neighbor in the tree
     * \param[in] x       The x coordinate of the search point
     * \param[in] y       The y coordinate of the search point
     * \param[in] z       The z coordinate of the search point
     */
    size_t find_nearest3d( const double x, const double y, const double z ) const;


private: // Internal data
    uint8_t d_dim;
    size_t d_N;
    void *d_tree;

    // Find the nearest point in the tree to x
    size_t find_nearest2( const double *x, double &dist, double *pos ) const;
};


} // namespace AMP


#endif
