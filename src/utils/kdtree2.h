#ifndef included_AMP_kdtree2
#define included_AMP_kdtree2

#include <array>
#include <vector>


namespace AMP {


/**
 * \class kdtree2
 * \brief A class used to to perform kd-tree based operations
 *
 * \details  This class provides routines for creating and search a kd-tree.
 *  The kdtree allows for log(N) nearest neighbor search.  The memory requirements
 *  for this class are O(N) on the number of points.  The search performance for small
 *  dimension data is O(log(N)).  For high dimension data, the number of points
 *  should be N >> 2^d where d is the dimension.  Otherwise the performance will degrade
 *  approaching O(N).
 */
template<uint8_t NDIM, class TYPE>
class kdtree2
{
public: // Convience typedef
    using Point = std::array<double, NDIM>;

public:
    /**
     * \brief   Default constructor
     * \details  This is the default constructor for creating the kdtree
     * \param[in] N     The number of points in the tree
     * \param[in] x     The coordinates of each point in the tree
     * \param[in] data  Data to associate with the nodes
     */
    kdtree2( size_t N, const Point *x, const TYPE *data );

    //!  Destructor
    ~kdtree2();

    //! Copy constructor
    kdtree2( const kdtree2 & ) = delete;

    //! Move constructor
    kdtree2( kdtree2 && );

    //! Assignment operator
    kdtree2 &operator=( const kdtree2 & ) = delete;

    //! Move operator
    kdtree2 &operator=( kdtree2 && );


    //! Function to return the bounding box for the tree
    std::array<double, 2 * NDIM> box() const;

    //! Function to return the number of entries stored in the tree
    inline size_t size() const { return d_N; }

    /**
     * \brief  Function to get the current memory usage
     * \details  This function returns the current number of bytes in use by the structures.
     *     Note: This is the total number of bytes used
     */
    size_t memory_usage() const;

    /**
     * \brief   Add a point
     * \details  This will add a point to the kdtree.
     *    Note that no rebalancing will be performed
     * \param[in] x       The coordinates of the point
     * \param[in] data    The data to add
     */
    void add( const Point &x, const TYPE &data );

    /**
     * \brief   Search the tree for the nearest neighbor point
     * \details  This will return the index of the nearest neighbor in the tree
     * \param[in] x       The coordinates of the point to search (NDIM)
     * @return            Returns a tuple containing the point and the data
     */
    std::tuple<Point, TYPE> findNearest( const Point &x ) const;


private: // Internal data
    // Structure used to store point data in the lowest leaf
    struct data_struct {
        std::vector<Point> x;
        std::vector<TYPE> data;
    };

    // Internal data
    size_t d_N;
    uint8_t d_split_dim;
    double d_split;
    Point d_lb, d_ub;
    kdtree2 *d_left, *d_right;
    data_struct *d_data;


private: // Internal functions
    static size_t find_split( size_t N, const double *x );
    void splitData( size_t N, const Point *x, const TYPE *data );
    void checkNearest( const Point &x, std::tuple<Point, TYPE> &nearest ) const;
    static constexpr double norm( const Point &x, const Point &y );
};


} // namespace AMP


#endif