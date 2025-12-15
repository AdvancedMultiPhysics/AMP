#ifndef included_AMP_kdtree2
#define included_AMP_kdtree2

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
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
     * \brief   Constructor
     * \details  This is the default constructor for creating the kdtree
     * \param[in] x     The coordinates of each point in the tree
     * \param[in] data  Data to associate with the nodes
     */
    kdtree2( const std::vector<Point> &x, const std::vector<TYPE> &data );

    //! Return the bounding box for the tree
    std::array<double, 2 * NDIM> box() const;

    //! Return the number of entries stored in the tree
    inline size_t size() const { return d_N; }

    //! Return true if the tree is empty
    inline bool empty() const { return d_N == 0; }

    //! Return the points in the tree
    std::vector<Point> getPoints() const;

    //! Return the points in the tree
    std::vector<std::pair<Point, TYPE>> getPointsAndData() const;

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
     * \details  This will return the point and data for the nearest neighbor in the tree
     * \param[in] x       The coordinates of the point to search (NDIM)
     * @return            Returns a tuple containing the point and the data
     */
    std::tuple<Point, TYPE> findNearest( const Point &x ) const;

    /**
     * \brief   Search the tree for the nearest neighbor points
     * \details  This will return the point and data for the nearest N neighbors in the tree
     * \param[in] x       The coordinates of the point to search (NDIM)
     * \param[in] N       The number of points to return
     * @return            Returns a vector of tuples containing the points and the data
     */
    std::vector<std::tuple<Point, TYPE>> findNearest( const Point &x, int N ) const;

    /**
     * \brief   Search the tree for the nearest neighbor points
     * \details  This will return the point and data for the all the points within the
     *     distance to the point x
     * \param[in] x       The coordinates of the point to search (NDIM)
     * \param[in] dist    The distance to the point
     * @return            Returns a vector of tuples containing the points and the data
     */
    std::vector<std::tuple<Point, TYPE>> findNearest( const Point &x, double dist ) const;

    /**
     * \brief   Search the tree for the nearest points to a ray
     * \details  This function will return all points within the given distance to the ray
     * \param[in] x       The coordinates of the starting point (NDIM)
     * \param[in] dir     The direction vector (NDIM)
     * \param[in] dist    The distance to search
     * @return            Returns a vector of candidates for the nearest points to a ray.
     */
    std::vector<std::tuple<Point, TYPE>>
    findNearestRay( const Point &x, const Point &dir, double dist ) const;


public: // Copy/assignment operators
    kdtree2()                  = default;
    ~kdtree2()                 = default;
    kdtree2( const kdtree2 & ) = delete;
    kdtree2( kdtree2 && )      = default;
    kdtree2 &operator=( const kdtree2 & ) = delete;
    kdtree2 &operator=( kdtree2 && ) = default;


private: // Internal data
    // Structure used to store point data in the lowest leaf
    struct data_struct {
        size_t N   = 0;
        Point *x   = nullptr;
        TYPE *data = nullptr;
        data_struct( size_t N );
        ~data_struct();
        void add( const Point &x2, const TYPE &d2 );
    };

    // Internal data
    size_t d_N          = 0;
    uint8_t d_split_dim = 0;
    double d_split      = 0;
    Point d_lb          = { 0 };
    Point d_ub          = { 0 };
    std::unique_ptr<kdtree2> d_left;
    std::unique_ptr<kdtree2> d_right;
    std::unique_ptr<data_struct> d_data;


private: // Internal functions
    kdtree2( size_t N, Point *x, TYPE *data );
    void initialize( size_t N, Point *x, TYPE *data );
    static size_t find_split( size_t N, const double *x );
    void splitData( size_t N, Point *x, TYPE *data );
    bool intersect( const Point &x, double dist2 ) const;
    void getPoints( std::vector<Point> &x ) const;
    void getPoints( std::vector<std::pair<Point, TYPE>> &x ) const;
    void
    findNearest( const Point &x, size_t N, std::tuple<Point, TYPE> *nearest, double *dist ) const;
    void findNearest( const Point &x,
                      double dist2,
                      std::vector<std::tuple<Point, TYPE>> &nearest ) const;
    void findNearestRay( const Point &x,
                         const Point &dir,
                         const Point &n_inv,
                         double dist2,
                         std::vector<std::tuple<Point, TYPE>> &nearest ) const;
    void
    checkNearest( const Point &x, size_t N, std::tuple<Point, TYPE> *nearest, double *dist ) const;
    static double distanceToBox( const std::array<double, NDIM> &pos,
                                 const std::array<double, NDIM> &ang,
                                 const std::array<double, NDIM> &lb,
                                 const std::array<double, NDIM> &ub );
};


} // namespace AMP


#endif
