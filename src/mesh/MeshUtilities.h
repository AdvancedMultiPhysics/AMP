#ifndef included_AMP_MeshUtilities
#define included_AMP_MeshUtilities


#include "AMP/geometry/Geometry.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/kdtree2.h"


namespace AMP::Mesh {


/**
 * \brief    Get points in the mesh
 * \details  This function returns points in the mesh at the given resolution.
 *     Note: The nodes are always used.
 * \param mesh  Mesh to sample
 * \param dx    Resolution to use
 *              Note: if dx=0, then an automatic resolution is used.
 */
std::tuple<std::vector<Point>, std::vector<MeshElementID>> sample( const Mesh &mesh, double dx );


/**
 * \brief    Get points in the element
 * \details  This function returns points in the mesh element at the given resolution.
 * \param elem  Element to sample
 * \param dx    Resolution to use.
 */
std::vector<Point> sample( const MeshElement &elem, double dx );


/**
 * \brief    Get a grid with the overlap with the given geometry
 * \details  This function returns an Array with the given resolution in which
 *     each cell contains the volume of the underlying geometry that intersects with the cell.
 * \param geom  Geometry to use
 * \param N     Number of cells to use
 */
Array<double> volumeOverlap( const AMP::Geometry::Geometry &geom, const std::vector<int> &N );


/**
 * \class ElementFinder
 * \brief A class used to help find the nearest element to a point
 * \details  This class provides functions to help find the nearest element to a point
 */
class ElementFinder
{
public:
    //! Empty constructor
    ElementFinder() : d_pos_hash( 0 ), d_dist( 0 ) {}

    //! Empty constructor
    ElementFinder( std::shared_ptr<AMP::Mesh::Mesh> mesh );

    //! Empty constructor
    ElementFinder( std::shared_ptr<AMP::Mesh::Mesh> mesh, AMP::Mesh::MeshIterator it );

    //! Copy constructor
    ElementFinder( const ElementFinder & ) = delete;

    //! Move constructor
    ElementFinder( ElementFinder && ) = default;

    //! Move operator
    ElementFinder &operator=( ElementFinder && ) = default;

    //! Get the nearest element and point
    std::pair<std::unique_ptr<MeshElement>, Point> nearest( const Point &x ) const;

    /**
     * \brief    Calculate the distance to the object given a ray
     * \details  This function computes the distance to the object given a ray.
     *     If the ray is inside the object, this distance is negitive.  If the
     *     ray will never intersect the object, this distance is inf.
     * \param[in] pos   Current position of ray
     * \param[in] dir   Direction of ray (should be normalized for most uses)
     * @return          Returns the surface and distance to the nearest surface
     *                  (intersection = pos + dir*distance)
     */
    double distance( const Point &pos, const Point &dir ) const;

    //! Return the underlying tree
    inline const auto &getTree() const { return d_tree; }

    //! Ensure the tree is up-to-date
    void update() const;

private:
    void initialize() const;

private:
    std::shared_ptr<AMP::Mesh::Mesh> d_mesh;
    mutable uint64_t d_pos_hash;
    AMP::Mesh::MeshIterator d_elements;
    mutable double d_dist;
    mutable kdtree2<3, AMP::Mesh::MeshElementID> d_tree;
};


} // namespace AMP::Mesh


#endif
