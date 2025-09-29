#ifndef included_AMP_DelaunayFaceList
#define included_AMP_DelaunayFaceList

#include <array>
#include <cstdint>
#include <stdint.h>
#include <stdlib.h>
#include <vector>

#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/extended_int.h"


namespace AMP::DelaunayTessellation {


/*! Class for storing the faces on the convex hull
 *  Storing the list of faces on the convex hull is necessary, but requires maintaining
 *  a list of triangles and their faces.  This class simplifies this storage.
 *  In addition, this class will and new nodes to the convex hull, returning the triangles
 *  that were created.
 *  Note:  Seperate instantiations of this class are thread safe, but a single instance is not.
 */
template<int NDIM>
class FaceList
{
public:
    using Point    = std::array<int, NDIM>;
    using Triangle = std::array<int, NDIM + 1>;

    /*! @brief  Standard constructor
     *  @details  Default constructor to be used
     * @param N         The number of vertices
     * @param x         The coordinates of the nodes (NDIM x N)
     *                  Note: coordinates must not change or be deleted during lifetime of
     *                  FaceList.
     * @param tri_id    The initial triangle id
     * @param tri       The triangle list (NDIM+1)
     */
    FaceList( const int N, const Point *x, const int tri_id, const Triangle &tri );

    //! Function to get the number of faces on the convex hull
    int get_N_face() { return data.size(); }

    /*! @brief  Function to add a node to the convex hull
     *  @details  This function will add a new node to the convex hull.  In the process of
     *     of adding the node, some faces will be removed, new faces will be added,
     *     and new triangles will be generated.  This function will return the new triangles
     *     which must be added.  Note: the new triangles are not necessarilly Delaunay.
     * @param[in] node_id       The vertex index to add
     * @param[in,out] unused    A list of unused triangle ids
     * @param[in,out] N_tri     The number of triangles
     * @param[out] new_tri_id   A list of valid ids to use for new triangles
     * @param[out] new_tri      The list of new triangles that were created
     * @param[out] new_tri_nab  The list of triangle neighbors for the new triangles
     * @param[out] neighbor     The list of existing triangles that are neighbors to the new
     * triangles
     * @param[out] face_id      The list of existing triangle faces that are neighbors to the
     * new triangles
     */
    void add_node( const int node_id,
                   std::vector<size_t> &unused,
                   size_t &N_tri,
                   std::vector<uint32_t> &new_tri_id,
                   std::vector<Triangle> &new_tri,
                   std::vector<Triangle> &new_tri_nab,
                   std::vector<int> &neighbor,
                   std::vector<int> &face_id );


    //! Function to update faces on the convex hull
    /*! This function will update faces on the convex hull.  This is necessary if the flips
     * changed the faces that lie on the convex hull.  There are two possibilites, the triangles
     * could have been changed but the faces are the same, in which case we need to update the
     * triangle numbers, face ids, and internal strucutures.  The second possiblity is the
     * entire
     * face configuration could change (eg a 2-2 flip in 3d).  This requires updating the faces
     * on
     * the convex hull.
     * Note: the number of faces on the convex hull should never change due to flips.
     * @param N         The number of faces that have changed
     * @param old_tid   The old triangle numbers (N)
     * @param old_fid   The old face ids (N)
     * @param new_tid   The new triangle numbers (N)
     * @param new_fid   The new face ids (N)
     * @param tri       The complete triangle list ( N_tri x NDIM+1 )
     */
    void update_face( const int N,
                      const int old_tid[],
                      const int old_fid[],
                      const int new_tid[],
                      const int new_fid[],
                      const Triangle *tri );

private:
    static std::array<int64_t, NDIM> calc_surface_normal( const std::array<int, NDIM> x[] );

private:
    // Private constructors
    FaceList();                              // Empty constructor.
    FaceList( const FaceList & );            // no implementation for copy
    FaceList &operator=( const FaceList & ); // no implementation for copy

    // Structure to store face information
    struct face_data_struct {
        int prev        = -1;
        int next        = -1;
        int tri_id      = -1;        // Triangle id
        int face_id     = -1;        // Face id
        int index[NDIM] = { -1 };    // Indicies of the face vertices
        Point x[NDIM]   = { { 0 } }; // Coordinates of the face vertices
    };

    // Data members
    const int Nx;         // The number of vertices
    int hash_table[1024]; // Internal hash table to improve performance when search for a given face
    const Point *x0;      // The vertex coordinates
    double xc[NDIM];      // A point within the centroid
    std::vector<face_data_struct> data; // The stored data

    // Function that determines the location of the triangle
    bool outside_triangle( const Point x[NDIM], const Point &xi ) const;

    // Function to get a unique index for each face
    inline size_t get_face_index( int face, int tri ) { return face + tri * ( NDIM + 1 ); }

    // Function to delete a set of faces
    void delete_faces( std::vector<int> &ids );

    // Function to check that the internal data is valid
    void check_data();
};

} // namespace AMP::DelaunayTessellation

#endif
