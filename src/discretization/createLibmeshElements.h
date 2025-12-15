#ifndef included_AMP_createLibmeshElements
#define included_AMP_createLibmeshElements

// AMP headers
#include "AMP/AMP_TPLs.h"
#include "AMP/mesh/MeshIterator.h"


#ifdef AMP_USE_LIBMESH


// Libmesh headers
DISABLE_WARNINGS
    #include "libmesh/libmesh_config.h"
    #undef LIBMESH_ENABLE_REFERENCE_COUNTING
    #include "libmesh/elem.h"
    #include "libmesh/enum_quadrature_type.h"
    #include "libmesh/fe_base.h"
    #include "libmesh/fe_type.h"
ENABLE_WARNINGS


namespace AMP::Discretization {


/**
  This is a helper class to create libmesh elements given a MeshIterator.
  It will cache the elements and allow for fast O(log(n)) access to the
  libmesh element given the MeshElementID.
*/
class createLibmeshElements
{
public:
    //! Empty constructor
    createLibmeshElements();

    //! De-constructor
    virtual ~createLibmeshElements();

    /**
     *  This function initializes / re-intializes the class to create the libmesh elements
     *     for all of the MeshElements in the given iterator.
     *     Note: This version only caches the libmesh elements
     *  \param[in] iterator     MeshElementIterator containing the elements of interest
     */
    void reinit( const AMP::Mesh::MeshIterator &iterator );

    /**
     *  This function initializes / re-intializes the class to create the libmesh elements
     *     and the FE base for all of the MeshElements in the given iterator.
     *     Note: This version caches the libmesh elements and the rule and base, reinitializing
     *     with the element.  This requires more memory.
     *  \param[in] iterator     MeshElementIterator containing the elements of interest
     *  \param[in] qtype        Type of the quadrature rule to use for the elements
     *  \param[in] qorder       Order of the quadrature rule to use for the elements
     *  \param[in] type         FE type to use for the elements
     *  \param[in] cache_fe     Do we want to cahce fe operators (true) or reinitialize the the
     * libmesh object (false).
     *                          If we cache the operators it will increase performance
     *                          but significantly increase memory requirements.
     *                          Note: these functions are not thread safe.
     */
    void reinit( const AMP::Mesh::MeshIterator &iterator,
                 libMeshEnums::QuadratureType qtype,
                 libMeshEnums::Order qorder,
                 std::shared_ptr<const libMesh::FEType> type,
                 bool cache_fe = false );

    /**
     *  This function returns the libmesh element given a MeshElementID
     *  \param[in] id           MeshElementID for the element of interest
     */
    const libMesh::Elem *getElement( const AMP::Mesh::MeshElementID &id ) const;

    /**
     *  This function returns the FE base given a MeshElementID
     *  \param[in] id           MeshElementID for the element of interest
     */
    const libMesh::FEBase *getFEBase( const AMP::Mesh::MeshElementID &id ) const;

    /**
     *  This function returns the quadrature rule used to build the elements
     *  \param[in] id           MeshElementID for the element of interest
     */
    const libMesh::QBase *getQBase( const AMP::Mesh::MeshElementID &id ) const;

    /**
     *  This function returns the FE type used to build the elements
     */
    const libMesh::FEType *getFEType() const { return d_type.get(); }

    /**
     *  Create a libmesh element from an AMP element (user must deallocate)
     */
    static libMesh::Elem *createElement( const AMP::Mesh::MeshElement &elem );

private:
    std::vector<AMP::Mesh::MeshElementID> d_ids;
    std::vector<size_t> d_index;
    libMeshEnums::QuadratureType d_qtype;
    libMeshEnums::Order d_qorder;
    std::shared_ptr<const libMesh::FEType> d_type;
    mutable AMP::Mesh::MeshElementID d_last_id;
    std::shared_ptr<libMesh::FEBase> d_base;
    std::shared_ptr<libMesh::QBase> d_rule;
    std::vector<libMesh::Elem *> d_elements;
    std::vector<libMesh::FEBase *> d_base_element;
    std::vector<libMesh::QBase *> d_rule_element;
};
} // namespace AMP::Discretization

#endif
#endif
