#ifndef included_AMP_MeshVariable_H
#define included_AMP_MeshVariable_H

#include "AMP/mesh/Mesh.h"
#include "AMP/vectors/SubsetVariable.h"


namespace AMP::LinearAlgebra {


/** \class MeshVariable
 * \brief An AMP Variable that describes how to subset a DOF for a mesh
 * \see SubsetVector
 */
class MeshVariable : public SubsetVariable
{
public:
    /** \brief Constructor
     * \param[in] name         The name of the new variable
     * \param[in] mesh         The mesh to subset for
     * \param[in] useMeshComm  Use the comm of the mesh (otherwise use comm of parent)
     */
    MeshVariable( const std::string &name,
                  std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                  bool useMeshComm = true );

    std::shared_ptr<AMP::Discretization::DOFManager>
        getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> ) const override;

    AMP::AMP_MPI getComm( const AMP::AMP_MPI &comm ) const override;

public: // Functions inherited from Variable
    std::string className() const override { return "MeshVariable"; }
    uint64_t getID() const override;
    std::shared_ptr<VectorSelector> createVectorSelector() const override;
    void writeRestart( int64_t ) const override;
    MeshVariable( int64_t );

private:
    MeshVariable();
    bool d_useMeshComm = false;
    std::shared_ptr<const AMP::Mesh::Mesh> d_mesh;
};


/** \class MeshItertorVariable
 * \brief An AMP Variable that describes how to subset a DOF for a mesh iterator
 * \see SubsetVector
 */
class MeshIteratorVariable : public SubsetVariable
{
public:
    /** \brief Constructor
     * \param[in] name     The name of the new variable
     * \param[in] iterator The iterator over the mesh elements of interest
     * \param[in] comm     The communicator to subset for
     */
    MeshIteratorVariable( const std::string &name,
                          const AMP::Mesh::MeshIterator &iterator,
                          const AMP_MPI &comm );

    std::shared_ptr<AMP::Discretization::DOFManager>
        getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> ) const override;

    AMP::AMP_MPI getComm( const AMP::AMP_MPI &comm ) const override;

public: // Functions inherited from Variable
    std::string className() const override { return "MeshIteratorVariable"; }
    uint64_t getID() const override;
    std::shared_ptr<VectorSelector> createVectorSelector() const override;
    void writeRestart( int64_t ) const override;
    MeshIteratorVariable( int64_t );

private:
    MeshIteratorVariable();
    const AMP::AMP_MPI d_comm;
    const AMP::Mesh::MeshIterator d_iterator;
};

} // namespace AMP::LinearAlgebra

#endif
