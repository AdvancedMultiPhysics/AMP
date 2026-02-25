#ifndef included_AMP_DiffusionFD
#define included_AMP_DiffusionFD

#include "AMP/IO/PIO.h"
#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/geometry/Geometry.h"
#include "AMP/geometry/shapes/Box.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/Operator.h"
#include "AMP/utils/ArraySize.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "ProfilerApp.h"


namespace AMP::Operator {


/* ------------------------------------------------------------------------------------------
    Implementation of a finite-difference discretization of a constant-coefficient diffusion
operator
------------------------------------------------------------------------------------------- */

/* Specifically, given a set of coefficients {cij}, the discretized operators are:
    - In 1D:
        -cxx*u_xx (3-point stencil)
    - In 2D:
        -(cxx*u_xx + cyy*u_yy + cyx*u_yx) (9-point stencil)
    - In 3D:
        -(cxx*u_xx + cyy*u_yy + czz*u_zz + cyx*u_yx + czx*u_zx + czy*u_yz) (19-point stencil)

    - Mixed derivatives are discretized with central finite differences (as opposed to upwind)
    - The incoming database must specify the constants cij
    - The mesh in the incoming OperatorParamaters must be a BoxMesh build from a cube generator
    - This class provides the functionality to construct a RHS vector, given the source term on the
   RHS of the PDE and Dirichlet boundary function
    - Dirichlet boundary conditions are not eliminated; instead, boundary rows in the matrix are
   some scaled identity (with the constant equal to the diagonal entry in the discretization
   stencil)
    - Note that, technically, any second-order, constant-coefficient PDEs of the form specific above
   can be discretized with this class. I.e., this class does not enforce any requirement that the
   PDE be elliptic (determinant[PDE operator] < 0).
*/

class DiffusionFDOperator : public LinearOperator
{

    //
private:
    //! Problem dimension
    size_t d_dim = static_cast<size_t>( -1 );

    //! Mesh
    std::shared_ptr<AMP::Mesh::BoxMesh> d_BoxMesh = nullptr;

    //! Convenience member
    static constexpr auto VertexGeom = AMP::Mesh::GeomType::Vertex;

    //! Mesh sizes, hx, hy, hz. We compute these based on the incoming mesh
    std::vector<double> d_h;

    //
public:
    std::shared_ptr<AMP::Database> d_db;
    std::shared_ptr<AMP::Discretization::boxMeshDOFManager> d_DOFMan;

    // Constructor
    DiffusionFDOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ );

    // Destructor
    virtual ~DiffusionFDOperator() {}

    // Used by OperatorFactory to create a DiffusionFDOperator
    static std::unique_ptr<AMP::Operator::Operator>
    create( std::shared_ptr<AMP::Operator::OperatorParameters> params )
    {
        return std::make_unique<DiffusionFDOperator>( params );
    };

    // Used to register this operator in a factory
    std::string type() const override { return "DiffusionFDOperator"; }

    /* Create RHS vector consistent with the linear operator. Here:
    1. PDESourceFun is a function returning the PDE source term
    2. boundaryFun  is a function returning the Dirichlet boundary value at the given location on
    the given boundary */
    std::shared_ptr<AMP::LinearAlgebra::Vector> createRHSVector(
        std::function<double( const AMP::Mesh::Point & )> PDESourceFun,
        std::function<double( const AMP::Mesh::Point &, int boundary_id )> boundaryFun );

    // Populate a vector with the given function
    void fillVectorWithFunction( std::shared_ptr<AMP::LinearAlgebra::Vector> u,
                                 std::function<double( const AMP::Mesh::Point & )> fun ) const;

    // Vector of hx, hy, hz
    const std::vector<double> &getMeshSize() const;


    // Data
private:
    //! FD coefficients
    std::shared_ptr<std::vector<double>> d_stencil = nullptr;

    //! Local grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_localBox = nullptr;

    //! Global grid index box w/ zero ghosts
    std::shared_ptr<AMP::Mesh::BoxMesh::Box> d_globalBox = nullptr;

    //! ArraySize of the local box
    std::shared_ptr<AMP::ArraySize> d_localArraySize = nullptr;

    //! Placeholder array of grid indices
    std::array<size_t, 5> d_ijk;

    //
private:
    // Build and set d_DOFMan
    void setDOFManager();

    // FD stencil
    std::vector<double> createStencil() const;

    //! Pack cols+data vectors in given row.
    void getCSRData( size_t row, std::vector<size_t> &cols, std::vector<double> &data );

    /** Pack cols+data for the given row, which is a local row index, corresponding to a local DOF
     * that's on the interior of the current process. The corresponding local ijk mesh indices are
     * in ijkLocal
     */
    void getCSRDataInterior( std::array<size_t, 5> &ijkLocal,
                             size_t rowLocal,
                             std::vector<size_t> &cols,
                             std::vector<double> &data ) const;

    /** Pack cols+data vectors in given row. Note that, despite the name, this function works for
     * any global row, not just those on the a (physical or processor) boundary, but for a DOF on
     * the interior of a process it's less efficient than "getCSRDataInterior"
     */
    void
    getCSRDataBoundary( size_t row, std::vector<size_t> &cols, std::vector<double> &data ) const;


    // Fill CSR matrix with data
    void fillMatrixWithCSRData( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix );

    // Assemble matrix
    std::shared_ptr<AMP::LinearAlgebra::Matrix> createDiscretizationMatrix();

    //! Map from grid index ijk to the corresponding DOF
    size_t gridIndsToDOF( std::array<int, 3> ijk ) const;

    //! Get grid indices corresponding to the DOF
    std::array<int, 3> DOFToGridInds( size_t dof ) const;

    /** Convert a global element box to a global node box.
     * Modified from src/mesh/test/test_BoxMeshIndex.cpp by removing the possibility of any of the
     * grid dimensions being periodic.
     */
    AMP::Mesh::BoxMesh::Box getGlobalNodeBox() const;

    //! Convert a local element box to a local node box.
    AMP::Mesh::BoxMesh::Box getLocalNodeBox() const;
};
} // namespace AMP::Operator


#endif
