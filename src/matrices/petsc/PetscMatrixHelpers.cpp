#include "petsc/private/vecimpl.h"
#include "petscmat.h"
#include "petscvec.h"

#include "AMP/matrices/Matrix.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/petsc/PetscHelpers.h"
#include "AMP/vectors/petsc/PetscVector.h"


#if PETSC_VERSION_LT( 3, 7, 5 )
    #error AMP only supports PETSc 3.7.5 or greater
#endif


namespace PETSC {


static void reset_mat_ops( Mat m );


/********************************************************
 * Wrapper class for an AMP vector for PETSc Vec         *
 ********************************************************/
static uint32_t globalHash = AMP::Utilities::hash_char( "PetscMatrixWrapper" );
class PetscMatrixWrapper
{
public:
    PetscMatrixWrapper()                             = delete;
    PetscMatrixWrapper( const PetscMatrixWrapper & ) = delete;
    explicit PetscMatrixWrapper( std::shared_ptr<AMP::LinearAlgebra::Matrix> mat );
    ~PetscMatrixWrapper();
    inline Mat &getMat() { return d_petscMat; }
    inline auto getAMP() { return d_mat; }
    inline bool check() const { return hash == globalHash; }
    Mat duplicate();

protected:
    Mat d_petscMat;
    uint32_t hash;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_mat;
};
PetscMatrixWrapper::PetscMatrixWrapper( std::shared_ptr<AMP::LinearAlgebra::Matrix> mat )
    : hash( globalHash ), d_mat( mat )
{
    if ( mat->getComm().isNull() )
        AMP_ERROR( "Null communicator detected in " + mat->type() );
    MPI_Comm petsc_comm = mat->getComm().getCommunicator();
    size_t N_col_local  = mat->numLocalColumns();
    size_t N_row_local  = mat->numLocalRows();
    size_t N_col_global = mat->numGlobalColumns();
    size_t N_row_global = mat->numGlobalRows();
    MatCreateShell( petsc_comm,
                    N_row_local,
                    N_col_local,
                    N_row_global,
                    N_col_global,
                    static_cast<void *>( this ),
                    &d_petscMat );
    reset_mat_ops( d_petscMat );
}
PetscMatrixWrapper::~PetscMatrixWrapper()
{
    int refct = ( (PetscObject) d_petscMat )->refct;
    AMP_INSIST( refct <= 1, "Deleting a matrix still held by PETSc" );
    hash = 0;
}


/********************************************************
 * Get the AMP vector from the PETSc Mat                 *
 ********************************************************/
static inline PetscMatrixWrapper *getWrapper( Mat m )
{
    void *ctx;
    MatShellGetContext( m, &ctx );
    auto p = reinterpret_cast<PetscMatrixWrapper *>( ctx );
    AMP_ASSERT( p->check() );
    return p;
}
std::shared_ptr<AMP::LinearAlgebra::Matrix> getAMP( Mat m )
{
    auto p = getWrapper( m );
    return p->getAMP();
}
Mat getMat( std::shared_ptr<AMP::LinearAlgebra::Matrix> m )
{
    auto ptr = new PetscMatrixWrapper( m );
    return ptr->getMat();
}


/********************************************************
 * Reset the PETSc matrix operations                     *
 ********************************************************/
PetscErrorCode _AMP_Mult( Mat m, Vec i, Vec o )
{
    auto mat    = PETSC::getAMP( m );
    auto vecIn  = PETSC::getAMP( i );
    auto vecOut = PETSC::getAMP( o );
    vecIn->makeConsistent(); // PETSc controls VecIn which may not be consistent
    mat->mult( vecIn, vecOut );
    return 0;
}
PetscErrorCode _AMP_Mult_add( Mat m, Vec i, Vec a, Vec o )
{
    auto mat    = PETSC::getAMP( m );
    auto vecIn  = PETSC::getAMP( i );
    auto vecAdd = PETSC::getAMP( a );
    auto vecOut = PETSC::getAMP( o );
    vecIn->makeConsistent(); // PETSc controls VecIn which may not be consistent
    mat->mult( vecIn, vecOut );
    vecOut->add( *vecOut, *vecAdd );
    return 0;
}
PetscErrorCode _AMP_GetDiagonal( Mat m, Vec d )
{
    auto mat = PETSC::getAMP( m );
    auto vec = PETSC::getAMP( d );
    mat->extractDiagonal( vec );
    return 0;
}
PetscErrorCode _AMP_GetVecs( Mat m, Vec *right, Vec *left )
{
    auto mat = PETSC::getAMP( m );
    if ( right != nullptr ) {
        auto pRight = AMP::LinearAlgebra::PetscVector::view( mat->createInputVector() );
        VecDuplicate( pRight->getVec(), right );
    }
    if ( left != nullptr ) {
        auto pLeft = AMP::LinearAlgebra::PetscVector::view( mat->createOutputVector() );
        VecDuplicate( pLeft->getVec(), left );
    }
    return 0;
}
PetscErrorCode _AMP_AXPY( Mat y, PetscScalar alpha, Mat x, MatStructure )
{
    auto matX = PETSC::getAMP( x );
    auto matY = PETSC::getAMP( y );
    matY->axpy( alpha, *matX );
    return 0;
}
PetscErrorCode _AMP_Scale( Mat x, PetscScalar alpha )
{
    auto mat = PETSC::getAMP( x );
    mat->scale( alpha );
    return 0;
}
PetscErrorCode _AMP_Destroy( Mat x )
{
    auto p = getWrapper( x );
    delete p;
    return 0;
}
void reset_mat_ops( Mat M )
{
#if PETSC_VERSION_GE( 3, 12, 2 )
    // BP: have not checked for versions above 3.7.5
    MatShellSetManageScalingShifts( M );
#endif
    MatShellSetOperation( M, MATOP_MULT, (void ( * )()) _AMP_Mult );
#if PETSC_VERSION_GE( 3, 12, 2 )
    MatShellSetOperation( M, MATOP_CREATE_VECS, (void ( * )()) _AMP_GetVecs );
#elif PETSC_VERSION_LE( 3, 7, 5 )
    MatShellSetOperation( M, MATOP_GET_VECS, (void ( * )()) _AMP_GetVecs );
#else
    #error Not programmed for this version of petsc
#endif
    MatShellSetOperation( M, MATOP_GET_DIAGONAL, (void ( * )()) _AMP_GetDiagonal );
    MatShellSetOperation( M, MATOP_MULT_ADD, (void ( * )()) _AMP_Mult_add );
    MatShellSetOperation( M, MATOP_AXPY, (void ( * )()) _AMP_AXPY );
    MatShellSetOperation( M, MATOP_SCALE, (void ( * )()) _AMP_Scale );
    MatShellSetOperation( M, MATOP_DESTROY, (void ( * )()) _AMP_Destroy );
}


} // namespace PETSC
