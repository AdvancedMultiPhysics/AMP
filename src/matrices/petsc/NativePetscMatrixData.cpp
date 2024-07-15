#include "AMP/matrices/petsc/NativePetscMatrixData.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/petsc/NativePetscVectorData.h"
#include "AMP/vectors/petsc/PetscVector.h"

#include "petscmat.h"
#include "petscvec.h"


namespace AMP::LinearAlgebra {

/********************************************************
 * Constructors                                          *
 ********************************************************/
NativePetscMatrixData::NativePetscMatrixData() : MatrixData() {}

NativePetscMatrixData::NativePetscMatrixData( Mat m, bool internally_created )
{
    d_Mat                  = m;
    d_MatCreatedInternally = internally_created;
}

NativePetscMatrixData::NativePetscMatrixData( std::shared_ptr<MatrixParametersBase> params,
					      const std::function<std::vector<size_t>( size_t )> &getRow )
    : MatrixData( params )
{
    auto parameters = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    AMP_ASSERT( parameters );
    const auto &comm = parameters->getComm().getCommunicator();
    const auto nrows = parameters->getLocalNumberOfRows();
    const auto ncols = parameters->getLocalNumberOfColumns();
    MatCreate( comm, &d_Mat );
    MatSetType( d_Mat, MATAIJ );
    MatSetFromOptions( d_Mat );
    MatSetSizes( d_Mat, nrows, ncols, PETSC_DETERMINE, PETSC_DETERMINE );
    
    auto row_nnz           = parameters->entryList();
    const auto max_row_nnz = *( std::max_element( row_nnz, row_nnz + nrows ) );

    // this allocates twice as much memory as needed
    // parameters->entryList() is just nnz per row, but whether it is on/off diag
    // is not known from here. This allocates equal space in both on and off diag
    // to guarantee insertions work
    std::vector<PetscInt> petsc_row_nnz(nrows,0);
    std::transform(row_nnz, row_nnz + nrows, petsc_row_nnz.begin(),
		   []( size_t r ) -> PetscInt { return r; } );
    AMP::pout << "petsc nnz: ";
    for ( auto&& n : petsc_row_nnz) {
      AMP::pout << n << " ";
    }
    AMP::pout << std::endl;
    // Safe to call both preallocation routines
    // Only first used for serial runs, only second used for multiprocess runs
    MatSeqAIJSetPreallocation( d_Mat, 0, petsc_row_nnz.data() );
    MatMPIAIJSetPreallocation( d_Mat,
			       0, petsc_row_nnz.data(),
			       0, petsc_row_nnz.data() );
    MatSetUp( d_Mat );

    // Insert zeros into all rows explicitly, sets the non-zero structure
    auto rowDOFs = parameters->getLeftDOFManager();
    AMP_ASSERT( rowDOFs );
    const auto srow  = rowDOFs->beginDOF();
    std::vector<PetscInt> petsc_cols(max_row_nnz,0);
    for ( size_t i = 0; i < nrows; ++i ) {
        PetscInt global_row = srow + i;
        const auto nvals    = row_nnz[i];
	auto cols = getRow( global_row );
	std::transform( cols.begin(), cols.end(), petsc_cols.begin(),
			[]( size_t c ) -> PetscInt { return c; } );
        std::vector<double> vals( nvals, 0.0 );
        MatSetValues( d_Mat, 1, &global_row, nvals,
		      petsc_cols.data(), vals.data(), INSERT_VALUES );
    }
    MatAssemblyBegin( d_Mat, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( d_Mat, MAT_FINAL_ASSEMBLY );

    d_MatCreatedInternally = true;
}

NativePetscMatrixData::~NativePetscMatrixData()
{
    if ( d_MatCreatedInternally )
        PETSC::matDestroy( &d_Mat );
}


AMP_MPI NativePetscMatrixData::getComm() const
{
    if ( d_pParameters )
        return d_pParameters->getComm();

    AMP_ASSERT( d_Mat );
    MPI_Comm comm;
    PetscObjectGetComm( reinterpret_cast<PetscObject>( d_Mat ), &comm );
    return AMP_MPI( comm );
}

/********************************************************
 * Get the left/right Vector/DOFManager                  *
 ********************************************************/
Vector::shared_ptr NativePetscMatrixData::getRightVector() const
{
    Vec a;
    MatCreateVecs( d_Mat, &a, nullptr );
    return createVector( a, true );
}
Vector::shared_ptr NativePetscMatrixData::getLeftVector() const
{
    Vec a;
    MatCreateVecs( d_Mat, nullptr, &a );
    return createVector( a, true );
}

std::shared_ptr<Discretization::DOFManager> NativePetscMatrixData::getRightDOFManager() const
{
    auto parameters = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    if ( parameters )
        return parameters->getRightDOFManager();

    return getRightVector()->getDOFManager();
}

std::shared_ptr<Discretization::DOFManager> NativePetscMatrixData::getLeftDOFManager() const
{
    auto parameters = std::dynamic_pointer_cast<MatrixParameters>( d_pParameters );
    if ( parameters )
        return parameters->getLeftDOFManager();

    return getLeftVector()->getDOFManager();
}


/********************************************************
 * Get values/row by global id                           *
 ********************************************************/
size_t NativePetscMatrixData::numGlobalRows() const
{
    int rows, cols;
    MatGetSize( d_Mat, &rows, &cols );
    return (size_t) rows;
}
size_t NativePetscMatrixData::numGlobalColumns() const
{
    int rows, cols;
    MatGetSize( d_Mat, &rows, &cols );
    return (size_t) cols;
}
void NativePetscMatrixData::getRowByGlobalID( size_t row,
                                              std::vector<size_t> &cols,
                                              std::vector<double> &values ) const
{
    PetscInt numCols;
    MatGetRow( d_Mat, row, &numCols, nullptr, nullptr );
    cols.resize( numCols );
    values.resize( numCols );
    MatRestoreRow( d_Mat, row, &numCols, nullptr, nullptr );
    if ( cols.size() ) { // the restore zeros out nCols
        const PetscInt *out_cols;
        const PetscScalar *out_vals;
        MatGetRow( d_Mat, row, &numCols, &out_cols, &out_vals );
        std::copy(
            (unsigned int *) out_cols, (unsigned int *) ( out_cols + numCols ), cols.begin() );
        std::copy( (double *) out_vals, (double *) ( out_vals + numCols ), values.begin() );
        MatRestoreRow( d_Mat, row, &numCols, &out_cols, &out_vals );
    }
}
std::vector<size_t> NativePetscMatrixData::getColumnIDs( size_t row ) const
{
    PetscInt numCols;
    MatGetRow( d_Mat, row, &numCols, nullptr, nullptr );
    std::vector<size_t> cols( numCols );
    MatRestoreRow( d_Mat, row, &numCols, nullptr, nullptr );

    if ( cols.size() ) { // the restore zeros out nCols
        const PetscInt *out_cols;
        MatGetRow( d_Mat, row, &numCols, &out_cols, nullptr );
        std::copy(
            (unsigned int *) out_cols, (unsigned int *) ( out_cols + numCols ), cols.begin() );
        MatRestoreRow( d_Mat, row, &numCols, &out_cols, nullptr );
    }

    return cols;
}
void NativePetscMatrixData::addValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    std::vector<PetscInt> petsc_rows( num_rows );
    std::vector<PetscInt> petsc_cols( num_cols );
    std::copy( rows, rows + num_rows, petsc_rows.begin() );
    std::copy( cols, cols + num_cols, petsc_cols.begin() );

    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<const double *>( vals );
        MatSetValues(
            d_Mat, num_rows, &petsc_rows[0], num_cols, &petsc_cols[0], values, ADD_VALUES );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}
void NativePetscMatrixData::setValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    std::vector<PetscInt> petsc_rows( num_rows );
    std::vector<PetscInt> petsc_cols( num_cols );
    std::copy( rows, rows + num_rows, petsc_rows.begin() );
    std::copy( cols, cols + num_cols, petsc_cols.begin() );

    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<const double *>( vals );
        MatSetValues(
            d_Mat, num_rows, &petsc_rows[0], num_cols, &petsc_cols[0], values, INSERT_VALUES );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}
void NativePetscMatrixData::getValuesByGlobalID( size_t num_rows,
                                                 size_t num_cols,
                                                 size_t *rows,
                                                 size_t *cols,
                                                 void *vals,
                                                 const typeID &id ) const
{
    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<double *>( vals );
        // Zero out the data in values
        for ( size_t i = 0; i < num_rows * num_cols; i++ )
            values[i] = 0.0;
        // Get the data for each row
        auto leftDOFManager = getLeftDOFManager();
        size_t firstRow     = leftDOFManager->beginDOF();
        size_t numRows      = leftDOFManager->numLocalDOF();

        for ( size_t i = 0; i < num_rows; i++ ) {
            if ( rows[i] < firstRow || rows[i] >= firstRow + numRows )
                continue;
            PetscInt numCols = 0;
            MatGetRow( d_Mat, rows[i], &numCols, nullptr, nullptr );
            if ( numCols == 0 )
                continue;
            MatRestoreRow( d_Mat, rows[i], &numCols, nullptr, nullptr );
            const PetscInt *out_cols;
            const PetscScalar *out_vals;
            MatGetRow( d_Mat, rows[i], &numCols, &out_cols, &out_vals );
            for ( size_t j1 = 0; j1 < num_cols; j1++ ) {
                for ( int j2 = 0; j2 < numCols; j2++ ) {
                    if ( (int) cols[j1] == out_cols[j2] )
                        values[i * num_cols + j1] = out_vals[j2];
                }
            }
            MatRestoreRow( d_Mat, rows[i], &numCols, &out_cols, &out_vals );
        }
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}


/********************************************************
 * Clone the matrix                                      *
 ********************************************************/
std::shared_ptr<MatrixData> NativePetscMatrixData::cloneMatrixData() const
{
    Mat new_mat;
    MatDuplicate( d_Mat, MAT_DO_NOT_COPY_VALUES, &new_mat );
    return std::make_shared<NativePetscMatrixData>( new_mat, true );
}
std::shared_ptr<MatrixData> NativePetscMatrixData::transpose() const
{
    Mat new_mat;
    // note that PETSc documentation states that the new Mat is
    // not actually formed. Instead the transposed operations are
    // done with the original matrix. This could cause memory leaks
    /// for our wrappers potentially
    MatCreateTranspose( d_Mat, &new_mat );
    return std::make_shared<NativePetscMatrixData>( new_mat, true );
}
std::shared_ptr<MatrixData> NativePetscMatrixData::duplicateMat( Mat m )
{
    Mat newMat;
    MatDuplicate( m, MAT_DO_NOT_COPY_VALUES, &newMat );
    return std::make_shared<NativePetscMatrixData>( newMat, true );
}

void NativePetscMatrixData::extractDiagonal( std::shared_ptr<Vector> v ) const
{
    auto data = std::dynamic_pointer_cast<NativePetscVectorData>( v->getVectorData() );
    MatGetDiagonal( d_Mat, data->getVec() );
}


/********************************************************
 * Copy                                                  *
 ********************************************************/
void NativePetscMatrixData::copyFromMat( Mat m ) { MatCopy( m, d_Mat, SAME_NONZERO_PATTERN ); }

/********************************************************
 * makeConsistent                                        *
 ********************************************************/
void NativePetscMatrixData::makeConsistent( AMP::LinearAlgebra::ScatterType )
{
    MatAssemblyBegin( d_Mat, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( d_Mat, MAT_FINAL_ASSEMBLY );
}


} // namespace AMP::LinearAlgebra
