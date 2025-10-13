#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/testHelpers/test_MatrixVectorFactory.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"


namespace AMP::LinearAlgebra {


void testBasics( AMP::UnitTest &ut, const std::string &type )
{
    // Test creating a non-square matrix and ensure it is the proper size
    auto left  = AMP::LinearAlgebra::createSimpleVector<double>( 5, "left" );
    auto right = AMP::LinearAlgebra::createSimpleVector<double>( 10, "right" );
    auto mat   = AMP::LinearAlgebra::createMatrix(
        right, left, type, []( size_t row ) { return std::vector<size_t>( 1, row ); } );
    int rows = mat->numGlobalRows();
    int cols = mat->numGlobalColumns();
    if ( rows == 5 && cols == 10 )
        ut.passes( "Created non-square matrix (" + type + ")" );
    else
        ut.failure( "Failed non-square matrix (" + type + ")" );
}


void fillWithPseudoLaplacian( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix,
                              std::shared_ptr<AMP::Discretization::DOFManager> dofmap )
{
    matrix->enableModifications();
    if ( matrix->type() == "NativePetscMatrix" ) {
        std::map<size_t, std::vector<size_t>> allCols;
        std::map<size_t, std::vector<double>> allVals;
        for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
            const auto cols  = matrix->getColumnIDs( i );
            const auto ncols = cols.size();
            std::vector<double> vals( ncols );
            for ( size_t j = 0; j != ncols; j++ ) {
                if ( cols[j] == i )
                    vals[j] = static_cast<double>( ncols );
                else
                    vals[j] = -1;
            }
            allVals[i] = vals;
            allCols[i] = cols;
        }
        for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
            auto &cols = allCols[i];
            auto &vals = allVals[i];
            matrix->setValuesByGlobalID( 1, cols.size(), &i, cols.data(), vals.data() );
        }
    } else {

        for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
            auto cols        = matrix->getColumnIDs( i );
            const auto ncols = cols.size();
            std::vector<double> vals( ncols );
            for ( size_t j = 0; j != ncols; j++ ) {
                if ( cols[j] == i )
                    vals[j] = static_cast<double>( ncols );
                else
                    vals[j] = -1;
            }
            if ( ncols ) {
                matrix->setValuesByGlobalID<double>( 1, ncols, &i, cols.data(), vals.data() );
            }
        }
    }

    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    AMP::pout << *matrix;
}

static void fillWithPseudoLaplacian( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix,
                                     std::shared_ptr<const MatrixFactory> factory )
{
    auto dofmap = factory->getDOFMap();
    fillWithPseudoLaplacian( matrix, dofmap );
}

void MatrixTests::InstantiateMatrix( AMP::UnitTest *utils )
{
    PROFILE( "InstantiateMatrix" );
    auto matrix = d_factory->getMatrix();
    if ( matrix )
        utils->passes( "created " + matrix->type() );
    else
        utils->failure( "created " );
}

std::shared_ptr<AMP::LinearAlgebra::Matrix>
MatrixTests::getCopyMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
{
    // if the copy factory does not exist return the input matrix
    if ( d_copy_factory ) {
        auto copyMatrix = d_copy_factory->getMatrix();
        copyMatrix->zero(); // src and dst matricies may store zeros differently
        copyMatrix->copy( matrix );
        return copyMatrix;
    } else {
        return matrix;
    }
}


void MatrixTests::VerifyGetLeftRightVector( AMP::UnitTest *utils )
{
    PROFILE( "VerifyGetLeftRightVector" );
    auto matrix   = d_factory->getMatrix();
    matrix        = getCopyMatrix( matrix );
    auto factory1 = std::make_shared<AmpInterfaceRightVectorFactory>( matrix );
    auto factory2 = std::make_shared<AmpInterfaceLeftVectorFactory>( matrix );
    VectorTests tests1( factory1 );
    VectorTests tests2( factory2 );
    tests1.testBasicVector( utils );
    tests2.testBasicVector( utils );
#if defined( AMP_USE_PETSC )
    auto factory3 = std::make_shared<PETScInterfaceRightVectorFactory>( matrix );
    auto factory4 = std::make_shared<PETScInterfaceLeftVectorFactory>( matrix );
    VectorTests tests3( factory3 );
    VectorTests tests4( factory4 );
    tests3.testPetsc( utils );
    tests4.testPetsc( utils );
#endif
}


void MatrixTests::VerifyGetSetValuesMatrix( AMP::UnitTest *utils )
{
    PROFILE( "VerifyGetSetValuesMatrix" );
    auto matrix = d_factory->getMatrix();
    auto dofmap = d_factory->getDOFMap();

    //    matrix->makeConsistent();
    fillWithPseudoLaplacian( matrix, d_factory );
    matrix = getCopyMatrix( matrix );
    for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
        std::vector<size_t> cols;
        std::vector<double> vals;
        matrix->getRowByGlobalID( i, cols, vals );
        for ( size_t j = 0; j != cols.size(); j++ ) {
            double ans   = ( i == cols[j] ) ? cols.size() : -1.;
            double value = matrix->getValueByGlobalID( i, cols[j] );
            if ( vals[j] != ans || value != vals[j] ) {
                utils->failure( "bad value in matrix " + matrix->type() );
                return;
            }
        }
    }
    utils->passes( "verify get and set" + matrix->type() );
}


void MatrixTests::VerifyAXPYMatrix( AMP::UnitTest *utils )
{
    PROFILE( "VerifyAXPYMatrix" );

    // Create vectors/matricies from the factory
    auto matrix1 = d_factory->getMatrix();
    auto matrix2 = d_factory->getMatrix();

    fillWithPseudoLaplacian( matrix1, d_factory );
    fillWithPseudoLaplacian( matrix2, d_factory );
    matrix1 = getCopyMatrix( matrix1 );
    matrix2 = getCopyMatrix( matrix2 );

    // Test axpy
    matrix2->axpy( -2., matrix1 ); // matrix2 = -matrix1

    auto vector1lhs   = matrix1->createInputVector();
    auto vector2lhs   = matrix2->createInputVector();
    auto vector1rhs   = matrix1->createInputVector();
    auto vector2rhs   = matrix2->createInputVector();
    auto vectorresult = matrix2->createInputVector();

    vector1lhs->setRandomValues();
    vector1lhs->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    vector2lhs->copyVector( vector1lhs );

    matrix1->mult( vector1lhs, vector1rhs );
    matrix2->mult( vector2lhs, vector2rhs ); // vector2rhs = - vector1rhs
    vectorresult->add( *vector1rhs, *vector2rhs );
    if ( vectorresult->L1Norm() < 0.000001 )
        utils->passes( "matrices are opposite " + matrix1->type() );
    else
        utils->failure( "matrices are not opposite " + matrix1->type() );
    if ( vector1rhs->L1Norm() > 0.00001 )
        utils->passes( "non-trivial vector " + matrix1->type() );
    else
        utils->passes( "trivial vector " + matrix1->type() );

        // currently can't seem to correctly catch signal that PETSc throws
#if !defined( AMP_USE_PETSC )
    // Test that axpy failes with different sized matricies
    std::vector<size_t> row( 7 );
    for ( size_t i = 0; i < row.size(); i++ )
        row[i] = i;
    auto smallVec = AMP::LinearAlgebra::createSimpleVector<double>( 7, vector1lhs->getVariable() );
    auto smallMat = AMP::LinearAlgebra::createMatrix(
        smallVec, smallVec, d_factory->type(), [row]( size_t ) { return row; } );
    try {
        matrix2->axpy( -2., smallMat ); // matrix2 = -matrix1
        utils->failure( "axpy did not crash with different sized matrices " + matrix1->type() );
    } catch ( std::exception &err ) {
        utils->passes( "axpy correctly fails with different sized matrices " + matrix1->type() );
    } catch ( ... ) {
        utils->failure( "axpy fails with different sized matrices (unknown failure) " +
                        matrix1->type() );
    }
#endif
}

void MatrixTests::VerifyCopyMatrix( AMP::UnitTest *utils )
{
    PROFILE( "VerifyCopyMatrix" );

    // Create vectors/matrices from the factory
    auto matrix1 = d_factory->getMatrix();
    fillWithPseudoLaplacian( matrix1, d_factory );
    auto matrix2 = getCopyMatrix( matrix1 );

    auto u1 = matrix1->createInputVector();
    auto v1 = matrix1->createInputVector();

    auto u2           = matrix2->createInputVector();
    auto v2           = matrix2->createInputVector();
    auto vectorresult = matrix2->createInputVector();

    u1->setRandomValues();
    u1->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    u2->copyVector( u1 );
    u2->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    matrix1->mult( u1, v1 );
    matrix2->mult( u2, v2 ); // v2 = v1
    vectorresult->copyVector( v1 );
    vectorresult->subtract( *vectorresult, *v2 );
    if ( vectorresult->L1Norm() < 0.000001 )
        utils->passes( "matrices are not equal after copy " + matrix1->type() );
    else
        utils->failure( "matrices are not equal after copy " + matrix1->type() );
    if ( v1->L1Norm() > 0.00001 )
        utils->passes( "non-trivial vector " + matrix1->type() );
    else
        utils->passes( "trivial vector " + matrix1->type() );

        // currently can't seem to correctly catch signal that PETSc throws
#if !defined( AMP_USE_PETSC )
    // Test that copy fails with different sized matricies
    std::vector<size_t> row( 7 );
    for ( size_t i = 0; i < row.size(); i++ )
        row[i] = i;
    auto smallVec = AMP::LinearAlgebra::createSimpleVector<double>( 7, u1->getVariable() );
    auto smallMat = AMP::LinearAlgebra::createMatrix(
        smallVec, smallVec, d_factory->type(), [row]( size_t ) { return row; } );
    try {
        matrix2->copy( smallMat ); // matrix2 = matrix1
        utils->failure( "copy did not crash with different sized matrices " + matrix1->type() );
    } catch ( std::exception &err ) {
        utils->passes( "copy correctly fails with different sized matrices " + matrix1->type() );
    } catch ( ... ) {
        utils->failure( "copy fails with different sized matrices (unknown failure) " +
                        matrix1->type() );
    }
#endif
}


void MatrixTests::VerifyScaleMatrix( AMP::UnitTest *utils )
{
    PROFILE( "VerifyScaleMatrix" );
    auto matrix1 = d_factory->getMatrix();
    auto matrix2 = d_factory->getMatrix();

    fillWithPseudoLaplacian( matrix1, d_factory );
    fillWithPseudoLaplacian( matrix2, d_factory );
    matrix1 = getCopyMatrix( matrix1 );
    matrix2 = getCopyMatrix( matrix2 );

    matrix2->scale( 1.234567 ); // matrix2 = matrix1

    auto vector1lhs   = matrix1->createInputVector();
    auto vector2lhs   = matrix2->createInputVector();
    auto vector1rhs   = matrix1->createInputVector();
    auto vector2rhs   = matrix2->createInputVector();
    auto vectorresult = matrix2->createInputVector();

    vector1lhs->setRandomValues();
    vector1lhs->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    vector2lhs->copyVector( vector1lhs );

    matrix1->mult( vector1lhs, vector1rhs );
    matrix2->mult( vector2lhs, vector2rhs ); // vector2rhs = 1.234567vector1rhs

    vector1rhs->scale( 1.234567 );
    vectorresult->subtract( *vector1rhs, *vector2rhs );

    if ( vectorresult->L1Norm() < 0.000001 )
        utils->passes( "matrices are equally scaled " + matrix1->type() );
    else
        utils->failure( "matrices are equally scaled " + matrix1->type() );
    if ( vector1rhs->L1Norm() > 0.00001 )
        utils->passes( "non-trivial vector " + matrix1->type() );
    else
        utils->passes( "trivial vector " + matrix1->type() );
}


void MatrixTests::VerifyExtractDiagonal( AMP::UnitTest *utils )
{
    PROFILE( "VerifyExtractDiagonal" );
    auto matrix = d_factory->getMatrix();
    //    matrix->makeConsistent(); // required by PETSc
    //    auto vector     = matrix->createInputVector();
    //    size_t firstRow = vector->getCommunicationList()->getStartGID();
    size_t maxCols = matrix->numGlobalColumns();
    matrix->enableModifications();
    for ( size_t row = matrix->beginRow(); row < matrix->endRow(); ++row ) {
        if ( row >= maxCols )
            break;
        matrix->setValueByGlobalID( row, row, static_cast<double>( row + 1 ) );
    }
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD ); // required by PETSc
    AMP::pout << *matrix;
    matrix            = getCopyMatrix( matrix );
    auto diag         = matrix->extractDiagonal();
    double l1norm     = static_cast<double>( diag->L1Norm() );
    auto numRows      = static_cast<double>( matrix->numGlobalRows() );
    auto numCols      = static_cast<double>( matrix->numGlobalColumns() );
    double compareVal = std::min( numRows, numCols );
    compareVal        = compareVal * ( 1. + compareVal ) / 2.;
    if ( fabs( l1norm - compareVal ) < 0.000001 )
        utils->passes( "Verify extractDiagonal " + matrix->type() );
    else
        utils->failure( "Verify extractDiagonal " + matrix->type() );
}


void MatrixTests::VerifyMultMatrix( AMP::UnitTest *utils )
{
    PROFILE( "VerifyMultMatrix" );
    auto matrix = d_factory->getMatrix();

    matrix->zero();

    // Verify mult with 0 matrix
    matrix = getCopyMatrix( matrix );
    matrix->zero();
    auto vectorlhs = matrix->createInputVector();
    auto vectorrhs = matrix->createInputVector();
    double normlhs, normrhs;

    vectorlhs->setRandomValues();
    vectorlhs->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    matrix->mult( vectorlhs, vectorrhs );
    normlhs = static_cast<double>( vectorlhs->L2Norm() );
    normrhs = static_cast<double>( vectorrhs->L2Norm() );
    if ( ( normlhs > 0 ) && ( normrhs < 0.0000001 ) )
        utils->passes( "mult by 0 matrix " + matrix->type() );
    else
        utils->failure( "mult by 0 matrix " + matrix->type() );

    // Verify mult with identity
    vectorlhs->setToScalar( 1.0 );
    matrix->setDiagonal( vectorlhs );
    vectorlhs->setRandomValues();
    vectorlhs->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    matrix->mult( vectorlhs, vectorrhs );
    normlhs = static_cast<double>( vectorlhs->L2Norm() );
    vectorrhs->subtract( *vectorlhs, *vectorrhs );
    normrhs = static_cast<double>( vectorrhs->L2Norm() );
    if ( ( normlhs > 0 ) && ( normrhs < 0.0000001 ) )
        utils->passes( "mult by I matrix " + matrix->type() );
    else
        utils->failure( "mult by I matrix " + matrix->type() );

    // Try the non-trivial matrix
    fillWithPseudoLaplacian( matrix, d_factory );
    vectorlhs->setRandomValues();
    vectorlhs->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    matrix->mult( vectorlhs, vectorrhs );
    normlhs = static_cast<double>( vectorlhs->L2Norm() );
    normrhs = static_cast<double>( vectorrhs->L2Norm() );
    if ( ( normlhs > 0 ) && ( normrhs > 0 ) )
        utils->passes( "mult by other matrix " + matrix->type() );
    else
        utils->failure( "mult by other matrix " + matrix->type() );
}

void MatrixTests::VerifyMatMultMatrix_IA( AMP::UnitTest *utils )
{
    PROFILE( "VerifyMatMultMatrix_IA" );

    auto matLap = d_factory->getMatrix();
    fillWithPseudoLaplacian( matLap, d_factory );
    matLap = getCopyMatrix( matLap );

    auto x = matLap->createInputVector();
    auto y = matLap->createOutputVector();

    x->setToScalar( 1.0 );
    const auto l1x = static_cast<double>( x->L1Norm() );
    AMP_ASSERT( l1x > 0.0 );

    auto matId = d_factory->getMatrix();
    matId      = getCopyMatrix( matId );
    matId->zero();
    matId->setDiagonal( x );

    matLap->mult( x, y );
    const auto l1y = static_cast<double>( y->L1Norm() );

    auto matProd = AMP::LinearAlgebra::Matrix::matMatMult( matId, matLap );
    auto xp      = matProd->createInputVector();
    auto yp      = matProd->createOutputVector();
    xp->setToScalar( 1.0 );
    matProd->mult( xp, yp );
    auto l1yp = static_cast<double>( yp->L1Norm() );

    if ( AMP::Utilities::approx_equal( l1y, l1yp ) ) {
        utils->passes( "matMatMult I*A " + matProd->type() );
    } else {
        AMP::pout << "matMatMult I*A(" << matProd->type() << "), || A * x || = " << l1y
                  << ", || (I * A) * x || = " << l1yp << ", || x || = " << l1x << std::endl;
        utils->failure( "matMatMult I*A  " + matProd->type() );
    }
}

void MatrixTests::VerifyMatMultMatrix_AI( AMP::UnitTest *utils )
{
    PROFILE( "VerifyMatMultMatrix_AI" );

    auto matLap = d_factory->getMatrix();
    fillWithPseudoLaplacian( matLap, d_factory );
    matLap = getCopyMatrix( matLap );

    auto x = matLap->createInputVector();
    auto y = matLap->createOutputVector();

    x->setToScalar( 1.0 );
    const auto l1x = static_cast<double>( x->L1Norm() );
    AMP_ASSERT( l1x > 0.0 );

    auto matId = d_factory->getMatrix();
    matId      = getCopyMatrix( matId );
    matId->zero();
    matId->setDiagonal( x );

    matLap->mult( x, y );
    const auto l1y = static_cast<double>( y->L1Norm() );

    auto matProd = AMP::LinearAlgebra::Matrix::matMatMult( matLap, matId );
    auto xp      = matProd->createInputVector();
    auto yp      = matProd->createOutputVector();
    xp->setToScalar( 1.0 );
    matProd->mult( xp, yp );
    auto l1yp = static_cast<double>( yp->L1Norm() );

    if ( AMP::Utilities::approx_equal( l1y, l1yp ) ) {
        utils->passes( "matMatMult A*I " + matProd->type() );
    } else {
        AMP::pout << "matMatMult A*I(" << matProd->type() << "), || A * x || = " << l1y
                  << ", || (A * I) * x || = " << l1yp << ", || x || = " << l1x << std::endl;
        utils->failure( "matMatMult A*I  " + matProd->type() );
    }
}

void MatrixTests::VerifyMatMultMatrix_AA( AMP::UnitTest *utils )
{
    PROFILE( "VerifyMatMultMatrix_AA" );

    auto matLap = d_factory->getMatrix();
    fillWithPseudoLaplacian( matLap, d_factory );
    matLap = getCopyMatrix( matLap );

    auto x = matLap->createInputVector();
    auto y = matLap->createOutputVector();

    x->setToScalar( 1.0 );
    const auto l1x = static_cast<double>( x->L1Norm() );
    AMP_ASSERT( l1x > 0.0 );

    matLap->mult( x, y );
    const auto l1y = static_cast<double>( y->L1Norm() );

    auto matProd = AMP::LinearAlgebra::Matrix::matMatMult( matLap, matLap );
    auto xp      = matProd->createInputVector();
    auto yp      = matProd->createOutputVector();
    xp->setToScalar( 1.0 );
    matProd->mult( xp, yp );
    auto l1yp = static_cast<double>( yp->L1Norm() );

    if ( AMP::Utilities::approx_equal( l1y, l1yp ) ) {
        utils->passes( "matMatMult A*A " + matProd->type() );
    } else {
        AMP::pout << "matMatMult(" << matProd->type() << ") A*A, || A * ( A * x ) || = " << l1y
                  << ", || ( A * A ) * x || = " << l1yp << ", || x || = " << l1x << std::endl;
        utils->failure( "matMatMult A*A  " + matProd->type() );
    }
}


// Test matrix-matrix multiplication (this tests takes a long time for large matrices)
void MatrixTests::VerifyMatMultMatrix( AMP::UnitTest *utils )
{
    PROFILE( "VerifyMatMultMatrix" );
    auto matZero   = d_factory->getMatrix();
    auto matLaplac = d_factory->getMatrix();

    if ( matLaplac->numGlobalRows() > 1000 && matZero->type() == "DenseSerialMatrix" ) {
        // Matrix-matrix multiplies take a long time (skip it)
        return;
    }

    matZero->zero();
    matZero = getCopyMatrix( matZero );
    fillWithPseudoLaplacian( matLaplac, d_factory );
    matLaplac = getCopyMatrix( matLaplac );

    // Verify matMatMult with 0 matrix
    auto matProd = AMP::LinearAlgebra::Matrix::matMatMult( matZero, matLaplac );
    if ( matProd->LinfNorm() == 0.0 ) {
        //    if ( true ) {
        utils->passes( "matMatMult 0*A " + matZero->type() );
    } else {
        utils->failure( "matMatMult 0*A " + matZero->type() );
    }

    // Verify mult with identity on left
    VerifyMatMultMatrix_IA( utils );

    // Verify mult with identity on right
    VerifyMatMultMatrix_AI( utils );

    // Verify mult with two pseudo-Laplacian matrices
    VerifyMatMultMatrix_AA( utils );
}


void MatrixTests::VerifyAddElementNode( AMP::UnitTest *utils )
{
    PROFILE( "VerifyAddElementNode" );
    auto mesh   = d_factory->getMesh();
    auto dofmap = d_factory->getDOFMap();
    auto matrix = d_factory->getMatrix();
    matrix->zero();
    matrix->enableModifications();

    // Fill all the node-node entries
    auto it  = mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    auto end = it.end();
    std::vector<size_t> dofs;
    dofs.reserve( 40 );
    while ( it != end ) {
        auto nodes = it->getElements( AMP::Mesh::GeomType::Vertex );
        dofs.clear();
        for ( auto &node : nodes ) {
            std::vector<size_t> dofsNode;
            dofmap->getDOFs( node.globalID(), dofsNode );
            for ( auto &elem : dofsNode )
                dofs.push_back( elem );
#if 0            
            const auto row = node.globalID().meshID().getData();
            for ( size_t c = 0; c < dofs.size(); ++c ) {
                double val = -1.0;
                if ( row == c )
                    val = dofs.size() - 1;
                matrix->addValueByGlobalID( row, dofs[c], val );
            }
#endif
        }
#if 1
        for ( size_t r = 0; r < dofs.size(); r++ ) {
            for ( size_t c = 0; c < dofs.size(); c++ ) {
                double val = -1.0;
                if ( r == c )
                    val = dofs.size() - 1;
                matrix->addValueByGlobalID( dofs[r], dofs[c], val );
            }
        }
#endif
        ++it;
    }
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    // Call makeConsistent a second time
    // This can illustrate a bug where the fill pattern of remote data has changed
    //   and epetra maintains the list of remote rows, but updates the columns
    //   resulting in an access error using the std::vector
    // Another example of this bug can be found in extra_tests/test_Epetra_FECrsMatrix_bug
    // Note: there is no point in catching this bug with a try catch since a failure
    //   will cause asymettric behavior that create a deadlock with one process waiting
    //   for the failed process
    // The current workaround is to disable the GLIBCXX_DEBUG flags?
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    // Check the values
    bool pass = true;
    it        = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    end       = it.end();
    std::vector<size_t> cols;
    std::vector<double> values;
    while ( it != end ) {
        dofmap->getDOFs( it->globalID(), dofs );
        for ( auto &dof : dofs ) {
            matrix->getRowByGlobalID( dof, cols, values );
            double sum = 0.0;
            for ( auto &value : values )
                sum += value;
            if ( fabs( sum ) > 1e-14 || cols.empty() )
                pass = false;
        }
        ++it;
    }
    if ( pass )
        utils->passes( "VerifyAddElementNode " + matrix->type() );
    else
        utils->failure( "VerifyAddElementNode " + matrix->type() );
}


void test_matrix_loop( AMP::UnitTest &ut, std::shared_ptr<MatrixTests> tests )
{
    tests->InstantiateMatrix( &ut );
    tests->VerifyGetSetValuesMatrix( &ut );
    //    tests->VerifyAXPYMatrix( &ut );
    tests->VerifyCopyMatrix( &ut );
    tests->VerifyScaleMatrix( &ut );
    tests->VerifyGetLeftRightVector( &ut );
    tests->VerifyExtractDiagonal( &ut );
    tests->VerifyMultMatrix( &ut );
    tests->VerifyMatMultMatrix( &ut );
    tests->VerifyAddElementNode( &ut );
}


void test_matrix_loop( AMP::UnitTest &ut,
                       std::shared_ptr<MatrixFactory> factory,
                       std::shared_ptr<MatrixFactory> copy_factory )
{
    std::string name = factory->name();
    if ( copy_factory )
        name += "->" + copy_factory->name();
    PROFILE2( name );
    auto tests = std::make_shared<MatrixTests>( factory, copy_factory );
    test_matrix_loop( ut, tests );
}


} // namespace AMP::LinearAlgebra
