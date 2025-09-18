#include "AMP/vectors/testHelpers/petsc/PetscVectorTests.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/petsc/PetscHelpers.h"
#include "AMP/vectors/testHelpers/petsc/PetscVectorFactory.h"

#include "petsc/private/vecimpl.h"

#include "string"
#include <algorithm>


namespace AMP::LinearAlgebra {


#define PASS_FAIL( test, MSG )                                                    \
    do {                                                                          \
        if ( test )                                                               \
            ut->passes( d_factory->name() + " - " + __FUNCTION__ + ": " + MSG );  \
        else                                                                      \
            ut->failure( d_factory->name() + " - " + __FUNCTION__ + ": " + MSG ); \
    } while ( 0 )


void checkPetscError( AMP::UnitTest *ut, PetscErrorCode i )
{
    if ( i ) {
        char *ans;
        PetscErrorMessage( i, nullptr, &ans );
        ut->failure( ans );
        delete[] ans;
    }
}


void PetscVectorTests::testPetscVector( AMP::UnitTest *ut )
{
    DuplicatePetscVector( ut );
    VerifyDotPetscVector( ut );
    VerifyNormsPetscVector( ut );
    VerifyScalePetscVector( ut );
    VerifyAXPYPetscVector( ut );
    VerifyAbsPetscVector( ut );
    VerifyMaxPointwiseDividePetscVector( ut );
    VerifyGetSizePetscVector( ut );
    VerifySwapPetscVector( ut );
    VerifyAXPBYPetscVector( ut );
    VerifySetPetscVector( ut );
    VerifySetRandomPetscVector( ut );
    VerifySqrtPetscVector( ut );
    VerifyPointwiseMultPetscVector( ut );
    VerifyPointwiseDividePetscVector( ut );
    VerifyPointwiseMaxPetscVector( ut );
    VerifyPointwiseMinPetscVector( ut );
    VerifyPointwiseMaxAbsPetscVector( ut );
    VerifyLogPetscVector( ut );
    VerifyExpPetscVector( ut );
    VerifyAYPXPetscVector( ut );
    VerifyAXPBYPCZPetscVector( ut );
    Bug_612( ut );
}


void PetscVectorTests::Bug_612( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setToScalar( 5.0 );
        vectorb->setToScalar( 0.0 );

        PetscReal result;
        VecMaxPointwiseDivide( *veca, *vecb, &result );
        PASS_FAIL( result == 5.0, "computation 1" );

        vectorb->setToScalar( 5.0 );
        vectorb->getRawDataBlock<PetscScalar>( 0 )[0] = 0.0;

        VecMaxPointwiseDivide( *veca, *vecb, &result );
        PASS_FAIL( result == 5.0, "computation 2" );

        vectorb->setToScalar( 0.5 );
        vectorb->getRawDataBlock<PetscReal>( 0 )[0] = 0.0;

        VecMaxPointwiseDivide( *veca, *vecb, &result );
        PASS_FAIL( result == 10.0, "computation 3" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::DuplicatePetscVector( AMP::UnitTest *ut )
{
    auto vectora   = d_factory->getVector();
    auto petsc_vec = d_factory->getVec( vectora );
    Vec another_vec;
    checkPetscError( ut, VecDuplicate( *petsc_vec, &another_vec ) );
    auto dup = PETSC::getAMP( another_vec );
    ut->passes( "managed duplicated" );
    auto name1 = vectora->getName();
    auto name2 = dup->getName();
    bool test1 = ( dup->getGlobalSize() == vectora->getGlobalSize() ) &&
                 ( dup->getLocalSize() == vectora->getLocalSize() );
    PASS_FAIL( test1, "Allocated sizes are the same" );
    PASS_FAIL( name1 == name2, "Associated variables are the same: " + name1 + " - " + name2 );
    checkPetscError( ut, PETSC::vecDestroy( &another_vec ) );
    ut->passes( "managed duplicated destroyed" );
    checkPetscError( ut, VecDestroy( &another_vec ) );
}


void PetscVectorTests::VerifyPointwiseMaxAbsPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    auto vectore = d_factory->getVector();
    auto vectorf = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );
    auto vecc = d_factory->getVec( vectorc );
    auto vecd = d_factory->getVec( vectord );
    auto vece = d_factory->getVec( vectore );
    auto vecf = d_factory->getVec( vectorf );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( .65 );
        vectord->copyVector( vectora );
        vectore->setToScalar( .65 );

        checkPetscError( ut, VecPointwiseMaxAbs( *vecc, *veca, *vecb ) );
        checkPetscError( ut, VecPointwiseMaxAbs( *vecf, *vecd, *vece ) );

        PASS_FAIL( vectorc->equals( *vectorf ), "VecPointwiseMaxAbs test" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for double" );
    }
}


void PetscVectorTests::VerifyPointwiseMaxPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    auto vectore = d_factory->getVector();
    auto vectorf = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );
    auto vecc = d_factory->getVec( vectorc );
    auto vecd = d_factory->getVec( vectord );
    auto vece = d_factory->getVec( vectore );
    auto vecf = d_factory->getVec( vectorf );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( .35 );
        vectord->copyVector( vectora );
        vectore->setToScalar( .35 );

        checkPetscError( ut, VecPointwiseMax( *vecc, *veca, *vecb ) );
        checkPetscError( ut, VecPointwiseMax( *vecf, *vecd, *vece ) );

        PASS_FAIL( vectorc->equals( *vectorf ), "VecPointwiseMax test" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyPointwiseMinPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    auto vectore = d_factory->getVector();
    auto vectorf = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );
    auto vecc = d_factory->getVec( vectorc );
    auto vecd = d_factory->getVec( vectord );
    auto vece = d_factory->getVec( vectore );
    auto vecf = d_factory->getVec( vectorf );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( .35 );
        vectord->copyVector( vectora );
        vectore->setToScalar( .35 );

        checkPetscError( ut, VecPointwiseMin( *vecc, *veca, *vecb ) );
        checkPetscError( ut, VecPointwiseMin( *vecf, *vecd, *vece ) );

        PASS_FAIL( vectorc->equals( *vectorf ), "VecPointwiseMin test" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyAXPBYPCZPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    auto vectore = d_factory->getVector();
    auto vectorf = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );
    auto vecc = d_factory->getVec( vectorc );
    auto vecd = d_factory->getVec( vectord );
    auto vece = d_factory->getVec( vectore );
    auto vecf = d_factory->getVec( vectorf );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( -.5 );
        vectorc->setToScalar( 3.45678 );
        vectord->copyVector( vectora );
        vectore->setToScalar( -.5 );
        vectorf->setToScalar( 3.45678 );

        checkPetscError( ut, VecAXPBYPCZ( *veca, 3.14159, 1.414, 2.1727, *vecb, *vecc ) );
        checkPetscError( ut, VecAXPBYPCZ( *vecd, 3.14159, 1.414, 2.1727, *vece, *vecf ) );

        PASS_FAIL( vectora->equals( *vectord ), "VecAXPBYPCZ test" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyAYPXPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );
    auto vecc = d_factory->getVec( vectorc );
    auto vecd = d_factory->getVec( vectord );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( -.5 );
        vectorc->copyVector( vectora );
        vectord->copyVector( vectorb );

        checkPetscError( ut, VecAYPX( *veca, 2, *vecb ) );
        checkPetscError( ut, VecAYPX( *vecc, 2, *vecd ) );

        PASS_FAIL( vectora->equals( *vectorc ), "VecAYPX test" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyExpPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectora->abs( *vectora );
        vectorb->copyVector( vectora );

        checkPetscError( ut, VecExp( *veca ) );
        checkPetscError( ut, VecExp( *vecb ) );

        PASS_FAIL( vectora->equals( *vectorb ), "VecExp test" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyLogPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectora->abs( *vectora );
        vectorb->copyVector( vectora );

        checkPetscError( ut, VecLog( *veca ) );
        checkPetscError( ut, VecLog( *vecb ) );

        PASS_FAIL( vectora->equals( *vectorb ), "VecLog test" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyNormsPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto veca    = d_factory->getVec( vectora );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        PetscReal l1norm_a1, l1norm_a2;
        PetscReal l2norm_a1, l2norm_a2;
        PetscReal infnorm_a1, infnorm_a2;
        checkPetscError( ut, VecNorm( *veca, NORM_1, &l1norm_a1 ) );
        checkPetscError( ut, VecNorm( *veca, NORM_2, &l2norm_a1 ) );
        checkPetscError( ut, VecNorm( *veca, NORM_INFINITY, &infnorm_a1 ) );
        l1norm_a2  = static_cast<PetscReal>( vectora->L1Norm() );
        l2norm_a2  = static_cast<PetscReal>( vectora->L2Norm() );
        infnorm_a2 = static_cast<PetscReal>( vectora->maxNorm() );
        PASS_FAIL( fabs( l1norm_a1 - l1norm_a2 ) < 1e-12,
                   "l1 norm: native norm equals interface norm 1" );
        PASS_FAIL( fabs( l2norm_a1 - l2norm_a2 ) < 1e-15,
                   "l2 norm: native norm equals interface norm 1" );
        PASS_FAIL( fabs( infnorm_a1 - infnorm_a2 ) < 1e-15,
                   "inf norm: native norm equals interface norm 1" );

        auto vectorc = d_factory->getVector();
        vectorc->copyVector( vectora );

        auto vecc = d_factory->getVec( vectorc );
        PetscReal l1norm_c1, l1norm_c2;
        PetscReal l2norm_c1, l2norm_c2;
        PetscReal infnorm_c1, infnorm_c2;
        checkPetscError( ut, VecNorm( *vecc, NORM_1, &l1norm_c1 ) );
        checkPetscError( ut, VecNorm( *vecc, NORM_2, &l2norm_c1 ) );
        checkPetscError( ut, VecNorm( *vecc, NORM_INFINITY, &infnorm_c1 ) );
        l1norm_c2  = static_cast<PetscReal>( vectorc->L1Norm() );
        l2norm_c2  = static_cast<PetscReal>( vectorc->L2Norm() );
        infnorm_c2 = static_cast<PetscReal>( vectorc->maxNorm() );
        PASS_FAIL( fabs( l1norm_c1 - l1norm_c2 ) < 1e-12,
                   "l1 norm: native norm equals interface norm 2" );
        PASS_FAIL( fabs( l2norm_c1 - l2norm_c2 ) < 1e-15,
                   "l2 norm: native norm equals interface norm 2" );
        PASS_FAIL( fabs( infnorm_c1 - infnorm_c2 ) < 1e-15,
                   "inf norm: native norm equals interface norm 2" );
        PASS_FAIL( fabs( l1norm_a1 - l1norm_c1 ) < 0.0000001, "l1 norms equal" );
        PASS_FAIL( fabs( l2norm_a1 - l2norm_c1 ) < 0.0000001, "l2 norms equal" );
        PASS_FAIL( fabs( infnorm_a1 - infnorm_c1 ) < 0.0000001, "max norms equal" );
        checkPetscError( ut, VecNormBegin( *vecc, NORM_1, &l1norm_c1 ) );
        checkPetscError( ut, VecNormBegin( *vecc, NORM_2, &l2norm_c1 ) );
        checkPetscError( ut, VecNormBegin( *vecc, NORM_INFINITY, &infnorm_c1 ) );
        checkPetscError( ut, VecNormEnd( *vecc, NORM_1, &l1norm_c1 ) );
        checkPetscError( ut, VecNormEnd( *vecc, NORM_2, &l2norm_c1 ) );
        checkPetscError( ut, VecNormEnd( *vecc, NORM_INFINITY, &infnorm_c1 ) );
        l1norm_c2  = static_cast<PetscReal>( vectorc->L1Norm() );
        l2norm_c2  = static_cast<PetscReal>( vectorc->L2Norm() );
        infnorm_c2 = static_cast<PetscReal>( vectorc->maxNorm() );
        PASS_FAIL( fabs( l1norm_c1 - l1norm_c2 ) < 0.00001,
                   "l1 norm: native norm equals interface norm 3" );
        PASS_FAIL( fabs( l2norm_c1 - l2norm_c2 ) < 0.00001,
                   "l2 norm: native norm equals interface norm 3" );
        PASS_FAIL( fabs( infnorm_c1 - infnorm_c2 ) < 0.00001,
                   "inf norm: native norm equals interface norm 3" );
        PASS_FAIL( fabs( l1norm_a1 - l1norm_c1 ) < 0.0000001, "l1 norms equal" );
        PASS_FAIL( fabs( l2norm_a1 - l2norm_c1 ) < 0.0000001, "l2 norms equal" );
        PASS_FAIL( fabs( infnorm_a1 - infnorm_c1 ) < 0.0000001, "max norms equal" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscReal" );
    }
}


void PetscVectorTests::VerifyAXPBYPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( 2.0 );

        auto vectorc = d_factory->getVector();
        vectorc->copyVector( vectora );
        auto vectord = d_factory->getVector();
        vectord->copyVector( vectorb );

        auto veca = d_factory->getVec( vectora );
        auto vecb = d_factory->getVec( vectorb );
        auto vecc = d_factory->getVec( vectorc );
        auto vecd = d_factory->getVec( vectord );

        checkPetscError( ut, VecAXPBY( *veca, 1.234, 2.345, *vecb ) );
        checkPetscError( ut, VecAXPBY( *vecc, 1.234, 2.345, *vecd ) );

        PASS_FAIL( vectora->L2Norm() != 0, "Trivial axpby computed" );
        PASS_FAIL( vectora->equals( *vectorc ), "Native axpby matches managed axpby" );

        vectora->axpby( 1.234, 2.345, *vectorb );
        vectorc->axpby( 1.234, 2.345, *vectord );

        PASS_FAIL( vectora->L2Norm() != 0, "Trivial axpby computed" );
        PASS_FAIL( vectora->equals( *vectorc ), "Native axpby matches managed axpby" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscReal" );
    }
}


void PetscVectorTests::VerifySwapPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );

    PetscReal norm1, norm2, norm3, norm4;

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setToScalar( 1 );
        vectorb->setToScalar( 2 );
        vectora->swapVectors( vectorb );
        norm1 = static_cast<PetscReal>( vectora->maxNorm() );
        norm2 = static_cast<PetscReal>( vectorb->maxNorm() );
        checkPetscError( ut, VecNorm( *veca, NORM_INFINITY, &norm3 ) );
        checkPetscError( ut, VecNorm( *vecb, NORM_INFINITY, &norm4 ) );
        bool test1 = norm1 == 2 && norm2 == 1;
        bool test2 = norm3 == 2 && norm4 == 1;
        PASS_FAIL( test1, "Swap vectors AMP interface works with AMP Vector" );
        PASS_FAIL( test2, "Swap vectors AMP interface works with PETSc Vec" );

        vectora->setToScalar( 3 );
        vectorb->setToScalar( 4 );
        checkPetscError( ut, VecSwap( *veca, *vecb ) );
        PetscObjectStateIncrease( reinterpret_cast<::PetscObject>( *veca ) );
        PetscObjectStateIncrease( reinterpret_cast<::PetscObject>( *vecb ) );
        norm1 = static_cast<PetscReal>( vectora->maxNorm() );
        norm2 = static_cast<PetscReal>( vectorb->maxNorm() );
        checkPetscError( ut, VecNorm( *veca, NORM_INFINITY, &norm3 ) );
        checkPetscError( ut, VecNorm( *vecb, NORM_INFINITY, &norm4 ) );
        test1 = norm1 == 4 && norm2 == 3;
        test2 = norm3 == 4 && norm4 == 3;
        PASS_FAIL( test1, "Swap vectors PETSc interface works with AMP Vector" );
        PASS_FAIL( test2, "Swap vectors PETSc interface works with PETSc Vec" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscReal" );
    }
}


void PetscVectorTests::VerifyGetSizePetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );

    PetscInt sizea1, sizea2, sizeb1, sizeb2;
    if ( vectora->isType<PetscScalar>( 0 ) ) {
        sizea1 = static_cast<PetscInt>( vectora->getGlobalSize() );
        checkPetscError( ut, VecGetSize( *veca, &sizea2 ) );
        sizeb1 = static_cast<PetscInt>( vectorb->getGlobalSize() );
        checkPetscError( ut, VecGetSize( *vecb, &sizeb2 ) );

        PASS_FAIL( sizea1 == sizea2, "Native PETSc: Native interface matches AMP interface" );
        PASS_FAIL( sizeb1 == sizeb2, "Managed PETSc: Native interface matches AMP interface" );
        if ( sizea1 == sizeb1 )
            ut->passes( "Managed PETSc matches native PETSc" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyMaxPointwiseDividePetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( 3.14159 );

        auto vectorc = d_factory->getVector();
        vectorc->copyVector( vectora );
        auto vectord = d_factory->getVector();
        vectord->copyVector( vectorb );

        auto veca = d_factory->getVec( vectora );
        auto vecb = d_factory->getVec( vectorb );
        auto vecc = d_factory->getVec( vectorc );
        auto vecd = d_factory->getVec( vectord );

        PetscReal ans1, ans2;

        checkPetscError( ut, VecMaxPointwiseDivide( *veca, *vecb, &ans1 ) );
        checkPetscError( ut, VecMaxPointwiseDivide( *vecc, *vecd, &ans2 ) );

        PASS_FAIL( fabs( ans1 - ans2 ) < 0.000001,
                   "VecMaxPointwiseDivide working for both vector types" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscReal" );
    }
}


void PetscVectorTests::VerifyAbsPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto veca    = d_factory->getVec( vectora );
    auto vecb    = d_factory->getVec( vectorb );
    if ( !veca || !vecb )
        ut->failure( "PETSC abs create" );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectora->addScalar( *vectora, 1. );
        vectorb->copyVector( vectora );
        vectora->scale( -1. );
        checkPetscError( ut, VecAbs( *veca ) );
        vectorb->subtract( *vectora, *vectorb );
        PASS_FAIL( vectorb->L1Norm() < 0.000001, "native interface on native petsc abs works" );

        auto vectorc = d_factory->getVector();
        auto vectord = d_factory->getVector();

        auto vecc = d_factory->getVec( vectorc );
        auto vecd = d_factory->getVec( vectord );
        if ( !vecc || !vecd )
            ut->failure( "PETSC abs create" );

        vectorc->setRandomValues();
        vectorc->addScalar( *vectorc, 1. );
        vectord->copyVector( vectorc );
        vectorc->scale( -1. );
        checkPetscError( ut, VecAbs( *vecc ) );
        vectord->subtract( *vectorc, *vectord );
        PASS_FAIL( vectord->L1Norm() < 0.000001, "managed interface on native petsc abs works" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyPointwiseMultPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    if ( !vectora || !vectorb || !vectorc || !vectord )
        ut->failure( "PointwiseMult create" );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( 4.567 );
        vectorc->copyVector( vectora );
        vectord->copyVector( vectorb );

        auto veca = d_factory->getVec( vectora );
        auto vecb = d_factory->getVec( vectorb );

        checkPetscError( ut, VecPointwiseMult( *veca, *veca, *vecb ) );
        vectorc->multiply( *vectorc, *vectord );
        vectorc->subtract( *vectorc, *vectora );
        PASS_FAIL( vectorc->L1Norm() < 0.000001, "managed interface for native vector" );

        auto vectore = d_factory->getVector();
        auto vectorf = d_factory->getVector();
        auto vectorg = d_factory->getVector();
        auto vectorh = d_factory->getVector();

        vectore->setRandomValues();
        vectorf->setToScalar( 4.567 );
        vectorg->copyVector( vectore );
        vectorh->copyVector( vectorf );

        auto vece = d_factory->getVec( vectore );
        auto vecf = d_factory->getVec( vectorf );

        checkPetscError( ut, VecPointwiseMult( *vece, *vece, *vecf ) );
        vectorg->multiply( *vectorg, *vectorh );
        vectorg->subtract( *vectorg, *vectore );

        PASS_FAIL( vectorg->L1Norm() < 0.000001, "native interface for managed vector" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifyPointwiseDividePetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto vectorc = d_factory->getVector();
    auto vectord = d_factory->getVector();
    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setToScalar( 4.567 );
        vectorc->copyVector( vectora );
        vectord->copyVector( vectorb );

        auto veca = d_factory->getVec( vectora );
        auto vecb = d_factory->getVec( vectorb );

        checkPetscError( ut, VecPointwiseDivide( *veca, *veca, *vecb ) );
        vectorc->divide( *vectorc, *vectord );
        vectorc->subtract( *vectorc, *vectora );
        PASS_FAIL( vectorc->L1Norm() < 0.000001, "managed interface for native vector" );

        auto vectore = d_factory->getVector();
        auto vectorf = d_factory->getVector();
        auto vectorg = d_factory->getVector();
        auto vectorh = d_factory->getVector();

        vectore->setRandomValues();
        vectorf->setToScalar( 4.567 );
        vectorg->copyVector( vectore );
        vectorh->copyVector( vectorf );

        auto vece = d_factory->getVec( vectore );
        auto vecf = d_factory->getVec( vectorf );

        checkPetscError( ut, VecPointwiseDivide( *vece, *vece, *vecf ) );
        vectorg->divide( *vectorg, *vectorh );
        vectorg->subtract( *vectorg, *vectore );

        PASS_FAIL( vectorg->L1Norm() < 0.000001, "native interface for managed vector" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifySqrtPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();

        auto vectorb = d_factory->getVector();
        vectorb->copyVector( vectora );

        auto veca = d_factory->getVec( vectora );
        auto vecb = d_factory->getVec( vectorb );
        checkPetscError( ut, VecSqrtAbs( *veca ) );
        checkPetscError( ut, VecSqrtAbs( *vecb ) );
        bool equal = vectora->equals( *vectorb );
        PASS_FAIL( equal, "Vector square root" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifySetRandomPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setToScalar( 10.0 );
        vectorb->setToScalar( 10.0 );

        checkPetscError( ut, VecSetRandom( *veca, nullptr ) );
        checkPetscError( ut, VecSetRandom( *vecb, nullptr ) );

        PASS_FAIL( vectora->maxNorm() < 1.0, "VecSetRandom for native petsc" );
        PASS_FAIL( vectorb->maxNorm() < 1.0, "VecSetRandom for managed petsc" );

        vectora->setToScalar( 5.0 );
        vectorb->setToScalar( 6.0 );
        vectora->setRandomValues();
        vectorb->setRandomValues();

        PASS_FAIL( vectora->maxNorm() < 1.0, "setToScalar for native petsc" );
        PASS_FAIL( vectorb->maxNorm() < 1.0, "setToScalar for managed petsc" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscScalar" );
    }
}


void PetscVectorTests::VerifySetPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();

    auto veca = d_factory->getVec( vectora );
    auto vecb = d_factory->getVec( vectorb );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        checkPetscError( ut, VecSet( *veca, 2.0 ) );
        checkPetscError( ut, VecSet( *vecb, 3.0 ) );

        auto aL1   = static_cast<PetscReal>( vectora->L1Norm() );
        auto bL1   = static_cast<PetscReal>( vectorb->L1Norm() );
        auto amax  = static_cast<PetscReal>( vectora->maxNorm() );
        auto bmax  = static_cast<PetscReal>( vectorb->maxNorm() );
        auto asize = vectora->getGlobalSize();
        auto bsize = vectorb->getGlobalSize();
        bool test1 = ( fabs( aL1 - asize * 2.0 ) < 0.000001 ) && ( fabs( amax - 2.0 ) < 0.000001 );
        bool test2 = ( fabs( bL1 - bsize * 3.0 ) < 0.000001 ) && ( fabs( bmax - 3.0 ) < 0.000001 );
        vectora->setToScalar( 5.0 );
        vectorb->setToScalar( 6.0 );
        aL1        = static_cast<PetscReal>( vectora->L1Norm() );
        bL1        = static_cast<PetscReal>( vectorb->L1Norm() );
        amax       = static_cast<PetscReal>( vectora->maxNorm() );
        bmax       = static_cast<PetscReal>( vectorb->maxNorm() );
        bool test3 = ( fabs( aL1 - asize * 5.0 ) < 0.000001 ) && ( fabs( amax - 5.0 ) < 0.000001 );
        bool test4 = ( fabs( bL1 - bsize * 6.0 ) < 0.000001 ) && ( fabs( bmax - 6.0 ) < 0.000001 );
        PASS_FAIL( test1, "VecSet for native petsc" );
        PASS_FAIL( test2, "VecSet for managed petsc" );
        PASS_FAIL( test3, "setToScalar for native petsc" );
        PASS_FAIL( test4, "setToScalar for managed petsc" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscReal" );
    }
}


void PetscVectorTests::VerifyAXPYPetscVector( AMP::UnitTest *ut )
{
    auto vectora      = d_factory->getVector();
    auto vectora2     = d_factory->getVector();
    auto vectora_orig = d_factory->getVector();
    auto vectorb      = d_factory->getVector();
    auto vectorb2     = d_factory->getVector();
    auto veca         = d_factory->getVec( vectora );
    auto vecb         = d_factory->getVec( vectorb );
    auto veca2        = d_factory->getVec( vectora2 );
    auto veca_orig    = d_factory->getVec( vectora_orig );
    if ( !veca || !vecb || !*veca || !*veca_orig )
        ut->failure( "PETSc AXPY create" );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setRandomValues();
        vectora_orig->copyVector( vectora );
        vectora2->copyVector( vectora );
        vectorb2->copyVector( vectorb );
        checkPetscError( ut, VecAXPY( *veca, 1.23456, *vecb ) );
        vectora2->axpy( 1.23456, *vectorb2, *vectora2 );
        PASS_FAIL( vectora->equals( *vectora2, 1.0e-13 ), "PETSc VecAXPY and AMP::axpy equal " );
        auto norma  = static_cast<PetscReal>( vectora->L2Norm() );
        auto norma2 = static_cast<PetscReal>( vectora2->L2Norm() );
        PASS_FAIL( fabs( norma - norma2 ) < 1e-14, "native interface on native petsc axpy works" );

        auto vectorc  = d_factory->getVector();
        auto vectorc2 = d_factory->getVector();
        auto vectord  = d_factory->getVector();
        auto vectord2 = d_factory->getVector();

        vectorc->copyVector( vectora_orig );
        vectorc2->copyVector( vectora_orig );
        vectord->copyVector( vectorb );
        vectord2->copyVector( vectorb2 );

        auto vecc  = d_factory->getVec( vectorc );
        auto vecd  = d_factory->getVec( vectord );
        auto vecc2 = d_factory->getVec( vectorc2 );
        auto vecd2 = d_factory->getVec( vectord2 );
        checkPetscError( ut, VecAXPY( *vecc, 1.23456, *vecd ) );
        vectorc2->axpy( 1.23456, *vectord2, *vectorc2 );
        if ( !vecc || !vecd || !vecc2 || !vecd2 )
            ut->failure( "PETSC AXPY create" );

        PASS_FAIL( ( vectorc->L1Norm() - vectorc2->L1Norm() ).abs() < 0.000001,
                   "managed interface l1 norm test of axpy" );
        PASS_FAIL( ( vectorc->L2Norm() - vectorc2->L2Norm() ).abs() < 1e-15,
                   "managed interface l2 norm test of axpy" );
        PASS_FAIL( ( vectorc->maxNorm() - vectorc2->maxNorm() ).abs() < 1e-15,
                   "managed interface inf norm test of axpy" );
        PASS_FAIL( ( vectorc->L1Norm() - vectora->L1Norm() ).abs() < 0.000001,
                   "managed and native L1 norms the same" );
        PASS_FAIL( ( vectorc->L2Norm() - vectora->L2Norm() ).abs() < 0.000001,
                   "managed and native L2 norms the same" );
        PASS_FAIL( ( vectorc->maxNorm() - vectora->maxNorm() ).abs() < 0.000001,
                   "managed and native inf norms the same" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscReal" );
    }
}


void PetscVectorTests::VerifyScalePetscVector( AMP::UnitTest *ut )
{
    auto vectora  = d_factory->getVector();
    auto vectora2 = d_factory->getVector();
    auto vectorb  = d_factory->getVector();
    auto veca     = d_factory->getVec( vectora );
    auto vecb     = d_factory->getVec( vectorb );
    auto veca2    = d_factory->getVec( vectora2 );
    if ( !*veca || !*veca2 || !*vecb )
        ut->failure( "PETSc scale create" );

    if ( vectora->isType<double>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->copyVector( vectora );
        vectora2->copyVector( vectora );
        checkPetscError( ut, VecScale( *veca, 1.23456 ) );
        auto norm1 = static_cast<PetscReal>( vectora->L2Norm() );
        auto norm2 = 1.23456 * static_cast<PetscReal>( vectorb->L2Norm() );
        PASS_FAIL( fabs( norm1 - norm2 ) < 0.000001,
                   "native interface on native petsc scaling works" );
        vectora->scale( 1. / 1.23456 );
        PetscReal norma( vectora->L2Norm() );
        PetscReal normb( vectorb->L2Norm() );
        PASS_FAIL( fabs( norma - normb ) < 0.000001,
                   "AMP interface on native petsc scaling works" );
        checkPetscError( ut, VecScale( *veca2, 1.234567 ) );
        checkPetscError( ut, VecScale( *veca2, 99.99 ) );
        PetscReal norma2( vectora2->L2Norm() );
        PetscReal normb2( vectorb->L2Norm() );
        PASS_FAIL( fabs( norma2 - 99.99 * 1.234567 * normb2 ) < 0.000001,
                   "Multiple scales working in native petsc" );

        auto vectorc = d_factory->getVector();
        auto vectord = d_factory->getVector();
        vectorc->copyVector( vectora );
        vectord->copyVector( vectora );

        auto vecc = d_factory->getVec( vectorc );
        PetscReal norm3, norm4;
        checkPetscError( ut, VecScale( *vecc, 1.23456 ) );
        norm3 = static_cast<PetscReal>( vectorc->L2Norm() );
        norm4 = 1.23456 * static_cast<PetscReal>( vectord->L2Norm() );
        PASS_FAIL( fabs( norm3 - norm4 ) < 0.000001,
                   "native interface on managed petsc scaling works" );
        vectorc->scale( 1. / 1.23456 );
        PetscReal normc( vectorc->L2Norm() );
        PetscReal normd( vectord->L2Norm() );
        PASS_FAIL( fabs( normc - normd ) < 0.000001,
                   "AMP interface on managed petsc scaling works" );
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscReal" );
    }
}


void PetscVectorTests::VerifyDotPetscVector( AMP::UnitTest *ut )
{
    auto vectora = d_factory->getVector();
    auto vectorb = d_factory->getVector();
    auto veca    = d_factory->getVec( vectora );
    auto vecb    = d_factory->getVec( vectorb );

    if ( vectora->isType<PetscScalar>( 0 ) ) {
        vectora->setRandomValues();
        vectorb->setRandomValues();
        PetscReal dot1, dot2, dot12;
        checkPetscError( ut, VecDot( *veca, *vecb, &dot1 ) );
        checkPetscError( ut, VecDot( *veca, *vecb, &dot12 ) );
        dot2 = static_cast<PetscReal>( vectora->dot( *vectorb ) );
        PASS_FAIL( fabs( dot1 - dot2 ) < 1e-12,
                   "native dot equals interface dot for native vector" );
        PASS_FAIL( fabs( dot1 - dot12 ) < 1e-12, "multiple native dot" );

        auto vectorc = d_factory->getVector();
        auto vectord = d_factory->getVector();
        vectorc->copyVector( vectora );
        vectord->copyVector( vectorb );
        auto vecc = d_factory->getVec( vectorc );
        auto vecd = d_factory->getVec( vectord );
        PetscReal dot3, dot4;
        checkPetscError( ut, VecDot( *vecc, *vecd, &dot3 ) );
        dot4 = static_cast<PetscReal>( vectorc->dot( *vectord ) );
        PASS_FAIL( fabs( dot3 - dot4 ) < 1e-12,
                   "native dot equals interface dot for managed vector" );
        PASS_FAIL( fabs( dot3 - dot1 ) < 0.00000001, "native dot equals managed dot" );

        auto vectore = d_factory->getVector();
        auto vectorf = d_factory->getVector();
        vectore->copyVector( vectora );
        vectorf->copyVector( vectorb );
        auto vece = d_factory->getVec( vectore );
        auto vecf = d_factory->getVec( vectorf );
        PetscReal dot5, dot6;
        checkPetscError( ut, VecDot( *vece, *vecf, &dot5 ) );
        dot6 = static_cast<PetscReal>( vectore->dot( *vectorf ) );
        PASS_FAIL( fabs( dot5 - dot6 ) < 1e-12,
                   "native dot equals interface dot for managed alloc vector" );
        PASS_FAIL( fabs( dot3 - dot5 ) < 1e-12, "native alloc dot equals managed alloc dot" );

        /**** Need to test for failures at some point...
        try {
          checkPetscError ( ut , VecDot ( *veca , *vecc , &dot1 ) );  // This
        should fail
          ut->failure ( "incorrect failure of checkPetscError ( ut , VecDot" )
        );
        } catch ( ... ) {
          ut->passes ( "correct failure of checkPetscError ( ut , VecDot" ) );
        }
        ***/
    } else {
        ut->expected_failure( "Petsc tests currently only work for PetscReal" );
    }
}


} // namespace AMP::LinearAlgebra

