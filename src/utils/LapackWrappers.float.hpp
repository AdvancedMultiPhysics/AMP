#ifndef USE_LAPACK_FLOAT_WRAPPER_HPP
#define USE_LAPACK_FLOAT_WRAPPER_HPP

#include "blas_lapack.h"
#include "utils/LapackWrappers.h"

#include <stdexcept>


// Define macro to handle name mangling
#ifndef FORTRAN_WRAPPER
    #if defined(USE_ACML)
        #define FORTRAN_WRAPPER(x) x ## _
    #elif defined(_WIN32) || defined(__hpux) || defined(USE_MKL)
        #define FORTRAN_WRAPPER(x) x
    #else
        #define FORTRAN_WRAPPER(x) x ## _
    #endif
#endif


namespace AMP {


// Define the member functions
#undef scopy
template <>
inline void Lapack<float>::copy( int N, const float *DX, int INCX, float *DY, int INCY ) 
{
    #ifdef USE_ATLAS
        cblas_scopy(N,DX,INCX,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Nl=N, INCXl=INCX, INCYl=INCY;
        FORTRAN_WRAPPER(::scopy)(&Nl,(float*)DX,&INCXl,DY,&INCYl);
    #else
        FORTRAN_WRAPPER(::scopy)(&N,(float*)DX,&INCX,DY,&INCY);
    #endif
}
// Define the member functions
#undef sswap
template <>
inline void Lapack<float>::swap( int N, float *DX, int INCX, float *DY, int INCY ) 
{
    #ifdef USE_ATLAS
        cblas_sswap(N,DX,INCX,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Nl=N, INCXl=INCX, INCYl=INCY;
        FORTRAN_WRAPPER(::sswap)(&Nl,DX,&INCXl,DY,&INCYl);
    #else
        FORTRAN_WRAPPER(::sswap)(&N,DX,&INCX,DY,&INCY);
    #endif
}
#undef sscal
template <>
inline void Lapack<float>::scal( int N, float DA, float *DX, int INCX ) 
{
    #ifdef USE_ATLAS
        cblas_sscal(N,DA,DX,INCX);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX;
        FORTRAN_WRAPPER(::sscal)(&Np,&DA,DX,&INCXp);
    #else
        FORTRAN_WRAPPER(::sscal)(&N,&DA,DX,&INCX);
    #endif
}
#undef snrm2
template <>
inline float Lapack<float>::nrm2( int N, const float *DX, int INCX )
{
    #ifdef USE_ATLAS
        return cblas_snrm2(N,DX,INCX);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX;
        return FORTRAN_WRAPPER(::snrm2)(&Np,(float*)DX,&INCXp);
    #else
        return FORTRAN_WRAPPER(::snrm2)(&N,(float*)DX,&INCX);
    #endif
}
#undef isamax
template <>
inline int Lapack<float>::iamax( int N, const float *DX, int INCX )
{
    #ifdef USE_ATLAS
        return cblas_isamax(N,DX,INCX)-1;
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX;
        return FORTRAN_WRAPPER(::isamax)(&Np,(float*)DX,&INCXp)-1;
    #else
        return FORTRAN_WRAPPER(::isamax)(&N,(float*)DX,&INCX)-1;
    #endif
}
#undef saxpy
template <>
inline void Lapack<float>::axpy( int N, float DA, const float *DX, int INCX, float *DY, int INCY ) 
{
    #ifdef USE_ATLAS
        cblas_saxpy(N,DA,DX,INCX,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX, INCYp=INCY;
        FORTRAN_WRAPPER(::saxpy)(&Np,&DA,(float*)DX,&INCXp,DY,&INCYp);
    #else
        FORTRAN_WRAPPER(::saxpy)(&N,&DA,(float*)DX,&INCX,DY,&INCY);
    #endif
}
#undef sgemv
template <>
inline void Lapack<float>::gemv( char TRANS, int M, int N, float ALPHA, 
    const float *A, int LDA, const float *DX, int INCX, float BETA, float *DY, int INCY )
{
    #ifdef USE_ATLAS
        cblas_sgemv(CblasColMajor,(CBLAS_TRANSPOSE)TRANS,M,N,ALPHA,A,LDA,DX,INCX,BETA,DY,INCY);
    #elif defined(USE_ACML)
        //FORTRAN_WRAPPER(::sgemv)(&TRANS,&M,&N,&ALPHA,(float*)A,&LDA,(float*)DX,&INCX,&BETA,DY,&INCY,1);
        ::sgemv(TRANS,M,N,ALPHA,(float*)A,LDA,(float*)DX,INCX,BETA,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Mp=M, Np=N, LDAp=LDA, INCXp=INCX, INCYp=INCY;
        FORTRAN_WRAPPER(::sgemv)(&TRANS,&Mp,&Np,&ALPHA,(float*)A,&LDAp,(float*)DX,&INCXp,&BETA,DY,&INCYp);
    #else
        FORTRAN_WRAPPER(::sgemv)(&TRANS,&M,&N,&ALPHA,(float*)A,&LDA,(float*)DX,&INCX,&BETA,DY,&INCY);
    #endif
}
#undef sgemm
template <>
inline void Lapack<float>::gemm( char TRANSA, char TRANSB, int M, int N, int K, float ALPHA, 
    const float *A, int LDA, const float *B, int LDB, float BETA, float *C, int LDC )
{
    #ifdef USE_ATLAS
        cblas_sgemm(CblasColMajor,(CBLAS_TRANSPOSE)TRANSA,(CBLAS_TRANSPOSE)TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);
    #elif defined(USE_ACML)
        FORTRAN_WRAPPER(::sgemm)(&TRANSA,&TRANSB,&M,&N,&K,&ALPHA,(float*)A,&LDA,(float*)B,&LDB,&BETA,C,&LDC,1,1);
        //::sgemm(TRANSA,TRANSA,M,N,K,ALPHA,(float*)A,LDA,(float*)B,LDB,BETA,C,LDC);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Mp=M, Np=N, Kp=K, LDAp=LDA, LDBp=LDB, LDCp=LDC;
        FORTRAN_WRAPPER(::sgemm)(&TRANSA,&TRANSB,&Mp,&Np,&Kp,&ALPHA,(float*)A,&LDAp,(float*)B,&LDBp,&BETA,C,&LDCp);
    #else
        FORTRAN_WRAPPER(::sgemm)(&TRANSA,&TRANSB,&M,&N,&K,&ALPHA,(float*)A,&LDA,(float*)B,&LDB,&BETA,C,&LDC);
    #endif
}
#undef sasum
template <>
inline float Lapack<float>::asum( int N, const float *DX, int INCX )
{
    #ifdef USE_ATLAS
        return cblas_sasum(N,DX,INCX);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX;
        return FORTRAN_WRAPPER(::sasum)(&Np,(float*)DX,&INCXp);
    #else
        return FORTRAN_WRAPPER(::sasum)(&N,(float*)DX,&INCX);
    #endif
}
#undef sdot
template <>
inline float Lapack<float>::dot( int N, const float *DX, int INCX, const float *DY, int INCY )
{
    #ifdef USE_ATLAS
        return cblas_sdot(N,DX,INCX,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX, INCYp=INCY;
        return FORTRAN_WRAPPER(::sdot)(&Np,(float*)DX,&INCXp,(float*)DY,&INCYp);
    #else
        return FORTRAN_WRAPPER(::sdot)(&N,(float*)DX,&INCX,(float*)DY,&INCY);
    #endif
}
#undef sger
template <>
inline void Lapack<float>::ger( int N, int M, float alpha, const float *x, int INCX, const float *y, int INCY, float *A, int LDA )
{
    #ifdef USE_ATLAS
        cblas_sger(N,M,alpha,x,INCX,y,INCY,A,LDA);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, Mp=M, INCXp=INCX, INCYp=INCY, LDAp=LDA;
        FORTRAN_WRAPPER(::sger)(&Np,&Mp,&alpha,(float*)x,&INCXp,(float*)y,&INCYp,A,&LDAp);
    #else
        FORTRAN_WRAPPER(::sger)(&N,&M,&alpha,(float*)x,&INCX,(float*)y,&INCY,A,&LDA);
    #endif
}
#undef sgesv
template <>
inline void Lapack<float>::gesv( int N, int NRHS, float *A, int LDA, int *IPIV, float *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        INFO = clapack_sgesv(CblasColMajor,N,NRHS,A,LDA,IPIV,B,LDB);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, NRHSp=NRHS, LDAp=LDA, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::sgesv)(&Np,&NRHSp,A,&LDAp,IPIVp,B,&LDBp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::sgesv)(&N,&NRHS,(float*)A,&LDA,IPIV,B,&LDB,&INFO);
    #endif
}
#undef sgtsv
template <>
inline void Lapack<float>::gtsv( int N, int NRHS, float *DL, float *D, float *DU, float *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support sgtsv");
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t N2=N, NRHS2=NRHS, LDB2=LDB, INFOp;
        FORTRAN_WRAPPER(::sgtsv)(&N2,&NRHS2,DL,D,DU,B,&LDB2,&INFOp);
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::sgtsv)(&N,&NRHS,DL,D,DU,B,&LDB,&INFO);
    #endif
}
#undef sgbsv
template <>
inline void Lapack<float>::gbsv( int N, int KL, int KU, int NRHS, float *AB, int LDAB, int *IPIV, float *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support sgbsv");
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, KLp=KL, KUp=KU, NRHSp=NRHS, LDABp=LDAB, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::sgbsv)(&Np,&KLp,&KUp,&NRHSp,AB,&LDABp,IPIVp,B,&LDBp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::sgbsv)(&N,&KL,&KU,&NRHS,AB,&LDAB,IPIV,B,&LDB,&INFO);
    #endif
}
#undef sgetrf
template <>
inline void Lapack<float>::getrf( int M, int N, float *A, int LDA, int *IPIV, int &INFO )
{
    #ifdef USE_ATLAS
        INFO = clapack_sgetrf(CblasColMajor,M,N,A,LDA,IPIV);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, Mp=M, LDAp=LDA, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::sgetrf)(&Mp,&Np,A,&LDAp,IPIVp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::sgetrf)(&M,&N,A,&LDA,IPIV,&INFO);
    #endif
}
#undef sgttrf
template <>
inline void Lapack<float>::gttrf( int N, float *DL, float *D, float *DU, float *DU2, int *IPIV, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support sgttrf");
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::sgttrf)(&Np,DL,D,DU,DU2,IPIVp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::sgttrf)(&N,DL,D,DU,DU2,IPIV,&INFO);
    #endif
}
#undef sgbtrf
template <>
inline void Lapack<float>::gbtrf( int M, int N, int KL, int KU, float *AB, int LDAB, int *IPIV, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support sgbtrf");
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Mp=M, Np=N, KLp=KL, KUp=KU, LDABp=LDAB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::sgbtrf)(&Mp,&Np,&KLp,&KUp,AB,&LDABp,IPIVp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #elif defined(USE_ACML)
        get_lock();
        FORTRAN_WRAPPER(::sgbtrf)(&M,&N,&KL,&KU,AB,&LDAB,IPIV,&INFO);
        release_lock();
    #else
        FORTRAN_WRAPPER(::sgbtrf)(&M,&N,&KL,&KU,AB,&LDAB,IPIV,&INFO);
    #endif
}
#undef sgetrs
template <>
inline void Lapack<float>::getrs( char TRANS, int N, int NRHS, const float *A, int LDA, const int *IPIV, float *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        INFO = clapack_sgetrs(CblasColMajor,(CBLAS_TRANSPOSE)TRANS,N,NRHS,A,LDA,IPIV,B,LDB);
    #elif defined(USE_ACML)
        ::sgetrs(TRANS,N,NRHS,(float*)A,LDA,(int*)IPIV,B,LDB,&INFO);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, NRHSp=NRHS, LDAp=LDA, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        for (int i=0; i<N; i++) { IPIVp[i] = IPIV[i]; }
        FORTRAN_WRAPPER(::sgetrs)(&TRANS,&Np,&NRHSp,(float*)A,&LDAp,IPIVp,B,&LDBp,&INFOp);
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::sgetrs)(&TRANS,&N,&NRHS,(float*)A,&LDA,(int*)IPIV,B,&LDB,&INFO);
    #endif
}
#undef sgttrs
template <>
inline void Lapack<float>::gttrs( char TRANS, int N, int NRHS, const float *DL, const float *D,
    const float *DU, const float *DU2, const int *IPIV, float *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support sgttrs");
    #elif defined(USE_ACML)
        ::sgttrs(TRANS,N,NRHS,(float*)DL,(float*)D,(float*)DU,(float*)DU2,(int*)IPIV,B,LDB,&INFO);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, NRHSp=NRHS, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        for (int i=0; i<N; i++) { IPIVp[i] = IPIV[i]; }
        FORTRAN_WRAPPER(::sgttrs)(&TRANS,&Np,&NRHSp,(float*)DL,(float*)D,(float*)DU,(float*)DU2,IPIVp,B,&LDBp,&INFOp);
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::sgttrs)(&TRANS,&N,&NRHS,(float*)DL,(float*)D,(float*)DU,(float*)DU2,(int*)IPIV,B,&LDB,&INFO);
    #endif
}
#undef sgbtrs
template <>
inline void Lapack<float>::gbtrs( char TRANS, int N, int KL, int KU, int NRHS, const float *AB, 
    int LDAB, const int *IPIV, float *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support sgbtrs");
    #elif defined(USE_ACML)
        ::sgbtrs(TRANS,N,KL,KU,NRHS,(float*)AB,LDAB,(int*)IPIV,B,LDB,&INFO);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, KLp=KL, KUp=KU, NRHSp=NRHS, LDABp=LDAB, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        for (int i=0; i<N; i++) { IPIVp[i] = IPIV[i]; }
        FORTRAN_WRAPPER(::sgbtrs)(&TRANS,&Np,&KLp,&KUp,&NRHSp,(float*)AB,&LDABp,IPIVp,B,&LDBp,&INFOp);
        INFO = static_cast<int>(INFOp);
        delete [] IPIVp;
    #else
        FORTRAN_WRAPPER(::sgbtrs)(&TRANS,&N,&KL,&KU,&NRHS,(float*)AB,&LDAB,(int*)IPIV,B,&LDB,&INFO);
    #endif
}
#undef sgetri
template <>
inline void Lapack<float>::getri( int N, float *A, int LDA, const int *IPIV, float *WORK, int LWORK, int &INFO )
{
    #ifdef USE_ATLAS
        INFO = clapack_dgetri(CblasColMajor,N,A,LDA,IPIV);
    #elif defined(USE_ACML)
        ::sgetri_(&N,A,&LDA,(int*)IPIV,WORK,&LWORK,&INFO);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, LDAp=LDA, LWORKp=LWORK, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        for (int i=0; i<N; i++) { IPIVp[i] = IPIV[i]; }
        FORTRAN_WRAPPER(::sgetri)(&Np,A,&LDAp,IPIVp,WORK,&LWORKp,&INFOp);
        INFO = static_cast<int>(INFOp);
        delete [] IPIVp;
    #else
        FORTRAN_WRAPPER(::sgetri)(&N,A,&LDA,(int*)IPIV,WORK,&LWORK,&INFO);
    #endif
}
#undef strsm
template <>
inline void Lapack<float>::trsm( char SIDE, char UPLO, char TRANS, char DIAG,
        int M, int N, float ALPHA, const float *A, int LDA, float *B, int LDB )
{
    #ifdef USE_ATLAS
        throw std::logic_error("strsm not implemented for ATLAS");
    #elif defined(USE_ACML)
        char SIDE2[2]={SIDE,0}, UPLO2[2]={UPLO,0}, TRANS2[2]={TRANS,0}, DIAG2[2]={DIAG,0};
        ::strsm_(SIDE2,UPLO2,TRANS2,DIAG2,&M,&N,&ALPHA,(float*)A,&LDA,B,&LDB,1,1,1,1);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Mp=M, Np=N, LDAp=LDA, LDBp=LDB;
        FORTRAN_WRAPPER(::strsm)(&SIDE,&UPLO,&TRANS,&DIAG,&Mp,&Np,&ALPHA,(float*)A,&LDAp,B,&LDBp);
    #else
        FORTRAN_WRAPPER(::strsm)(&SIDE,&UPLO,&TRANS,&DIAG,&M,&N,&ALPHA,(float*)A,&LDA,B,&LDB);
    #endif
}
#undef slamch
template <>
inline float Lapack<float>::lamch( char cmach )
{
    #ifdef USE_ATLAS
        return clapack_slamch(cmach);
    #elif defined(USE_ACML)
        return ::slamch(cmach);
    #else
        return FORTRAN_WRAPPER(::slamch)(&cmach);
    #endif
}


} // namespace

#endif
