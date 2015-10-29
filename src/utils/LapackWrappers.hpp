#ifndef USE_LAPACK_WRAPPER_HPP
#define USE_LAPACK_WRAPPER_HPP

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
#undef dcopy
inline void Lapack::dcopy( int N, const double *DX, int INCX, double *DY, int INCY ) 
{
    #ifdef USE_ATLAS
        cblas_dcopy(N,DX,INCX,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Nl=N, INCXl=INCX, INCYl=INCY;
        FORTRAN_WRAPPER(::dcopy)(&Nl,(double*)DX,&INCXl,DY,&INCYl);
    #else
        FORTRAN_WRAPPER(::dcopy)(&N,(double*)DX,&INCX,DY,&INCY);
    #endif
}
// Define the member functions
#undef dswap
inline void Lapack::dswap( int N, double *DX, int INCX, double *DY, int INCY ) 
{
    #ifdef USE_ATLAS
        cblas_dswap(N,DX,INCX,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Nl=N, INCXl=INCX, INCYl=INCY;
        FORTRAN_WRAPPER(::dswap)(&Nl,DX,&INCXl,DY,&INCYl);
    #else
        FORTRAN_WRAPPER(::dswap)(&N,DX,&INCX,DY,&INCY);
    #endif
}
#undef dscal
inline void Lapack::dscal( int N, double DA, double *DX, int INCX ) 
{
    #ifdef USE_ATLAS
        cblas_dscal(N,DA,DX,INCX);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX;
        FORTRAN_WRAPPER(::dscal)(&Np,&DA,DX,&INCXp);
    #else
        FORTRAN_WRAPPER(::dscal)(&N,&DA,DX,&INCX);
    #endif
}
#undef dnrm2
inline double Lapack::dnrm2( int N, const double *DX, int INCX )
{
    #ifdef USE_ATLAS
        return cblas_dnrm2(N,DX,INCX);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX;
        return FORTRAN_WRAPPER(::dnrm2)(&Np,(double*)DX,&INCXp);
    #else
        return FORTRAN_WRAPPER(::dnrm2)(&N,(double*)DX,&INCX);
    #endif
}
#undef idamax
inline int Lapack::idamax( int N, const double *DX, int INCX )
{
    #ifdef USE_ATLAS
        return cblas_idamax(N,DX,INCX)-1;
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX;
        return FORTRAN_WRAPPER(::idamax)(&Np,(double*)DX,&INCXp)-1;
    #else
        return FORTRAN_WRAPPER(::idamax)(&N,(double*)DX,&INCX)-1;
    #endif
}
#undef daxpy
inline void Lapack::daxpy( int N, double DA, const double *DX, int INCX, double *DY, int INCY ) 
{
    #ifdef USE_ATLAS
        cblas_daxpy(N,DA,DX,INCX,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX, INCYp=INCY;
        FORTRAN_WRAPPER(::daxpy)(&Np,&DA,(double*)DX,&INCXp,DY,&INCYp);
    #else
        FORTRAN_WRAPPER(::daxpy)(&N,&DA,(double*)DX,&INCX,DY,&INCY);
    #endif
}
#undef dgemv
inline void Lapack::dgemv( char TRANS, int M, int N, double ALPHA, 
    const double *A, int LDA, const double *DX, int INCX, double BETA, double *DY, int INCY )
{
    #ifdef USE_ATLAS
        cblas_dgemv(CblasColMajor,(CBLAS_TRANSPOSE)TRANS,M,N,ALPHA,A,LDA,DX,INCX,BETA,DY,INCY);
    #elif defined(USE_ACML)
        //FORTRAN_WRAPPER(::dgemv)(&TRANS,&M,&N,&ALPHA,(double*)A,&LDA,(double*)DX,&INCX,&BETA,DY,&INCY,1);
        ::dgemv(TRANS,M,N,ALPHA,(double*)A,LDA,(double*)DX,INCX,BETA,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Mp=M, Np=N, LDAp=LDA, INCXp=INCX, INCYp=INCY;
        FORTRAN_WRAPPER(::dgemv)(&TRANS,&Mp,&Np,&ALPHA,(double*)A,&LDAp,(double*)DX,&INCXp,&BETA,DY,&INCYp);
    #else
        FORTRAN_WRAPPER(::dgemv)(&TRANS,&M,&N,&ALPHA,(double*)A,&LDA,(double*)DX,&INCX,&BETA,DY,&INCY);
    #endif
}
#undef dgemm
inline void Lapack::dgemm( char TRANSA, char TRANSB, int M, int N, int K, double ALPHA, 
    const double *A, int LDA, const double *B, int LDB, double BETA, double *C, int LDC )
{
    #ifdef USE_ATLAS
        cblas_dgemm(CblasColMajor,(CBLAS_TRANSPOSE)TRANSA,(CBLAS_TRANSPOSE)TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);
    #elif defined(USE_ACML)
        FORTRAN_WRAPPER(::dgemm)(&TRANSA,&TRANSB,&M,&N,&K,&ALPHA,(double*)A,&LDA,(double*)B,&LDB,&BETA,C,&LDC,1,1);
        //::dgemm(TRANSA,TRANSA,M,N,K,ALPHA,(double*)A,LDA,(double*)B,LDB,BETA,C,LDC);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Mp=M, Np=N, Kp=K, LDAp=LDA, LDBp=LDB, LDCp=LDC;
        FORTRAN_WRAPPER(::dgemm)(&TRANSA,&TRANSB,&Mp,&Np,&Kp,&ALPHA,(double*)A,&LDAp,(double*)B,&LDBp,&BETA,C,&LDCp);
    #else
        FORTRAN_WRAPPER(::dgemm)(&TRANSA,&TRANSB,&M,&N,&K,&ALPHA,(double*)A,&LDA,(double*)B,&LDB,&BETA,C,&LDC);
    #endif
}
#undef dasum
inline double Lapack::dasum( int N, const double *DX, int INCX )
{
    #ifdef USE_ATLAS
        return cblas_dasum(N,DX,INCX);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX;
        return FORTRAN_WRAPPER(::dasum)(&Np,(double*)DX,&INCXp);
    #else
        return FORTRAN_WRAPPER(::dasum)(&N,(double*)DX,&INCX);
    #endif
}
#undef ddot
inline double Lapack::ddot( int N, const double *DX, int INCX, const double *DY, int INCY )
{
    #ifdef USE_ATLAS
        return cblas_ddot(N,DX,INCX,DY,INCY);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX, INCYp=INCY;
        return FORTRAN_WRAPPER(::ddot)(&Np,(double*)DX,&INCXp,(double*)DY,&INCYp);
    #else
        return FORTRAN_WRAPPER(::ddot)(&N,(double*)DX,&INCX,(double*)DY,&INCY);
    #endif
}
#undef dger
inline void Lapack::dger( int N, int M, double alpha, const double *x, int INCX, const double *y, int INCY, double *A, int LDA )
{
    #ifdef USE_ATLAS
        cblas_dger(N,M,alpha,x,INCX,y,INCY,A,LDA);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INCXp=INCX, INCYp=INCY;
        FORTRAN_WRAPPER(::dger)(&N,&M,&alpha,(double*)x,&INCX,(double*)y,&INCY,A,&LDA);
    #else
        FORTRAN_WRAPPER(::dger)(&N,&M,&alpha,(double*)x,&INCX,(double*)y,&INCY,A,&LDA);
    #endif
}
#undef dgesv
inline void Lapack::dgesv( int N, int NRHS, double *A, int LDA, int *IPIV, double *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        INFO = clapack_dgesv(CblasColMajor,N,NRHS,A,LDA,IPIV,B,LDB);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, NRHSp=NRHS, LDAp=LDA, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::dgesv)(&Np,&NRHSp,A,&LDAp,IPIVp,B,&LDBp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::dgesv)(&N,&NRHS,(double*)A,&LDA,IPIV,B,&LDB,&INFO);
    #endif
}
#undef dgtsv
inline void Lapack::dgtsv( int N, int NRHS, double *DL, double *D, double *DU, double *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support dgtsv");
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t N2=N, NRHS2=NRHS, LDB2=LDB, INFOp;
        FORTRAN_WRAPPER(::dgtsv)(&N2,&NRHS2,DL,D,DU,B,&LDB2,&INFOp);
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::dgtsv)(&N,&NRHS,DL,D,DU,B,&LDB,&INFO);
    #endif
}
#undef dgbsv
inline void Lapack::dgbsv( int N, int KL, int KU, int NRHS, double *AB, int LDAB, int *IPIV, double *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support dgbsv");
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, KLp=KL, KUp=KU, NRHSp=NRHS, LDABp=LDAB, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::dgbsv)(&Np,&KLp,&KUp,&NRHSp,AB,&LDABp,IPIVp,B,&LDBp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::dgbsv)(&N,&KL,&KU,&NRHS,AB,&LDAB,IPIV,B,&LDB,&INFO);
    #endif
}
#undef dgetrf
inline void Lapack::dgetrf( int M, int N, double *A, int LDA, int *IPIV, int &INFO )
{
    #ifdef USE_ATLAS
        INFO = clapack_dgetrf(CblasColMajor,M,N,A,LDA,IPIV);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, Mp=M, LDAp=LDA, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::dgetrf)(&Mp,&Np,A,&LDAp,IPIVp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::dgetrf)(&M,&N,A,&LDA,IPIV,&INFO);
    #endif
}
#undef dgttrf
inline void Lapack::dgttrf( int N, double *DL, double *D, double *DU, double *DU2, int *IPIV, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support dgttrf");
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::dgttrf)(&Np,DL,D,DU,DU2,IPIVp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::dgttrf)(&N,DL,D,DU,DU2,IPIV,&INFO);
    #endif
}
#undef dgbtrf
inline void Lapack::dgbtrf( int M, int N, int KL, int KU, double *AB, int LDAB, int *IPIV, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support dgbtrf");
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Mp=M, Np=N, KLp=KL, KUp=KU, LDABp=LDAB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        FORTRAN_WRAPPER(::dgbtrf)(&Mp,&Np,&KLp,&KUp,AB,&LDABp,IPIVp,&INFOp);
        for (int i=0; i<N; i++) { IPIV[i] = static_cast<int>(IPIVp[i]); }
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #elif defined(USE_ACML)
        get_lock();
        FORTRAN_WRAPPER(::dgbtrf)(&M,&N,&KL,&KU,AB,&LDAB,IPIV,&INFO);
        release_lock();
    #else
        FORTRAN_WRAPPER(::dgbtrf)(&M,&N,&KL,&KU,AB,&LDAB,IPIV,&INFO);
    #endif
}
#undef dgetrs
inline void Lapack::dgetrs( char TRANS, int N, int NRHS, const double *A, int LDA, const int *IPIV, double *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        INFO = clapack_dgetrs(CblasColMajor,(CBLAS_TRANSPOSE)TRANS,N,NRHS,A,LDA,IPIV,B,LDB);
    #elif defined(USE_ACML)
        ::dgetrs(TRANS,N,NRHS,(double*)A,LDA,(int*)IPIV,B,LDB,&INFO);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, NRHSp=NRHS, LDAp=LDA, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        for (int i=0; i<N; i++) { IPIVp[i] = IPIV[i]; }
        FORTRAN_WRAPPER(::dgetrs)(&TRANS,&Np,&NRHSp,(double*)A,&LDAp,IPIVp,B,&LDBp,&INFOp);
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::dgetrs)(&TRANS,&N,&NRHS,(double*)A,&LDA,(int*)IPIV,B,&LDB,&INFO);
    #endif
}
#undef dgttrs
inline void Lapack::dgttrs( char TRANS, int N, int NRHS, const double *DL, const double *D,
    const double *DU, const double *DU2, const int *IPIV, double *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support dgttrs");
    #elif defined(USE_ACML)
        ::dgttrs(TRANS,N,NRHS,(double*)DL,(double*)D,(double*)DU,(double*)DU2,(int*)IPIV,B,LDB,&INFO);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, NRHSp=NRHS, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        for (int i=0; i<N; i++) { IPIVp[i] = IPIV[i]; }
        FORTRAN_WRAPPER(::dgttrs)(&TRANS,&Np,&NRHSp,(double*)DL,(double*)D,(double*)DU,(double*)DU2,IPIVp,B,&LDBp,&INFOp);
        delete [] IPIVp;
        INFO = static_cast<int>(INFOp);
    #else
        FORTRAN_WRAPPER(::dgttrs)(&TRANS,&N,&NRHS,(double*)DL,(double*)D,(double*)DU,(double*)DU2,(int*)IPIV,B,&LDB,&INFO);
    #endif
}
#undef dgbtrs
inline void Lapack::dgbtrs( char TRANS, int N, int KL, int KU, int NRHS, const double *AB, 
    int LDAB, const int *IPIV, double *B, int LDB, int &INFO )
{
    #ifdef USE_ATLAS
        throw std::logic_error("ATLAS does not support dgbtrs");
    #elif defined(USE_ACML)
        ::dgbtrs(TRANS,N,KL,KU,NRHS,(double*)AB,LDAB,(int*)IPIV,B,LDB,&INFO);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, KLp=KL, KUp=KU, NRHSp=NRHS, LDABp=LDAB, LDBp=LDB, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        for (int i=0; i<N; i++) { IPIVp[i] = IPIV[i]; }
        FORTRAN_WRAPPER(::dgbtrs)(&TRANS,&Np,&KLp,&KUp,&NRHSp,(double*)AB,&LDABp,IPIVp,B,&LDBp,&INFOp);
        INFO = static_cast<int>(INFOp);
        delete [] IPIVp;
    #else
        FORTRAN_WRAPPER(::dgbtrs)(&TRANS,&N,&KL,&KU,&NRHS,(double*)AB,&LDAB,(int*)IPIV,B,&LDB,&INFO);
    #endif
}
#undef dgetri
inline void Lapack::dgetri( int N, double *A, int LDA, const int *IPIV, double *WORK, int LWORK, int &INFO )
{
    #ifdef USE_ATLAS
        INFO = clapack_dgetri(CblasColMajor,N,A,LDA,IPIV);
    #elif defined(USE_ACML)
        ::dgetri_(&N,A,&LDA,(int*)IPIV,WORK,&LWORK,&INFO);
    #elif defined(USE_MATLAB_LAPACK)
        ptrdiff_t Np=N, LDAp=LDA, LWORKp=LWORK, INFOp;
        ptrdiff_t *IPIVp = new ptrdiff_t[N];
        for (int i=0; i<N; i++) { IPIVp[i] = IPIV[i]; }
        FORTRAN_WRAPPER(::dgetri)(&Np,A,&LDAp,IPIVp,WORK,&LWORKp,&INFOp);
        INFO = static_cast<int>(INFOp);
        delete [] IPIVp;
    #else
        FORTRAN_WRAPPER(::dgetri)(&N,A,&LDA,(int*)IPIV,WORK,&LWORK,&INFO);
    #endif
}
#undef dtrsm
inline void Lapack::dtrsm( char SIDE, char UPLO, char TRANS, char DIAG,
        int M, int N, double ALPHA, const double *A, int LDA, double *B, int LDB )
{
    #ifdef USE_ATLAS
        throw std::logic_error("dtrsm not implimented for ATLAS");
    #elif defined(USE_ACML)
        char SIDE2[2]={SIDE,0}, UPLO2[2]={UPLO,0}, TRANS2[2]={TRANS,0}, DIAG2[2]={DIAG,0};
        ::dtrsm_(SIDE2,UPLO2,TRANS2,DIAG2,&M,&N,&ALPHA,(double*)A,&LDA,B,&LDB,1,1,1,1);
    #elif defined(USE_MATLAB_LAPACK)
        FORTRAN_WRAPPER(::dtrsm)(&SIDE,&UPLO,&TRANS,&DIAG,&M,&N,&ALPHA,(double*)A,&LDA,B,&LDB);
    #else
        FORTRAN_WRAPPER(::dtrsm)(&SIDE,&UPLO,&TRANS,&DIAG,&M,&N,&ALPHA,(double*)A,&LDA,B,&LDB);
    #endif
}
#undef dgetri
inline double Lapack::dlamch( char cmach )
{
    #ifdef USE_ATLAS
        return clapack_dlamch(cmach);
    #elif defined(USE_ACML)
        return ::dlamch(cmach);
    #else
        return FORTRAN_WRAPPER(::dlamch)(&cmach);
    #endif
}


} // namespace

#endif

