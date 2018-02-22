namespace AMP {
namespace LinearAlgebra {


inline PetscVector::PetscVector() : d_PetscRandom( 0 ), d_petscVec( nullptr ) {}


inline PetscRandom &PetscVector::getPetscRandom( const AMP_MPI &comm )
{
    if ( d_PetscRandom == 0 ) {
        d_PetscRandom = new PetscRandom;
        PetscRandomCreate( comm.getCommunicator(), d_PetscRandom );
        PetscRandomSetType( *d_PetscRandom, PETSCRAND48 ); // This is a horrible RNG for
                                                           // stochastic simulation.  Do not
                                                           // use.
    }
    return *d_PetscRandom;
}


inline Vec &PetscVector::getVec() { return d_petscVec; }


inline const Vec &PetscVector::getVec() const { return d_petscVec; }


inline PetscVector::~PetscVector()
{
    if ( d_PetscRandom ) {
        PETSC::randomDestroy( d_PetscRandom );
        delete d_PetscRandom;
    }
}


} // namespace LinearAlgebra
} // namespace AMP
