#ifndef included_AMP_NativePetscVectorData
#define included_AMP_NativePetscVectorData

#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/data/GhostDataHelper.hpp"
#include "AMP/vectors/data/VectorData.h"

#include "petscvec.h"


namespace AMP::LinearAlgebra {


/*! \struct Vec
    \brief PETSc vector
*/


/** \class NativePetscVector
 * \brief An AMP Vector that uses PETSc for parallel data management, linear algebra,
 * etc.
 * \details  This is an AMP wrapper to PETSc.   Rather, it wraps these calls.
 * This class is used when PETSc is chosen as the default linear algebra engine.
 */
class NativePetscVectorData : public GhostDataHelper<PetscScalar>
{
public:
    /** \brief Construct a wrapper for a PETSc Vec from a set of parameters
     * \param[in] v             The Vec to wrap
     * \param[in] deleteable    If true, ~NativePetscVector() will call VecDestroy()
     * \param[in] comm          The communicator associated with the Vec
     */
    explicit NativePetscVectorData( Vec v, bool deleteable, AMP_MPI comm = AMP_MPI() );

    //! Destructor
    virtual ~NativePetscVectorData();

    // Overrides to makeConsistent ensure that the internal Petsc Vector
    // is in a valid state by calling assemble/resetArray as needed
    void makeConsistent( ScatterType t ) override;
    void makeConsistent() override;

    std::string VectorDataName() const override { return "NativePetscVector"; }
    size_t numberOfDataBlocks() const override;
    size_t sizeOfDataBlock( size_t i ) const override;
    void putRawData( const void *, const typeID & ) override;
    void getRawData( void *, const typeID & ) const override;

    void setValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void addValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getValuesByLocalID( size_t, const size_t *, void *, const typeID & ) const override;

    // Return the id of the data
    uint64_t getDataID() const override
    {
        return reinterpret_cast<uint64_t>( getRawDataBlockAsVoid( 0 ) );
    }
    void *getRawDataBlockAsVoid( size_t i ) override;
    const void *getRawDataBlockAsVoid( size_t i ) const override;
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( PetscScalar ); }
    typeID getType( size_t ) const override { return getTypeID<PetscScalar>(); }
    void swapData( VectorData &rhs ) override;

    /** \brief Clone the data
     */
    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;

    void assemble() override;

    //! Get the PETSc vector
    Vec &getVec() { return d_petscVec; }

    //! Get the PETSc vector
    const Vec &getVec() const { return d_petscVec; }

protected:
    void resetArray();
    void resetArray() const;

private:
    friend class NativePetscVectorOperations;
    bool d_bDeleteMe                        = false;
    Vec d_petscVec                          = nullptr;
    mutable PetscScalar *d_pArray           = nullptr; // mutable so that we can cache the value
    mutable const PetscScalar *d_pArrayRead = nullptr;
};


} // namespace AMP::LinearAlgebra

#endif
