#ifndef included_AMP_EpetraVectorData
#define included_AMP_EpetraVectorData

#include "AMP/vectors/data/GhostDataHelper.hpp"
#include "AMP/vectors/data/VectorData.h"

DISABLE_WARNINGS
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {

/** \class EpetraVectorEngineParameters
 * \brief Class that details how to construct an EpetraVectorEngine
 */
class EpetraVectorEngineParameters
{
public:
    /** \brief Constructor
     * \param[in] N_local   Number of local entries
     * \param[in] comm      Communicator to use
     * \param[in] commList  communication list
     */
    EpetraVectorEngineParameters( size_t N_local,
                                  const AMP_MPI &comm,
                                  std::shared_ptr<CommunicationList> commList );

    //! Destructor
    virtual ~EpetraVectorEngineParameters();

    /** \brief  Return the Epetra_Map for this engine
     * \return  The Epetra_Map
     */
    Epetra_Map &getEpetraMap();

    //! Return the local size
    inline size_t getLocalSize() const { return d_end - d_begin; }

    //! Return the local size
    inline size_t getGlobalSize() const { return d_global; }

    //! Return the first DOF on this core
    inline size_t beginDOF() const { return d_begin; }

    /** \brief  Return the Epetra_MpiComm for this engine
     * \return The Epetra_MpiComm
     */
    inline AMP_MPI getComm() const { return d_comm; }

    //! The CommunicationList for a vector
    std::shared_ptr<CommunicationList> d_CommList = nullptr;

private:
    size_t d_begin  = 0;                          // Starting DOF
    size_t d_end    = 0;                          // Ending DOF
    size_t d_global = 0;                          // Number of global DOFs
    AMP_MPI d_comm;                               // Comm
    std::shared_ptr<Epetra_Map> d_emap = nullptr; // Epetra map
};

/** \class EpetraVectorData
 * \brief A linear algebra engine that uses Epetra
 * \details  Use the Epetra implementation of the L1 BLAS routines.  Unlike other
 * libraries, it is very difficult to separate the data from the engine.  For this
 * reason, the EpetraVectorEngine contains the Epetra_Vector to operate on.
 */
class EpetraVectorData : public GhostDataHelper<double>
{

public: // Virtual functions
    //! Virtual destructor
    virtual ~EpetraVectorData() {}
    static std::shared_ptr<EpetraVectorData>
    create( std::shared_ptr<EpetraVectorEngineParameters> alias, std::shared_ptr<VectorData> buf );

    std::string VectorDataName() const override { return "EpetraVectorData"; }
    size_t numberOfDataBlocks() const override { return 1; }
    size_t sizeOfDataBlock( size_t i ) const override { return i == 0 ? d_localSize : 0; }
    void setValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void addValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getValuesByLocalID( size_t, const size_t *, void *, const typeID & ) const override;
    void putRawData( const void *in, const typeID &id ) override;
    void getRawData( void *out, const typeID &id ) const override;
    uint64_t getDataID() const override
    {
        return reinterpret_cast<uint64_t>( getRawDataBlockAsVoid( 0 ) );
    }
    void *getRawDataBlockAsVoid( size_t i ) override;
    const void *getRawDataBlockAsVoid( size_t i ) const override;
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( double ); }
    typeID getType( size_t ) const override { return getTypeID<double>(); }
    void swapData( VectorData & ) override;
    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;

    /** \brief  Get the raw Epetra_Vector
     * \return  The Epetra_Vector currently used by this engine
     */
    inline Epetra_Vector &getEpetra_Vector() { return d_epetraVector; }

    /** \brief  Get the raw Epetra_Vector
     * \return  The Epetra_Vector currently used by this engine
     */
    inline const Epetra_Vector &getEpetra_Vector() const { return d_epetraVector; }

    EpetraVectorData( std::shared_ptr<EpetraVectorEngineParameters> alias,
                      Epetra_DataAccess,
                      const Epetra_BlockMap &,
                      std::shared_ptr<VectorData>,
                      int,
                      int,
                      int );

protected:
    //! The Epetra_Vector to perform work on
    Epetra_Vector d_epetraVector;

    //! A shared ptr to the buffer (to prevent it from leaving scope)
    std::shared_ptr<VectorData> d_buf_scope;
};


} // namespace AMP::LinearAlgebra


#endif
