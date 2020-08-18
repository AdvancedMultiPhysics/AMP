#ifndef included_AMP_NullVector
#define included_AMP_NullVector

#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/VectorDataNull.h"
#include <string>


namespace AMP {
namespace LinearAlgebra {


/** \brief An empty vector
 * \details Some operators do not require vectors for application.  In these
 * circumstances, a NullVector is used.  This stores no data and performs no
 * work.
 */
template<class TYPE = double>
class NullVector : public Vector
{
public: // Public constructors
    /**
     *  \brief Create a NullVector
     *  \param[in]  name  Name of variable to associate with this NullVector
     *  \return Vector shared pointer to a NullVector
     */
    static inline Vector::shared_ptr create( const std::string &name )
    {
        return create( std::make_shared<Variable>( name ) );
    }

    /**
     *  \brief Create a NullVector
     *  \param[in]  name  Variable to associate with this NullVector
     *  \return Vector shared pointer to a NullVector
     */
    static inline Vector::shared_ptr create( const Variable::shared_ptr name )
    {
        return std::shared_ptr<NullVector>( new NullVector<TYPE>( name ) );
    }

    virtual ~NullVector() = default;


 /****************************************************************
 * VectorData operations -- will move to Vector eventually      *
 ****************************************************************/
public:
    std::string VectorDataName() const override { return d_VectorData->VectorDataName(); }
    size_t numberOfDataBlocks() const override { return d_VectorData->numberOfDataBlocks(); }
    size_t sizeOfDataBlock( size_t i = 0 ) const override { return d_VectorData->sizeOfDataBlock(i); }
    void putRawData( const double *buf ) override { d_VectorData->putRawData(buf); }
    void copyOutRawData( double *buf ) const override { d_VectorData->copyOutRawData(buf); } 
    size_t getLocalSize() const override { return d_VectorData->getLocalSize(); } 
    size_t getGlobalSize() const override { return d_VectorData->getGlobalSize(); } 
    size_t getLocalStartID() const override { return d_VectorData->getLocalStartID(); } 
    void setValuesByLocalID( int num, size_t *indices, const double *vals ) override { d_VectorData->setValuesByLocalID(num, indices, vals); }
    void setLocalValuesByGlobalID( int num, size_t *indices, const double *vals ) override { d_VectorData->setLocalValuesByGlobalID(num, indices, vals); }
    void addValuesByLocalID( int num, size_t *indices, const double *vals ) override { d_VectorData->addValuesByLocalID(num, indices, vals); }
    void addLocalValuesByGlobalID( int num, size_t *indices, const double *vals ) override { d_VectorData->addLocalValuesByGlobalID(num, indices, vals); }
    void getLocalValuesByGlobalID( int num, size_t *indices, double *vals ) const override { d_VectorData->getLocalValuesByGlobalID(num, indices, vals); }
    uint64_t getDataID() const override { return d_VectorData->getDataID(); }
    void *getRawDataBlockAsVoid( size_t i ) override { return d_VectorData->getRawDataBlockAsVoid(i); }
    const void *getRawDataBlockAsVoid( size_t i ) const override { return d_VectorData->getRawDataBlockAsVoid(i); }
    size_t sizeofDataBlockType( size_t i ) const override { return d_VectorData->sizeofDataBlockType(i); }
    bool isTypeId( size_t hash, size_t block ) const override { return d_VectorData->isTypeId(hash, block); }
    void swapData( VectorData &rhs ) override { d_VectorData->swapData(rhs); }
    std::shared_ptr<VectorData> cloneData() const override { return d_VectorData->cloneData(); }
    UpdateState getUpdateStatus() const override { return d_VectorData->getUpdateStatus(); }
    void setUpdateStatus( UpdateState state ) override { d_VectorData->setUpdateStatus(state); }
    void makeConsistent( ScatterType t ) override { d_VectorData->makeConsistent(t); }
    //    AMP_MPI getComm() const override{ return d_VectorData->getComm(); }
    bool hasComm() const override { return d_VectorData->hasComm(); }
    AMP_MPI getComm() const override { return d_VectorData->getComm(); }
    //! Get the CommunicationList for this Vector
    CommunicationList::shared_ptr getCommunicationList() const override { return d_VectorData->getCommunicationList(); }
    void setCommunicationList( CommunicationList::shared_ptr comm ) override { d_VectorData->setCommunicationList(comm); }
    void dataChanged() override { return d_VectorData->dataChanged(); }
/****************************************************************
 ****************************************************************/

public: // Functions inherited from Vector
    inline std::string type() const override { return "Null Vector"; }
    inline std::shared_ptr<ParameterBase> getParameters() override
    {
        return std::shared_ptr<ParameterBase>();
    }
    inline shared_ptr cloneVector( const Variable::shared_ptr name ) const override
    {
        return create( name );
    }
    inline void swapVectors( Vector & ) override {}
    inline void aliasVector( Vector & ) override {}
    inline void assemble() override {}
    using Vector::cloneVector;
    using Vector::dot;


protected:
    virtual Vector::shared_ptr selectInto( const VectorSelector & ) override
    {
        return Vector::shared_ptr();
    }
    virtual Vector::const_shared_ptr selectInto( const VectorSelector & ) const override
    {
        return Vector::const_shared_ptr();
    }


private:
    explicit inline NullVector( Variable::shared_ptr var )
    {
        d_VectorData = new VectorDataNull<TYPE>();
	d_VectorOps = std::make_shared<VectorOperationsDefault<TYPE>>();
        setVariable( var );
        d_CommList = CommunicationList::createEmpty( 0, AMP_MPI( AMP_COMM_SELF ) );
        d_DOFManager.reset( new AMP::Discretization::DOFManager( 0, AMP_MPI( AMP_COMM_SELF ) ) );
    }
};


} // namespace LinearAlgebra
} // namespace AMP

#endif
