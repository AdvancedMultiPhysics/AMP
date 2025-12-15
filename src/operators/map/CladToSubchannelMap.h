// #ifndef included_AMP_CladToSubchannelMap
#define included_AMP_CladToSubchannelMap

#include "AMP/operators/map/ScalarZAxisMap.h"

namespace AMP::Operator {


typedef AsyncMapOperatorParameters CladToSubchannelMapParameters;


/**
 * \class  CladToSubchannelMap
 * \brief  A class used to map temperature from the clad meshes to the subchannel mesh
 * \details  This class maps a scalar quantity (eg. Temperature) from the outer surface
 *    of the clad meshes of a fuel assembly to a subchannel mesh.
 *    mesh1 - clad meshes
 *    mesh2 - subchannel mesh
 */
class CladToSubchannelMap : public AMP::Operator::AsyncMapOperator
{
public:
    /** \brief  Returns true if MapType = "CladToSubchannelMapParameters"
     * \param[in] s  A string extracted from the MapType line in a MeshToMeshMap db
     * \return  True iff s == "CladToSubchannelMapParameters"
     */
    static bool validMapType( const std::string &s );

    /** \brief  Typedef to identify the parameters class of this operator
     */
    typedef CladToSubchannelMapParameters Parameters;

    /** \brief   Standard constructor
     * \param[in] params  Input parameters
     */
    explicit CladToSubchannelMap( std::shared_ptr<const AMP::Operator::OperatorParameters> params );

    //! Destructor
    virtual ~CladToSubchannelMap();

    //! Return the name of the operator
    std::string type() const override { return "CladToSubchannelMap"; }

    /** \brief   Set a frozen vector for results of the apply operation.
     * \details  Set a frozen vector for results of the apply operation.
     * \param result    The results vector
     */
    void setVector( AMP::LinearAlgebra::Vector::shared_ptr result ) override;

    /** \brief   Start a communicative apply operation.
     * \details  Start a communicative apply operation.
     */
    void applyStart( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                     AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    /** \brief   Finish a communicative apply operation.
     * \details  Finish a communicative apply operation.
     */
    void applyFinish( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::LinearAlgebra::Vector::shared_ptr r ) override;

protected:
private:
    // The grid of the subchannel mesh
    void fillSubchannelGrid( std::shared_ptr<AMP::Mesh::Mesh> ); // Function to fill the subchannel
                                                                 // data for all processors
    size_t N_subchannels;                                        // The total number of subchannels
    std::vector<double> d_x, d_y, d_z; // The x, y, z grid for the subchannel
    std::vector<bool>
        d_ownSubChannel; // Which subchannels do I own (multple procs my own a subchannel)
    std::vector<std::vector<int>>
        d_subchannelRanks; // The processors that need each x-y point to fill the result vec
    std::vector<std::vector<int>>
        d_subchannelSend; // The processors that are sending data to fill each subchannel

    // Iterators over the mesh elemens of interest
    static AMP::Mesh::MeshIterator getSubchannelIterator( std::shared_ptr<AMP::Mesh::Mesh> );
    AMP::Mesh::MeshIterator d_iterator1;
    AMP::Mesh::MeshIterator d_iterator2;

    // The list of local MeshElements in each subchannel
    std::vector<std::vector<AMP::Mesh::MeshElementID>> d_elem;

    // Buffers to send/recv the data
    std::vector<AMP_MPI::Request> d_currRequests;
    std::vector<double *> d_sendBuffer;
    size_t d_sendMaxBufferSize;

    int getSubchannelIndex( double x, double y );
};


} // namespace AMP::Operator
