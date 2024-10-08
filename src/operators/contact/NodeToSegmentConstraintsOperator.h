
#ifndef included_AMP_NoteToSegmentConstraintsOperator
#define included_AMP_NoteToSegmentConstraintsOperator


#include "AMP/matrices/Matrix.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/contact/NodeToSegmentConstraintsOperatorParameters.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include <fstream>
#include <memory>
#include <vector>


namespace AMP::Operator {

/**
  An abstract base class for representing a linear operator. This class
  stores the matrix representation of the linear operator. It provides
  an implementation of the apply() function.
  @see Operator
  */
class NodeToSegmentConstraintsOperator : public Operator
{

public:
    /**
      Constructor. This resets the matrix shared pointer.
      @param [in] params
      */
    NodeToSegmentConstraintsOperator(
        std::shared_ptr<const NodeToSegmentConstraintsOperatorParameters> params )
        : Operator( params )
    {
        AMP_INSIST( params->d_db->keyExists( "InputVariable" ), "key not found" );
        std::string inpVarName = params->d_db->getString( "InputVariable" );
        d_InputVariable.reset( new AMP::LinearAlgebra::Variable( inpVarName ) );

        AMP_INSIST( params->d_db->keyExists( "OutputVariable" ), "key not found" );
        std::string outVarName = params->d_db->getString( "OutputVariable" );
        d_OutputVariable.reset( new AMP::LinearAlgebra::Variable( outVarName ) );

        d_GlobalComm  = ( params->d_GlobalComm );
        d_DOFsPerNode = ( params->d_DOFsPerNode );
        d_DOFManager  = ( params->d_DOFManager );

        std::vector<AMP::Mesh::MeshID> meshIDs = params->d_Mesh->getBaseMeshIDs();
        AMP_INSIST( params->d_db->keyExists( "MasterMeshIndex" ), "key not found" );
        d_MasterMeshID = meshIDs[params->d_db->getScalar<int>( "MasterMeshIndex" )];
        AMP_INSIST( params->d_db->keyExists( "SlaveMeshIndex" ), "key not found" );
        d_SlaveMeshID = meshIDs[params->d_db->getScalar<int>( "SlaveMeshIndex" )];

        AMP_INSIST( params->d_db->keyExists( "MasterBoundaryID" ), "key not found" );
        d_MasterBoundaryID = params->d_db->getScalar<int>( "MasterBoundaryID" );
        AMP_INSIST( params->d_db->keyExists( "SlaveBoundaryID" ), "key not found" );
        d_SlaveBoundaryID = params->d_db->getScalar<int>( "SlaveBoundaryID" );

        size_t rank          = d_GlobalComm.getRank();
        std::string fileName = "debug_operator_" + std::to_string( rank );
        d_fout.open( fileName.c_str(), std::fstream::out );
    }

    /**
      Destructor
      */
    virtual ~NodeToSegmentConstraintsOperator() { d_fout.close(); }

    /**
     * This function is useful for re-initializing/updating an operator
     * \param params
     *        parameter object containing parameters to change
     */
    virtual void reset( std::shared_ptr<const OperatorParameters> params );

    /**
      The apply function for this operator, A, performs the following operation:
      r = a*A(u) + b*f, if f is not NULL and r = a*A(u), if f is NULL.
      Here, A(u) is simply a Matrix-Vector multiplication.
      @param [in] f auxillary/rhs vector.
      @param [in] u input vector.
      @param [out] r residual/output vector.
      @param [in] a first constant used in the expression: r = a*A(u) + b*f. The default value is
      -1.
      @param [in] b second constant used in the expression: r = a*A(u) + b*f. The default value is
      1.
      */
    virtual void apply( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                        AMP::LinearAlgebra::Vector::const_shared_ptr u,
                        AMP::LinearAlgebra::Vector::shared_ptr r,
                        const double a = -1.0,
                        const double b = 1.0 );

    /**
      @return The variable for the input vector.
      */
    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable();

    /**
      @return The variable for the output vector
      */
    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable();

    // deprecated
    void applyResidualCorrection( AMP::LinearAlgebra::Vector::shared_ptr r );
    void applySolutionCorrection( AMP::LinearAlgebra::Vector::shared_ptr u );
    void getRhsCorrection( AMP::LinearAlgebra::Vector::shared_ptr d );
    void addShiftToSlave( AMP::LinearAlgebra::Vector::shared_ptr u );
    //
    size_t numLocalConstraints();
    size_t numGlobalConstraints();

    void setSlaveToZero( AMP::LinearAlgebra::Vector::shared_ptr u );
    void addSlaveToMaster( AMP::LinearAlgebra::Vector::shared_ptr u );
    void copyMasterToSlave( AMP::LinearAlgebra::Vector::shared_ptr u );

    AMP::Mesh::MeshID getMasterMeshID() const { return d_MasterMeshID; }
    AMP::Mesh::MeshID getSlaveMeshID() const { return d_SlaveMeshID; }

protected:
private:
    void getVectorIndicesFromGlobalIDs( const std::vector<AMP::Mesh::MeshElementID> &globalIDs,
                                        std::vector<size_t> &vectorIndices );

    AMP::AMP_MPI d_GlobalComm;
    std::shared_ptr<AMP::Discretization::DOFManager> d_DOFManager;
    size_t d_DOFsPerNode;

    AMP::Mesh::MeshID d_MasterMeshID;
    AMP::Mesh::MeshID d_SlaveMeshID;

    int d_MasterBoundaryID;
    int d_SlaveBoundaryID;

    std::vector<int> d_SendCnts;
    std::vector<int> d_SendDisps;
    std::vector<int> d_RecvCnts;
    std::vector<int> d_RecvDisps;
    std::vector<int> d_TransposeSendCnts;
    std::vector<int> d_TransposeSendDisps;
    std::vector<int> d_TransposeRecvCnts;
    std::vector<int> d_TransposeRecvDisps;

    // actually we don't need to store the meshelementids but maybe useful later to check whether
    // the active set has
    // changed
    std::vector<AMP::Mesh::MeshElementID> d_SlaveVerticesGlobalIDs;
    std::vector<AMP::Mesh::MeshElementID> d_RecvMasterVerticesGlobalIDs;

    std::vector<size_t> d_SlaveIndices;
    std::vector<size_t> d_RecvMasterIndices;
    std::vector<size_t> d_MasterVerticesMap;
    std::vector<size_t> d_MasterVerticesInverseMap;
    std::vector<size_t> d_MasterVerticesOwnerRanks;
    std::vector<double> d_MasterShapeFunctionsValues;
    std::vector<double> d_SlaveVerticesShift;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_InputVariable;  /**< Input variable */
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_OutputVariable; /**< Output variable */

    std::fstream d_fout;
};
} // namespace AMP::Operator

#endif
