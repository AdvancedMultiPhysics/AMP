
#ifndef included_AMP_ConstraintsEliminationOperator
#define included_AMP_ConstraintsEliminationOperator

#include "AMP/operators/Operator.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include <vector>

namespace AMP::Operator {

/**
  u^s = C u^m + d
  */
class ConstraintsEliminationOperator : public Operator
{

public:
    /**
      Constructor.
      @param [in] params
      */
    explicit ConstraintsEliminationOperator( std::shared_ptr<const OperatorParameters> params )
        : Operator( params )
    {
        AMP_INSIST( params->d_db->keyExists( "InputVariable" ), "key not found" );
        std::string inpVarName = params->d_db->getString( "InputVariable" );
        d_InputVariable.reset( new AMP::LinearAlgebra::Variable( inpVarName ) );

        AMP_INSIST( params->d_db->keyExists( "OutputVariable" ), "key not found" );
        std::string outVarName = params->d_db->getString( "OutputVariable" );
        d_OutputVariable.reset( new AMP::LinearAlgebra::Variable( outVarName ) );
    }

    /**
      Destructor
      */
    virtual ~ConstraintsEliminationOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "ConstraintsEliminationOperator"; }

    /**
      Calls first addSlaveToMaster(...) and second setSlaveToZero(...) on the residual vector:
      r^m = r^m + C^T r^s
      r^s = 0

      The apply function for this operator, A, performs the following operation:
      r = A(u)
      Here, A(u) is simply a Matrix-Vector multiplication.
      @param [in] u input vector.
      @param [out] f residual/output vector.
      */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    /**
      @return The variable for the input vector.
      */
    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

    /**
      @return The variable for the output vector.
      */
    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    /**
      u^m = u^m + C^T u^s
      */
    virtual void addSlaveToMaster( AMP::LinearAlgebra::Vector::shared_ptr u ) = 0;
    /**
      u^s = C u^m
      */
    virtual void copyMasterToSlave( AMP::LinearAlgebra::Vector::shared_ptr u ) = 0;
    /**
      u^s = 0
      */
    virtual void setSlaveToZero( AMP::LinearAlgebra::Vector::shared_ptr u );
    /**
      u^s = u^s + d
      */
    virtual void addShiftToSlave( AMP::LinearAlgebra::Vector::shared_ptr u );

protected:
    std::vector<size_t> d_SlaveIndices;
    std::vector<double> d_SlaveShift;

private:
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_InputVariable;  /**< Input variable */
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_OutputVariable; /**< Output variable */
};
} // namespace AMP::Operator

#endif
