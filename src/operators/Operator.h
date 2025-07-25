#ifndef included_AMP_Operator
#define included_AMP_Operator

#include <memory>

#include "AMP/operators/OperatorParameters.h"

#include "AMP/utils/Utilities.h"

#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"

#include <string>


namespace AMP::Operator {


/**
 * Class Operator is an abstract base class for representing
 * a discrete operator which may be linear or nonlinear.  Concrete
 * implementations must include an implementation of the apply() function.
 * The constructor for the class takes a pointer to
 * a OperatorParameters object.
 */
class Operator
{
public:
    typedef std::shared_ptr<AMP::Operator::Operator> shared_ptr;

    //! Default constructor
    Operator( void );

    //! Constructor
    explicit Operator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~Operator() {}

    //! Return the name of the operator
    virtual std::string type() const = 0;

    /**
     * This function is useful for re-initializing/updating an operator
     * \param params
     *    parameter object containing parameters to change
     */
    virtual void reset( std::shared_ptr<const OperatorParameters> params );

    /**
      This base class can not give a meaningful definition of apply. See the derived classes for
      how they define apply. Each operator is free to define apply in a way that is appropriate
      for that operator.
      \param u: shared pointer to const input vector u
      \param f: shared pointer to output vector storing result of applying this operator
      */
    virtual void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> f ) = 0;

    /**
     * Default base class implementation of the residual: f-L(u)
     * \param f: shared pointer to const vector rhs
     * \param u: shared pointer to const vector u
     * \param r: shared pointer to vector residual
     */
    virtual void residual( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                           std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                           std::shared_ptr<AMP::LinearAlgebra::Vector> r );

    /**
     * This function returns a OperatorParameters object
     * constructed by the operator which contains parameters from
     * which new operators can be created. Returning
     * a parameter object instead of an Operator itself is meant
     * to give users more flexibility. Examples of how this functionality
     * might be used would be the construction of Jacobian, frozen Jacobian,
     * preconditioner approximations to the Jacobian, adjoint operators etc
     * \param type: std:string specifying type of return operator parameters
     *      being requested. Currently the valid option is Jacobian
     * \param u: const pointer to current solution vector
     * \param params: pointer to additional parameters that might be required
     *      to construct the return parameters
     */
    virtual std::shared_ptr<OperatorParameters>
    getParameters( const std::string &type,
                   std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                   std::shared_ptr<OperatorParameters> params = nullptr );

    AMP::Utilities::MemoryType getMemoryLocation() const { return d_memory_location; }

    /**
     * Specify level of diagnostic information printed during iterations.
     * \param level
     *    zero prints none or minimal information, higher numbers provide increasingly
     *    verbose debugging information.
     */
    virtual void setDebugPrintInfoLevel( int level ) { d_iDebugPrintInfoLevel = level; }

    //! Return the output variable
    virtual std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const
    {
        return nullptr;
    }

    //! Return the input variable
    virtual std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const
    {
        return nullptr;
    }

    //! Return the selector for output vectors
    virtual std::shared_ptr<AMP::LinearAlgebra::VectorSelector> selectOutputVector() const;

    //! Return the selector for input vectors
    virtual std::shared_ptr<AMP::LinearAlgebra::VectorSelector> selectInputVector() const;

    //! Subset output vector
    std::shared_ptr<AMP::LinearAlgebra::Vector>
    subsetOutputVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec ) const;

    //! Subset output vector
    std::shared_ptr<const AMP::LinearAlgebra::Vector>
    subsetOutputVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> vec ) const;

    //! Subset input vector
    std::shared_ptr<AMP::LinearAlgebra::Vector>
    subsetInputVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec ) const;

    //! Subset input vector
    std::shared_ptr<const AMP::LinearAlgebra::Vector>
    subsetInputVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> vec ) const;

    //! given a vector return whether it is valid or not
    // default behavior is to return true;
    virtual bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ) { return true; }

    //! Return the mesh
    std::shared_ptr<AMP::Mesh::Mesh> getMesh() { return d_Mesh; }

    //! Return the mesh
    std::shared_ptr<const AMP::Mesh::Mesh> getMesh() const { return d_Mesh; }

    /**
     * virtual interface used to make a vector consistent in an operator defined
     * way. An example of where an operator is required to make a vector consistent is
     * in the context of AMR where ghost values on coarse-fine interfaces are filled
     * in an operator dependent way. The default implementation is to simply call the
     * vector makeConsistent(SET)
     */
    virtual void makeConsistent( std::shared_ptr<AMP::LinearAlgebra::Vector> vec );

    //! re-initialize a vector, e.g. after a regrid operation has happened.
    //! This is useful for example when numerical
    //! overshoots or undershoots have happened due to interpolation for example
    //! The default is a null op
    virtual void reInitializeVector( std::shared_ptr<AMP::LinearAlgebra::Vector> ) {}

protected:
    void getFromInput( std::shared_ptr<AMP::Database> db );

    /**
     * This function returns a OperatorParameters object
     * constructed by the operator which contains parameters from
     * which new Jacobian operators can be created. Returning
     * a parameter object instead of an Operator itself is meant
     * to give users more flexibility.
     */
    virtual std::shared_ptr<OperatorParameters>
    getJacobianParameters( std::shared_ptr<const AMP::LinearAlgebra::Vector> )
    {
        return nullptr;
    }

    void setMemoryAndBackendParameters( std::shared_ptr<AMP::Database> db );

    int d_iDebugPrintInfoLevel = 0;

    int d_iObject_id;

    static int d_iInstance_id;

    std::shared_ptr<AMP::Mesh::Mesh> d_Mesh;

    AMP::Utilities::MemoryType d_memory_location = AMP::Utilities::MemoryType::none;

    AMP::Utilities::Backend d_backend = AMP::Utilities::Backend::Serial;

private:
};
} // namespace AMP::Operator

#endif
