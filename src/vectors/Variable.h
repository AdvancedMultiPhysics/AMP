#ifndef included_AMP_Variable_h
#define included_AMP_Variable_h

#include "AMP/utils/Units.h"
#include "AMP/utils/enable_shared_from_this.h"

#include <memory>
#include <string_view>


namespace AMP::IO {
class RestartManager;
}


namespace AMP::LinearAlgebra {

/**\brief Flag to choose algorithm for makeConsistent
 *\see makeConsistent
 */
enum class ScatterType : uint8_t { CONSISTENT_ADD, CONSISTENT_SET };

/**\brief The four states a Vector can be in
 *\see makeConsistent
 */
enum class UpdateState : uint8_t {
    UNCHANGED     = 0,
    LOCAL_CHANGED = 1,
    ADDING        = 2,
    SETTING       = 3,
    MIXED         = 4
};


class VectorSelector;


/**
 * \class Variable
 * \brief A description of the data in the vector
 *
 * \details  This class stores information about the vector such as the name and units.
 *   It is used in subsetting operations.
 */
class Variable : public AMP::enable_shared_from_this<Variable>
{
public:
    // Empty constructor
    Variable() = default;

    /** \brief  Construct a variable with a name
     * \details  At the very least, a variable must have a name.
     *    Since this class is virtual, this constructor is used by derived classes.
     * \param  name  The name of the variable
     */
    explicit Variable( const std::string &name );


    //!  Destructor
    virtual ~Variable();


    //!  Set the units of this variable
    virtual void setUnits( const Units &u );


    //!  Get the units of this variable
    virtual const Units &getUnits() const;


    /** \brief  A function that returns the name of a variable
     * \details  This gives access to the name
     */
    inline auto &getName() const { return d_VariableName; }


    /** \brief  A function that returns the name of a variable
     * \details  This gives access to the name
     */
    virtual std::string className() const { return "Variable"; }


    /** \brief  Compares two variables for equality.
     * \details This operation compares the names.
     * \param  rhs  Variable to compare
     */
    virtual bool operator==( const Variable &rhs ) const;


    /** \brief  Inverse of ==
     * \details This function performs an equality check and negates it.
     * \param  rhs  Variable to compare
     */
    bool operator!=( const Variable &rhs ) const;


    /** \brief  Create a variable of the same type with a new name
     * \details This function will create a "deep" copy of this variable.
     */
    std::shared_ptr<Variable> clone() const;


    /** \brief  Create a variable of the same type with a new name
     * \details This function will create a "deep" copy of this variable.
     */
    virtual std::shared_ptr<Variable> clone( const std::string &name ) const;


    /** \brief  Create a VectorSelector
     * \details This function will create a VectorSelector that is able to subset
     *    a vector for the variable.  This is used by Vector::subsetVectorForVariable
     *    which then calls Vector::selectInto.
     */
    virtual std::shared_ptr<VectorSelector> createVectorSelector() const;


    //! Get a unique id hash for the vector
    virtual uint64_t getID() const;


public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will the variable to an HDF5 file
     * \param fid    File identifier to write
     */
    virtual void writeRestart( int64_t fid ) const;


    /**
     * \brief    Read restart data to file
     * \details  This function will create a variable from the restart file
     * \param fid    File identifier to write
     */
    Variable( int64_t fid );


protected:
    /** \brief  A name given to the variable
     * \details Variables have names for easy identification.
     *     For instance, some variables are called displacement, concentration,
     *     search direction, etc.
     */
    std::string d_VariableName;

    /** \brief  The units this variable is measured in
     */
    AMP::Units d_Units;
};

} // namespace AMP::LinearAlgebra

#endif
