#ifndef included_AMP_MultiVariable_h
#define included_AMP_MultiVariable_h

#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorSelector.h"

#include <vector>


namespace AMP::LinearAlgebra {


/** \brief  A class for combining variables.
 * \details  When physics are brought together, individual variables need
 * to be combined to generate a composition.  For instance, combining
 * temperature and displacement into a single variable.
 *
 * \see MultiVector
 */
class MultiVariable final : public Variable
{
public:
    /** \brief Get the first variable in the MultiVariable
     * \return An iterator pointing to the first variable
     */
    inline auto begin() { return d_vVariables.begin(); }

    /** \brief Get end of the MultiVariable array
     * \return An iterator pointing to the end
     */
    inline auto end() { return d_vVariables.end(); }

    /** \brief Get the first variable in the MultiVariable
     * \return An iterator pointing to the first variable
     */
    inline auto begin() const { return d_vVariables.begin(); }

    /** \brief Get end of the MultiVariable array
     * \return An iterator pointing to the end
     */
    inline auto end() const { return d_vVariables.end(); }

    /** \brief If there are multiple matching variables in the list, this
     *  will remove them.  Note that may change the etnry order and will remove any null entries.
     */
    void removeDuplicateVariables();

    /** \brief Constructor
     * \details Because a MultiVariable is a Variable, it must have a name.  This does
     *    not change the names of the variables in the list of vectors.
     * \param[in] name  The name of the MultiVariable
     * \param[in] vars  Optional list of variables in the MultiVariable
     *
     */
    explicit MultiVariable( const std::string &name,
                            std::vector<std::shared_ptr<Variable>> vars = {} );

    //! Create a multivariable from a single variable
    explicit MultiVariable( std::shared_ptr<Variable> var );


    /** \brief  Get a particular variable from the list of variables
      * \param  which  the index of the variable sought
      *
      * \details This is an alias for \code
        d_vVariables[which];
        \endcode It is bounds checked in
      * debug builds.
      */
    virtual std::shared_ptr<Variable> getVariable( size_t which );

    /** \brief  Get a particular variable from the list of variables
      * \param  which  the index of the variable sought
      *
      * \details This is an alias for \code
        d_vVariables[which];
        \endcode It is bounds checked in
      * debug builds.
      */
    virtual std::shared_ptr<const Variable> getVariable( size_t which ) const;

    /** \brief Returns the number of variables in the list
      *
      * \details This is an alias for
        \code
        d_vVariables.size();
        \endcode
      */
    virtual size_t numVariables() const;

    /** \brief Add a variable to the end of the variable list
      * \param  newVar  a shared pointer to the new variable
      *
      * \details This is an alias for
        \code
        d_vVariables.push_back ( newVar );
        \endcode
        unless newVar is a MultiVariable.  In order to keep
        heirarchies to a minimum, the members of newVar are added
        instead of newVar itself.
      */
    virtual void add( std::shared_ptr<Variable> newVar );

    /** \brief Set a particular variable in the list
      * \param i    index into the list
      * \param var  a shared pointer to the variable to be placed in the list
      *
      * \details  This is an alias for
        \code
        d_vVariables[i] = var;
        \endcode
        * This is bounds checked in debug builds
      */
    virtual void setVariable( size_t i, std::shared_ptr<Variable> &var );

    // These are adequately documented elsewhere.
    virtual bool operator==( const Variable &rhs ) const override;
    virtual std::shared_ptr<Variable> clone( const std::string &name ) const override;
    virtual void setUnits( const Units &units ) override;


public: // Default constructors
    MultiVariable() : Variable() {}
    MultiVariable( MultiVariable && )      = default;
    MultiVariable( const MultiVariable & ) = default;
    MultiVariable &operator=( MultiVariable && ) = default;
    MultiVariable &operator=( const MultiVariable & ) = default;
    virtual ~MultiVariable()                          = default;

public: // Functions inherited from Variable
    std::string className() const override { return "MultiVariable"; }
    std::shared_ptr<VectorSelector> createVectorSelector() const override;
    void writeRestart( int64_t ) const override;
    MultiVariable( int64_t, AMP::IO::RestartManager *manager );
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;

protected:
    //! List of variables comprising the MultiVariable
    std::vector<std::shared_ptr<Variable>> d_vVariables;
};


/** \brief  A class for selecting multi-variables.
 * \details  This class provides a selector for a multivariable
 */
class VS_MultiVariable : public VectorSelector
{

public:
    /** \brief Constructor
     * \param[in] var  The variable to subset on
     */
    explicit VS_MultiVariable( const std::shared_ptr<MultiVariable> &var );

    std::string getName() const;

public: // Functions inherited from VectorSelector
    bool isSelected( const Vector & ) const override;
    Vector::shared_ptr subset( Vector::shared_ptr vec ) const override;

protected:
    std::shared_ptr<MultiVariable> d_var;
};


} // namespace AMP::LinearAlgebra


#endif
