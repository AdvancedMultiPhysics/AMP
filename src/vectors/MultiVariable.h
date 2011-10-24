#ifndef included_AMP_MultiVariable_h
#define included_AMP_MultiVariable_h

#include "Variable.h"
#include <vector>

namespace AMP {
namespace LinearAlgebra {

  /** \brief  A class for combining variables.
    * \details  When physics are brought together, individual variables need
    * to be combined to generate a composition.  For instance, combining 
    * temperature and displacement into a single variable.
    *
    * \see MultiVector
    */

  class MultiVariable : public Variable
  {
    public:
      /** \brief A typedef for an iterator for a MultiVariable
        */
      typedef std::vector<Variable::shared_ptr>::iterator              iterator;
      /** \brief A typedef for an iterator for a MultiVariable
        */
      typedef std::vector<Variable::shared_ptr>::const_iterator  const_iterator;
    protected:
      /** \brief List of variables comprising the MultiVariable
        *
        */
      std::vector<Variable::shared_ptr>   d_vVariables;

    public:
      /** \brief Get the first variable in the MultiVariable
        * \return An iterator pointing to the first variable
        */
      inline iterator  beginVariable();
      /** \brief Get end of the MultiVariable array
        * \return An iterator pointing to the end
        */
      inline iterator  endVariable();
      /** \brief Get the first variable in the MultiVariable
        * \return An iterator pointing to the first variable
        */
      inline const_iterator  beginVariable() const;
      /** \brief Get end of the MultiVariable array
        * \return An iterator pointing to the end
        */
      inline const_iterator  endVariable() const;


      /** \brief If there are multiple matching variables in the list, this
        *  will remove them
        */
      inline void  removeDuplicateVariables ();

      /** \brief Given a vector of strings, this will sort the MultiVariable
        * to the given order
        * \param[in] v A list of names by which to sort the MultiVariable
        */
      void  sortVariablesByName ( const std::vector<std::string> &v );

      /** \brief Constructor
        * \param name The name of the MultiVariable
        *
        * \details Because a MultiVariable is a Variable, it must have a name.  This does
        * not change the names of the variables in the list of vectors.
        */
      MultiVariable ( const std::string &name );

      /** \brief Destructor
        *
        */
      virtual ~MultiVariable ();

      /** \brief  Get a particular variable from the list of variables
        * \param  which  the index of the variable sought
        *
        * \details This is an alias for \code
        d_vVariables[which];
        \endcode It is bounds checked in
        * debug builds.
        */
      virtual Variable::shared_ptr  getVariable ( size_t which );

      /** \brief  Throws an exception
        *
        * \details  Since this function is used to aid memory allocation, the
        * fact that it throws an exception should not affect general users
        */
      virtual  size_t  variableID () const;

      /** \brief Returns the number of variables in the list
        *
        * \details This is an alias for 
        \code
        d_vVariables.size();
        \endcode
        */
      virtual  size_t  numVariables ();

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
      virtual void   add ( Variable::shared_ptr newVar );

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
      virtual void setVariable ( size_t i , Variable::shared_ptr &var  );

      /** \brief  Throws an exception
        *
        *  As mentioned in the details of Variable::DOFsPerObject, the number of DOFs may
        *  be a field on the discretization.  It is possible to create such a variable with
        *  multivectors.  Since the return value is undefined for MultiVariables, this method
        *  throws an integer.
        */
			virtual  size_t   DOFsPerObject () const { throw 1; }


      // These are adequately documented elsewhere.
      virtual bool   operator == ( const Variable &rhs ) const;
      virtual Variable::shared_ptr  cloneVariable ( const std::string &name ) const;
      virtual void  setUnits ( const std::string &units );
      
  };

}
}

#include "MultiVariable.inline.h"
#endif
