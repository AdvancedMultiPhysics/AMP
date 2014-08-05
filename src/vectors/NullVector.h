#ifndef included_AMP_NullVector
#define included_AMP_NullVector

#include <string>
#include "Vector.h"

namespace AMP {
namespace LinearAlgebra {

  /** \brief An empty vector
    * \details Some operators do not require vectors for application.  In these
    * circumstances, a NullVector is used.  This stores no data and performs no
    * work.
    */

  class NullVector  : public Vector
  {
    private:
      NullVector( Variable::shared_ptr );

    public:
      /**
        *  \brief Create a NullVector
        *  \param[in]  name  Name of variable to associate with this NullVector
        *  \return Vector shared pointer to a NullVector
        */
      static Vector::shared_ptr   create ( const std::string &name );

      /**
        *  \brief Create a NullVector
        *  \param[in]  name  Variable to associate with this NullVector
        *  \return Vector shared pointer to a NullVector
        */
      static Vector::shared_ptr   create ( const Variable::shared_ptr name );

      virtual ~NullVector();

      virtual std::string type() const { return "Null Vector"; }
      virtual boost::shared_ptr<ParameterBase> getParameters ();

      virtual shared_ptr cloneVector(const Variable::shared_ptr name) const;
      template <typename RETURN_TYPE>
      RETURN_TYPE *  getRawDataBlock ();

      template <typename RETURN_TYPE>
      const RETURN_TYPE *  getRawDataBlock() const;

      virtual void copyVector( Vector::const_shared_ptr rhs );
      virtual void swapVectors(Vector &);
      virtual void aliasVector(Vector &);

      virtual void setToScalar(double );
      virtual void scale(double , const VectorOperations &);
      virtual void scale(double );
      virtual void addScalar(const VectorOperations &, double );
      virtual void add(const VectorOperations &, const VectorOperations &);
      virtual void subtract(const VectorOperations &, const VectorOperations &);
      virtual void multiply( const VectorOperations &, const VectorOperations &);
      virtual void divide( const VectorOperations &, const VectorOperations &);
      virtual void reciprocal(const VectorOperations &);
      virtual void linearSum(double , const VectorOperations &,
              double , const VectorOperations &);
      virtual void axpy(double , const VectorOperations &, const VectorOperations &);
      virtual void axpby(double , double, const VectorOperations &);
      virtual void abs(const VectorOperations &);
      virtual double min(void) const;
      virtual double max(void) const;
      virtual void setRandomValues(void);
      virtual void setValuesByLocalID ( int , size_t * , const double * );
      virtual void setLocalValuesByGlobalID ( int , size_t * , const double * );
      virtual void addValuesByLocalID ( int , size_t * , const double * );
      virtual void addLocalValuesByGlobalID ( int , size_t * , const double * );
      virtual void getLocalValuesByGlobalID ( int , size_t * , double * ) const;


      virtual void makeConsistent ( ScatterType  );

      virtual void assemble();
      virtual double L1Norm(void) const;
      virtual double L2Norm(void) const;
      virtual double maxNorm(void) const;
      virtual double dot(const VectorOperations &) const;

      virtual void putRawData ( const double *in );
      virtual void copyOutRawData ( double *out ) const;

      virtual size_t getLocalSize() const;
      virtual size_t getGlobalSize() const;
      virtual size_t getGhostSize() const;

      virtual size_t numberOfDataBlocks () const;
      virtual size_t sizeOfDataBlock ( size_t ) const;

      using Vector::cloneVector;
      using Vector::dot;

    protected:

      virtual Vector::shared_ptr  selectInto ( const VectorSelector & ) { return Vector::shared_ptr(); }
      virtual Vector::const_shared_ptr  selectInto ( const VectorSelector & ) const { return Vector::const_shared_ptr(); }

      virtual void *getRawDataBlockAsVoid ( size_t );
      virtual const void *getRawDataBlockAsVoid ( size_t ) const;


  };

}
}

#include "NullVector.tmpl.h"

#endif
