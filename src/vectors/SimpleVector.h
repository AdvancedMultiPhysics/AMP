#ifndef  included_AMP_SimpleVector
#define  included_AMP_SimpleVector

#include "Vector.h"

namespace AMP {
namespace LinearAlgebra {

  /** \brief A core-local vector
    * \details This is a Vector that implements the Vector interface for a std::vector<double>.
    */
  class SimpleVector : public Vector
  {
    private:
      std::vector<double>  d_Data;
      size_t d_startIndex;
      size_t d_globalSize;
      AMP_MPI d_comm;

      SimpleVector ();
      SimpleVector ( const SimpleVector & );

    public:
      /** \brief    Create a SimpleVector
        * \details  This is the factory method for the SimpleVector.  It returns the shared pointer
        * to be used in the code
        * \param    localSize  The number of elements in the vector on this processor
        * \param    var The variable associated with the new vector
        */
      static Vector::shared_ptr  create ( size_t localSize, Variable::shared_ptr var );

      /** \brief    Create a SimpleVector
        * \details  This is the factory method for the SimpleVector.  It returns the shared pointer
        * to be used in the code
        * \param    localSize  The number of elements in the vector on this processor
        * \param    var The variable associated with the new vector
        * \param    comm The variable associated with the new vector
        */
      static Vector::shared_ptr  create ( size_t localSize, Variable::shared_ptr var, AMP_MPI comm );

      /** \brief    Create a SimpleVector
        * \details  This is the factory method for the SimpleVector.  It returns the shared pointer
        * to be used in the code that spans a comm and contains ghost values.
        * \param    var The variable associated with the new vector
        * \param    DOFs The DOFManager
        * \param    commlist The communication list
        */
      static Vector::shared_ptr  create (  Variable::shared_ptr var,
        AMP::Discretization::DOFManager::shared_ptr DOFs, 
        AMP::LinearAlgebra::CommunicationList::shared_ptr commlist );

      /** \brief  Destructor
        */
      virtual ~SimpleVector () {}

      virtual std::string type() const { return "Simple Vector"; }
      virtual void setToScalar(double alpha);
      virtual void scale(double alpha, const VectorOperations &x);
      virtual void scale(double alpha); 
      virtual void add(const VectorOperations &x, const VectorOperations &y);
      virtual void subtract(const VectorOperations &x, const VectorOperations &y);
      virtual void multiply( const VectorOperations &x, const VectorOperations &y);
      virtual void divide( const VectorOperations &x, const VectorOperations &y);
      virtual void reciprocal(const VectorOperations &x);
      virtual void linearSum(double alpha, const VectorOperations &x,
              double beta, const VectorOperations &y);
      virtual void axpy(double alpha, const VectorOperations &x, const VectorOperations &y);
      virtual void axpby(double alpha, double beta, const VectorOperations &x);
      virtual void abs(const VectorOperations &x);

      virtual double min(void) const;

      virtual double max(void) const;

      virtual double L1Norm(void) const;

      virtual double L2Norm(void) const;
      virtual double maxNorm(void) const;

      using Vector::dot;
      virtual double dot(const VectorOperations &x) const;


      virtual boost::shared_ptr<ParameterBase> getParameters ();
      using Vector::cloneVector;
      virtual Vector::shared_ptr cloneVector(const Variable::shared_ptr name) const;
      virtual size_t  numberOfDataBlocks () const;
      virtual size_t  sizeOfDataBlock ( size_t i = 0 ) const;
      using Vector::copyVector;
      virtual void copyVector(const Vector &src_vec);
      virtual void swapVectors(Vector &other);
      virtual void aliasVector(Vector &other);
      virtual void setValuesByLocalID ( int num , size_t *indices , const double *vals );

      /** \brief Not implemented
        */
      virtual void setLocalValuesByGlobalID ( int num , size_t *indices , const double *vals );
      virtual void addValuesByLocalID ( int num , size_t *indices , const double *vals );

      /** \brief Not implemented
        */
      virtual void addLocalValuesByGlobalID ( int num , size_t *indices , const double *vals );

      /** \brief Not implemented
        */
      virtual void getLocalValuesByGlobalID ( int num , size_t *indices , double *vals ) const;
      virtual void assemble();
      virtual void putRawData ( const double *in );
      virtual void copyOutRawData ( double *out ) const;
      virtual size_t getLocalSize() const;
      virtual size_t getGlobalSize() const;
      virtual void *getRawDataBlockAsVoid ( size_t i );
      virtual const void *getRawDataBlockAsVoid ( size_t i ) const;

      double &operator[] ( size_t i );
      double  operator[] ( size_t i ) const ;

      /** \brief Resize this vector
        * \param[in] i The new size
        */
      void    resize ( size_t i );
  };

}
}

#include "SimpleVector.inline.h"

#endif
