#ifndef included_AMP_DisplacementMap
#define included_AMP_DisplacementMap


#include <vector>
#include <list>

#include "operators/map/AsyncMapOperator.h"
#include "operators/map/DisplacementMapParameters.h"
#include "vectors/Vector.h"
#include "ampmesh/Mesh.h"
#include "utils/AMP_MPI.h"


namespace AMP {
namespace Operator {

  class DisplacementMap : public AMP::Operator::AsyncMapOperator 
  {
    public:
      typedef  boost::shared_ptr<DisplacementMapParameters>   params_shared_ptr;

    private:
      std::vector<int>          d_MySurfaceIndicesSend;
      std::vector<int>          d_MySurfaceIndicesRecv;

      std::vector<int>          d_MySurfaceCountsS;
      std::vector<int>          d_MySurfaceDisplsS;
      std::vector<int>          d_MySurfaceCountsR;
      std::vector<int>          d_MySurfaceDisplsR;

      std::vector<double>       d_SendBuffer;
      std::vector<double>       d_RecvBuffer;

      AMP_MPI                   d_MapComm;

      AMP::LinearAlgebra::Vector::shared_ptr        d_OutputVector;
      AMP::LinearAlgebra::Variable::shared_ptr      d_inpVariable;

      int   d_SendTag;
      int   d_RecvTag;

      void  sendSurface ( params_shared_ptr );
      void  recvSurface ( params_shared_ptr );
      void  sendOrder ( params_shared_ptr );
      void  recvOrder ( params_shared_ptr );
      void  buildSendRecvList ( params_shared_ptr );
      void  finalizeCommunication ( params_shared_ptr );

      class Point
      {
        public:
          double _pos[3];
          size_t _id;
          // Allows me to push data to things in a multiset.
          mutable std::list<int>  _procs;

          static double _precision;

          Point ();
          Point ( const Point &rhs );
          bool operator == ( const Point &rhs ) const;
          bool operator <  ( const Point &rhs ) const;
      };

      class CommInfo
      {
        public:
          size_t           _remId;
          std::list<int>   _procs;
      };

    protected:
    public:


      /** \brief  Returns true if MapType = "GapSize"
        * \param[in] s  A string extracted from the MapType line in a MeshToMeshMap db
        * \return  True iff s == "GapSize"
        */
      static bool  validMapType ( const std::string & );

      /** \brief  Typedef to identify the parameters class of this operator
        */
      typedef DisplacementMapParameters   Parameters;

      /** \brief  The base tag used in communication.
        */
      enum { CommTagBase = 10000 };

      /** Constructor
        */
      DisplacementMap ( const boost::shared_ptr<AMP::Operator::OperatorParameters> & params );
      /** Destructor
        */
      virtual ~DisplacementMap ();

      virtual bool continueAsynchronousConstruction ( const boost::shared_ptr < AMP::Operator::OperatorParameters > &params );


      virtual void applyStart(const AMP::LinearAlgebra::Vector::shared_ptr &f, const AMP::LinearAlgebra::Vector::shared_ptr &u,
          AMP::LinearAlgebra::Vector::shared_ptr  &r, const double a = -1.0, const double b = 1.0);

      virtual void applyFinish(const AMP::LinearAlgebra::Vector::shared_ptr &f, const AMP::LinearAlgebra::Vector::shared_ptr &u,
          AMP::LinearAlgebra::Vector::shared_ptr  &r, const double a = -1.0, const double b = 1.0);

      virtual void  setVector ( AMP::LinearAlgebra::Vector::shared_ptr &p );

      virtual AMP::LinearAlgebra::Variable::shared_ptr  getInputVariable (int varId = -1) { return d_inpVariable; }
      virtual AMP::LinearAlgebra::Variable::shared_ptr  getOutputVariable () { return d_inpVariable; }
  };


}
}

#endif
