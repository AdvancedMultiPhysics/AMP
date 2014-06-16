
#ifndef included_AMP_PelletStackOperator
#define included_AMP_PelletStackOperator

#include "operators/libmesh/PelletStackOperatorParameters.h"

namespace AMP {
  namespace Operator {

    class PelletStackOperator : public Operator 
    {
      public :
        PelletStackOperator(const boost::shared_ptr<PelletStackOperatorParameters> & params);

        virtual ~PelletStackOperator() { }

        int getLocalIndexForPellet(unsigned int pellId);

        unsigned int getTotalNumberOfPellets();

        std::vector<AMP::Mesh::Mesh::shared_ptr> getLocalMeshes();

        std::vector<unsigned int> getLocalPelletIds();

        bool useSerial();

        bool onlyZcorrection();

        bool useScaling();

        void reset(const boost::shared_ptr<OperatorParameters>& params);

        void applyUnscaling(AMP::LinearAlgebra::Vector::shared_ptr f);

        void apply(AMP::LinearAlgebra::Vector::const_shared_ptr f, AMP::LinearAlgebra::Vector::const_shared_ptr u,
            AMP::LinearAlgebra::Vector::shared_ptr r, const double a = -1.0, const double b = 1.0);

      protected:
        void applySerial(AMP::LinearAlgebra::Vector::const_shared_ptr f, AMP::LinearAlgebra::Vector::const_shared_ptr u,
            AMP::LinearAlgebra::Vector::shared_ptr &r);

        void applyOnlyZcorrection(AMP::LinearAlgebra::Vector::shared_ptr &u);

        void applyXYZcorrection(AMP::LinearAlgebra::Vector::const_shared_ptr f, AMP::LinearAlgebra::Vector::const_shared_ptr u,
            AMP::LinearAlgebra::Vector::shared_ptr &r);

        void computeZscan(AMP::LinearAlgebra::Vector::const_shared_ptr u, std::vector<double> &finalMaxZdispsList);

        unsigned int d_totalNumberOfPellets;
        unsigned int d_currentPellet;
        bool d_useSerial;
        bool d_onlyZcorrection;
        bool d_useScaling;
        double d_scalingFactor;
        short int d_masterId;
        short int d_slaveId;
        std::vector<AMP::Mesh::Mesh::shared_ptr> d_meshes;
        std::vector<unsigned int> d_pelletIds;
        AMP::LinearAlgebra::Variable::shared_ptr d_var;
        AMP::LinearAlgebra::Vector::shared_ptr d_frozenVectorForMaps;
        bool d_frozenVectorSet;
        AMP_MPI d_pelletStackComm;
        boost::shared_ptr<AMP::Operator::AsyncMapColumnOperator>  d_n2nMaps;
    };

  }
}

#endif

