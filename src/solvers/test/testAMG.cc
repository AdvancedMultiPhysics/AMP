
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstdlib>

#include "ml_include.h"

const double phiCoeffs[4][4] = {
  {0.5, -0.75, 0, 0.25},
  {0.25, -0.25, -0.25, 0.25},
  {0.5, 0.75, 0, -0.25},
  {-0.25, -0.25, 0.25, 0.25}
};

const unsigned int elemDofMap[2][2] = {
  {0, 1},
  {2, 3}
};

const double gaussWts[] = {(8.0/9.0), (5.0/9.0), (5.0/9.0)};
const double gaussPts[] = {0.0, sqrt(3.0/5.0), -sqrt(3.0/5.0)};

double evalPhiPrime(int dofId, double psi) {
  return (phiCoeffs[dofId][1] + (2.0*phiCoeffs[dofId][2]*psi) + (3.0*phiCoeffs[dofId][3]*psi*psi));
}

struct GlobalData {
  int N;
  double** mat;
} myData;

void createMatrix() {
  typedef double* doublePtr;
  myData.mat = new doublePtr[2*(myData.N)];
  for(int i = 0; i < (2*(myData.N)); ++i) {
    myData.mat[i] = new double[2*(myData.N)];
    for(int j = 0; j < (2*(myData.N)); ++j) {
      myData.mat[i][j] = 0.0;
    }//end for j
  }//end for i
}

void computeMatrix() {
  double elementMat[4][4];
  for(int i = 0; i < 2; ++i) {
    for(int c = 0; c < 2; ++c) {
      for(int j = 0; j < 2; ++j) {
        for(int d = 0; d < 2; ++d) {
          elementMat[elemDofMap[i][c]][elemDofMap[j][d]] = 0.0;
          for(int g = 0; g < 3; ++g) {
            elementMat[elemDofMap[i][c]][elemDofMap[j][d]] += (gaussWts[g]*
                evalPhiPrime(elemDofMap[i][c], gaussPts[g])*evalPhiPrime(elemDofMap[j][d], gaussPts[g]));
          }//end g
        }//end for d
      }//end for j
    }//end for c
  }//end for i

  for(int i = 0; i < (myData.N - 1); ++i) {
    for(int r = 0; r < 4; ++r) {
      for(int c = 0; c < 4; ++c) {
        myData.mat[(2*i) + r][(2*i) + c] += (elementMat[r][c]);
      }//end for c
    }//end for r
  }//end for i

  for(int i = 0; i < (2*(myData.N)); ++i) {
    myData.mat[i][0] = 0.0;
    myData.mat[0][i] = 0.0;
    myData.mat[i][2*(myData.N - 1)] = 0.0;
    myData.mat[2*(myData.N - 1)][i] = 0.0;
  }//end for i

  myData.mat[0][0] = 1.0;
  myData.mat[2*(myData.N - 1)][2*(myData.N - 1)] = 1.0;
}

void freeMatrix() {
  for(int i = 0; i < (2*(myData.N)); ++i) {
    delete [] (myData.mat[i]);
    myData.mat[i] = NULL;
  }//end for i
  delete [] (myData.mat);
  myData.mat = NULL;
}

int myMatVec(ML_Operator *data, int in_length, double in[], int out_length, double out[]) {
  for(int i = 0; i < out_length; ++i) {
    out[i] = 0.0;
    for(int j = 0; j < in_length; ++j) {
      out[i] += ((myData.mat[i][j])*in[j]);
    }//end for j
  }//end for i
  return 0;
}

int myGetRow(ML_Operator *data, int N_requested_rows, int requested_rows[],
    int allocated_space, int columns[], double values[], int row_lengths[]) {
  int spaceRequired = 0;
  int cnt = 0;
  for(int i = 0; i < N_requested_rows; ++i) {
    int row = requested_rows[i];
    std::vector<unsigned int> cols;
    std::vector<double> vals;

    for(int j = 0; j < (2*(myData.N)); ++j) {
      if(fabs(myData.mat[row][j]) > 1.0e-15) {
        cols.push_back(j);
        vals.push_back(myData.mat[row][j]);
      }
    }//end for j

    spaceRequired += cols.size();
    if(allocated_space >= spaceRequired) {
      for(size_t j = 0; j < cols.size(); ++j) {
        columns[cnt] = cols[j];
        values[cnt] = vals[j];
        ++cnt;
      }//end for j
      row_lengths[i] = cols.size();
    } else {
      return 0;
    }
  }//end for i
  return 1;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  assert(argc > 1);
  myData.N = atoi(argv[1]);
  createMatrix();
  computeMatrix();

  const int numGrids = 10;
  const int numPDEs = 2;
  const int maxIterations = 1000;
  const int coarseSize = 8;

  ML_set_random_seed(123456);
  ML* ml_object;
  ML_Create(&ml_object, numGrids);

  ML_Init_Amatrix(ml_object, 0, (2*(myData.N)), (2*(myData.N)), &myData);
  ML_Set_Amatrix_Getrow(ml_object, 0, &myGetRow, NULL, (2*(myData.N)));
  ML_Set_Amatrix_Matvec(ml_object, 0, &myMatVec);
  ML_Set_MaxIterations(ml_object, maxIterations);
  ML_Set_ResidualOutputFrequency(ml_object, 1);
  ML_Set_PrintLevel(10);
  ML_Set_OutputLevel(ml_object, 10);

  ML_Aggregate* agg_object;
  ML_Aggregate_Create(&agg_object);
  agg_object->num_PDE_eqns = numPDEs;
  agg_object->nullspace_dim = 1;
  ML_Aggregate_Set_MaxCoarseSize(agg_object, coarseSize);
  ML_Aggregate_Set_CoarsenScheme_UncoupledMIS(agg_object);

  const int nlevels = ML_Gen_MGHierarchy_UsingAggregation(ml_object, 0, ML_INCREASING, agg_object);
  std::cout<<"Number of actual levels: "<<nlevels<<std::endl;

  for(int lev = 0; lev < (nlevels - 1); ++lev) {
    //ML_Gen_Smoother_SymGaussSeidel(ml_object, lev, ML_BOTH, 2, 1.0);
    ML_Gen_Smoother_Jacobi(ml_object, lev, ML_BOTH, 2, 0.8);
  }
  ML_Gen_Smoother_Amesos(ml_object, (nlevels - 1), ML_AMESOS_KLU, -1, 0.0);

  ML_Gen_Solver(ml_object, ML_MGV, 0, (nlevels-1));

  double* solArr = new double[2*(myData.N)];
  double* rhsArr = new double[2*(myData.N)];

  for(int i = 0; i < (2*(myData.N)); i++) {
    solArr[i] = (static_cast<double>(rand()))/(static_cast<double>(RAND_MAX));
  }//end for i

  myMatVec(NULL, (2*myData.N), solArr, (2*myData.N), rhsArr);

  for(int i = 0; i < (2*(myData.N)); i++) {
    solArr[i] = 0.0;
  }//end for i

  ML_Iterate(ml_object, solArr, rhsArr);

  ML_Aggregate_Destroy(&agg_object);
  ML_Destroy(&ml_object);

  delete [] solArr;
  delete [] rhsArr;
  freeMatrix();
  MPI_Finalize();
}  



