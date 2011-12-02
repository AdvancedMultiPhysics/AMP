
#include "MechanicsNonlinearUpdatedLagrangianElement.h"
#include "utils/Utilities.h"

namespace AMP {
namespace Operator {

  void MechanicsNonlinearUpdatedLagrangianElement :: computeStressAndStrain(const std::vector<std::vector<double> > & elementInputVectors, 
      std::vector<double> & stressVec, std::vector<double> & strainVec) {
    //const std::vector<Real> & JxW = (*d_JxW);

    const std::vector<std::vector<RealGradient> > & dphi = (*d_dphi);

    const std::vector<std::vector<Real> > & phi = (*d_phi);

    //const std::vector<Point> & xyz = (*d_xyz);

    d_fe->reinit(d_elem);

    d_materialModel->preNonlinearAssemblyElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) {
      d_materialModel->preNonlinearAssemblyGaussPointOperation();

      /* Compute Strain From Given Displacement */

      double dudx = 0;
      double dudy = 0;
      double dudz = 0;
      double dvdx = 0;
      double dvdy = 0;
      double dvdz = 0;
      double dwdx = 0;
      double dwdy = 0;
      double dwdz = 0;

      for(unsigned int k = 0; k < num_nodes; k++) {
        dudx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](0));
        dudy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](1));
        dudz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](2));

        dvdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](0));
        dvdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](1));
        dvdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](2));

        dwdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](0));
        dwdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](1));
        dwdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](2));
      } //end for k

      std::vector<std::vector<double> > fieldsAtGaussPt(Mechanics::TOTAL_NUMBER_OF_VARIABLES); 

      //Strain
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dudx);
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dvdy);
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dwdz);
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dvdz + dwdy));
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudz + dwdx));
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudy + dvdx));

      if( !(elementInputVectors[Mechanics::TEMPERATURE].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::TEMPERATURE][k]*phi[k][qp]);
        } //end for k
        fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::BURNUP].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::BURNUP][k]*phi[k][qp]);
        } //end for k
        fieldsAtGaussPt[Mechanics::BURNUP].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k]*phi[k][qp]);
        } //end for k
        fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::LHGR].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::LHGR][k]*phi[k][qp]);
        } //end for k
        fieldsAtGaussPt[Mechanics::LHGR].push_back(valAtGaussPt);
      }

      double* currStress;

      /* Compute Stress Corresponding to Given Strain, Temperature, Burnup, ... */

      d_materialModel->getInternalStress(fieldsAtGaussPt, currStress);

      for(int i = 0; i < 6; i++) {
        stressVec[(6*qp) + i] = currStress[i];
        strainVec[(6*qp) + i] = fieldsAtGaussPt[Mechanics::DISPLACEMENT][i];
      }//end for i

      d_materialModel->postNonlinearAssemblyGaussPointOperation();
    }//end for qp

    d_materialModel->postNonlinearAssemblyElementOperation();
  }


  void MechanicsNonlinearUpdatedLagrangianElement :: printStressAndStrain(FILE* fp, const 
      std::vector<std::vector<double> > & elementInputVectors) {
    //const std::vector<Real> & JxW = (*d_JxW);

    const std::vector<std::vector<RealGradient> > & dphi = (*d_dphi);

    const std::vector<std::vector<Real> > & phi = (*d_phi);

    const std::vector<Point> & xyz = (*d_xyz);

    d_fe->reinit(d_elem);

    d_materialModel->preNonlinearAssemblyElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) {
      d_materialModel->preNonlinearAssemblyGaussPointOperation();

      /* Compute Strain From Given Displacement */

      double dudx = 0;
      double dudy = 0;
      double dudz = 0;
      double dvdx = 0;
      double dvdy = 0;
      double dvdz = 0;
      double dwdx = 0;
      double dwdy = 0;
      double dwdz = 0;

      for(unsigned int k = 0; k < num_nodes; k++) {
        dudx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](0));
        dudy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](1));
        dudz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 0]*dphi[k][qp](2));

        dvdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](0));
        dvdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](1));
        dvdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 1]*dphi[k][qp](2));

        dwdx += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](0));
        dwdy += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](1));
        dwdz += (elementInputVectors[Mechanics::DISPLACEMENT][(3*k) + 2]*dphi[k][qp](2));
      } //end for k

      std::vector<std::vector<double> > fieldsAtGaussPt(Mechanics::TOTAL_NUMBER_OF_VARIABLES); 

      //Strain
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dudx);
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dvdy);
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(dwdz);
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dvdz + dwdy));
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudz + dwdx));
      fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(0.5*(dudy + dvdx));

      if( !(elementInputVectors[Mechanics::TEMPERATURE].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::TEMPERATURE][k]*phi[k][qp]);
        } //end for k
        fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::BURNUP].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::BURNUP][k]*phi[k][qp]);
        } //end for k
        fieldsAtGaussPt[Mechanics::BURNUP].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k]*phi[k][qp]);
        } //end for k
        fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::LHGR].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::LHGR][k]*phi[k][qp]);
        } //end for k
        fieldsAtGaussPt[Mechanics::LHGR].push_back(valAtGaussPt);
      }

      double* currStress;

      /* Compute Stress Corresponding to Given Strain, Temperature, Burnup, ... */

      d_materialModel->getInternalStress(fieldsAtGaussPt, currStress);

      fprintf(fp, "%.12f %.12f %.12f \n", xyz[qp](0), xyz[qp](1), xyz[qp](2));
      fprintf(fp, "%.12f %.12f %.12f %.12f %.12f %.12f \n", currStress[0], currStress[1], currStress[2], 
          currStress[3], currStress[4], currStress[5]);
      fprintf(fp, "%.12f %.12f %.12f %.12f %.12f %.12f \n", fieldsAtGaussPt[Mechanics::DISPLACEMENT][0],
          fieldsAtGaussPt[Mechanics::DISPLACEMENT][1], fieldsAtGaussPt[Mechanics::DISPLACEMENT][2],
          fieldsAtGaussPt[Mechanics::DISPLACEMENT][3], fieldsAtGaussPt[Mechanics::DISPLACEMENT][4],
          fieldsAtGaussPt[Mechanics::DISPLACEMENT][5]);

      d_materialModel->postNonlinearAssemblyGaussPointOperation();
    }//end for qp

    d_materialModel->postNonlinearAssemblyElementOperation();
  }

  void MechanicsNonlinearUpdatedLagrangianElement :: initMaterialModel( const std::vector<double> & initTempVector ) 
  {
    const std::vector<std::vector<Real> > & phi = (*d_phi);

    d_fe->reinit(d_elem);

    d_materialModel->preNonlinearInitElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) {
      d_materialModel->preNonlinearInitGaussPointOperation();

      double tempAtGaussPt = 0;
      if( !(initTempVector.empty()) ) {
        for(unsigned int k = 0; k < num_nodes; k++) {
          tempAtGaussPt += (initTempVector[k]*phi[k][qp]);
        } //end for k
      }

      d_materialModel->nonlinearInitGaussPointOperation(tempAtGaussPt);

      d_materialModel->postNonlinearInitGaussPointOperation();

      if((d_useJaumannRate == false) && (d_useFlanaganTaylorElem == true)) {
        for(unsigned int i = 0; i < 3; i++) {
          for(unsigned int j = 0; j < 3; j++) {

            if(i == j) {
              d_leftStretchV_n.push_back(1.0);
            } else {
              d_leftStretchV_n.push_back(0.0);
            }

            if(i == j) {
              d_leftStretchV_np1.push_back(1.0);
            } else {
              d_leftStretchV_np1.push_back(0.0);
            }

            if(i == j) {
              d_rotationR_n.push_back(1.0);
            } else {
              d_rotationR_n.push_back(0.0);
            }

            if(i == j) {
              d_rotationR_np1.push_back(1.0);
            } else {
              d_rotationR_np1.push_back(0.0);
            }
          }
        }

        d_gaussPtCnt++;
      }
    }//end for qp

    d_materialModel->postNonlinearInitElementOperation();
  }

  void MechanicsNonlinearUpdatedLagrangianElement :: apply_Normal() 
  {
    //const std::vector<Real> & JxW = (*d_JxW);

    std::vector<std::vector<double> > & elementInputVectors = d_elementInputVectors;

    std::vector<std::vector<double> > & elementInputVectors_pre = d_elementInputVectors_pre;

    std::vector<double> & elementOutputVector = (*d_elementOutputVector);

    std::vector<Point> xyz, xyz_n, xyz_np1, xyz_np1o2;

    d_fe->reinit(d_elem);

    d_materialModel->preNonlinearAssemblyElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    xyz.resize(num_nodes);
    xyz_n.resize(num_nodes);
    xyz_np1.resize(num_nodes);
    xyz_np1o2.resize(num_nodes);

    double refX[8], refY[8], refZ[8], dNdX[8], dNdY[8], dNdZ[8], detJ_0[1];

    Point p1;
    for(unsigned int ijk = 0; ijk < num_nodes; ijk++) {
      p1 = d_elem->point(ijk);
      //xyz[ijk] = p1;
      refX[ijk] = xyz[ijk](0) = d_elementRefXYZ[(3 * ijk) + 0];
      refY[ijk] = xyz[ijk](1) = d_elementRefXYZ[(3 * ijk) + 1];
      refZ[ijk] = xyz[ijk](2) = d_elementRefXYZ[(3 * ijk) + 2];
    }

    double currX[8], currY[8], currZ[8], dNdx[8], dNdy[8], dNdz[8], detJ[1], delta_u[8], delta_v[8], delta_w[8];
    double x_np1o2[8], y_np1o2[8], z_np1o2[8], prevX[8], prevY[8], prevZ[8], N[8];
    double rsq3 = (1.0 / sqrt(3.0));
    const double currXi[8] = {-rsq3, rsq3, -rsq3, rsq3, -rsq3, rsq3, -rsq3, rsq3};
    const double currEta[8] = {-rsq3, -rsq3, rsq3, rsq3, -rsq3, -rsq3, rsq3, rsq3};
    const double currZeta[8] = {-rsq3, -rsq3, -rsq3, -rsq3, rsq3, rsq3, rsq3, rsq3};
    
    for(unsigned int ijk = 0; ijk < num_nodes; ijk++) {
      prevX[ijk] = xyz_n[ijk](0) = xyz[ijk](0) + elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 0];
      prevY[ijk] = xyz_n[ijk](1) = xyz[ijk](1) + elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 1];
      prevZ[ijk] = xyz_n[ijk](2) = xyz[ijk](2) + elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 2];

      currX[ijk] = xyz_np1[ijk](0) = xyz[ijk](0) + elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 0];
      currY[ijk] = xyz_np1[ijk](1) = xyz[ijk](1) + elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 1];
      currZ[ijk] = xyz_np1[ijk](2) = xyz[ijk](2) + elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 2];

      delta_u[ijk] = elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 0] - elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 0];
      delta_v[ijk] = elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 1] - elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 1];
      delta_w[ijk] = elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 2] - elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 2];

      x_np1o2[ijk] = xyz_np1o2[ijk](0) = xyz_n[ijk](0) + (delta_u[ijk] / 2.0);
      y_np1o2[ijk] = xyz_np1o2[ijk](1) = xyz_n[ijk](1) + (delta_v[ijk] / 2.0);
      z_np1o2[ijk] = xyz_np1o2[ijk](2) = xyz_n[ijk](2) + (delta_w[ijk] / 2.0);
    }

    if(d_iDebugPrintInfoLevel > 11) {
      for(int i = 0; i < 8; i++) {
        std::cout<<"delta_u["<<i<<"]="<<delta_u[i]<<" delta_v["<<i<<"]="<<delta_v[i]<<" delta_w["<<i<<"]="<<delta_w[i]<<std::endl;
        std::cout<<"elementInputVectors["<<(3*i)+0<<"]="<<elementInputVectors[Mechanics::DISPLACEMENT][(3*i) + 0]<<" elementInputVectors["<<(3*i)+1<<"]="<<elementInputVectors[Mechanics::DISPLACEMENT][(3*i) + 1]<<" elementInputVectors["<<(3*i)+2<<"]="<<elementInputVectors[Mechanics::DISPLACEMENT][(3*i) + 2]<<std::endl;
        std::cout<<"elementInputVectors_pre["<<(3*i)+0<<"]="<<elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*i) + 0]<<" elementInputVectors_pre["<<(3*i)+1<<"]="<<elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*i) + 1]<<" elementInputVectors_pre["<<(3*i)+2<<"]="<<elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*i) + 2]<<std::endl;
      }
    }

    for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) {
      d_materialModel->preNonlinearAssemblyGaussPointOperation();

      double F_n[3][3], F_np1[3][3], F_np1o2[3][3], U_n[3][3], U_np1[3][3], U_np1o2[3][3], R_n[3][3], R_np1[3][3], R_np1o2[3][3];
      //double Identity[3][3], e_np1o2_tilda_rotated[3][3], Bl_np1[6][24], spin_np1[3][3], d_np1[3][3], el_np1[6], detF = 1.0, detF_np1, detF_n;
      double Identity[3][3], e_np1o2_tilda_rotated[3][3], Bl_np1[6][24], spin_np1[3][3], d_np1[3][3], el_np1[6], detF = 1.0;
      double dN_dxnp1o2[8], dN_dynp1o2[8], dN_dznp1o2[8], detJ_np1o2[1], d_np1o2[3][3], d_np1o2_temp[3][3];
     
      computeShapeFunctions(N, currXi[qp], currEta[qp], currZeta[qp]);

      // The value of radius is very important.
      //double radius = sqrt((xyz[qp](0) * xyz[qp](0)) + (xyz[qp](1) * xyz[qp](1)));

      // Constructing an identity matrix.
      for(unsigned int i = 0; i < 3; i++) {
        for(unsigned int j = 0; j < 3; j++) {
          Identity[i][j] = 0.0;
        }
        Identity[i][i] = 1.0;
      }
     
      // The algorithm given in Box-8.1 of the book Simo and Hughes (1998) is
      // implemented here. Some matrices are calculated in a different way.

      // The deformation gradients are computed in the next three lines.
      constructShapeFunctionDerivatives(dNdX, dNdY, dNdZ, refX, refY, refZ, currXi[qp], currEta[qp], currZeta[qp], detJ_0);
      //computeDeformationGradient(dphi, xyz_n, num_nodes, qp, F_n);
      computeGradient(dNdX, dNdY, dNdZ, prevX, prevY, prevZ, num_nodes, F_n);
      //computeDeformationGradient(dphi, xyz_np1, num_nodes, qp, F_np1);
      computeGradient(dNdX, dNdY, dNdZ, currX, currY, currZ, num_nodes, F_np1);
      //computeDeformationGradient(dphi, xyz_np1o2, num_nodes, qp, F_np1o2);
      computeGradient(dNdX, dNdY, dNdZ, x_np1o2, y_np1o2, z_np1o2, num_nodes, F_np1o2);
  
      if(d_useJaumannRate == false) {
        double difference = 0.0;
        for(int i = 0; i < 3; i++) {
          for(int j = 0; j < 3; j++) {
            difference += fabs((0.5*(F_n[i][j]+F_np1[i][j]))-F_np1o2[i][j]);
          }
        }
        //std::cout<<"difference in F's = "<<difference<<std::endl;

        if(d_useFlanaganTaylorElem == false) {
          // Polar decomposition (F=RU) of the deformation gradient is conducted here.
          //std::cout<<"Will do it for the n-th step."<<std::endl;
          polarDecompositionFeqRU_Simo(F_n, R_n, U_n);
          //polarDecomposeRU(F_n, R_n, U_n);
          polarDecompositionFeqRU_Simo(F_np1, R_np1, U_np1);
          //polarDecomposeRU(F_np1, R_np1, U_np1);
          polarDecompositionFeqRU_Simo(F_np1o2, R_np1o2, U_np1o2);
          //polarDecomposeRU(F_np1o2, R_np1o2, U_np1o2);
        } else {
          // The Flanagan-Taylor element formulation is being implemented here.
          double R_curr[3][3], V_curr[3][3], R_prev[3][3], V_prev[3][3];
          double L_curr[3][3], D_curr[3][3], W_curr[3][3], Omega[3][3], tempMat[3][3], invTempMat[3][3];
          double ImhO[3][3], IphO[3][3], invImhO[3][3], IphOR_prev[3][3];
          double DpWV[3][3], VO[3][3], delta_V[3][3], DV[3][3], VD[3][3], Z_curr[3][3];
          double z[3], w[3], omega[3], tempVec[3];
          double traceV_curr;

          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              R_curr[i][j] = d_rotationR_np1[(9 * d_gaussPtCnt) + (3 * i) + j];
              R_prev[i][j] = d_rotationR_n[(9 * d_gaussPtCnt) + (3 * i) + j];
              V_curr[i][j] = d_leftStretchV_np1[(9 * d_gaussPtCnt) + (3 * i) + j];
              V_prev[i][j] = d_leftStretchV_n[(9 * d_gaussPtCnt) + (3 * i) + j];
              Omega[i][j] = 0.0;
            }
          }

          constructShapeFunctionDerivatives(dNdx, dNdy, dNdz, currX, currY, currZ, currXi[qp], currEta[qp], currZeta[qp], detJ);
          computeGradient(dNdx, dNdy, dNdz, delta_u, delta_v, delta_w, num_nodes, L_curr);

          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              D_curr[i][j] = 0.5 * (L_curr[i][j] + L_curr[j][i]);
              W_curr[i][j] = 0.5 * (L_curr[i][j] - L_curr[j][i]);
            }
          }

          w[0] = W_curr[2][1];
          w[1] = W_curr[0][2];
          w[2] = W_curr[1][0];

          matMatMultiply(D_curr, V_curr, DV);

          matMatMultiply(V_curr, D_curr, VD);

          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              Z_curr[i][j] = DV[i][j] - VD[i][j];
              //AMP::pout<<"D_curr["<<i<<"]["<<j<<"]="<<D_curr[i][j]<<"    V_curr["<<i<<"]["<<j<<"]="<<V_curr[i][j]<<"    Z_curr["<<i<<"]["<<j<<"]="<<Z_curr[i][j]<<std::endl;
            }
          }

          z[0] = z[1] = z[2] = 0.0;
          for(int i = 0; i < 3; i++) {
            //z[0] = (D_curr[2][i] * V_curr[i][1]) - (D_curr[1][i] * V_curr[i][2]);
            z[0] = Z_curr[2][1];
            //z[1] = - (D_curr[2][i] * V_curr[i][0]) + (D_curr[0][i] * V_curr[i][2]);
            z[1] = Z_curr[0][2];
            //z[2] = (D_curr[1][i] * V_curr[i][0]) - (D_curr[0][i] * V_curr[i][1]);
            z[2] = Z_curr[1][0];
          }

          traceV_curr = V_curr[0][0] + V_curr[1][1] + V_curr[2][2];
          for(int i = 0; i < 3; i++) 
            for(int j = 0; j < 3; j++) 
              tempMat[i][j] = ((Identity[i][j] * traceV_curr) - V_curr[i][j]);

          matInverse(tempMat, invTempMat);

          matVecMultiply(invTempMat, z, tempVec);
          for(int i = 0; i < 3; i++) 
            tempVec[i] = 1.0 * tempVec[i];

          vecVecAddition(w, tempVec, omega);
          for(int i = 0; i < 3; i++) 
            omega[i] = 1.0 * omega[i];

          if(d_iDebugPrintInfoLevel > 11) {
            for(int i = 0; i < 3; i++) { 
              AMP::pout<<"tempVec["<<i<<"]="<<tempVec[i]<<"  z["<<i<<"]="<<z[i]<<std::endl;
            }
          }
        
          Omega[0][1] = -omega[2];
          Omega[0][2] = omega[1];
          Omega[1][0] = omega[2];
          Omega[1][2] = -omega[0];
          Omega[2][0] = -omega[1];
          Omega[2][1] = omega[0];

          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              ImhO[i][j] = Identity[i][j] - (0.5 * Omega[i][j]);
              IphO[i][j] = Identity[i][j] + (0.5 * Omega[i][j]);
            }
          }

          matMatMultiply(IphO, R_prev, IphOR_prev);

          matInverse(ImhO, invImhO);

          if(d_iDebugPrintInfoLevel > 11) {
            for(int i = 0; i < 3; i++) { 
              for(int j = 0; j < 3; j++) {
                AMP::pout<<"ImhO["<<i<<"]["<<j<<"]="<<ImhO[i][j]<<"  invImhO["<<i<<"]["<<j<<"]="<<invImhO[i][j]<<std::endl;
              }
            }
          }
        
          matMatMultiply(invImhO, IphOR_prev, R_curr);

          matMatMultiply(L_curr, V_curr, DpWV);

          matMatMultiply(V_curr, Omega, VO);

          for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
              delta_V[i][j] = DpWV[i][j] - VO[i][j];

          for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
              V_curr[i][j] = V_prev[i][j] + delta_V[i][j];

          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              //AMP::pout<<"delta_V["<<i<<"]["<<j<<"]="<<delta_V[i][j]<<"    DpWV["<<i<<"]["<<j<<"]="<<DpWV[i][j]<<"    VO["<<i<<"]["<<j<<"]="<<VO[i][j]<<std::endl;
              d_rotationR_np1[(9 * d_gaussPtCnt) + (3 * i) + j] = R_curr[i][j];
              d_leftStretchV_np1[(9 * d_gaussPtCnt) + (3 * i) + j] = V_curr[i][j];
              // The rotation matrices are updated here.
              R_n[i][j] = R_prev[i][j];
              R_np1o2[i][j] = R_curr[i][j]; // Mid-step rotation tensor gets the value of the current rotation tensor.
              R_np1[i][j] = R_curr[i][j];
            }
          }

          d_gaussPtCnt++;
        } // Flanagan-Taylor element formulation ends here.

        if(d_iDebugPrintInfoLevel > 11) {
          for(int i = 0; i < 3; i++) { 
            for(int j = 0; j < 3; j++) {
              //R_np1o2[i][j] = R_np1[i][j];
              AMP::pout<<"R_n["<<i<<"]["<<j<<"]="<<R_n[i][j]<<"  R_np1["<<i<<"]["<<j<<"]="<<R_np1[i][j]<<std::endl;
            }
          }
        }
        
        // Gradient of the incremental displacement with respect to the np1o2 configuration.
        constructShapeFunctionDerivatives(dN_dxnp1o2, dN_dynp1o2, dN_dznp1o2, x_np1o2, y_np1o2, z_np1o2, currXi[qp], currEta[qp], currZeta[qp], detJ_np1o2);
        
        // Calculate the rate of deformation tensor with respect to the np1o2 configuration.
        computeGradient(dN_dxnp1o2, dN_dynp1o2, dN_dznp1o2, delta_u, delta_v, delta_w, num_nodes, d_np1o2_temp);
        for(int i = 0; i < 3; i++) {
          for(int j = 0; j < 3; j++) {
            d_np1o2[i][j] = 0.5 * (d_np1o2_temp[i][j] + d_np1o2_temp[j][i]);
          }
        }

        // Rotate the rate of deformation to the unrotated configuration.
        pullbackCorotational(R_np1o2, d_np1o2, e_np1o2_tilda_rotated);
        if(d_iDebugPrintInfoLevel > 11) {
          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              AMP::pout<<"e_np1o2_tilda_rotated["<<i<<"]["<<j<<"]="<<e_np1o2_tilda_rotated[i][j]<<std::endl;
            }
          }
        }
      }

      // Calculate the derivatives of the shape functions at the current coordinate. 
      constructShapeFunctionDerivatives(dNdx, dNdy, dNdz, currX, currY, currZ, currXi[qp], currEta[qp], currZeta[qp], detJ);
      //constructShapeFunctionDerivatives(dNdx, dNdy, dNdz, prevX, prevY, prevZ, currXi[qp], currEta[qp], currZeta[qp], detJ);

      for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 24; j++) {
          Bl_np1[i][j] = 0.0;
        }
      }
      
      for(unsigned int i = 0; i < num_nodes; i++) {
        Bl_np1[0][(3 * i) + 0] = dNdx[i];
        Bl_np1[1][(3 * i) + 1] = dNdy[i];
        Bl_np1[2][(3 * i) + 2] = dNdz[i];
        Bl_np1[3][(3 * i) + 1] = dNdz[i];
        Bl_np1[3][(3 * i) + 2] = dNdy[i];
        Bl_np1[4][(3 * i) + 0] = dNdz[i];
        Bl_np1[4][(3 * i) + 2] = dNdx[i];
        Bl_np1[5][(3 * i) + 0] = dNdy[i];
        Bl_np1[5][(3 * i) + 1] = dNdx[i];
      }

      if(d_useJaumannRate == true) {
        computeGradient(dNdx, dNdy, dNdz, delta_u, delta_v, delta_w, num_nodes, d_np1);
        for(int i = 0; i < 3; i++) {
          for(int j = 0; j < 3; j++) {
            spin_np1[i][j] = 0.5 * (d_np1[j][i] - d_np1[i][j]);
          }
        }

        for(int i = 0; i < 6; i++) {
          el_np1[i] = 0.0;
        }
  
        for(int i = 0; i < 6; i++) {
          for(int j = 0; j < 8; j++) {
            el_np1[i] += (Bl_np1[i][(3 * j) + 0] * delta_u[j]);
            el_np1[i] += (Bl_np1[i][(3 * j) + 1] * delta_v[j]);
            el_np1[i] += (Bl_np1[i][(3 * j) + 2] * delta_w[j]);
          }
        }

        //detF_np1 = matDeterminant(F_np1);
        //detF_n = matDeterminant(F_n);
        //detF = detF_np1 / detF_n;
        detF = 1.0; // Based on a test by Bathe, this transformation is not required. So temporary hack, passing unity.
      }

/*      for(int i = 0; i < num_nodes; i++) {
        std::cout<<"Nonlinear-->dN_dxn["<<i<<"]="<<dN_dxn[i]<<" dN_dyn["<<i<<"]="<<dN_dyn[i]<<" dN_dzn["<<i<<"]="<<dN_dzn[i]<<std::endl;
      }
*/

      // Check whether the strain increment tensor is symmetric or not.
      if(d_useJaumannRate == false) {
        double check_symmetry = 0.0, tol = 1.0e-7;
        check_symmetry += ((e_np1o2_tilda_rotated[0][1] - e_np1o2_tilda_rotated[1][0]) * (e_np1o2_tilda_rotated[0][1] - e_np1o2_tilda_rotated[1][0]));
        check_symmetry += ((e_np1o2_tilda_rotated[0][2] - e_np1o2_tilda_rotated[2][0]) * (e_np1o2_tilda_rotated[0][2] - e_np1o2_tilda_rotated[2][0]));
        check_symmetry += ((e_np1o2_tilda_rotated[1][2] - e_np1o2_tilda_rotated[2][1]) * (e_np1o2_tilda_rotated[1][2] - e_np1o2_tilda_rotated[2][1]));
        AMP_INSIST((check_symmetry < tol), "The incremental strain tensor for updated lagrangian is not symmetric.");
      }
  
      std::vector<std::vector<double> > fieldsAtGaussPt(Mechanics::TOTAL_NUMBER_OF_VARIABLES); 

      //Incremental Strain
      if(d_useJaumannRate == false) {
        if(d_iDebugPrintInfoLevel > 11) {
          for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
              AMP::pout<<"e_np1o2_tilda_rotated["<<i<<"]["<<j<<"]="<<e_np1o2_tilda_rotated[i][j]<<std::endl;
          AMP::pout<<"\n"<<std::endl;
        }

        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(e_np1o2_tilda_rotated[0][0]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(e_np1o2_tilda_rotated[1][1]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(e_np1o2_tilda_rotated[2][2]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * (e_np1o2_tilda_rotated[1][2] + e_np1o2_tilda_rotated[2][1]));
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * (e_np1o2_tilda_rotated[0][2] + e_np1o2_tilda_rotated[2][0]));
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * (e_np1o2_tilda_rotated[0][1] + e_np1o2_tilda_rotated[1][0]));
      }

      if(d_useJaumannRate == true) {
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[0]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[1]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[2]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * el_np1[3]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * el_np1[4]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * el_np1[5]);
      }
      if(d_iDebugPrintInfoLevel > 11) {
        for(int i = 0; i < 6; i++) {
          std::cout<<"strain["<<i<<"]="<<fieldsAtGaussPt[Mechanics::DISPLACEMENT][i]<<std::endl;
        }
      }

      if( !(elementInputVectors[Mechanics::TEMPERATURE].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::TEMPERATURE][k] * N[k]);
        } //end for k
        //valAtGaussPt = ((valAtGaussPt - 301.0) * (1.25 - (radius * radius))) + 301.0;
        fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::BURNUP].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::BURNUP][k] * N[k]);
        } //end for k
        //valAtGaussPt = (valAtGaussPt / 2.0) * exp(radius);
        fieldsAtGaussPt[Mechanics::BURNUP].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k] * N[k]);
        } //end for k
        fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::LHGR].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::LHGR][k] * N[k]);
        } //end for k
        fieldsAtGaussPt[Mechanics::LHGR].push_back(valAtGaussPt);
      }

      /* Compute Stress Corresponding to Given Strain, Temperature, Burnup, ... */
      double* currStress;
      if(d_useJaumannRate == false) {
        d_materialModel->getInternalStress_UL(fieldsAtGaussPt, currStress, R_n, R_np1, detF);
        if(d_iDebugPrintInfoLevel > 9) {
          AMP::pout<<"currStress[0] = "<<currStress[0]<<std::endl;
          AMP::pout<<"currStress[1] = "<<currStress[1]<<std::endl;
          AMP::pout<<"currStress[2] = "<<currStress[2]<<std::endl;
          AMP::pout<<"currStress[3] = "<<currStress[3]<<std::endl;
          AMP::pout<<"currStress[4] = "<<currStress[4]<<std::endl;
          AMP::pout<<"currStress[5] = "<<currStress[5]<<std::endl;
          AMP::pout<<"el_np1[0] = "<<d_np1o2[0][0]<<std::endl;
          AMP::pout<<"el_np1[1] = "<<d_np1o2[1][1]<<std::endl;
          AMP::pout<<"el_np1[2] = "<<d_np1o2[2][2]<<std::endl;
          AMP::pout<<"el_np1[3] = "<<(d_np1o2[1][2] + d_np1o2[2][1])<<std::endl;
          AMP::pout<<"el_np1[4] = "<<(d_np1o2[0][2] + d_np1o2[2][0])<<std::endl;
          AMP::pout<<"el_np1[5] = "<<(d_np1o2[0][1] + d_np1o2[1][0])<<std::endl;
        }
      } else {
        d_materialModel->getInternalStress_UL(fieldsAtGaussPt, currStress, spin_np1, Identity, detF);
        if(d_iDebugPrintInfoLevel > 9) {
          AMP::pout<<"currStress[0] = "<<currStress[0]<<std::endl;
          AMP::pout<<"currStress[1] = "<<currStress[1]<<std::endl;
          AMP::pout<<"currStress[2] = "<<currStress[2]<<std::endl;
          AMP::pout<<"currStress[3] = "<<currStress[3]<<std::endl;
          AMP::pout<<"currStress[4] = "<<currStress[4]<<std::endl;
          AMP::pout<<"currStress[5] = "<<currStress[5]<<std::endl;
          AMP::pout<<"el_np1[0] = "<<el_np1[0]<<std::endl;
          AMP::pout<<"el_np1[1] = "<<el_np1[1]<<std::endl;
          AMP::pout<<"el_np1[2] = "<<el_np1[2]<<std::endl;
          AMP::pout<<"el_np1[3] = "<<el_np1[3]<<std::endl;
          AMP::pout<<"el_np1[4] = "<<el_np1[4]<<std::endl;
          AMP::pout<<"el_np1[5] = "<<el_np1[5]<<std::endl;
          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              std::cout<<"Spin["<<i<<"]["<<j<<"]="<<spin_np1[i][j]<<std::endl;
            }
          }
          std::cout<<"\n\n"<<std::endl;
        }
      }
      
      for(unsigned int j = 0; j < num_nodes; j++) {
        for(unsigned int d = 0; d < 3; d++) {
          double tmp = 0.0;
          for(unsigned int m = 0; m < 6; m++) {
            tmp += (Bl_np1[m][(3 * j) + d] * currStress[m]);
          }

          elementOutputVector[(3*j) + d] += (detJ[0] * tmp);

        }//end for d
      }//end for j

      d_materialModel->postNonlinearAssemblyGaussPointOperation();
    }//end for qp

    d_materialModel->postNonlinearAssemblyElementOperation();
  }

  void MechanicsNonlinearUpdatedLagrangianElement :: apply_Reduced() 
  {
    //AMP_INSIST((d_useJaumannRate == true), "Reduced integration has only been implemented for Jaumann rate.");
    
    std::vector<std::vector<double> > & elementInputVectors = d_elementInputVectors;

    std::vector<std::vector<double> > & elementInputVectors_pre = d_elementInputVectors_pre;

    std::vector<double> & elementOutputVector = (*d_elementOutputVector);

    std::vector<Point> xyz, xyz_n, xyz_np1, xyz_np1o2;

    d_fe->reinit(d_elem);

    d_materialModel->preNonlinearAssemblyElementOperation();

    const unsigned int num_nodes = d_elem->n_nodes();

    xyz.resize(num_nodes);
    xyz_n.resize(num_nodes);
    xyz_np1.resize(num_nodes);
    xyz_np1o2.resize(num_nodes);

    double refX[8], refY[8], refZ[8];
    double dNdX[8], dNdY[8], dNdZ[8], detJ_0[1];

    Point p1;
    for(unsigned int ijk = 0; ijk < num_nodes; ijk++) {
      p1 = d_elem->point(ijk);
      //xyz[ijk] = p1;
      refX[ijk] = xyz[ijk](0) = d_elementRefXYZ[(3 * ijk) + 0];
      refY[ijk] = xyz[ijk](1) = d_elementRefXYZ[(3 * ijk) + 1];
      refZ[ijk] = xyz[ijk](2) = d_elementRefXYZ[(3 * ijk) + 2];
    }

    double currX[8], currY[8], currZ[8], dNdx[8], dNdy[8], dNdz[8], detJ[1], delta_u[8], delta_v[8], delta_w[8], N[8];
    double x_np1o2[8], y_np1o2[8], z_np1o2[8], prevX[8], prevY[8], prevZ[8];
    double rsq3 = (1.0 / sqrt(3.0));
    const double currXi[8] = {-rsq3, rsq3, -rsq3, rsq3, -rsq3, rsq3, -rsq3, rsq3};
    const double currEta[8] = {-rsq3, -rsq3, rsq3, rsq3, -rsq3, -rsq3, rsq3, rsq3};
    const double currZeta[8] = {-rsq3, -rsq3, -rsq3, -rsq3, rsq3, rsq3, rsq3, rsq3};
    double Bl_np1_bar[6][24], sum_detJ, Bl_center[6][24];
    
    for(unsigned int ijk = 0; ijk < num_nodes; ijk++) {
      prevX[ijk] = xyz_n[ijk](0) = xyz[ijk](0) + elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 0];
      prevY[ijk] = xyz_n[ijk](1) = xyz[ijk](1) + elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 1];
      prevZ[ijk] = xyz_n[ijk](2) = xyz[ijk](2) + elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 2];

      currX[ijk] = xyz_np1[ijk](0) = xyz[ijk](0) + elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 0];
      currY[ijk] = xyz_np1[ijk](1) = xyz[ijk](1) + elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 1];
      currZ[ijk] = xyz_np1[ijk](2) = xyz[ijk](2) + elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 2];

      delta_u[ijk] = elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 0] - elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 0];
      delta_v[ijk] = elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 1] - elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 1];
      delta_w[ijk] = elementInputVectors[Mechanics::DISPLACEMENT][(3*ijk) + 2] - elementInputVectors_pre[Mechanics::DISPLACEMENT][(3*ijk) + 2];

      x_np1o2[ijk] = xyz_np1o2[ijk](0) = xyz_n[ijk](0) + (delta_u[ijk] / 2.0);
      y_np1o2[ijk] = xyz_np1o2[ijk](1) = xyz_n[ijk](1) + (delta_v[ijk] / 2.0);
      z_np1o2[ijk] = xyz_np1o2[ijk](2) = xyz_n[ijk](2) + (delta_w[ijk] / 2.0);
    }

    if(d_iDebugPrintInfoLevel > 11) {
      for(int i = 0; i < 8; i++) {
        std::cout<<"delta_u["<<i<<"]="<<delta_u[i]<<" delta_v["<<i<<"]="<<delta_v[i]<<" delta_w["<<i<<"]="<<delta_w[i]<<std::endl;
      }
    }

    double ebar_np1o2[3][3], sum_detJ_np1o2 = 0.0, avg_dil_strain = 0.0;
    if(d_useJaumannRate == false) {
      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          ebar_np1o2[i][j] = 0.0;
        }
      }

      for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) {
        double F_np1o2[3][3], R_np1o2[3][3], U_np1o2[3][3];
        //double dN_dxnp1o2[8], dN_dynp1o2[8], dN_dznp1o2[8], detJ_np1o2[1], d_np1o2[3][3], d_np1o2_temp[3][3], e_np1o2_tilda_rotated[3][3];
        double dN_dxnp1o2[8], dN_dynp1o2[8], dN_dznp1o2[8], detJ_np1o2[1], d_np1o2_temp[3][3];
  
        // The deformation gradients are computed in the next three lines.
        constructShapeFunctionDerivatives(dNdX, dNdY, dNdZ, refX, refY, refZ, currXi[qp], currEta[qp], currZeta[qp], detJ_0);
        //computeDeformationGradient(dphi, xyz_np1o2, num_nodes, qp, F_np1o2);
        computeGradient(dNdX, dNdY, dNdZ, x_np1o2, y_np1o2, z_np1o2, num_nodes, F_np1o2);
  
        if(d_useFlanaganTaylorElem == false) {
          // Polar decomposition (F=RU) of the deformation gradient is conducted here.
          //std::cout<<"Will do it for the n-th step."<<std::endl;
          polarDecompositionFeqRU_Simo(F_np1o2, R_np1o2, U_np1o2);
          //polarDecomposeRU(F_np1o2, R_np1o2, U_np1o2);
        }
        
        // Gradient of the incremental displacement with respect to the np1o2 configuration.
        constructShapeFunctionDerivatives(dN_dxnp1o2, dN_dynp1o2, dN_dznp1o2, x_np1o2, y_np1o2, z_np1o2, currXi[qp], currEta[qp], currZeta[qp], detJ_np1o2);
        
        // Calculate the rate of deformation tensor with respect to the np1o2 configuration.
        computeGradient(dN_dxnp1o2, dN_dynp1o2, dN_dznp1o2, delta_u, delta_v, delta_w, num_nodes, d_np1o2_temp);

        for(int i = 0; i < 3; i++) {
          ebar_np1o2[i][i] += (d_np1o2_temp[i][i] * detJ_np1o2[0]);
        }
        sum_detJ_np1o2 += detJ_np1o2[0];
      }

      for(int i = 0; i < 3; i++) {
        ebar_np1o2[i][i] = ebar_np1o2[i][i] / sum_detJ_np1o2;
      }
      avg_dil_strain = ((1.0 / 3.0) * (ebar_np1o2[0][0] + ebar_np1o2[1][1] + ebar_np1o2[2][2]));
    } // end of if - condition......

    sum_detJ = 0.0;
    for(unsigned int i = 0; i < 6; i++) {
      for(unsigned int j = 0; j < (3 * num_nodes); j++) {
        Bl_np1_bar[i][j] = 0.0;
        Bl_center[i][j] = 0.0;
      }
    }

    for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) {
      constructShapeFunctionDerivatives(dNdx, dNdy, dNdz, currX, currY, currZ, currXi[qp], currEta[qp], currZeta[qp], detJ);
      sum_detJ += detJ[0];

      for(unsigned int i = 0; i < 8; i++) {
        Bl_np1_bar[0][(3 * i) + 0] += (dNdx[i] * detJ[0]);
        Bl_np1_bar[1][(3 * i) + 0] += (dNdx[i] * detJ[0]);
        Bl_np1_bar[2][(3 * i) + 0] += (dNdx[i] * detJ[0]);

        Bl_np1_bar[0][(3 * i) + 1] += (dNdy[i] * detJ[0]);
        Bl_np1_bar[1][(3 * i) + 1] += (dNdy[i] * detJ[0]);
        Bl_np1_bar[2][(3 * i) + 1] += (dNdy[i] * detJ[0]);

        Bl_np1_bar[0][(3 * i) + 2] += (dNdz[i] * detJ[0]);
        Bl_np1_bar[1][(3 * i) + 2] += (dNdz[i] * detJ[0]);
        Bl_np1_bar[2][(3 * i) + 2] += (dNdz[i] * detJ[0]);
      }
    }

    if(d_iDebugPrintInfoLevel > 11) {
      std::cout<<"sum_detJ="<<sum_detJ<<std::endl;
    }

    double one3TimesSumDetJ = 1.0 / (3.0 * sum_detJ);
    for(unsigned int i = 0; i < 6; i++) {
      for(unsigned int j = 0; j < (3 * num_nodes); j++) {
        Bl_np1_bar[i][j] = Bl_np1_bar[i][j] * one3TimesSumDetJ;

        if(d_iDebugPrintInfoLevel > 11) {
          std::cout<<"Bl_np1_bar["<<i<<"]["<<j<<"]="<<Bl_np1_bar[i][j]<<std::endl;
        }
      }
    }

    constructShapeFunctionDerivatives(dNdx, dNdy, dNdz, currX, currY, currZ, 0.0, 0.0, 0.0, detJ);
    for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) {
      Bl_center[0][(3 * qp) + 0] += (dNdx[qp]);
      Bl_center[1][(3 * qp) + 1] += (dNdy[qp]);
      Bl_center[2][(3 * qp) + 2] += (dNdz[qp]);
      Bl_center[3][(3 * qp) + 1] += (dNdz[qp]);
      Bl_center[3][(3 * qp) + 2] += (dNdy[qp]);
      Bl_center[4][(3 * qp) + 0] += (dNdz[qp]);
      Bl_center[4][(3 * qp) + 2] += (dNdx[qp]);
      Bl_center[5][(3 * qp) + 0] += (dNdy[qp]);
      Bl_center[5][(3 * qp) + 1] += (dNdx[qp]);
    }

    for(unsigned int qp = 0; qp < d_qrule->n_points(); qp++) {
      d_materialModel->preNonlinearAssemblyGaussPointOperation();

      double Bl_np1[6][24], d_np1[3][3], spin_np1[3][3], el_np1[6], Bl_dil[6][24], F_np1o2[3][3], R_np1o2[3][3], U_np1o2[3][3];
      double F_n[3][3], R_n[3][3], U_n[3][3], F_np1[3][3], R_np1[3][3], U_np1[3][3];
      double dN_dxnp1o2[8], dN_dynp1o2[8], dN_dznp1o2[8], detJ_np1o2[1], d_np1o2[3][3], d_np1o2_temp[3][3], e_np1o2_tilda_rotated[3][3];
      double detF = 1.0;

      computeShapeFunctions(N, currXi[qp], currEta[qp], currZeta[qp]);

      // The deformation gradients are computed in the next three lines.
      constructShapeFunctionDerivatives(dNdX, dNdY, dNdZ, refX, refY, refZ, currXi[qp], currEta[qp], currZeta[qp], detJ_0);
      //computeDeformationGradient(dphi, xyz_n, num_nodes, qp, F_n);
      computeGradient(dNdX, dNdY, dNdZ, prevX, prevY, prevZ, num_nodes, F_n);
      //computeDeformationGradient(dphi, xyz_np1, num_nodes, qp, F_np1);
      computeGradient(dNdX, dNdY, dNdZ, currX, currY, currZ, num_nodes, F_np1);
      //computeDeformationGradient(dphi, xyz_np1o2, num_nodes, qp, F_np1o2);
      computeGradient(dNdX, dNdY, dNdZ, x_np1o2, y_np1o2, z_np1o2, num_nodes, F_np1o2);
  
      if(d_useJaumannRate == false) {
        if(d_useFlanaganTaylorElem == false) {
          // Polar decomposition (F=RU) of the deformation gradient is conducted here.
          //std::cout<<"Will do it for the n-th step."<<std::endl;
          polarDecompositionFeqRU_Simo(F_n, R_n, U_n);
          //polarDecomposeRU(F_n, R_n, U_n);
          polarDecompositionFeqRU_Simo(F_np1, R_np1, U_np1);
          //polarDecomposeRU(F_np1, R_np1, U_np1);
          polarDecompositionFeqRU_Simo(F_np1o2, R_np1o2, U_np1o2);
          //polarDecomposeRU(F_np1o2, R_np1o2, U_np1o2);
        }
        
        // Gradient of the incremental displacement with respect to the np1o2 configuration.
        constructShapeFunctionDerivatives(dN_dxnp1o2, dN_dynp1o2, dN_dznp1o2, x_np1o2, y_np1o2, z_np1o2, currXi[qp], currEta[qp], currZeta[qp], detJ_np1o2);
        
        // Calculate the rate of deformation tensor with respect to the np1o2 configuration.
        computeGradient(dN_dxnp1o2, dN_dynp1o2, dN_dznp1o2, delta_u, delta_v, delta_w, num_nodes, d_np1o2_temp);
        for(int i = 0; i < 3; i++) {
          for(int j = 0; j < 3; j++) {
            d_np1o2[i][j] = 0.5 * (d_np1o2_temp[i][j] + d_np1o2_temp[j][i]);
          }
        }

        double term1 = (1.0 / 3.0) * (d_np1o2[0][0] + d_np1o2[1][1] + d_np1o2[2][2]);
        for(int i = 0; i < 3; i++) {
          d_np1o2[i][i] += (avg_dil_strain - term1);
        }

        // Rotate the rate of deformation to the unrotated configuration.
        pullbackCorotational(R_np1o2, d_np1o2, e_np1o2_tilda_rotated);
        if(d_iDebugPrintInfoLevel > 11) {
          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              AMP::pout<<"e_np1o2_tilda_rotated["<<i<<"]["<<j<<"]="<<e_np1o2_tilda_rotated[i][j]<<std::endl;
            }
          }
        }

        //double term1 = (1.0 / 3.0) * (e_np1o2_tilda_rotated[0][0] + e_np1o2_tilda_rotated[1][1] + e_np1o2_tilda_rotated[2][2]);
        //for(int i = 0; i < 3; i++) {
        //  e_np1o2_tilda_rotated[i][i] += (avg_dil_strain - term1);
        //}
      } // end of if(d_useJaumannRate == false)

      // Calculate the derivatives of the shape functions at the current coordinate.
      constructShapeFunctionDerivatives(dNdx, dNdy, dNdz, currX, currY, currZ, currXi[qp], currEta[qp], currZeta[qp], detJ);

      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          d_np1[i][j] = 0.0;
          spin_np1[i][j] = 0.0;
        }
      }

      for(int i = 0; i < 6; i++) {
        el_np1[i] = 0.0;
        for(int j = 0; j < 24; j++) {
          Bl_np1[i][j] = 0.0;
          Bl_dil[i][j] = 0.0;
        }
      }
      
      for(unsigned int i = 0; i < num_nodes; i++) {
        Bl_np1[0][(3 * i) + 0] = dNdx[i];
        Bl_np1[1][(3 * i) + 1] = dNdy[i];
        Bl_np1[2][(3 * i) + 2] = dNdz[i];
        Bl_np1[3][(3 * i) + 1] = dNdz[i];
        Bl_np1[3][(3 * i) + 2] = dNdy[i];
        Bl_np1[4][(3 * i) + 0] = dNdz[i];
        Bl_np1[4][(3 * i) + 2] = dNdx[i];
        Bl_np1[5][(3 * i) + 0] = dNdy[i];
        Bl_np1[5][(3 * i) + 1] = dNdx[i];

        double one3 = 1.0 / 3.0;
        for(int j = 0; j < 3; j++) {
          Bl_dil[j][(3 * i) + 0] = dNdx[i] * one3;
          Bl_dil[j][(3 * i) + 1] = dNdy[i] * one3;
          Bl_dil[j][(3 * i) + 2] = dNdz[i] * one3;
        }
      }
      
      // Calculate the spin tensor for the jaumann rate.
      computeGradient(dNdx, dNdy, dNdz, delta_u, delta_v, delta_w, num_nodes, d_np1);
      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          spin_np1[i][j] = 0.5 * (d_np1[j][i] - d_np1[i][j]);
        }
      }

/*      for(int i = 0; i < 3; i++) {
        for(unsigned int j = 0; j < (3 * num_nodes); j++) {
          Bl_np1[i][j] = Bl_np1[i][j] - Bl_dil[i][j] + Bl_center[i][j];
        }
      }
*/      
      for(int i = 0; i < 6; i++) {
        for(unsigned int j = 0; j < (3 * num_nodes); j++) {
          Bl_np1[i][j] = Bl_np1[i][j] - Bl_dil[i][j] + Bl_np1_bar[i][j];

          if(d_iDebugPrintInfoLevel > 11) {
            std::cout<<"Bl_np1["<<i<<"]["<<j<<"]="<<Bl_np1[i][j]<<std::endl;
          }
        }
      }

      if(d_onePointShearIntegration) {
        for(int i = 3; i < 6; i++) {
          for(unsigned int j = 0; j < (3 * num_nodes); j++) {
            Bl_np1[i][j] = Bl_center[i][j];
          }
        }
      }

      for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 8; j++) {
          el_np1[i] += (Bl_np1[i][(3 * j) + 0] * delta_u[j]);
          el_np1[i] += (Bl_np1[i][(3 * j) + 1] * delta_v[j]);
          el_np1[i] += (Bl_np1[i][(3 * j) + 2] * delta_w[j]);
        }
      }

      /* Compute Strain From Given Displacement */

      std::vector<std::vector<double> > fieldsAtGaussPt(Mechanics::TOTAL_NUMBER_OF_VARIABLES); 

      //Strain
      if(d_useJaumannRate == true) {
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[0]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[1]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[2]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[3]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[4]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(el_np1[5]);
      }
 
      if(d_useJaumannRate == false) {
        if(d_iDebugPrintInfoLevel > 11) {
          for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
              AMP::pout<<"e_np1o2_tilda_rotated["<<i<<"]["<<j<<"]="<<e_np1o2_tilda_rotated[i][j]<<std::endl;
          AMP::pout<<"\n"<<std::endl;
        }

        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(e_np1o2_tilda_rotated[0][0]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(e_np1o2_tilda_rotated[1][1]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(e_np1o2_tilda_rotated[2][2]);
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * (e_np1o2_tilda_rotated[1][2] + e_np1o2_tilda_rotated[2][1]));
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * (e_np1o2_tilda_rotated[0][2] + e_np1o2_tilda_rotated[2][0]));
        fieldsAtGaussPt[Mechanics::DISPLACEMENT].push_back(1.0 * (e_np1o2_tilda_rotated[0][1] + e_np1o2_tilda_rotated[1][0]));
      }

      if( !(elementInputVectors[Mechanics::TEMPERATURE].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::TEMPERATURE][k] * N[k]);
        } //end for k
        fieldsAtGaussPt[Mechanics::TEMPERATURE].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::BURNUP].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::BURNUP][k] * N[k]);
        } //end for k
        fieldsAtGaussPt[Mechanics::BURNUP].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::OXYGEN_CONCENTRATION].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][k] * N[k]);
        } //end for k
        fieldsAtGaussPt[Mechanics::OXYGEN_CONCENTRATION].push_back(valAtGaussPt);
      }

      if( !(elementInputVectors[Mechanics::LHGR].empty()) ) {
        double valAtGaussPt = 0;
        for(unsigned int k = 0; k < num_nodes; k++) {
          valAtGaussPt += (elementInputVectors[Mechanics::LHGR][k] * N[k]);
        } //end for k
        fieldsAtGaussPt[Mechanics::LHGR].push_back(valAtGaussPt);
      }

      // Constructing an identity matrix.
      double Identity[3][3];
      for(unsigned int i = 0; i < 3; i++) {
        for(unsigned int j = 0; j < 3; j++) {
          Identity[i][j] = 0.0;
        }
        Identity[i][i] = 1.0;
      }
     
      double* currStress;
      /* Compute Stress Corresponding to Given Strain, Temperature, Burnup, ... */
      if(d_useJaumannRate == false) {
        d_materialModel->getInternalStress_UL(fieldsAtGaussPt, currStress, R_n, R_np1, detF);
      } else {
        d_materialModel->getInternalStress_UL(fieldsAtGaussPt, currStress, spin_np1, Identity, detF);
      }

      for(unsigned int j = 0; j < num_nodes; j++) {
        for(unsigned int d = 0; d < 3; d++) {
          double tmp = 0.0;
          for(unsigned int m = 0; m < 6; m++) {
            tmp += (Bl_np1[m][(3 * j) + d] * currStress[m]);
          }

          elementOutputVector[(3*j) + d] += (detJ[0] * tmp);
        }//end for d
      }//end for j

      d_materialModel->postNonlinearAssemblyGaussPointOperation();
    }//end for qp

    d_materialModel->postNonlinearAssemblyElementOperation();
  }

  void MechanicsNonlinearUpdatedLagrangianElement :: computeDeformationGradient(const std::vector<std::vector<RealGradient> > & dphi, 
      const std::vector<Point> xyz, unsigned int num_nodes, unsigned int qp, double F[3][3])
  {
    for(unsigned int i = 0; i < 3; i++) {
      for(unsigned int j = 0; j < 3; j++) {
        F[i][j] = 0.0;
      }
    }

    for(unsigned int k = 0; k < num_nodes; k++) {
      F[0][0] += (xyz[k](0) * dphi[k][qp](0));
      F[0][1] += (xyz[k](0) * dphi[k][qp](1));
      F[0][2] += (xyz[k](0) * dphi[k][qp](2));
      F[1][0] += (xyz[k](1) * dphi[k][qp](0));
      F[1][1] += (xyz[k](1) * dphi[k][qp](1));
      F[1][2] += (xyz[k](1) * dphi[k][qp](2));
      F[2][0] += (xyz[k](2) * dphi[k][qp](0));
      F[2][1] += (xyz[k](2) * dphi[k][qp](1));
      F[2][2] += (xyz[k](2) * dphi[k][qp](2));
    }
  }

  void MechanicsNonlinearUpdatedLagrangianElement :: initializeReferenceXYZ(std::vector<double> & elementRefXYZ)
  {
    std::vector<Point> xyz;

    d_fe->reinit(d_elem);

    const unsigned int num_nodes = d_elem->n_nodes();

    xyz.resize(num_nodes);
    
    Point p1;
    for(unsigned int ijk = 0; ijk < num_nodes; ijk++) {
      p1 = d_elem->point(ijk);
      xyz[ijk] = p1;

      elementRefXYZ[(3 * ijk) + 0] = xyz[ijk](0);
      elementRefXYZ[(3 * ijk) + 1] = xyz[ijk](1);
      elementRefXYZ[(3 * ijk) + 2] = xyz[ijk](2);
    }

  }

  void MechanicsNonlinearUpdatedLagrangianElement :: preNonlinearElementInit()
  {
    d_leftStretchV_n.clear();

    d_leftStretchV_np1.clear();

    d_rotationR_n.clear();

    d_rotationR_np1.clear();

    d_gaussPtCnt = 0;
  }

}
}

