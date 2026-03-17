// Code developed by Huibin Ke from UW-Madison for the evolution of Mn-Ni-Si
// precipitates in Reactor Pressure Vessel stees.
#include <cvode/cvode.h> /* main integrator header file */
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
// #include <cvode/cvode_band.h>        /* band solver header */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fct. and macros */
#include <sundials/sundials_math.h>  /* contains the macros ABS, SQR, and EXP */
#include <sundials/sundials_types.h> /* definition of realtype */

// 现代 SUNDIALS 使用线性代数模块替代了旧的 cvode_band.h
#include <sunlinsol/sunlinsol_band.h> /* 线性解法器定义 */
#include <sunmatrix/sunmatrix_band.h> /* 矩阵定义 */

#include <iostream>

#include "Constants.h" /*Constants header file*/
#include "Input.h"     /*Input parameter header file*/

InputCondition* ICond;    /*Defined in Input.h, including irradiation conditions*/
InputMaterial* IMaterial; /*Defined in Input.h, including material information*/
InputProperty*
    IProp; /*Defined in Input.h, including all other parameters used in model*/

using namespace std;

/*function defs*/

struct UserDataType {
  realtype* size;
  realtype** radClust;
  realtype** beta;
  realtype** delG;
  realtype* J;

  // 构造函数：安全地分块动态分配内存
  UserDataType() {
    size = new realtype[numClass]();
    radClust = new realtype*[numPhase];
    beta = new realtype*[numPhase];
    delG = new realtype*[numPhase];
    
    for(int p = 0; p < numPhase; p++) {
      radClust[p] = new realtype[numClass]();
      beta[p] = new realtype[numClass]();
      delG[p] = new realtype[numClass]();
    }
    J = new realtype[numCalcPhase * (numClass + 1)]();
  }

  // 析构函数：防止内存泄漏
  ~UserDataType() {
    delete[] size;
    for(int p = 0; p < numPhase; p++) {
      delete[] radClust[p];
      delete[] beta[p];
      delete[] delG[p];
    }
    delete[] radClust;
    delete[] beta;
    delete[] delG;
    delete[] J;
  }
};

typedef UserDataType* UserData;

// 替换这 5 行声明
static void loadData(UserData data);
static void getbeta(realtype* size, realtype** beta);
static void getSize(realtype* size);
static void getRadClust(realtype* size, realtype** radClust);
static void getDelG(realtype* size, realtype** radClust, realtype** delG);
static void getInitVals(realtype y0[neq]);
static void initParams();
static void GetRED(realtype D[numComp], realtype Flux);
static void printYVector(N_Vector y);
static void getOutput(N_Vector y, realtype radM1[numCalcPhase],
                      realtype radM2[numCalcPhase],
                      realtype rhoC[numCalcPhase]);
static void getFlux(UserData data, N_Vector y, realtype J[]);
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
void int_to_string(int i, string &a, int base);

/*global problem parameters*/
realtype D[numComp], aP[numPhase],
    nu[numPhase][numComp]; /*Diffusion coefficient,effective precipitate lattice
                              constant, square of precipitate composition*/
realtype rhoC[numCalcPhase], radM1[numCalcPhase],
    radM2[numCalcPhase];          /*precipitate number density, two kinds of mean radius
                                     (see readme for more detail)*/
realtype Flux, solProd[numPhase]; /*Irradiation flux, solute product */

int main() {
  // ✅ 彻底解决栈溢出：使用 vector 在堆上分配 1.6MB 的数组
  std::vector<realtype> y0data_vec(neq, ZERO);
  realtype* y0data = y0data_vec.data();
  realtype t;

  realtype tout = 1E0;
  double ts = 0.0;

  N_Vector y0 = NULL;
  UserData data = NULL;
  void *cvode_mem = NULL;
  int flag, mxsteps = 2000000;
  long int mu = 3, ml = 3;
  realtype *yd;

  SUNContext sunctx;
  SUNContext_Create(SUN_COMM_NULL, &sunctx);

  // ✅ 抛弃 malloc，改用 C++ 的 new，确保内存大小分配绝对精准且防空指针
  ICond = new InputCondition();
  IMaterial = new InputMaterial();
  IProp = new InputProperty();
  
  LoadInput(ICond, IMaterial, IProp);
  initParams();
  getInitVals(y0data);
  y0 = N_VMake_Serial(neq, y0data, sunctx);

  /*Write the output file*/
  ofstream O_file;
  O_file.open("Output");
  O_file << "Run\tCalcTime(s)\tTime(s)\tFluence(n/m2s)\t";
  for (int p = 0; p < numPhase; p++) {
    string phaseStr;
    int_to_string(p + 1, phaseStr, 10);
    cout << p << " " << p + 1 << " " << phaseStr << endl;
    O_file << "Mean_Radius_of_Phase_" + phaseStr +
                  "_Homo(m)\tNumber_Density_of_Phase_" + phaseStr +
                  "_Homo(1/m3)\t";
  }
  for (int p = 0; p < numPhase; p++) {
    string phaseStr;
    int_to_string(p + 1, phaseStr, 10);
    O_file << "Mean_Radius_of_Phase_" + phaseStr +
                  "_Heter(m)\tNumber_Density_of_Phase_" + phaseStr +
                  "_Heter(1/m3)\t";
  }
  O_file << "Mn\tNi\tSi" << endl;

  /*Solving all the equations*/
  for (int i = 0; i < runs; i++) {
    cvode_mem = CVodeCreate(CV_BDF, sunctx);
    
    // ✅ 用 new 安全创建 data 对象，每次循环重新分配
    data = new UserDataType();
    loadData(data); // 此时进入 loadData，指针绝对是安全的了
    
    flag = CVodeSetUserData(cvode_mem, data);
    flag = CVodeInit(cvode_mem, f, T0, y0);
    flag = CVodeSStolerances(cvode_mem, RTOL, ATOL);

    SUNMatrix A = SUNBandMatrix(neq, mu, ml, sunctx);
    SUNLinearSolver LS = SUNLinSol_Band(y0, A, sunctx);
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    flag = CVodeSetMaxNumSteps(cvode_mem, mxsteps);
    
    time_t tik, tok;
    time(&tik);
    flag = CVode(cvode_mem, tout, y0, &t, CV_NORMAL);
    time(&tok);
    
    getOutput(y0, radM1, radM2, rhoC);
    yd = NV_DATA_S(y0);
    
    O_file << i << "\t" << difftime(tok, tik) << "\t" << ts + tout
           << "\t" << (ts + tout) * Flux << "\t";
    for (int p = 0; p < numCalcPhase; p++) {
      O_file << RadiusCalc[p] << "\t" << rhoC[p] << "\t";
    }
    O_file << yd[neq - 3] << "\t" << yd[neq - 2] << "\t" << yd[neq - 1] << endl;
    
    printYVector(y0);

    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    CVodeFree(&cvode_mem);
    
    // ✅ 配合 new 使用 delete 释放内存，防止内存泄漏
    delete data;

    ts = ts + tout;
    tout = pow(10, i / 9);
  }
  
  N_VDestroy_Serial(y0);
  SUNContext_Free(&sunctx);

  // 清理全局指针
  delete ICond;
  delete IMaterial;
  delete IProp;

  return 0;
}

/*************************************************

This function initialize parameters used in model

*************************************************/
static void initParams()
{
  Flux = ICond->Flux; /*Irradiatio flux*/
  for (int i = 0; i < numComp; i++)
  {
    D[i] = IMaterial->D[i]; /*Thermal diffusion coefficients*/
  }
  GetRED(D, Flux); /*Calculate radiation enhanced diffusion coefficients*/
  for (int p = 0; p < numPhase; p++)
  {
    aP[p] = pow((3 * IMaterial->cVol[p]) / (4 * pi),
                1. / 3.); /*Effective atomic radius in precipitate*/
  }
  for (int p = 0; p < numPhase; p++)
  {
    for (int c = 0; c < numComp; c++)
    {
      nu[p][c] =
          pow(IMaterial->X[p][c], 2); /*Square of precipitate composition*/
    }
  }

  return;
}

/*****************************************************************

This function gives the radiation enhanced diffusion coefficients
Described in SI Sec. C

*****************************************************************/
static void GetRED(realtype D[numComp], realtype Flux)
{
  realtype Eta, Gs, Cvr;

  /*When flux is higher than reference flux, use p-scaling (Eq. SI-18) to
   * calculate gs*/
  if (Flux > IProp->Rflux)
  {
    Eta = 16 * pi * IProp->rv * IProp->DCB * IProp->SigmaDpa * IProp->Rflux /
          IMaterial->aVol / IProp->DV / pow(IProp->DDP, 2);
    Gs = 2.0 / Eta * (pow(1 + Eta, 0.5) - 1.0) *
         pow((IProp->Rflux / Flux), IProp->p_factor);
  } /*When flux is lower than reference flux, use Eq. SI-19 and SI-20 to
       calculate gs*/
  else
  {
    Eta = 16 * pi * IProp->rv * IProp->DCB * IProp->SigmaDpa * Flux /
          IMaterial->aVol / IProp->DV / pow(IProp->DDP, 2);
    Gs = 2.0 / Eta * (pow(1 + Eta, 0.5) - 1.0);
  }
  Cvr = IProp->DCB * Flux * IProp->SigmaDpa * Gs /
        (IProp->DV * IProp->DDP); /*Calculate vacancy concentration under
                                     irradiation, Eq. SI-17*/
  for (int i = 0; i < numComp; i++)
  {
    D[i] =
        D[i] +
        IProp->DV * Cvr * D[i] /
            IMaterial
                ->DFe; /*Radiation enhanced diffusion coefficients, Eq. SI-16*/
  }
}

/*****************************************************************

This function loads data that is as a function of cluster size
All values are calcuate via next few functions.

*****************************************************************/
static void loadData(UserData data) {
  getSize(data->size);
  getRadClust(data->size, data->radClust);
  getbeta(data->size, data->beta);
  getDelG(data->size, data->radClust, data->delG);
  return;
}

/*****************************************************************

This function calculates part of the absorption rate

*****************************************************************/
static void getbeta(realtype* size, realtype** beta) {
  for (int p = 0; p < numPhase; p++) {
    for (int i = 0; i < numClass; i++) {
      beta[p][i] = (4 * pi * aP[p] * pow(size[i], (1. / 3.))) / IMaterial->aVol;
    }
  }
  return;
}

/*****************************************************************

This function calculates number of atoms (size) in each cluster

*****************************************************************/

static void getSize(realtype* size) {
  for (int i = 0; i < numClass; i++) {
    size[i] = double(i + 1);
  }
  return;
}

/*****************************************************************

This function calculates radius of each cluster

*****************************************************************/

static void getRadClust(realtype* size, realtype** radClust) {
  for (int p = 0; p < numPhase; p++) {
    for (int i = 0; i < numClass; i++) {
      radClust[p][i] =
          pow(3 * IMaterial->cVol[p] * size[i] / (4 * pi), 1. / 3.);
    }
  }
  return;
}

/*********************************************************************************

This function calculates difference of interfacial energy between adjacent
clusters

**********************************************************************************/

static void getDelG(realtype* size, realtype** radClust, realtype** delG) {
  realtype delta;
  for (int p = 0; p < numPhase; p++) {
    for (int i = 1; i < numClass; i++) {
      delta = -((IMaterial->sig[p] * 4 * pi * pow(radClust[p][i - 1], TWO)) -
                (IMaterial->sig[p] * 4 * pi * pow(radClust[p][i], TWO)));
      delG[p][i] = exp(delta / (kb * ICond->Temp));
    }
  }
  return;
}

/*****************************************************************

This function gives the inital values of number density of each cluster
1E-30 for clusters >= 2 atoms
monomer concentration is calculated based on Eq. SI-15

*****************************************************************/

static void getInitVals(realtype y0[neq])
{
  realtype base = 1E-30;
  for (int i = 0; i < neq; i++)
  {
    y0[i] = base;
  }
  for (int c = 0; c < numComp; c++)
  {
    y0[neq - numComp + c] =
        IMaterial->C0[c]; /*Concentration of solute in matrix*/
  }
  for (int p = 0; p < numPhase; p++)
  {
    solProd[p] = 1;
    for (int c = 0; c < numComp; c++)
    {
      solProd[p] = solProd[p] * pow(IMaterial->C0[c], IMaterial->X[p][c]);
    }
    y0[p * numClass] =
        solProd[p]; /*Effective monomer concentration, based on Eq. SI-15*/
  }
  return;
}

/**************************************************************************

This function calculates flux (Jn->n+1) between adjacent clusters, Eq. SI-2

****************************************************************************/

static int fluxIndex(int phase, int cluster)
{
  return phase * (numClass + 1) + cluster;
}

static void getFlux(UserData data, N_Vector y, realtype J[])
{
  realtype *yd;
  yd = NV_DATA_S(y);
  realtype solP[numCalcPhase], wp[numComp], sumwp, wpEff;
  int pref; /*pref is used to refer to the real phase for both homo and heter
               nucleated phases, pref=0 is T3 phase, pref=1 is T6 phase*/
  for (int p = 0; p < numPhase; p++)
  {
    solP[p] = 1;
    for (int c = 0; c < numComp; c++)
    {
      solP[p] = solP[p] * pow(yd[neq - numComp + c],
                              IMaterial->X[p][c]); /*Calculate solute product*/
    }
  }
  for (int p = numPhase; p < numCalcPhase; p++)
  {
    solP[p] = 1E-30; /*solute product for heterogeneous nucleation phase, used
                        to calcuate effective monomer concentration*/
  }
  for (int p = 0; p < numCalcPhase; p++)
  {
    pref = p % numPhase;
    J[fluxIndex(p, 0)] = ZERO;
    yd[p * numClass] = solP[p]; /*Effective monomer concentration*/
    sumwp = 0;
    for (int c = 0; c < numComp; c++)
    {
      wp[c] = yd[neq - numComp + c] * D[c] * data->beta[pref][0];
      sumwp = sumwp + (nu[pref][c] / wp[c]);
    }
    wpEff = 1.0 / sumwp; /*Absorption rate for momoner to dimer*/
    J[fluxIndex(p, 1)] =
        wpEff * (yd[p * numClass] / numPhase -
                 (IMaterial->solPBar[pref] / solP[pref]) * data->delG[pref][1] *
                     yd[p * numClass + 1]); /*flux from monomer to dimer*/
    for (int i = 2; i < numClass; i++)
    {
      sumwp = 0;
      for (int c = 0; c < numComp; c++)
      {
        wp[c] = yd[neq - numComp + c] * D[c] * data->beta[pref][i - 1];
        sumwp = sumwp + (nu[pref][c] / wp[c]);
      }
      wpEff = 1.0 / sumwp;
      J[fluxIndex(p, i)] =
          wpEff *
          (yd[p * numClass + i - 1] -
           (IMaterial->solPBar[pref] / solP[pref]) * data->delG[pref][i] *
               yd[p * numClass + i]); /*flux from size i to size i+1*/
    }
    J[fluxIndex(p, numClass)] = ZERO;
  }
  return;
}

/**********************************************************************************

This function calculates Eq. SI-1, the change of concentration of each cluster
size

***********************************************************************************/

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype *yd, *ydotd;
  yd = NV_DATA_S(y);
  ydotd = NV_DATA_S(ydot);
  UserData data;

  int pref;
  realtype solP[numPhase];
  realtype GR;

  data = (UserData)user_data;

  // 使用在 UserData 中预分配的 J 数组，而不是每次都动态分配
  getFlux(data, y, data->J);

  realtype sumNdot[numCalcPhase], Cdot;
  for (int p = 0; p < numCalcPhase; p++)
  {
    sumNdot[p] = ZERO;
    ydotd[p * numClass] = ZERO;
    for (int i = 1; i < numClass; i++)
    {
      ydotd[p * numClass + i] =
          data->J[fluxIndex(p, i)] -
          data->J[fluxIndex(p, i + 1)]; /*Eq. 1 in Sec. 2.1 without Rhet term*/
      sumNdot[p] = sumNdot[p] +
                   ydotd[p * numClass + i] * data->size[i]; /*Monomer consumed*/
    }
  }
  for (int p = numPhase; p < numCalcPhase; p++)
  {
    pref = p % numPhase;
    solP[pref] = 1;
    for (int c = 0; c < numComp; c++)
    {
      solP[pref] = solP[pref] * pow(yd[neq - numComp + c],
                                    IMaterial->X[pref][c]); /*solute product*/
    }
  }

  /*The next three lines is the generation of clusters in cascade damage*/
  GR = IProp->Alpha * Flux * IProp->ccs * solP[IProp->HGPhase] /
       IProp->RsolP; /*Generation rate of clusters, Eq. 5 in Sec. 2.2*/
  ydotd[(IProp->HGPhase + numPhase) * numClass + IProp->HGSize - 1] +=
      GR; /*Add Rhet term to Eq. (1) in Sec. 2.1*/
  sumNdot[(IProp->HGPhase + numPhase)] +=
      GR * data->size[IProp->HGSize - 1]; /*Calculate the monomers consumed in
                                             heterogeneous nucleation*/

  for (int c = 0; c < numComp; c++)
  {
    Cdot = 0;
    for (int p = 0; p < numCalcPhase; p++)
    {
      pref = p % numPhase;
      Cdot = Cdot + IMaterial->X[pref][c] * sumNdot[p];
    }
    ydotd[neq - numComp + c] = -Cdot; /*change of solute/monomer in matrix*/
  }

  return (0);
}

/*****************************************************************

This function calculates mean cluster radius and cluster density

*****************************************************************/

static void getOutput(N_Vector y, realtype radM1[numCalcPhase],
                      realtype radM2[numCalcPhase],
                      realtype rhoC[numCalcPhase]) {
  realtype *yd, numC, numCxSize1, numCxSize2;
  yd = NV_DATA_S(y);
  int pref;
  for (int p = 0; p < numCalcPhase; p++) {
    pref = p % numPhase;
    numC = 0; 
    numCxSize1 = 0;
    numCxSize2 = 0;
    radM1[p] = 0;
    radM2[p] = 0;
    rhoC[p] = 0;
    for (int i = CutoffSize; i < numClass; i++) {
      realtype clusterSize = static_cast<realtype>(i + 1);
      // 直接计算半径，绝对安全，不需要调用 getRadClust 也不需要大数组
      realtype clusterRadius =
          pow(3 * IMaterial->cVol[pref] * clusterSize / (4 * pi), 1. / 3.);
      numC = numC + yd[p * numClass + i];
      numCxSize1 = numCxSize1 + yd[p * numClass + i] * clusterRadius;
      numCxSize2 = numCxSize2 + yd[p * numClass + i] * clusterSize;
    }
    radM1[p] = numCxSize1 / numC;
    radM2[p] =
        pow((numCxSize2 / numC * IMaterial->cVol[pref]) / ((4. / 3.) * pi),
            (1. / 3.));
    rhoC[p] = numC / IMaterial->aVol; 
  }
  return;
}

/**********************************************************************************************

This function prints cluster size distribution in the file Profile for the final
solution time.

***********************************************************************************************/
static void printYVector(N_Vector y) {
  realtype *yd;
  ofstream P_file;
  yd = NV_DATA_S(y);
  int pref;

  for (int p = 0; p < numCalcPhase; p++) {
    pref = p % numPhase;
    string profStr = "Profile_";
    string phaseStr;
    int_to_string(p, phaseStr, 10);
    profStr.append(phaseStr);
    P_file.open(profStr.c_str());
    P_file << "cluster size (# atoms)\tcluster radius (m)\tcluster density (1/m3)" << endl;
    for (int i = 0; i < numClass; i++) {
      realtype clusterSize = static_cast<realtype>(i + 1);
      // 同样，直接计算即可
      realtype clusterRadius =
          pow(3 * IMaterial->cVol[pref] * clusterSize / (4 * pi), 1. / 3.);
      P_file << clusterSize << "\t" << clusterRadius << "\t"
             << yd[p * numClass + i] / IMaterial->aVol << endl;
    }
    P_file.close();
  }
  return;
}

void int_to_string(int i, string &a, int base)
{
  int ii = i;
  string aa;
  int remain = ii % base;

  if (ii == 0)
  {
    a.push_back(ii + 48);
    return;
  }

  while (ii > 0)
  {
    aa.push_back(ii % base + 48);
    ii = (ii - remain) / base;
    remain = ii % base;
  }
  for (ii = aa.size() - 1; ii >= 0; ii--)
  {
    a.push_back(aa[ii]);
  }
  return;
}
