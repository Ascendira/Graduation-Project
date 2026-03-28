// Code developed by Huibin Ke from UW-Madison for the evolution of Mn-Ni-Si
// precipitates. Modified to implement Binary Grouping Method (Method of
// Moments)

#include <filesystem>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <cvode/cvode.h>             /* main integrator header file */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fct. and macros */
#include <sundials/sundials_math.h>  /* contains the macros ABS, SQR, and EXP */
#include <sundials/sundials_types.h> /* definition of realtype */

#include <sunlinsol/sunlinsol_spgmr.h>

#include "Constants.h" /*Constants header file*/
#include "Input.h"     /*Input parameter header file*/

using namespace std;

InputCondition *ICond;
InputMaterial *IMaterial;
InputProperty *IProp;

realtype D[numComp], aP[numPhase], surfTerm[numPhase], nu[numPhase][numComp];
realtype Flux, solProd[numPhase];

GroupMap GMap[numGroups];

// 供 CVODE 传递给导数函数 f 的自定义数据结构
struct UserDataType
{
  realtype *n_center;
  realtype **radGroup;
  realtype **beta_group;
  realtype *J_M;

  UserDataType()
  {
    n_center = new realtype[numGroups]();
    radGroup = new realtype *[numPhase];
    beta_group = new realtype *[numPhase];
    for (int p = 0; p < numPhase; p++)
    {
      radGroup[p] = new realtype[numGroups]();
      beta_group[p] = new realtype[numGroups]();
    }
    J_M = new realtype[numCalcPhase * (numGroups + 1)]();
  }
  ~UserDataType()
  {
    delete[] n_center;
    for (int p = 0; p < numPhase; p++)
    {
      delete[] radGroup[p];
      delete[] beta_group[p];
    }
    delete[] radGroup;
    delete[] beta_group;
    delete[] J_M;
  }
};

/* =========================================================================
 * 函数原型声明
 * ========================================================================= */
static void initParams();
static void GetRED(realtype D[numComp], realtype Flux);
static void initGrouping();
static realtype getAvgN(realtype M0, realtype M1, int groupIdx);
static int fluxIndex(int phase, int cluster);
static int cluNumTGroupIndex(realtype n);
static void getGroupFlux(UserDataType *data, N_Vector y, realtype *J_M);
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static void getOutput(N_Vector y, realtype radM1[numCalcPhase],
                      realtype radM2[numCalcPhase],
                      realtype rhoC[numCalcPhase]);
static void printYVector(N_Vector y, int runIdx);
void int_to_string(int i, string &a, int base);

/* =========================================================================
 * 主函数 MAIN
 * ========================================================================= */
int main()
{
  realtype t, tout = 1E0;
  double ts = 0.0;
  int mxsteps = 2000000;

  // 1. 初始化 SUNDIALS 上下文
  SUNContext sunctx;
  SUNContext_Create(SUN_COMM_NULL, &sunctx);

  ICond = new InputCondition();
  IMaterial = new InputMaterial();
  IProp = new InputProperty();

  LoadInput(ICond, IMaterial, IProp);

  // 初始化物理参数和分组地图
  initParams();
  initGrouping();

  // 2. 初始化解向量 N_Vector
  N_Vector y0 = N_VNew_Serial(neq_groups, sunctx);
  realtype *yd = NV_DATA_S(y0);

  // 初始值填充 (极小值避免除0错误)
  for (int p = 0; p < numCalcPhase; p++)
  {
    for (int g = 0; g < numGroups; g++)
    {
      int p_base = p * numGroups * 2;
      yd[p_base + g] = 1E-30;
      yd[p_base + numGroups + g] = 1E-30 * GMap[g].n_center;
    }
  }

  // 矩阵溶质初始浓度
  for (int c = 0; c < numComp; c++)
  {
    yd[neq_groups - numComp + c] = IMaterial->C0[c];
  }

  // 单体初始平衡浓度映射到每相的第一个组
  for (int p = 0; p < numPhase; p++)
  {
    solProd[p] = 1.0;
    for (int c = 0; c < numComp; c++)
    {
      solProd[p] *= pow(IMaterial->C0[c], IMaterial->X[p][c]);
    }
    int p_base = p * numGroups * 2;
    yd[p_base] = solProd[p];             // 数量 M0_g0
    yd[p_base + numGroups] = solProd[p]; // 原子总数 M1_g0 (因为单体 n=1)
  }

  // 3. 配置求解器和数据
  void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
  UserDataType *data = new UserDataType();
  for (int g = 0; g < numGroups; g++)
  {
    data->n_center[g] = GMap[g].n_center;
  }
  CVodeSetUserData(cvode_mem, data);
  CVodeInit(cvode_mem, f, T0, y0);
  CVodeSStolerances(cvode_mem, RTOL, ATOL);
  CVodeSetMaxNumSteps(cvode_mem, mxsteps);

  // 配置带状线性求解器 (分组后带状特性依然保持)
  // SUNMatrix A = SUNBandMatrix(neq_groups, numGroups * 2, numGroups * 2, sunctx);
  // SUNLinearSolver LS = SUNLinSol_Band(y0, A, sunctx);
  // CVodeSetLinearSolver(cvode_mem, LS, A);

  // 创建 SPGMR 线性求解器。参数 0 表示使用默认的 Krylov 子空间最大维度 (通常为 5)
  // SUN_PREC_NONE 表示暂时不使用预条件器 (Preconditioner)
  SUNLinearSolver LS = SUNLinSol_SPGMR(y0, SUN_PREC_NONE, 0, sunctx);

  // 将迭代求解器附加到 CVODE 内存块中，注意这里不需要传入矩阵对象 (传 NULL)
  CVodeSetLinearSolver(cvode_mem, LS, NULL);

  // 4. 打开主输出文件
  string dir = "../data/output";
  std::filesystem::path dirPath(dir);
  if (!std::filesystem::exists(dirPath))
  {
    if (!std::filesystem::create_directories(dirPath))
      return 0;
  }

  ofstream O_file(dir + "/Output_Grouping.txt");
  O_file << "Run\tCalcTime(s)\tTime(s)\tFluence(n/m2s)\t";
  for (int p = 0; p < numPhase; p++)
  {
    string ps;
    int_to_string(p + 1, ps, 10);
    O_file << "Rad_P" + ps + "_Homo(m)\tRho_P" + ps + "_Homo(1/m3)\t";
  }
  for (int p = 0; p < numPhase; p++)
  {
    string ps;
    int_to_string(p + 1, ps, 10);
    O_file << "Rad_P" + ps + "_Heter(m)\tRho_P" + ps + "_Heter(1/m3)\t";
  }
  O_file << "Mn\tNi\tSi" << endl;

  realtype radM1_out[numCalcPhase], radM2_out[numCalcPhase],
      rhoC_out[numCalcPhase];

  // 5. 主循环计算
  for (int i = 0; i < runs; i++)
  {
    time_t tik, tok;
    time(&tik);

    int flag = CVode(cvode_mem, tout, y0, &t, CV_NORMAL);

    time(&tok);

    if (flag < 0)
    {
      cerr << "CVODE Error: flag=" << flag << " at run " << i << endl;
      break;
    }

    getOutput(y0, radM1_out, radM2_out, rhoC_out);
    printYVector(y0, i);

    realtype *yd_final = NV_DATA_S(y0);
    O_file << i << "\t" << difftime(tok, tik) << "\t" << t << "\t" << t * Flux
           << "\t";

    for (int p = 0; p < numCalcPhase; p++)
    {
      O_file << radM2_out[p] << "\t" << rhoC_out[p] << "\t";
    }

    O_file << yd_final[neq_groups - 3] << "\t" << yd_final[neq_groups - 2]
           << "\t" << yd_final[neq_groups - 1] << endl;

    tout *= 1.5;
  }

  // 清理内存
  N_VDestroy(y0);
  delete data;
  delete ICond;
  delete IMaterial;
  delete IProp;
  CVodeFree(&cvode_mem);
  SUNLinSolFree(LS);
  SUNContext_Free(&sunctx);

  return 0;
}

/* =========================================================================
 * 辅助与初始化函数
 * ========================================================================= */
static void initParams()
{
  Flux = ICond->Flux;
  for (int i = 0; i < numComp; i++)
    D[i] = IMaterial->D[i];
  GetRED(D, Flux);
  for (int p = 0; p < numPhase; p++)
  {
    aP[p] = pow((3 * IMaterial->cVol[p]) / (4 * pi), 1. / 3.);
    surfTerm[p] = 4.0 * pi * IMaterial->sig[p] *
                  pow((3.0 * IMaterial->cVol[p]) / (4.0 * pi), 2.0 / 3.0) /
                  (kb * ICond->Temp);
    for (int c = 0; c < numComp; c++)
    {
      nu[p][c] = pow(IMaterial->X[p][c], 2);
    }
  }
}

static void GetRED(realtype D[numComp], realtype Flux)
{
  realtype Eta, Gs, Cvr;
  if (Flux > IProp->Rflux)
  {
    Eta = 16 * pi * IProp->rv * IProp->DCB * IProp->SigmaDpa * IProp->Rflux /
          IMaterial->aVol / IProp->DV / pow(IProp->DDP, 2);
    Gs = 2.0 / Eta * (pow(1 + Eta, 0.5) - 1.0) *
         pow((IProp->Rflux / Flux), IProp->p_factor);
  }
  else
  {
    Eta = 16 * pi * IProp->rv * IProp->DCB * IProp->SigmaDpa * Flux /
          IMaterial->aVol / IProp->DV / pow(IProp->DDP, 2);
    Gs = 2.0 / Eta * (pow(1 + Eta, 0.5) - 1.0);
  }
  Cvr = IProp->DCB * Flux * IProp->SigmaDpa * Gs / (IProp->DV * IProp->DDP);
  for (int i = 0; i < numComp; i++)
  {
    D[i] = D[i] + IProp->DV * Cvr * D[i] / IMaterial->DFe;
  }
}

static void initGrouping()
{
  realtype current_n = 1.0;
  realtype current_width = 1.0;
  for (int g = 0; g < numGroups; g++)
  {
    GMap[g].n_min = current_n;
    if (g < numDiscrete)
    {
      GMap[g].width = 1.0;
    }
    else
    {
      current_width *= groupingFactor;
      GMap[g].width = std::max(1.0, std::round(current_width));
    }
    GMap[g].n_max = GMap[g].n_min + GMap[g].width - 1.0;
    GMap[g].n_center = (GMap[g].n_min + GMap[g].n_max) / 2.0;
    current_n = GMap[g].n_max + 1.0;
  }
  cout << "Grouping init complete. Max cluster size: "
       << GMap[numGroups - 1].n_max << " atoms" << endl;
}

static realtype getAvgN(realtype M0, realtype M1, int groupIdx)
{
  if (groupIdx < numDiscrete || M0 < 1e-35)
    return GMap[groupIdx].n_center;
  realtype n_avg = M1 / M0;
  if (n_avg < GMap[groupIdx].n_min)
    return GMap[groupIdx].n_min;
  if (n_avg > GMap[groupIdx].n_max)
    return GMap[groupIdx].n_max;
  return n_avg;
}

static int cluNumTGroupIndex(realtype n)
{
  if (n < 1)
    return 0;
  for (int g = 0; g < numGroups; g++)
  {
    if (n >= GMap[g].n_min && n <= GMap[g].n_max)
      return g;
  }
  return numGroups - 1;
}

/* =========================================================================
 * 核心：分组法通量与 ODE 方程 (getGroupFlux & f)
 * ========================================================================= */
static int fluxIndex(int phase, int groupIdx)
{
  return phase * (numGroups + 1) + groupIdx;
}

static void getGroupFlux(UserDataType *data, N_Vector y, realtype *J_M)
{
  realtype *yd = NV_DATA_S(y);
  int M1_off = numGroups;
  realtype solP[numCalcPhase], sumwp, wp, wpEff;

  realtype C_sol[numComp];
  for (int c = 0; c < numComp; c++)
  {
    C_sol[c] = std::max(1e-30, yd[neq_groups - numComp + c]);
  }

  for (int p = 0; p < numPhase; p++)
  {
    solP[p] = 1.0;
    for (int c = 0; c < numComp; c++)
    {
      solP[p] *= pow(C_sol[c], IMaterial->X[p][c]);
    }
  }
  for (int p = numPhase; p < numCalcPhase; p++)
    solP[p] = 1E-30;

  for (int p = 0; p < numCalcPhase; p++)
  {
    int pref = p % numPhase;
    int p_base = p * numGroups * 2;

    J_M[fluxIndex(p, 0)] = ZERO;

    realtype r_b_0 =
        pow(3.0 * IMaterial->cVol[pref] * 1.0 / (4.0 * pi), 1.0 / 3.0);
    sumwp = 0;
    for (int c = 0; c < numComp; c++)
    {
      realtype wp = C_sol[c] * D[c] *
                    ((4.0 * pi * r_b_0) / IMaterial->aVol);
      sumwp += (nu[pref][c] / wp);
    }
    wpEff = 1.0 / sumwp;

    realtype C_right_0 = yd[p_base] / numPhase;
    realtype C_left_1 = yd[p_base + 1];

    realtype delG_boundary_0 =
        exp(surfTerm[pref] * (pow(2.0, 2.0 / 3.0) - pow(1.0, 2.0 / 3.0)));
    realtype emission_ratio_0 =
        (IMaterial->solPBar[pref] / std::max(solP[pref], 1e-30)) *
        delG_boundary_0;

    realtype forward_flux_0 = wpEff * C_right_0;
    realtype backward_flux_0 = wpEff * emission_ratio_0 * C_left_1;

    J_M[fluxIndex(p, 1)] = forward_flux_0 - backward_flux_0;
    // fluxIndex indicates the flux form group i t group i+1

    for (int g = 2; g < numGroups; g++)
    {
      realtype M0_gm1 = yd[p_base + g - 1];
      realtype M1_gm1 = yd[p_base + g + M1_off - 1];

      realtype M0_g = yd[p_base + g];
      realtype M1_g = yd[p_base + g + M1_off];

      // 1. 评估界面处的吸收速率 (wpEff) 和发射速率修正 (emission_ratio)
      realtype n_b = GMap[g - 1].n_max;
      realtype r_b =
          pow(3.0 * IMaterial->cVol[pref] * n_b / (4.0 * pi), 1.0 / 3.0);

      sumwp = 0;
      for (int c = 0; c < numComp; c++)
      {
        wp = C_sol[c] * D[c] * ((4.0 * pi * r_b) / IMaterial->aVol);
        sumwp += (nu[pref][c] / wp);
      }
      wpEff = 1.0 / sumwp;

      // 2. 评估界面左侧浓度 (C_right_g) -> 驱动向右的生长通量
      realtype C_right_g = 0.0;
      if (M0_gm1 > 1e-35)
      {
        if (g < numDiscrete)
        {
          // 【精确区】：浓度就是 M0 本身，不需要插值
          C_right_g = M0_gm1;
        }
        else
        {
          // 【分组区】：使用差值法线性重构右边界浓度
          realtype n_avg_gm1 = getAvgN(M0_gm1, M1_gm1, g - 1);
          realtype delta_n_gm1 = n_avg_gm1 - GMap[g - 1].n_center;
          realtype slope_gm1 =
              12.0 * delta_n_gm1 / (GMap[g - 1].width * GMap[g - 1].width);
          C_right_g = M0_gm1 * (1.0 + slope_gm1 * (GMap[g - 1].n_max - GMap[g - 1].n_center));
        }
        if (C_right_g < 0.0)
          C_right_g = 1E-30;
      }

      // 3. 评估界面右侧浓度 (C_left_gp1) -> 驱动向左的溶解通量
      realtype C_left_g = 0.0;
      if (M0_g > 1e-35)
      {
        if (g < numDiscrete)
        {
          // 下一组依然是精确区
          C_left_g = M0_g;
        }
        else
        {
          // 下一组是分组区 (完美涵盖了交界处情况)
          realtype n_avg_g = getAvgN(M0_g, M1_g, g);
          realtype delta_n_g = n_avg_g - GMap[g].n_center;
          realtype slope_g = 12.0 * delta_n_g / (GMap[g].width * GMap[g].width);
          C_left_g = M0_g * (1.0 + slope_g * (GMap[g].n_min - GMap[g].n_center));
        }
        if (C_left_g < 0.0)
          C_left_g = 1E-30;
      }

      // 4. 界面能修正与通量计算 (完美迎风格式)
      realtype delG_boundary = exp(
          surfTerm[pref] * (pow(n_b + 1.0, 2.0 / 3.0) - pow(n_b, 2.0 / 3.0)));
      realtype emission_ratio =
          (IMaterial->solPBar[pref] / solP[pref]) * delG_boundary;

      realtype forward_flux = wpEff * C_right_g;
      realtype backward_flux = wpEff * emission_ratio * C_left_g;

      J_M[fluxIndex(p, g)] = forward_flux - backward_flux;
    }

    J_M[fluxIndex(p, numGroups)] = ZERO;
  }
}

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype *yd = NV_DATA_S(y);
  realtype *ydotd = NV_DATA_S(ydot);
  UserDataType *data = static_cast<UserDataType *>(user_data);

  getGroupFlux(data, y, data->J_M);

  realtype total_sol_cons[numComp] = {0.0, 0.0, 0.0};

  for (int p = 0; p < numCalcPhase; p++)
  {
    int pref = p % numPhase;
    int M0_base = p * numGroups * 2;
    int M1_base = M0_base + numGroups;

    ydotd[M0_base] = -data->J_M[fluxIndex(p, 1)];
    ydotd[M1_base] = -data->J_M[fluxIndex(p, 1)];

    for (int g = 1; g < numGroups; g++)
    {

      ydotd[M0_base + g] = data->J_M[fluxIndex(p, g)] - data->J_M[fluxIndex(p, g + 1)];
      ydotd[M1_base + g] = data->J_M[fluxIndex(p, g)] * GMap[g].n_min - data->J_M[fluxIndex(p, g + 1)] * GMap[g].n_max;

      for (int c = 0; c < numComp; c++)
      {
        total_sol_cons[c] += IMaterial->X[pref][c] * ydotd[M1_base + g];
      }
    }
  }

  realtype solP_pref = 1.0;
  for (int c = 0; c < numComp; c++)
  {
    realtype safe_C = std::max(1e-30, yd[neq_groups - numComp + c]);
    solP_pref *= pow(safe_C, IMaterial->X[IProp->HGPhase][c]);
  }
  realtype GR = IProp->Alpha * Flux * IProp->ccs * solP_pref / IProp->RsolP;

  int g_hg = cluNumTGroupIndex(IProp->HGSize);
  int p_hg = IProp->HGPhase + numPhase;

  ydotd[p_hg * numGroups * 2 + g_hg] += GR;
  ydotd[p_hg * numGroups * 2 + numGroups + g_hg] += IProp->HGSize * GR;

  realtype ydotdM0_g = ydotd[p_hg * numGroups * 2 + g_hg];
  realtype ydotdM1_g = ydotd[p_hg * numGroups * 2 + numGroups + g_hg];

  for (int c = 0; c < numComp; c++)
  {
    total_sol_cons[c] += IMaterial->X[IProp->HGPhase][c] * (IProp->HGSize * GR);
  }

  for (int c = 0; c < numComp; c++)
  {
    ydotd[neq_groups - numComp + c] = -total_sol_cons[c];
  }

  return 0;
}

/* =========================================================================
 * 输出与格式化函数
 * ========================================================================= */
static void getOutput(N_Vector y, realtype radM1[numCalcPhase],
                      realtype radM2[numCalcPhase],
                      realtype rhoC[numCalcPhase])
{
  realtype *yd = NV_DATA_S(y);
  int M1_off = numGroups;

  for (int p = 0; p < numCalcPhase; p++)
  {
    int pref = p % numPhase;
    int p_base = p * numGroups * 2;

    realtype total_M0 = 0.0;
    realtype sum_Ri_M0 = 0.0;
    realtype total_M1 = 0.0;

    for (int g = 0; g < numGroups; g++)
    {
      if (GMap[g].n_max < CutoffSize)
        continue;

      realtype M0 = yd[p_base + g];
      realtype M1 = yd[p_base + g + M1_off];

      if (M0 < 1e-35)
        continue;

      realtype n_avg = M1 / M0;
      realtype r_avg =
          pow(3.0 * IMaterial->cVol[pref] * n_avg / (4.0 * pi), 1.0 / 3.0);

      total_M0 += M0;
      sum_Ri_M0 += r_avg * M0;
      total_M1 += M1;
    }

    if (total_M0 > 1e-35)
    {
      radM1[p] = sum_Ri_M0 / total_M0;
      realtype n_mean = total_M1 / total_M0;
      radM2[p] =
          pow((n_mean * IMaterial->cVol[pref]) / ((4.0 / 3.0) * pi), 1.0 / 3.0);
      rhoC[p] = total_M0 / IMaterial->aVol;
    }
    else
    {
      radM1[p] = 0.0;
      radM2[p] = 0.0;
      rhoC[p] = 0.0;
    }
  }
}

static void printYVector(N_Vector y, int runIdx)
{
  realtype *yd = NV_DATA_S(y);
  int M1_off = numGroups;

  for (int p = 0; p < numCalcPhase; p++)
  {
    string dir = "../data/output/Phase_" + to_string(p);
    std::filesystem::path dirPath(dir);
    if (!std::filesystem::exists(dirPath))
    {
      if (!std::filesystem::create_directories(dirPath))
        return;
    }
    string profStr = dir + "/Run" + to_string(runIdx) + ".txt";

    ofstream P_file(profStr.c_str());
    if (!P_file.is_open())
      return;
    P_file << "n_min\tn_max\tn_avg\tradius(m)\tdensity(1/m3)" << endl;

    int pref = p % numPhase;
    int p_base = p * numGroups * 2;
    for (int g = 0; g < numGroups; g++)
    {
      realtype M0 = yd[p_base + g];
      realtype M1 = yd[p_base + g + M1_off];

      realtype n_avg = (M0 > 1e-35) ? (M1 / M0) : GMap[g].n_center;
      realtype r_avg = pow(3.0 * IMaterial->cVol[pref] * n_avg / (4.0 * pi), 1.0 / 3.0);

      P_file << GMap[g].n_min << "\t" << GMap[g].n_max << "\t" << n_avg << "\t"
             << r_avg << "\t" << (M0 / GMap[g].width) / IMaterial->aVol << endl;
    }
    P_file.close();
  }
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
  for (int j = aa.size() - 1; j >= 0; j--)
  {
    a.push_back(aa[j]);
  }
}