// Code developed by Huibin Ke from UW-Madison for the evolution of Mn-Ni-Si
// precipitates. Modified to implement Binary Grouping Method (Method of
// Moments)

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <cvode/cvode.h>             /* main integrator header file */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fct. and macros */
#include <sundials/sundials_math.h>  /* contains the macros ABS, SQR, and EXP */
#include <sundials/sundials_types.h> /* definition of realtype */

// 重新引入 SPGMR 求解器头文件
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
static string gOutputDir = "../data/output";
static ofstream gLogFile;

static ostream &logStream()
{
  if (gLogFile.is_open())
    return gLogFile;
  return cout;
}

// 供 CVODE 传递给导数函数 f 的自定义数据结构
struct UserDataType
{
  realtype *n_center;
  realtype **radGroup;
  realtype **beta_group;
  realtype *J_M;

  // 诊断信息: 用于定位 CVode 后期变慢/“卡住”的根因
  realtype min_solP_seen;
  realtype max_emission_ratio_seen;
  realtype min_wpEff_seen;
  realtype max_abs_flux_seen;
  long int non_finite_count;
  long int non_positive_solP_count;
  long int rhs_eval_counter;

  // 当前 CVode 调用对应的外层循环信息
  int current_run;
  realtype current_target_t;

  // 允许在 RHS 诊断输出中读取 CVode 累计统计量
  void *cvode_mem;

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
    resetDiagnostics();
  }

  void resetDiagnostics()
  {
    min_solP_seen = std::numeric_limits<realtype>::max();
    max_emission_ratio_seen = ZERO;
    min_wpEff_seen = std::numeric_limits<realtype>::max();
    max_abs_flux_seen = ZERO;
    non_finite_count = 0;
    non_positive_solP_count = 0;
    rhs_eval_counter = 0;
    current_run = -1;
    current_target_t = ZERO;
    cvode_mem = nullptr;
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

  // 统一输出目录与日志文件
  if (std::filesystem::exists("data"))
    gOutputDir = "data/output";
  else
    gOutputDir = "../data/output";

  std::filesystem::path dirPath(gOutputDir);
  if (!std::filesystem::exists(dirPath))
  {
    if (!std::filesystem::create_directories(dirPath))
      return 0;
  }

  gLogFile.open(gOutputDir + "/log.txt", ios::out | ios::trunc);
  if (!gLogFile.is_open())
  {
    cerr << "Cannot open log file: " << (gOutputDir + "/log.txt") << endl;
    return 0;
  }

  // 初始化物理参数和分组地图
  initParams();
  initGrouping();

  // 2. 初始化解向量 N_Vector
  N_Vector y0 = N_VNew_Serial(neq_groups, sunctx);
  realtype *yd = NV_DATA_S(y0);

  // 初始值填充 (交错排列: M0_0, M1_0, M0_1, M1_1, ...)
  for (int p = 0; p < numCalcPhase; p++)
  {
    int p_base = p * numGroups * 2;
    for (int g = 0; g < numGroups; g++)
    {
      yd[p_base + 2 * g] = 1E-30;                        // M0
      yd[p_base + 2 * g + 1] = 1E-30 * GMap[g].n_center; // M1
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
    yd[p_base] = solProd[p];     // 数量 M0_g0
    yd[p_base + 1] = solProd[p]; // 原子总数 M1_g0 (因为单体 n=1)
  }

  // 3. 配置求解器和数据
  void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
  UserDataType *data = new UserDataType();
  data->cvode_mem = cvode_mem;
  for (int g = 0; g < numGroups; g++)
  {
    data->n_center[g] = GMap[g].n_center;
  }
  CVodeSetUserData(cvode_mem, data);
  CVodeInit(cvode_mem, f, T0, y0);
  CVodeSStolerances(cvode_mem, RTOL, ATOL);
  CVodeSetMaxNumSteps(cvode_mem, mxsteps);

  // 配置 SPGMR 线性求解器 (不使用矩阵对象，完美适配全局耦合)
  SUNLinearSolver LS = SUNLinSol_SPGMR(y0, SUN_PREC_NONE, 0, sunctx);
  CVodeSetLinearSolver(cvode_mem, LS, NULL);

  // 4. 打开主输出文件
  ofstream O_file(gOutputDir + "/Output_Grouping.txt");
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

  realtype radM1_out[numCalcPhase], radM2_out[numCalcPhase], rhoC_out[numCalcPhase];
  long int nst_last = 0, nfe_last = 0, nli_last = 0;

  // 5. 主循环计算
  for (int i = 0; i < runs; i++)
  {
    time_t tik, tok;
    time(&tik);

    data->resetDiagnostics();
    data->current_run = i;
    data->current_target_t = tout;

    int flag = CVode(cvode_mem, tout, y0, &t, CV_NORMAL);

    time(&tok);

    long int nst_total = 0, nfe_total = 0, nli_total = 0;
    CVodeGetNumSteps(cvode_mem, &nst_total);
    CVodeGetNumRhsEvals(cvode_mem, &nfe_total);
    CVodeGetNumLinIters(cvode_mem, &nli_total);

    logStream() << scientific << setprecision(6)
                << "[diag] run=" << i
                << " target_t=" << tout
                << " reached_t=" << t
                << " flag=" << flag
                << " dSteps=" << (nst_total - nst_last)
                << " dRhs=" << (nfe_total - nfe_last)
                << " dLinIters=" << (nli_total - nli_last)
                << " minSolP=" << data->min_solP_seen
                << " maxEmissionRatio=" << data->max_emission_ratio_seen
                << " minWpEff=" << data->min_wpEff_seen
                << " maxAbsFlux=" << data->max_abs_flux_seen
                << " nonFinite=" << data->non_finite_count
                << " nonPositiveSolP=" << data->non_positive_solP_count
                << " rhsCallsInRun=" << data->rhs_eval_counter
                << " wall_s=" << difftime(tok, tik) << endl;

    nst_last = nst_total;
    nfe_last = nfe_total;
    nli_last = nli_total;

    if (flag < 0)
    {
      logStream() << "CVODE Error: flag=" << flag << " at run " << i << endl;
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
  if (gLogFile.is_open())
    gLogFile.close();

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
  logStream() << "Grouping init complete. Max cluster size: "
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

    if (solP[p] < data->min_solP_seen)
      data->min_solP_seen = solP[p];
    if (solP[p] <= ZERO)
      data->non_positive_solP_count++;
    if (!std::isfinite(solP[p]))
      data->non_finite_count++;
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
      realtype wp_temp = C_sol[c] * D[c] *
                         ((4.0 * pi * r_b_0) / IMaterial->aVol);
      sumwp += (nu[pref][c] / wp_temp);
    }
    wpEff = 1.0 / sumwp;
    if (wpEff < data->min_wpEff_seen)
      data->min_wpEff_seen = wpEff;
    if (!std::isfinite(wpEff) || !std::isfinite(sumwp))
      data->non_finite_count++;

    // 交错排列下获取 M0_0 和 M0_1
    realtype C_right_0 = yd[p_base] / numPhase;
    realtype C_left_1 = yd[p_base + 2]; // M0_1 的索引为 p_base + 2

    realtype delG_boundary_0 =
        exp(surfTerm[pref] * (pow(2.0, 2.0 / 3.0) - pow(1.0, 2.0 / 3.0)));
    realtype emission_ratio_0 =
        (IMaterial->solPBar[pref] / std::max(solP[pref], 1e-30)) *
        delG_boundary_0;
    if (emission_ratio_0 > data->max_emission_ratio_seen)
      data->max_emission_ratio_seen = emission_ratio_0;
    if (!std::isfinite(emission_ratio_0))
      data->non_finite_count++;

    realtype forward_flux_0 = wpEff * C_right_0;
    realtype backward_flux_0 = wpEff * emission_ratio_0 * C_left_1;

    J_M[fluxIndex(p, 1)] = forward_flux_0 - backward_flux_0;
    realtype abs_flux_0 = abs(J_M[fluxIndex(p, 1)]);
    if (abs_flux_0 > data->max_abs_flux_seen)
      data->max_abs_flux_seen = abs_flux_0;
    if (!std::isfinite(J_M[fluxIndex(p, 1)]))
      data->non_finite_count++;

    for (int g = 2; g < numGroups; g++)
    {
      // 交错排列下的 M0 和 M1 获取方式
      realtype M0_gm1 = yd[p_base + 2 * (g - 1)];
      realtype M1_gm1 = yd[p_base + 2 * (g - 1) + 1];

      realtype M0_g = yd[p_base + 2 * g];
      realtype M1_g = yd[p_base + 2 * g + 1];

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
      if (wpEff < data->min_wpEff_seen)
        data->min_wpEff_seen = wpEff;
      if (!std::isfinite(wpEff) || !std::isfinite(sumwp))
        data->non_finite_count++;

      // 2. 评估界面左侧浓度 (C_right_g) -> 驱动向右的生长通量
      realtype C_right_g = 0.0;
      if (M0_gm1 > 1e-35)
      {
        if (g < numDiscrete)
        {
          C_right_g = M0_gm1;
        }
        else
        {
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
          C_left_g = M0_g;
        }
        else
        {
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
      if (emission_ratio > data->max_emission_ratio_seen)
        data->max_emission_ratio_seen = emission_ratio;
      if (!std::isfinite(emission_ratio))
        data->non_finite_count++;

      realtype forward_flux = wpEff * C_right_g;
      realtype backward_flux = wpEff * emission_ratio * C_left_g;

      J_M[fluxIndex(p, g)] = forward_flux - backward_flux;
      realtype abs_flux = abs(J_M[fluxIndex(p, g)]);
      if (abs_flux > data->max_abs_flux_seen)
        data->max_abs_flux_seen = abs_flux;
      if (!std::isfinite(J_M[fluxIndex(p, g)]))
        data->non_finite_count++;
    }

    J_M[fluxIndex(p, numGroups)] = ZERO;
  }
}

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype *yd = NV_DATA_S(y);
  realtype *ydotd = NV_DATA_S(ydot);
  UserDataType *data = static_cast<UserDataType *>(user_data);

  data->rhs_eval_counter++;
  if (data->rhs_eval_counter % 500000 == 0)
  {
    realtype progress = ZERO;
    if (data->current_target_t > ZERO)
      progress = t / data->current_target_t;

    long int nst_total = 0;
    long int nli_total = 0;
    if (data->cvode_mem != nullptr)
    {
      CVodeGetNumSteps(data->cvode_mem, &nst_total);
      CVodeGetNumLinIters(data->cvode_mem, &nli_total);
    }

    logStream() << scientific << setprecision(6)
                << "[diag-rhs] eval=" << data->rhs_eval_counter
                << " t=" << t
                << " minSolP=" << data->min_solP_seen
                << " maxEmissionRatio=" << data->max_emission_ratio_seen
                << " nonFinite=" << data->non_finite_count << endl;

    logStream() << scientific << setprecision(6)
                << "[diag-rhs-detail] run=" << data->current_run
                << " target_t=" << data->current_target_t
                << " progress=" << progress
                << " totalSteps=" << nst_total
                << " totalLinIters=" << nli_total << endl;
  }

  getGroupFlux(data, y, data->J_M);

  realtype total_sol_cons[numComp] = {0.0, 0.0, 0.0};

  for (int p = 0; p < numCalcPhase; p++)
  {
    int pref = p % numPhase;
    int p_base = p * numGroups * 2; // 交错排列基底索引

    // 组 0
    ydotd[p_base] = -data->J_M[fluxIndex(p, 1)];     // M0_0
    ydotd[p_base + 1] = -data->J_M[fluxIndex(p, 1)]; // M1_0

    for (int g = 1; g < numGroups; g++)
    {
      // 交错排布赋值
      ydotd[p_base + 2 * g + 1] = data->J_M[fluxIndex(p, g)] * GMap[g].n_min - data->J_M[fluxIndex(p, g + 1)] * GMap[g].n_max;
      ydotd[p_base + 2 * g] = ydotd[p_base + 2 * g + 1] / GMap[g].n_center;
      
      for (int c = 0; c < numComp; c++)
      {
        total_sol_cons[c] += IMaterial->X[pref][c] * ydotd[p_base + 2 * g + 1];
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
  int hg_base = p_hg * numGroups * 2;

  // 非均相形核的生成项交错排布索引
  ydotd[hg_base + 2 * g_hg] += GR;
  ydotd[hg_base + 2 * g_hg + 1] += IProp->HGSize * GR;

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

  for (int p = 0; p < numCalcPhase; p++)
  {
    int pref = p % numPhase;
    int p_base = p * numGroups * 2;

    realtype total_M0 = 0.0;
    realtype sum_Ri_M0 = 0.0;
    realtype total_M1 = 0.0;

    for (int g = 0; g < numGroups; g++)
    {
      realtype M0, M1;
      if (std::floor(GMap[g].n_max) < CutoffSize)
        continue;
      else if (std::ceil(GMap[g].n_min) < CutoffSize)
      {
        // 交错获取 M0, M1
        M0 = yd[p_base + 2 * g] * ((std::floor(GMap[g].n_min) - CutoffSize + 1.0));
        M1 = yd[p_base + 2 * g + 1] * ((std::floor(GMap[g].n_min) - CutoffSize + 1.0));
      }
      else
      {
        M0 = yd[p_base + 2 * g] * GMap[g].width;
        M1 = yd[p_base + 2 * g + 1] * GMap[g].width;
      }

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

  for (int p = 0; p < numCalcPhase; p++)
  {
    string dir = gOutputDir + "/Phase_" + to_string(p);
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
      // 交错获取 M0, M1
      realtype M0 = yd[p_base + 2 * g] * GMap[g].width;
      realtype M1 = yd[p_base + 2 * g + 1] * GMap[g].width;

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