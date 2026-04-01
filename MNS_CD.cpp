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
#include <algorithm>

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

realtype D[numComp], aP[numPhase], surfTerm[numPhase + 1], nu[numPhase][numComp];
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
static realtype getDelgDisl(int p, realtype n, int pref);
static int fluxIndex(int phase, int cluster);
static int cluNumTGroupIndex(realtype n);
static void getGroupFlux(UserDataType *data, N_Vector y, realtype *J_M);
static bool ensureGroupPlotScript(const std::filesystem::path &scriptPath);
static bool ensurePhasePlotScript(const std::filesystem::path &scriptPath);
static void plotPhaseProfile(const std::filesystem::path &txtPath,
                             int phaseIdx, int runIdx);
static void writeGroupPlotDataAndImages(UserDataType *data, realtype t,
                                        const realtype *ydotd);
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

  string phaseNames[numPhase] = {"T3", "T6", "Cu"};

  for (int p = 0; p < numPhase; p++)
  {
    O_file << "Rad_" << phaseNames[p] << "_Homo(m)\tRho_" << phaseNames[p] << "_Homo(1/m3)\t";
  }
  for (int p = 0; p < numPhase; p++)
  {
    O_file << "Rad_" << phaseNames[p] << "_Heter(m)\tRho_" << phaseNames[p] << "_Heter(1/m3)\t";
  }

  O_file << "Mn_matrix\tNi_matrix\tSi_matrix\tCu_matrix" << endl;

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
    O_file << i << "\t" << difftime(tok, tik) << "\t" << t << "\t" << t * Flux << "\t";

    for (int p = 0; p < numCalcPhase; p++)
    {
      O_file << radM2_out[p] << "\t" << rhoC_out[p] << "\t";
    }
    for (int c = 0; c < numComp; c++)
    {
      O_file << yd_final[neq_groups - numComp + c] << (c == numComp - 1 ? "" : "\t");
    }
    O_file << endl;

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
    realtype a = surfTerm[p];
    for (int c = 0; c < numComp; c++)
    {
      nu[p][c] = pow(IMaterial->X[p][c], 2);
    }
  }
  surfTerm[3] = 4.0 * pi * (IMaterial->sig[2] + IMaterial->sig[0]) / 2.0 *
                pow((3.0 * IMaterial->cVol[2]) / (4.0 * pi), 2.0 / 3.0) /
                (kb * ICond->Temp);
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

static realtype get_delG_disl(int p, realtype n, int pref)
{
  // 只有异质形核相 (p >= numPhase) 才计算位错释放应变能
  if (p < numPhase)
    return ZERO;

  // 提取论文 Table 2 中的物理参数
  realtype r_core = 0.4 * 1e-9;   // 位错核半径: 0.4 nm -> m
  realtype E_core = 0.937 * 1e10; // 单位: eV/m

  realtype b_vec = 0.248 * 1e-9;          // 伯格斯矢量: 0.248 nm -> m
  realtype mu = 80.0 * 1e9 / 1.60218e-19; // 单位: eV/m^3

  realtype cVol_pref = IMaterial->cVol[pref];
  realtype r_p = pow(3.0 * cVol_pref * n / (4.0 * pi), 1.0 / 3.0);

  realtype delG = 0.0;

  if (r_p < r_core)
  {
    delG = 2.0 * r_p * E_core;
  }
  else
  {
    realtype prefactor = (mu * b_vec * b_vec) / (2.0 * pi); // 注意这里合并了 4pi 和解析积出的 2
    realtype elastic_term = r_p * (log(2.0 * r_p / r_core) - 1.0);

    delG = 2.0 * r_p * E_core + prefactor * elastic_term;
  }

  // 应变能释放会降低系统总自由能，因此作为负值返回
  return -delG;
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

static bool ensureGroupPlotScript(const std::filesystem::path &scriptPath)
{
  ofstream pyFile(scriptPath.string(), ios::out | ios::trunc);
  if (!pyFile.is_open())
    return false;

  pyFile << "import csv\n";
  pyFile << "import math\n";
  pyFile << "import sys\n";
  pyFile << "import matplotlib\n";
  pyFile << "matplotlib.use('Agg')\n";
  pyFile << "import matplotlib.pyplot as plt\n\n";

  pyFile << "SCRIPT_VERSION = '2'\n\n";

  pyFile << "def read_line_csv(path):\n";
  pyFile << "    groups = []\n";
  pyFile << "    phase_names = []\n";
  pyFile << "    phase_values = []\n";
  pyFile << "    with open(path, newline='') as f:\n";
  pyFile << "        reader = csv.reader(f)\n";
  pyFile << "        header = next(reader)\n";
  pyFile << "        phase_names = header[1:]\n";
  pyFile << "        phase_values = [[] for _ in phase_names]\n";
  pyFile << "        for row in reader:\n";
  pyFile << "            if not row:\n";
  pyFile << "                continue\n";
  pyFile << "            groups.append(int(float(row[0])))\n";
  pyFile << "            for i, val in enumerate(row[1:]):\n";
  pyFile << "                try:\n";
  pyFile << "                    v = float(val)\n";
  pyFile << "                except ValueError:\n";
  pyFile << "                    v = 0.0\n";
  pyFile << "                if not math.isfinite(v):\n";
  pyFile << "                    v = 0.0\n";
  pyFile << "                phase_values[i].append(v)\n";
  pyFile << "    return groups, phase_names, phase_values\n\n";

  pyFile << "def read_bar_csv(path):\n";
  pyFile << "    groups = []\n";
  pyFile << "    n_max = []\n";
  pyFile << "    n_min = []\n";
  pyFile << "    with open(path, newline='') as f:\n";
  pyFile << "        reader = csv.reader(f)\n";
  pyFile << "        next(reader)\n";
  pyFile << "        for row in reader:\n";
  pyFile << "            if not row:\n";
  pyFile << "                continue\n";
  pyFile << "            try:\n";
  pyFile << "                g = int(float(row[0]))\n";
  pyFile << "                high = float(row[1])\n";
  pyFile << "                low = float(row[2])\n";
  pyFile << "            except (ValueError, IndexError):\n";
  pyFile << "                continue\n";
  pyFile << "            if not (math.isfinite(low) and math.isfinite(high)):\n";
  pyFile << "                continue\n";
  pyFile << "            if high < low:\n";
  pyFile << "                low, high = high, low\n";
  pyFile << "            groups.append(g)\n";
  pyFile << "            n_max.append(high)\n";
  pyFile << "            n_min.append(low)\n";
  pyFile << "    return groups, n_max, n_min\n\n";

  pyFile << "def main():\n";
  pyFile << "    if len(sys.argv) != 5:\n";
  pyFile << "        raise SystemExit('Usage: plot_groups.py <line_csv> <bar_csv> <line_png> <bar_png>')\n";
  pyFile << "    line_csv, bar_csv, line_png, bar_png = sys.argv[1:5]\n";
  pyFile << "    groups, phase_names, phase_values = read_line_csv(line_csv)\n";
  pyFile << "    bar_groups, bar_max, bar_min = read_bar_csv(bar_csv)\n\n";

  pyFile << "    plt.figure(figsize=(12, 6))\n";
  pyFile << "    for i, name in enumerate(phase_names):\n";
  pyFile << "        plt.plot(groups, phase_values[i], linewidth=1.8, label=name)\n";
  pyFile << "    plt.xlabel('numGroups')\n";
  pyFile << "    plt.ylabel('avg_n = ydot_M1 / ydot_M0')\n";
  pyFile << "    plt.title('avg_n by group and calc phase')\n";
  pyFile << "    plt.grid(True, linestyle='--', alpha=0.35)\n";
  pyFile << "    plt.legend(loc='best', ncol=2)\n";
  pyFile << "    plt.tight_layout()\n";
  pyFile << "    plt.savefig(line_png, dpi=180)\n";
  pyFile << "    plt.close()\n\n";

  pyFile << "    plt.figure(figsize=(12, 4.8))\n";
  pyFile << "    heights = [hi - lo for hi, lo in zip(bar_max, bar_min)]\n";
  pyFile << "    bottoms = bar_min\n";
  pyFile << "    plt.bar(bar_groups, heights, width=0.9, color='#4C78A8', bottom=bottoms)\n";
  pyFile << "    plt.xlabel('numGroups')\n";
  pyFile << "    plt.ylabel('cluster size n')\n";
  pyFile << "    plt.title('Group span diagnostic')\n";
  pyFile << "    plt.grid(True, axis='y', linestyle='--', alpha=0.35)\n";
  pyFile << "    for g, hi, lo in zip(bar_groups, bar_max, bar_min):\n";
  pyFile << "        plt.text(g, hi, f'{hi:.0f}', ha='center', va='bottom', fontsize=8)\n";
  pyFile << "        plt.text(g, lo, f'{lo:.0f}', ha='center', va='top', fontsize=8)\n";
  pyFile << "    plt.tight_layout()\n";
  pyFile << "    plt.savefig(bar_png, dpi=180)\n";
  pyFile << "    plt.close()\n\n";

  pyFile << "if __name__ == '__main__':\n";
  pyFile << "    main()\n";

  return true;
}

static bool ensurePhasePlotScript(const std::filesystem::path &scriptPath)
{
  std::filesystem::path parent = scriptPath.parent_path();
  if (!parent.empty() && !std::filesystem::exists(parent))
  {
    if (!std::filesystem::create_directories(parent))
      return false;
  }

  ofstream pyFile(scriptPath.string(), ios::out | ios::trunc);
  if (!pyFile.is_open())
    return false;

  pyFile << "import csv\n";
  pyFile << "import math\n";
  pyFile << "import sys\n";
  pyFile << "import matplotlib\n";
  pyFile << "matplotlib.use('Agg')\n";
  pyFile << "import matplotlib.pyplot as plt\n\n";

  pyFile << "SCRIPT_VERSION = '1'\n\n";

  pyFile << "def read_profile(path):\n";
  pyFile << "    n_avg = []\n";
  pyFile << "    radius = []\n";
  pyFile << "    density = []\n";
  pyFile << "    with open(path, newline='') as f:\n";
  pyFile << "        reader = csv.reader(f, delimiter='\t')\n";
  pyFile << "        next(reader, None)\n";
  pyFile << "        for row in reader:\n";
  pyFile << "            if len(row) < 5:\n";
  pyFile << "                continue\n";
  pyFile << "            try:\n";
  pyFile << "                n_val = float(row[2])\n";
  pyFile << "                r_val = float(row[3])\n";
  pyFile << "                d_val = float(row[4])\n";
  pyFile << "            except ValueError:\n";
  pyFile << "                continue\n";
  pyFile << "            if not (math.isfinite(n_val) and math.isfinite(r_val) and math.isfinite(d_val)):\n";
  pyFile << "                continue\n";
  pyFile << "            n_avg.append(n_val)\n";
  pyFile << "            radius.append(r_val)\n";
  pyFile << "            density.append(d_val)\n";
  pyFile << "    return n_avg, radius, density\n\n";

  pyFile << "def main():\n";
  pyFile << "    if len(sys.argv) != 5:\n";
  pyFile << "        raise SystemExit('Usage: plot_phase_profile.py <txt_path> <png_path> <phase_idx> <run_idx>')\n";
  pyFile << "    txt_path, png_path, phase_idx, run_idx = sys.argv[1:5]\n";
  pyFile << "    n_avg, radius, density = read_profile(txt_path)\n";
  pyFile << "    if not n_avg:\n";
  pyFile << "        return\n";
  pyFile << "    fig, ax1 = plt.subplots(figsize=(10, 5))\n";
  pyFile << "    ax1.plot(n_avg, radius, color='#1f77b4', linewidth=1.8, label='radius (m)')\n";
  pyFile << "    ax1.set_xlabel('n_avg')\n";
  pyFile << "    ax1.set_ylabel('radius (m)', color='#1f77b4')\n";
  pyFile << "    ax1.tick_params(axis='y', labelcolor='#1f77b4')\n";
  pyFile << "    ax1.grid(True, linestyle='--', alpha=0.35)\n";
  pyFile << "    ax2 = ax1.twinx()\n";
  pyFile << "    ax2.plot(n_avg, density, color='#d62728', linewidth=1.5, label='density (1/m3)')\n";
  pyFile << "    ax2.set_ylabel('density (1/m3)', color='#d62728')\n";
  pyFile << "    ax2.tick_params(axis='y', labelcolor='#d62728')\n";
  pyFile << "    try:\n";
  pyFile << "        phase_val = int(float(phase_idx))\n";
  pyFile << "    except ValueError:\n";
  pyFile << "        phase_val = 0\n";
  pyFile << "    try:\n";
  pyFile << "        run_val = int(float(run_idx))\n";
  pyFile << "    except ValueError:\n";
  pyFile << "        run_val = 0\n";
  pyFile << "    fig.suptitle(f'Phase {phase_val} Run {run_val}')\n";
  pyFile << "    fig.tight_layout(rect=[0, 0.03, 1, 0.95])\n";
  pyFile << "    fig.savefig(png_path, dpi=180)\n";
  pyFile << "    plt.close(fig)\n\n";

  pyFile << "if __name__ == '__main__':\n";
  pyFile << "    main()\n";

  return true;
}

static string quotePath(const std::filesystem::path &pathVal)
{
  return string("\"") + pathVal.string() + "\"";
}

static void plotPhaseProfile(const std::filesystem::path &txtPath,
                             int phaseIdx, int runIdx)
{
  std::filesystem::path plotRoot = std::filesystem::path(gOutputDir) / "plots";
  if (!std::filesystem::exists(plotRoot))
  {
    if (!std::filesystem::create_directories(plotRoot))
    {
      logStream() << "Phase plot skip: cannot create plot directory " << plotRoot << endl;
      return;
    }
  }

  std::filesystem::path scriptPath = plotRoot / "plot_phase_profile.py";
  if (!ensurePhasePlotScript(scriptPath))
  {
    logStream() << "Phase plot skip: cannot write python script " << scriptPath << endl;
    return;
  }

  std::filesystem::path pngPath = txtPath;
  if (pngPath.has_extension())
    pngPath.replace_extension(".png");
  else
    pngPath += ".png";

  const std::string pythonExe = "D:/Compiler/Anaconda3/python.exe";
  std::string cmd = pythonExe + " " + scriptPath.string() + " " +
                    txtPath.string() + " " + pngPath.string() + " " +
                    to_string(phaseIdx) + " " + to_string(runIdx);

  int ret = system(cmd.c_str());
  if (ret != 0)
  {
    logStream() << "Phase plot failed for phase " << phaseIdx
                << " run " << runIdx << ". Input file: " << txtPath << endl;
  }
}

static void writeGroupPlotDataAndImages(UserDataType *data, realtype t,
                                        const realtype *ydotd)
{
  if (data == nullptr || ydotd == nullptr)
    return;

  static int lastPlottedRun = -1;
  if (data->current_run < 0 || data->current_run == lastPlottedRun)
    return;

  if (data->current_target_t > ZERO && t < data->current_target_t * RCONST(0.999))
    return;

  lastPlottedRun = data->current_run;

  std::filesystem::path plotDir = std::filesystem::path(gOutputDir) / "plots";
  if (!std::filesystem::exists(plotDir))
  {
    if (!std::filesystem::create_directories(plotDir))
    {
      logStream() << "Plot skip: cannot create plot directory " << plotDir << endl;
      return;
    }
  }

  std::filesystem::path lineDir = plotDir / "avg_n";
  std::filesystem::path barDir = plotDir / "gmap_span";

  auto ensureDir = [](const std::filesystem::path &dir)
  {
    if (std::filesystem::exists(dir))
      return true;
    return std::filesystem::create_directories(dir);
  };

  if (!ensureDir(lineDir) || !ensureDir(barDir))
  {
    logStream() << "Plot skip: cannot create sub directories under " << plotDir << endl;
    return;
  }

  string runTag = to_string(data->current_run);
  std::filesystem::path lineCsv = lineDir / ("avg_n_run" + runTag + ".csv");
  std::filesystem::path barCsv = barDir / ("gmap_span_run" + runTag + ".csv");

  ofstream lineFile(lineCsv.string(), ios::out | ios::trunc);
  ofstream barFile(barCsv.string(), ios::out | ios::trunc);
  if (!lineFile.is_open() || !barFile.is_open())
  {
    logStream() << "Plot skip: cannot open CSV files for run " << data->current_run << endl;
    return;
  }

  lineFile << "group";
  for (int p = 0; p < numCalcPhase; p++)
    lineFile << ",phase_" << p;
  lineFile << endl;

  for (int g = 0; g < numGroups; g++)
  {
    lineFile << g;
    for (int p = 0; p < numCalcPhase; p++)
    {
      int p_base = p * numGroups * 2;
      realtype denominator = ydotd[p_base + 2 * g];
      realtype numerator = ydotd[p_base + 2 * g + 1];

      realtype avg_n = ZERO;
      if (std::isfinite(denominator) && std::isfinite(numerator) &&
          std::abs(denominator) > RCONST(1e-40))
      {
        avg_n = numerator / denominator;
      }
      lineFile << "," << avg_n;
    }
    lineFile << endl;
  }

  barFile << "group,n_max,n_min" << endl;
  for (int g = 0; g < numGroups; g++)
  {
    barFile << g << "," << GMap[g].n_max << "," << GMap[g].n_min << endl;
  }

  lineFile.close();
  barFile.close();

  std::filesystem::path scriptPath = plotDir / "plot_groups.py";
  if (!ensureGroupPlotScript(scriptPath))
  {
    logStream() << "Plot skip: cannot write python script " << scriptPath << endl;
    return;
  }

  std::filesystem::path linePng = lineDir / ("avg_n_run" + runTag + ".png");
  std::filesystem::path barPng = barDir / ("gmap_span_run" + runTag + ".png");

  const std::string pythonExe = "D:/Compiler/Anaconda3/python.exe";
  string cmd = pythonExe + " " + scriptPath.string() + " " +
               lineCsv.string() + " " + barCsv.string() + " " +
               linePng.string() + " " + barPng.string();
  int ret = system(cmd.c_str());

  if (ret == 0)
  {
    logStream() << "Plot generated for run " << data->current_run
                << ": " << linePng << " and " << barPng << endl;
  }
  else
  {
    logStream() << "Plot script failed for run " << data->current_run
                << ". CSV files kept: " << lineCsv << " and " << barCsv << endl;
  }
}

static void getGroupFlux(UserDataType *data, N_Vector y, realtype *J_M)
{
  realtype *yd = NV_DATA_S(y);
  realtype solP[numCalcPhase], sumwp, wp, wpEff;

  for (int p = 0; p < numPhase; p++)
  {
    solP[p] = 1.0;
    for (int c = 0; c < numComp; c++)
    {
      solP[p] *= pow(yd[neq_groups - numComp + c], IMaterial->X[p][c]);
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
      realtype wp_temp = yd[neq_groups - numComp + c] * D[c] *
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

    realtype dG_surf_0 = surfTerm[pref] * (pow(2.0, 2.0 / 3.0) - pow(1.0, 2.0 / 3.0));

    realtype dG_disl_0 = get_delG_disl(p, 2.0, pref) - get_delG_disl(p, 1.0, pref);
    realtype dG_disl_kT_0 = dG_disl_0 / (kb * ICond->Temp);

    realtype emission_ratio_0 = (IMaterial->solPBar[pref] / solP[pref]) * exp(dG_surf_0 + dG_disl_kT_0);
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
        wp = yd[neq_groups - numComp + c] * D[c] * ((4.0 * pi * r_b) / IMaterial->aVol);
        sumwp += (nu[pref][c] / wp);
      }
      wpEff = 1.0 / sumwp;
      if (wpEff < data->min_wpEff_seen)
        data->min_wpEff_seen = wpEff;
      if (!std::isfinite(wpEff) || !std::isfinite(sumwp))
        data->non_finite_count++;

      // 2. 评估界面左侧浓度 (C_right_g) -> 驱动向右的生长通量
      realtype C_right_g = 1e-30;
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
      }

      // 3. 评估界面右侧浓度 (C_left_gp1) -> 驱动向左的溶解通量
      realtype C_left_g = 1e-30;
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
      }

      // 4. 界面能修正与通量计算 (完美迎风格式)
      realtype current_surfTerm = surfTerm[pref];

      // Cu 相 (pref == 2) 跨越 CutoCRP (20原子) 时的动态界面能
      if (pref == 2 && n_b >= 20.0)
      {
        current_surfTerm = surfTerm[numPhase + 1];
        // 使用预设的跨越 CutoCRP 后的界面能参数
      }

      // 1. 表面能带来的势垒差 (无量纲)
      realtype dG_surf_g = current_surfTerm * (pow(n_b + 1.0, 2.0 / 3.0) - pow(n_b, 2.0 / 3.0));

      // 2. 位错应变能释放带来的势垒差 (无量纲)
      realtype dG_disl_g = get_delG_disl(p, n_b + 1.0, pref) - get_delG_disl(p, n_b, pref);
      realtype dG_disl_kT_g = dG_disl_g / (kb * ICond->Temp);

      // 3. 【统一合并】：物理意义为 exp([ΔG_surf + ΔG_disl] / kT)
      realtype emission_ratio = (IMaterial->solPBar[pref] / solP[pref]) * exp(dG_surf_g + dG_disl_kT_g);

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

    ydotd[p_base] = -data->J_M[fluxIndex(p, 1)];     // M0_0
    ydotd[p_base + 1] = -data->J_M[fluxIndex(p, 1)]; // M1_0

    for (int g = 1; g < numGroups; g++)
    {
      ydotd[p_base + 2 * g + 1] = data->J_M[fluxIndex(p, g)] * GMap[g].n_min - data->J_M[fluxIndex(p, g + 1)] * GMap[g].n_max;
      ydotd[p_base + 2 * g] = ydotd[p_base + 2 * g + 1] / GMap[g].n_center;

      for (int c = 0; c < numComp; c++)
      {
        total_sol_cons[c] += IMaterial->X[pref][c] * ydotd[p_base + 2 * g + 1];
      }
    }
  }

  // 在每个 run 的目标时刻附近导出一次分组数据并生成图像
  // writeGroupPlotDataAndImages(data, t, ydotd);

  // =========================================================================
  // 【核心机制融合】：统一计算 MNS 相 (T3 和 T6) 的热力学驱动力与实际浓度积
  // =========================================================================
  realtype driving_force_MNS[numPhase - 1] = {0.0};
  realtype solP_MNS[numPhase - 1] = {1.0, 1.0}; // 预存实际浓度积，避免重复使用 pow 计算
  realtype total_df_MNS = 0.0;

  for (int p_MNS = 0; p_MNS < numPhase - 1; p_MNS++)
  {
    for (int c = 0; c < numComp; c++)
    {
      solP_MNS[p_MNS] *= pow(yd[neq_groups - numComp + c], IMaterial->X[p_MNS][c]);
    }
    // 驱动力 = 实际溶度积 / 平衡溶度积。若小于 1 则说明未过饱和。
    realtype df = solP_MNS[p_MNS] / IMaterial->solPBar[p_MNS];
    driving_force_MNS[p_MNS] = (df > 1.0) ? (df - 1.0) : 0.0;
    total_df_MNS += driving_force_MNS[p_MNS];
  }

  // =========================================================================
  // 【回归原版设计的机制 1】：平行伴侣触发 (Cu 跨越 20 时，催化伴生一个 22 原子的 MNS)
  // =========================================================================
  int p_Cu = 2;
  realtype CutoCRP = 20.0;
  realtype MNS_to_Cu_ratio = 1.1;

  int g_Cu_crit = cluNumTGroupIndex(CutoCRP);
  realtype companion_mns_atoms = CutoCRP * MNS_to_Cu_ratio;
  int g_MNS_seed = cluNumTGroupIndex(companion_mns_atoms);

  // 获取 Cu 团簇跨越 20 的通量，用作“催化率”
  realtype coupling_flux = data->J_M[fluxIndex(p_Cu, g_Cu_crit)];

  if (coupling_flux > ZERO)
  {
    // 注意：绝对不要从 p_Cu 中减去 coupling_flux！
    // Cu 团簇继续保留在 p_Cu 中作为 Core 演化。

    if (total_df_MNS > ZERO)
    {
      for (int p_MNS = 0; p_MNS < numPhase - 1; p_MNS++)
      {
        realtype fraction = driving_force_MNS[p_MNS] / total_df_MNS;
        if (fraction > 1e-6)
        {
          realtype phase_flux = coupling_flux * fraction;
          int mns_base = p_MNS * numGroups * 2;

          // 在 MNS 相中生成一个独立的“伴侣/外壳”团簇
          ydotd[mns_base + 2 * g_MNS_seed] += phase_flux;
          ydotd[mns_base + 2 * g_MNS_seed + 1] += phase_flux * companion_mns_atoms;

          // 消耗基体的 Mn, Ni, Si 来形成这个伴侣
          for (int c = 0; c < numComp; c++)
          {
            total_sol_cons[c] += phase_flux * companion_mns_atoms * IMaterial->X[p_MNS][c];
          }
        }
      }
    }
    else
    {
      // Fallback: 驱动力不足时挂载给 T3
      int p_fallback = 0;
      int mns_base = p_fallback * numGroups * 2;

      ydotd[mns_base + 2 * g_MNS_seed] += coupling_flux;
      ydotd[mns_base + 2 * g_MNS_seed + 1] += coupling_flux * companion_mns_atoms;

      for (int c = 0; c < numComp; c++)
      {
        total_sol_cons[c] += coupling_flux * companion_mns_atoms * IMaterial->X[p_fallback][c];
      }
    }
  }

  // =========================================================================
  // 【机制 2】：MNS 相的级联直接形核 (复用相同的竞争比例)
  // =========================================================================
  if (total_df_MNS > ZERO)
  {
    for (int p_MNS = 0; p_MNS < numPhase - 1; p_MNS++)
    {
      realtype fraction = driving_force_MNS[p_MNS] / total_df_MNS;

      if (fraction > 1e-6)
      {
        // 直接复用之前算好的 solP_MNS[p_MNS] 结合 fraction 计算最终分配的生成率
        realtype GR_p = IProp->Alpha * Flux * IProp->ccs * solP_MNS[p_MNS] / IProp->RsolP * fraction;

        int g_hg = cluNumTGroupIndex(IProp->HGSize);
        int p_hg = p_MNS + numPhase; // 异质形核相索引 (T3对应3, T6对应4)
        int hg_base = p_hg * numGroups * 2;

        ydotd[hg_base + 2 * g_hg] += GR_p;
        ydotd[hg_base + 2 * g_hg + 1] += IProp->HGSize * GR_p;

        for (int c = 0; c < numComp; c++)
        {
          total_sol_cons[c] += IMaterial->X[p_MNS][c] * (IProp->HGSize * GR_p);
        }
      }
    }
  }

  // =========================================================================
  // 【机制 3】：纯 Cu 相的级联形核 (独立计算)
  // =========================================================================
  int HGPhase_Cu = 2;       // 纯 Cu 相的 pref 索引
  int HGSize_Cu = 16;       // 文献中 n_Cu-het = 16 atoms
  realtype Alpha_Cu = 0.03; // 文献中 alpha_Cu = 0.03

  realtype solP_Cu = 1.0;
  for (int c = 0; c < numComp; c++)
  {
    realtype safe_C = std::max(1e-30, yd[neq_groups - numComp + c]);
    solP_Cu *= pow(safe_C, IMaterial->X[HGPhase_Cu][c]);
  }

  // 按照驱动力比例缩放生成率
  realtype GR_Cu = Alpha_Cu * Flux * IProp->ccs * solP_Cu / IMaterial->solPBar[HGPhase_Cu];

  int g_hg_Cu = cluNumTGroupIndex(HGSize_Cu);
  int p_hg_Cu = HGPhase_Cu + numPhase;
  int hg_base_Cu = p_hg_Cu * numGroups * 2;

  ydotd[hg_base_Cu + 2 * g_hg_Cu] += GR_Cu;
  ydotd[hg_base_Cu + 2 * g_hg_Cu + 1] += HGSize_Cu * GR_Cu;

  for (int c = 0; c < numComp; c++)
  {
    total_sol_cons[c] += IMaterial->X[HGPhase_Cu][c] * (HGSize_Cu * GR_Cu);
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
    std::filesystem::path profilePath(profStr);

    ofstream P_file(profilePath.string(), ios::out | ios::trunc);
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

    // plotPhaseProfile(profilePath, p, runIdx);
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