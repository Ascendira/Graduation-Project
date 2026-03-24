// Code developed by Huibin Ke from UW-Madison for the evolution of Mn-Ni-Si
// precipitates in Reactor Pressure Vessel stees.
#include <cvode/cvode.h> /* main integrator header file */
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fct. and macros */
#include <sundials/sundials_math.h>  /* contains the macros ABS, SQR, and EXP */
#include <sundials/sundials_types.h> /* definition of realtype */

#include <sunlinsol/sunlinsol_band.h> /* 线性解法器定义 */
#include <sunmatrix/sunmatrix_band.h> /* 矩阵定义 */

#include <iostream>
#include <filesystem>

#include "Constants.h" /*Constants header file*/
#include "Input.h"     /*Input parameter header file*/

InputCondition *ICond;    /*Defined in Input.h, including irradiation conditions*/
InputMaterial *IMaterial; /*Defined in Input.h, including material information*/
InputProperty *IProp;     /*Defined in Input.h, including all other parameters used in model*/
GroupMap GMap[numGroups];

using namespace std;

/*function defs*/

struct UserDataType
{
    // 基础物理参数 (对应每个组的中心或边界)
    realtype *n_center;   // 每个组的中心原子数
    realtype **radGroup;  // 每个相、每个组对应的平均半径
    realtype **beta;      // 每个相、每个组的吸收系数部分
    
    // 通量缓存 (用于 getGroupFlux 和 f 函数之间的计算传递)
    realtype *J_M0;       // 0阶矩通量缓存 (大小: numCalcPhase * numGroups)
    realtype *J_M1;       // 1阶矩通量缓存 (大小: numCalcPhase * numGroups)

    // 构造函数：根据 numGroups 动态分配内存
    UserDataType()
    {
        // 1. 分配组相关的基础数组
        n_center = new realtype[numGroups]();
        
        // 2. 分配多相矩阵 (用于存储各相特有的组属性)
        radGroup = new realtype *[numPhase];
        beta     = new realtype *[numPhase];

        for (int p = 0; p < numPhase; p++)
        {
            radGroup[p] = new realtype[numGroups]();
            beta[p]     = new realtype[numGroups]();
        }

        // 3. 分配通量缓存
        // 必须覆盖所有计算相 (Homo + Heter)
        J_M0 = new realtype[numCalcPhase * numGroups]();
        J_M1 = new realtype[numCalcPhase * numGroups]();
    }

    // 析构函数：严格释放内存，防止长时间运行下的内存泄漏
    ~UserDataType()
    {
        delete[] n_center;

        for (int p = 0; p < numPhase; p++)
        {
            delete[] radGroup[p];
            delete[] beta[p];
        }
        
        delete[] radGroup;
        delete[] beta;

        delete[] J_M0;
        delete[] J_M1;
    }
};


static void initParams();
static void GetRED(realtype D[numComp], realtype Flux);
static void getOutput(N_Vector y, realtype radM1[numCalcPhase], realtype radM2[numCalcPhase], realtype rhoC[numCalcPhase]);
static void printYVector(N_Vector y, int runIdx);
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static void initGrouping();

/*global problem parameters*/
realtype D[numComp], aP[numPhase], nu[numPhase][numComp];
/*Diffusion coefficient,effective precipitate lattice constant, square of precipitate composition*/

realtype rhoC[numCalcPhase], radM1[numCalcPhase], radM2[numCalcPhase];
/*precipitate number density, two kinds of mean radius (see readme for more detail)*/

realtype Flux, solProd[numPhase];
/*Irradiation flux, solute product */

int main() {
    // 1. 初始化 SUNDIALS 上下文与物理参数
    SUNContext sunctx;
    SUNContext_Create(SUN_COMM_NULL, &sunctx); //

    // 分配输入参数内存
    ICond = new InputCondition();
    IMaterial = new InputMaterial();
    IProp = new InputProperty();

    LoadInput(ICond, IMaterial, IProp); // 加载材料常数
    initGrouping(); // 关键：必须先初始化分组地图，确定 numGroups
    initParams();   // 计算扩散系数 D

    // 2. 定义并初始化 N_Vector y (求解器的核心向量)
    // 这里的长度是 neq_groups = (4 * numGroups * 2) + 3
    N_Vector y = N_VNew_Serial(neq_groups, sunctx); 
    realtype *yd = NV_DATA_S(y);

    // 初始值填充：所有矩变量设为极小值，溶质设为初始浓度
    for (int i = 0; i < neq_groups; i++) yd[i] = 1.0E-30;

    // 设置矩阵溶质初始浓度 (位于向量末尾)
    for (int c = 0; c < numComp; c++) {
        yd[neq_groups - numComp + c] = IMaterial->C0[c];
    }

    // 设置单体 (n=1) 的初始矩：将有效单体浓度放入各相的第0个组
    for (int p = 0; p < numPhase; p++) {
        realtype solProd_init = 1.0;
        for (int c = 0; c < numComp; c++) 
            solProd_init *= pow(IMaterial->C0[c], IMaterial->X[p][c]);
        
        int p_base = p * numGroups * 2;
        yd[p_base] = solProd_init;               // M0_g0 (数量)
        yd[p_base + numGroups] = solProd_init * 1.0; // M1_g0 (质量，因为 n=1)
    }

    // 3. 配置 CVODE 求解器
    void *cvode_mem = CVodeCreate(CV_BDF, sunctx); //
    UserDataType *data = new UserDataType();      // 包含通量缓存的自定义结构
    for(int g=0; g<numGroups; g++) data->n_center[g] = GMap[g].n_center;

    CVodeSetUserData(cvode_mem, data);
    CVodeInit(cvode_mem, f, T0, y); // 关联导数函数 f
    CVodeSStolerances(cvode_mem, RTOL, ATOL); //

    // 线性解法器配置 (使用带状矩阵优化性能)
    SUNMatrix A = SUNBandMatrix(neq_groups, numGroups*2, numGroups*2, sunctx); 
    SUNLinearSolver LS = SUNLinSol_Band(y, A, sunctx);
    CVodeSetLinearSolver(cvode_mem, LS, A);

    // 4. 计算循环与输出
    realtype t, tout = 1.0; 
    ofstream O_file("Output_Grouping.txt");
    O_file << "Time(s)\tMn\tNi\tSi\tRad_Phase1\tRho_Phase1..." << endl;

    for (int i = 0; i < runs; i++) {
        int flag = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
        if (flag < 0) break;

        // --- 调用统计函数 ---
        getOutput(y, radM1, radM2, rhoC); // 计算 R 和 rho
        printYVector(y, i);              // 打印尺寸分布快照

        // 写入主输出
        O_file << t << "\t" << yd[neq_groups-3] << "\t" << yd[neq_groups-2] << "\t" << yd[neq_groups-1];
        for (int p = 0; p < numCalcPhase; p++) {
            O_file << "\t" << radM2[p] << "\t" << rhoC[p];
        }
        O_file << endl;

        tout *= 1.5; // 对数步长推进
    }

    // 5. 释放资源
    N_VDestroy(y);
    delete data;
    CVodeFree(&cvode_mem);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    SUNContext_Free(&sunctx);
    return 0;
}
/**
 * 初始化集群分组地图
 * 将 1 到 50000+ 的集群映射到有限数量的组 (numGroups) 中
 */
static void initGrouping() {
    realtype current_n = 1.0;
    realtype current_width = 1.0;

    for (int g = 0; g < numGroups; g++) {
        GMap[g].n_min = current_n;

        if (g < numDiscrete) {
            // 离散阶段：每组宽度为 1
            GMap[g].width = 1.0;
        } else {
            // 分组阶段：宽度按几何级数增长 (类似 Xolotl 的空间分布逻辑)
            // 确保宽度至少为 1 且为整数
            current_width *= groupingFactor;
            GMap[g].width = std::max(1.0, std::round(current_width));
        }

        GMap[g].n_max = GMap[g].n_min + GMap[g].width - 1.0;
        GMap[g].n_center = (GMap[g].n_min + GMap[g].n_max) / 2.0;

        // 为下一组准备起始位置
        current_n = GMap[g].n_max + 1.0;
    }
    
    // 自动更新最大集群尺寸
    printf("Grouping complete. Max cluster size considered: %.0f atoms\n", GMap[numGroups-1].n_max);
}

/**
 * 差值法：计算组内平均原子数 n_avg
 * $n_{avg} = \frac{M_1}{M_0}$
 */
static realtype getAvgN(realtype M0, realtype M1, int groupIdx) {
    if (M0 < 1e-40) return GMap[groupIdx].n_center;
    
    realtype n_avg = M1 / M0;
    
    // 边界检查：确保平均值不超出组的物理范围
    if (n_avg < GMap[groupIdx].n_min) return GMap[groupIdx].n_min;
    if (n_avg > GMap[groupIdx].n_max) return GMap[groupIdx].n_max;
    
    return n_avg;
}

/**
 * 查找特定尺寸 n 属于哪个组
 * 用于将初始浓度分布映射到组上
 */
static int findGroupIndex(realtype n) {
    if (n < 1) return 0;
    for (int g = 0; g < numGroups; g++) {
        if (n >= GMap[g].n_min && n <= GMap[g].n_max) {
            return g;
        }
    }
    return numGroups - 1;
}

/**
 * 基于 GroupMap 和矩重构计算组间通量
 * y 向量布局: [Phase0_M0(numGroups), Phase0_M1(numGroups), Phase1_M0... , Solutes(numComp)]
 */
static void getGroupFlux(UserDataType *data, N_Vector y, realtype *J_M0, realtype *J_M1) {
    realtype *yd = NV_DATA_S(y);
    int M1_offset = numGroups; // M1 变量相对于 M0 的偏移
    
    // 获取当前的矩阵溶质浓度 (最后 numComp 个元素)
    realtype solP[numCalcPhase];
    for (int p = 0; p < numCalcPhase; p++) {
        int pref = p % numPhase;
        solP[p] = 1.0;
        for (int c = 0; c < numComp; c++) {
            // 从 y 向量末尾提取溶质浓度
            realtype C_solute = yd[neq - numComp + c];
            solP[p] *= pow(C_solute, IMaterial->X[pref][c]);
        }
    }

    for (int p = 0; p < numCalcPhase; p++) {
        int pref = p % numPhase;
        int p_base = p * numGroups * 2;

        // 遍历所有组，计算 g -> g+1 的通量
        for (int g = 0; g < numGroups - 1; g++) {
            realtype M0_g = yd[p_base + g];
            realtype M1_g = yd[p_base + g + M1_offset];
            
            // 1. 计算组内平均尺寸和物理属性
            realtype n_avg = getAvgN(M0_g, M1_g, g);
            realtype r_avg = pow(3.0 * IMaterial->cVol[pref] * n_avg / (4.0 * pi), 1.0/3.0);
            
            // 吸收速率 beta (与半径成正比)
            realtype beta_g = (4.0 * pi * aP[pref] * r_avg * D[0]) / IMaterial->aVol; 

            // 2. 矩重构：计算组边界 n_max 处的有效浓度 C_boundary
            // 如果 M0 很小，浓度趋于 0
            realtype C_boundary = 0.0;
            if (M0_g > 1e-35) {
                // 计算偏离度：平均尺寸相对于中心点的偏移
                realtype delta_n = n_avg - GMap[g].n_center;
                // 计算斜率因子 (简化的线性近似)
                realtype slope = (GMap[g].width > 1) ? (12.0 * delta_n / (GMap[g].width * GMap[g].width)) : 0;
                // 边界处的浓度 = 平均浓度 * (1 + 相对斜率补正)
                C_boundary = (M0_g / GMap[g].width) * (1.0 + slope * (GMap[g].n_max - GMap[g].n_center));
            }

            // 3. 计算跨组净通量 J = 吸收 - 发射
            // 这里假设跨组是通过单原子吸附/脱附实现的
            realtype flux = beta_g * (solP[p] * C_boundary - 
                            (IMaterial->solPBar[pref] / solP[p]) * C_boundary); 

            // 4. 记录 0 阶和 1 阶通量 [cite: 101, 105, 106]
            J_M0[p * numGroups + g] = flux;
            // 1 阶通量代表原子总数的转移，需乘以边界处的原子数
            J_M1[p * numGroups + g] = GMap[g].n_max * flux; 
        }
        
        // 边界处理：最后一组通量设为 0
        J_M0[p * numGroups + numGroups - 1] = 0.0;
        J_M1[p * numGroups + numGroups - 1] = 0.0;
    }
}

/*************************************************

This function initialize parameters used in model

*************************************************/
static void initParams()
{
  Flux = ICond->Flux; 
  /*Irradiatio flux*/
  for (int i = 0; i < numComp; i++)
  {
    D[i] = IMaterial->D[i]; 
    /*Thermal diffusion coefficients*/
  }
  GetRED(D, Flux); 
  /*Calculate radiation enhanced diffusion coefficients*/
  for (int p = 0; p < numPhase; p++)
  {
    aP[p] = pow((3 * IMaterial->cVol[p]) / (4 * pi), 1. / 3.); 
    /*Effective atomic radius in precipitate*/
  }
  for (int p = 0; p < numPhase; p++)
  {
    for (int c = 0; c < numComp; c++)
    {
      nu[p][c] = pow(IMaterial->X[p][c], 2); 
      /*Square of precipitate composition*/
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

  /*When flux is higher than reference flux, use p-scaling (Eq. SI-18) to calculate gs*/
  if (Flux > IProp->Rflux)
  {
    Eta = 16 * pi * IProp->rv * IProp->DCB * IProp->SigmaDpa * IProp->Rflux /
          IMaterial->aVol / IProp->DV / pow(IProp->DDP, 2);
    Gs = 2.0 / Eta * (pow(1 + Eta, 0.5) - 1.0) *
         pow((IProp->Rflux / Flux), IProp->p_factor);
  }
  /*When flux is lower than reference flux, use Eq. SI-19 and SI-20 to calculate gs*/
  else
  {
    Eta = 16 * pi * IProp->rv * IProp->DCB * IProp->SigmaDpa * Flux /
          IMaterial->aVol / IProp->DV / pow(IProp->DDP, 2);
    Gs = 2.0 / Eta * (pow(1 + Eta, 0.5) - 1.0);
  }
  Cvr = IProp->DCB * Flux * IProp->SigmaDpa * Gs / (IProp->DV * IProp->DDP); 
  /*Calculate vacancy concentration under irradiation, Eq. SI-17*/
  for (int i = 0; i < numComp; i++)
  {
    D[i] = D[i] + IProp->DV * Cvr * D[i] / IMaterial ->DFe; /*Radiation enhanced diffusion coefficients, Eq. SI-16*/
  }
}

/**
 * 导数函数 f: 计算 dM0/dt, dM1/dt 和 dC_matrix/dt
 */
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
    realtype *yd = NV_DATA_S(y);
    realtype *ydotd = NV_DATA_S(ydot);
    UserDataType *data = static_cast<UserDataType *>(user_data);

    // 1. 初始化导数向量
    for (int i = 0; i < neq; i++) ydotd[i] = 0.0;

    // 2. 调用重写的 getGroupFlux 获取跨组通量
    // J_M0[p * numGroups + g] 是从组 g 到 g+1 的数量通量
    // J_M1[p * numGroups + g] 是从组 g 到 g+1 的原子质量通量
    getGroupFlux(data, y, data->J_M0, data->J_M1);

    // 用于计算溶质消耗的累加器 [Mn, Ni, Si]
    realtype total_solute_consumption[numComp] = {0.0, 0.0, 0.0};

    // 3. 遍历每个相 (Homo/Heter) 和每个组
    for (int p = 0; p < numCalcPhase; p++) {
        int pref = p % numPhase;
        int p_base = p * numGroups * 2;
        int M0_base = p_base;
        int M1_base = p_base + numGroups;

        for (int g = 0; g < numGroups; g++) {
            // 获取流入和流出的通量 (组 g 的输入来自 g-1)
            realtype J0_in  = (g == 0) ? 0.0 : data->J_M0[p * numGroups + g - 1];
            realtype J0_out = data->J_M0[p * numGroups + g];

            realtype J1_in  = (g == 0) ? 0.0 : data->J_M1[p * numGroups + g - 1];
            realtype J1_out = data->J_M1[p * numGroups + g];

            // --- 0阶矩方程: dM0/dt = J_in - J_out ---
            ydotd[M0_base + g] = J0_in - J0_out;

            // --- 1阶矩方程: dM1/dt = n_in*J_in - n_out*J_out ---
            ydotd[M1_base + g] = J1_in - J1_out;

            // --- 质量守恒累加 ---
            // 溶质原子从矩阵转移到沉淀中，消耗率 = \sum (X_p,c * dM1_p,g/dt)
            for (int c = 0; c < numComp; c++) {
                total_solute_consumption[c] += IMaterial->X[pref][c] * ydotd[M1_base + g];
            }
        }
    }

    // 4. 处理级联损伤产生 (Cascade Damage / Heterogeneous Nucleation)
    // 根据原代码逻辑计算 GR
    realtype solP_pref = 1.0; 
    for (int c = 0; c < numComp; c++) {
        solP_pref *= pow(yd[neq - numComp + c], IMaterial->X[IProp->HGPhase][c]);
    }
    realtype GR = IProp->Alpha * Flux * IProp->ccs * solP_pref / IProp->RsolP; // Eq. 5

    // 找到级联产生尺寸 HGSize 属于哪个组
    int g_hg = findGroupIndex(IProp->HGSize);
    int p_hg = IProp->HGPhase + numPhase; // 异质相索引布局
    
    // 更新该组的矩变量
    ydotd[p_hg * numGroups * 2 + g_hg] += GR;              // M0 增加
    ydotd[p_hg * numGroups * 2 + numGroups + g_hg] += IProp->HGSize * GR; // M1 增加 (原子总数)

    // 补充级联引起的溶质消耗
    for (int c = 0; c < numComp; c++) {
        total_solute_consumption[c] += IMaterial->X[IProp->HGPhase][c] * (IProp->HGSize * GR);
    }

    // 5. 更新矩阵溶质浓度: dC/dt = - 沉淀消耗
    for (int c = 0; c < numComp; c++) {
        ydotd[neq - numComp + c] = -total_solute_consumption[c];
    }

    return 0;
}

/*****************************************************************

This function calculates mean cluster radius and cluster density

*****************************************************************/

/**
 * 从矩变量计算物理输出：平均半径和数量密度
 */
static void getOutput(N_Vector y, realtype radM1[numCalcPhase], realtype radM2[numCalcPhase], realtype rhoC[numCalcPhase]) {
    realtype *yd = NV_DATA_S(y);
    int M1_off = numGroups;

    for (int p = 0; p < numCalcPhase; p++) {
        int p_base = p * numGroups * 2;
        int pref = p % numPhase;

        realtype total_M0 = 0.0;
        realtype sum_Ri_M0 = 0.0;
        realtype total_M1 = 0.0;

        // 遍历所有组，统计大于 CutoffSize 的集群
        for (int g = 0; g < numGroups; g++) {
            if (GMap[g].n_max < CutoffSize) continue;

            realtype M0 = yd[p_base + g];
            realtype M1 = yd[p_base + g + M1_off];
            
            if (M0 < 1e-35) continue;

            // 计算该组当前的平均尺寸 n_avg = M1 / M0
            realtype n_avg = M1 / M0;
            // 计算对应的半径 R(n_avg)
            realtype r_avg = pow(3.0 * IMaterial->cVol[pref] * n_avg / (4.0 * pi), 1.0 / 3.0);

            total_M0 += M0;
            sum_Ri_M0 += r_avg * M0;
            total_M1 += M1;
        }

        if (total_M0 > 1e-35) {
            // 方法 1: 算术平均半径 \sum(R_i * N_i) / \sum N_i
            radM1[p] = sum_Ri_M0 / total_M0;

            // 方法 2: 等效体积半径 R(\sum n_i * N_i / \sum N_i)
            realtype n_mean = total_M1 / total_M0;
            radM2[p] = pow((n_mean * IMaterial->cVol[pref]) / ((4.0 / 3.0) * pi), 1.0 / 3.0);

            // 数量密度 (单位: 1/m^3)
            rhoC[p] = total_M0 / IMaterial->aVol;
        } else {
            radM1[p] = 0.0; radM2[p] = 0.0; rhoC[p] = 0.0;
        }
    }
}

/**********************************************************************************************

This function prints cluster size distribution in the file Profile for the final
solution time.

***********************************************************************************************/
/**
 * 输出分组后的尺寸分布快照
 */
static void printYVector(N_Vector y, int runIdx) {
    realtype *yd = NV_DATA_S(y);
    int M1_off = numGroups;

    for (int p = 0; p < numCalcPhase; p++) {
        int pref = p % numPhase;
        string profStr = "Profile_P" + to_string(p) + "_Run" + to_string(runIdx) + ".txt";
        ofstream P_file(profStr);

        P_file << "n_min\tn_max\tn_avg\tradius(m)\tdensity(1/m3)" << endl;

        int p_base = p * numGroups * 2;
        for (int g = 0; g < numGroups; g++) {
            realtype M0 = yd[p_base + g];
            realtype M1 = yd[p_base + g + M1_off];
            
            realtype n_avg = (M0 > 1e-35) ? (M1 / M0) : GMap[g].n_center;
            realtype r_avg = pow(3.0 * IMaterial->cVol[pref] * n_avg / (4.0 * pi), 1.0 / 3.0);
            
            // 注意：输出密度时要除以组宽度，得到单位尺寸的浓度
            P_file << GMap[g].n_min << "\t" 
                   << GMap[g].n_max << "\t"
                   << n_avg << "\t"
                   << r_avg << "\t"
                   << (M0 / GMap[g].width) / IMaterial->aVol << endl;
        }
        P_file.close();
    }
}