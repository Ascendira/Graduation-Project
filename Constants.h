#ifndef Constant_h
#define Constant_h

#include <sundials/sundials_types.h>

// 兼容 SUNDIALS 7.0+ 版本的类型与宏变更
#ifndef RCONST
#define RCONST SUN_RCONST
typedef sunrealtype realtype;
#endif

// 二元分组法
#define numDiscrete 100     // 前100个集群保持离散（每组1个）
#define numGroups 5000       // 总分组数（包含离散和分组部分）
#define groupingFactor 1.002 // 分组宽度增长因子

struct GroupMap
{
  realtype n_min;    // 组内最小原子数
  realtype n_max;    // 组内最大原子数
  realtype width;    // 组宽度 (n_max - n_min + 1)
  realtype n_center; // 组中心尺寸 (用于简化计算或初始化)
};

#define ZERO RCONST(0.0) // zero
#define ONE RCONST(1.0)  // one
#define TWO RCONST(2.0)  // two

#define kb RCONST(8.617E-5) // Boltzmann
#define pi RCONST(3.141592) // pi

#define numPhase 2     // Number of precipitating phases
#define numComp 3      // Number of precipitating components
#define numClass 50000 // Number of cluster classes/maximum cluster size considered
#define runs 50        // Number of loops to run
#define CutoffSize 65  // Cutoff size used for output
#define RadiusCalc \
  radM2         // Method to calculate mean radius of precipitate, radM1 or radM2
#define T0 ZERO // Initial time

#define numCalcPhase (numPhase * 2) // number of calculating phases,including both home and heter nucleated phases
#define neq_groups (numCalcPhase * numGroups * 2 + numComp)

#define RTOL RCONST(1.0E-6)  // rel tolerance
#define ATOL RCONST(1.0E-30) // abs tolerance (non-zero to avoid over-solving tiny states)

#endif
