#ifndef Input_h
#define Input_h

// 确保在定义结构体之前包含 SUNDIALS 类型定义
#include <sundials/sundials_types.h>
#include "Constants.h"

// 在 C++ 中，定义结构体后不需要再用 typedef 声明指针（虽然可以，但建议分开写清楚）
struct InputMaterial
{
    realtype aLat, aVol, C0[numComp], D[numComp], DFe, X[numPhase][numComp], aP[numPhase], cVol[numPhase], sig[numPhase], solPBar[numPhase];
};

struct InputCondition
{
    realtype Temp, Flux;
};

struct InputProperty
{
    int HGSize, HGPhase;
    realtype Alpha, RsolP, ccs, Rflux, p_factor, DDP, DCB, SigmaDpa, DV, rv;
};
// Cascade efficiency, v-i recombination radius,binding energy of di-vacancyies, dislocation densities

void LoadInput(InputCondition *ICond, InputMaterial *IMaterial, InputProperty *IProp);

#endif