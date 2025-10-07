# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 15:53:25 2025

@author: chengf
"""

import numpy as np
import pandas as pd

#rho=pure rate of time preference
rho=0.5 #unit:%
#rho=0
#eta= absolute value of elasticity of marginal utility of consumption
eta=1 #no unit
#eta=1.45

base_year = 2025

path="C:/codes/spyder/"

df = pd.read_excel(path+"test_data.xlsx",header=None, skiprows=2)
df=np.array(df)
years = df[:,0]
#years = np.array(years)
#N=the number of years in the study period
N= np.size(years)
temp_baseline=df[:,1]

#test data are from 1950
gdp = df[:,3]
gdp=np.array(gdp)
population = df[:,4]
population = np.array(population)







# -------------------------
# Damage function from DICE, the fraction of GDP damage, so we need the GDP data
# -------------------------
def damage_dice(temp, delta1=0.0, delta2=0.002):
    """DICE-style: damage percent = delta1*T + delta2*T^2 (T in °C)"""
    return delta1 * temp + delta2 * (temp ** 2)

# -------------------------
# Helper: compute temp impulse from 1 tCO2 using Bern IRF + 1-box EBM
# -------------------------
def temp_impulse_from_1tco2(N, ecs=3.0, C_eff=10.0, alpha_rf=5.35, C0=280.0,
                            bern_a=None, bern_tau=None):
    """
    Return temp_impulse[t] (°C) for t=0..N-1 caused by adding 1 tCO2 at t=0.
    Notes:
      - 1 tCO2 -> GtCO2 = 1e-9
      - convert to GtC (divide by 44/12)
      - bern_a/bern_tau default: Bern-TAR-like approx
    """
    if bern_a is None:
        bern_a = np.array([0.152, 0.253, 0.279, 0.316])
    if bern_tau is None:
        bern_tau = np.array([np.inf, 171.0, 18.0, 2.57])
    # 1 tCO2 in GtCO2
    add_GtCO2 = 1.0e-9
    GtCO2_per_GtC = 44.0 / 12.0
    add_GtC = add_GtCO2 / GtCO2_per_GtC
    # IRF
    t = np.arange(N)
    irf = np.zeros(N)
    irf += bern_a[0]
    for i in range(1, len(bern_a)):
        tau = bern_tau[i]
        irf += bern_a[i] * np.exp(-t / tau)
    # concentration perturbation in ppm
    ppm_per_GtC = 1.0 / 2.13  # approx
    Cpert = add_GtC * ppm_per_GtC * irf  # ppm series
    Cseries = C0 + Cpert
    # radiative forcing
    F = alpha_rf * np.log(np.maximum(Cseries, 1e-6) / C0)
    # EBM integrate: lambda = F2x / ECS
    F2x = 3.7
    lam = F2x / ecs
    T = np.zeros(N)
    for i in range(N - 1):
        dTdt = (F[i] - lam * T[i]) / C_eff
        T[i + 1] = T[i] + dTdt  # dt=1 year
    # T is temperature anomaly caused by the pulse (°C)
    return T

#temperature repsonse from 1 tCO2 added in 2025
#because we need to caculate the scc in 2025
#the index of 2025 in years is 275
base_year_index = np.where(years == 2025)[0][0]
temp_impulse_per_tco2 = np.full(N,np.nan)
# temp_impulse_per_tco2[0:275]=0
# temp_impulse_per_tco2[275:]= temp_impulse_from_1tco2(N)[0:(751-275)]

temp_impulse_per_tco2[0:base_year_index]=0
temp_impulse_per_tco2[base_year_index:]= temp_impulse_from_1tco2(N)[0:(N-base_year_index)]


#the test data in the climate module is from 1750
#the base year in the ramsey rule is assumed to be 1950 here
#so we need to remove the number of nan data (n=200) from the number of years during the study period (N)
n= 200

# -------------------------
# Method A: Damage-based (pulse-response)
#constant discount rate
# -------------------------
def scc_damage_based(temp_baseline, temp_impulse_per_tco2, gdp,
                     damage_func, damage_kwargs=None,
                     discount_rate=0.03, add_tco2=1.0, damage_is_fraction=True):
    """
    temp_baseline: baseline temps (N)
    temp_impulse_per_tco2: temp response (°C) per 1 tCO2 (N)
    gdp: GDP series (N)
    damage_func: function(temp_array, **damage_kwargs) -> fraction of GDP loss (N)
    damage_is_fraction: if True, damage_func returns fraction; else absolute USD
    returns: SCC ($/tCO2), details dict
    """
    N = len(temp_baseline)
    if damage_kwargs is None:
        damage_kwargs = {}
    # perturbed temp: assume pulse occurs at t=0, scale by add_tco2
    temp_pert = np.array(temp_baseline) + np.array(temp_impulse_per_tco2) * add_tco2
    D_frac_base = damage_func(np.array(temp_baseline), **damage_kwargs)
    D_frac_pert = damage_func(temp_pert, **damage_kwargs)
    if damage_is_fraction:
        D_usd_base = D_frac_base * np.array(gdp)
        D_usd_pert = D_frac_pert * np.array(gdp)
    else:
        D_usd_base = D_frac_base
        D_usd_pert = D_frac_pert
    delta_D = D_usd_pert - D_usd_base  # $/year

    return D_usd_base,D_usd_pert

D_usd_base_2, D_usd_pert_2 = scc_damage_based(temp_baseline, temp_impulse_per_tco2, gdp,
                     damage_dice)

delta_D_2=(D_usd_pert_2-D_usd_base_2)*(10**12)



# -------------------------
# Discount helpers
# -------------------------
#r=the discount rate, constant discout rate
def discount_factors_const(N, r):
    """DF[t] = 1/(1+r)^t; t=0..N-1"""
    t = np.arange(N)
    return 1.0 / ((1.0 + r) ** t)


# discount to present value

base_year_index = np.where(years == 2025)[0]

# -------------------------
#r=the discount rate, constant discout rate
def discount_factors_const_relative(N, base_year_index,r):
    """DF[t] = 1/(1+r)^t; t=0..N-1"""
    base_year_index = base_year_index[0]
    DF = np.zeros(N)
    for t in range(N):
        if  t > base_year_index:
            #print(t)
            DF[t] = 1.0 /((1.0 + r) ** (t-base_year_index))
            #print(DF[t])
        elif t< base_year_index:
            DF[t] = 1.0 * (1.0 + r) ** (base_year_index-t)
        else:
            DF[t] = 1
    return DF

discount_rate=0.03
DF = discount_factors_const(N-n, discount_rate)

#scc = np.sum(delta_D_2[n:] * DF)
#scc = np.sum(delta_D_2[275:] * DF[75:])
scc = np.sum(delta_D_2[n:] * DF[:])

DF_relative = discount_factors_const_relative(N-n, base_year_index-n, discount_rate)
# scc_relative = np.sum(delta_D_2[n:] * DF_relative)

#scc_relative = np.sum(delta_D_2[275:] * DF_relative[75:])
scc_relative = np.sum(delta_D_2[n:] * DF_relative)