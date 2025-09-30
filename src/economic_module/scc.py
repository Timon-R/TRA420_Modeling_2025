# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:06:42 2025

@author: chengf
"""

import numpy as np
import pandas as pd

#the parameters and input

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
#the study period
years = df[:,0]
#years = np.array(years)
#N=the number of years in the study period
N= np.size(years)
#global mean temperature data during the study period under the baseline scenario
temp_baseline=df[:,1]
#test data are from 1950
#GDP data during the study period
gdp = df[:,3]
gdp=np.array(gdp)
#population data during the study period
population = df[:,4]
population = np.array(population)



# -------------------------
# Damage function from DICE, the fraction of GDP damage
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

#temperature repsonse from 1 tCO2
temp_impulse_per_tco2= temp_impulse_from_1tco2(N)


#D_usd_base: damage under baseline scenario
#D_usd_pert: damage under perturbation sceanrio
def damage(temp_baseline, temp_impulse_per_tco2, gdp,
                     damage_func, damage_kwargs=None,
                     add_tco2=1.0, damage_is_fraction=True):
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


D_usd_base, D_usd_pert = damage(temp_baseline, temp_impulse_per_tco2, gdp,
                     damage_dice)

#damage difference between baseline and pertubation scenario
delta_D=(D_usd_pert-D_usd_base)*(10**12)

#calculate gdp per capita
#the growth rate of gdp per capita could be used to act as the growth rate of per capita consumption
gdp_per_capita=np.full(N,np.nan)
gdp_per_capita_after_damage=np.full(N,np.nan)
growth_consumption =np.full(N,np.nan)
for n in range(N):
    print(n)
    print(population[0])
    gdp_per_capita[n]=gdp[n]*(10**12)/population[n]
    gdp_per_capita_after_damage[n]=(gdp[n]-D_usd_base[n])*(10**12)/population[n]
    if n>=1:
        growth_consumption[n] = (gdp_per_capita_after_damage[n]-gdp_per_capita_after_damage[n-1])/gdp_per_capita_after_damage[n-1]
    else:
        growth_consumption[n] = np.nan

#DF=the discount rate obtained by Ramsey rule
#growth assumption is the annual growth rate of per capita consumption 
#growth consumption is the input caculated from the GDP and damage
# growth consumption at year t = (GDP-Damage) at year t+1 minus (GDP-Damage) at year t 


#ramsey rule, the first year with GDP is set as the base year
def discount_factors_ramsey(N, rho, eta, growth_consumption):
    """
    Ramsey-style DF: DF[t] = 1 / prod_{s=0}^{t-1} (1 + r_s), r_s = rho + eta * g_c_s
    growth_consumption: array length N of consumption growth rates (g_c[0] unused)
    """
    DF = np.zeros(N)
    DF[0] = 1.0  #which year should be the base year?
    for t in range(1, N):
        r_s = rho + eta * growth_consumption[t+200]
        #print(r_s)
        DF[t] = DF[t-1] / (1.0 + r_s)
    return DF

    # DF_relative = DF/DF[275]
    # return DF,DF_relative

#the number of nan data in the deltaD gdp from 1950; base year 2025
n= 200
#n=200+75
discount_ramsey=discount_factors_ramsey(N-n, rho/100, eta, growth_consumption)


# discount to present value

base_year_index = np.where(years == 2025)[0]

##ramsey rule, 2025 is set as the base year
def discount_factors_ramsey_relative(N, n,base_year_index,rho, eta, growth_consumption):
    """
    Ramsey-style DF: DF[t] = 1 / prod_{s=0}^{t-1} (1 + r_s), r_s = rho + eta * g_c_s
    growth_consumption: array length N of consumption growth rates (g_c[0] unused)
    """
    DF = np.zeros(N)
    r_s = np.zeros(N)
    base_year_index=base_year_index[0]
    print(base_year_index)
    DF[base_year_index] = 1.0  #which year should be the base year?
    print(DF[base_year_index])
    for t in range(N):
        r_s[t] = rho + eta * growth_consumption[t+n]
        #print(r_s[t])
        if  t > base_year_index:
            #print(t)
            DF[t] = DF[t-1] / (1.0 + r_s[t])
            #print(DF[t])
        elif t< base_year_index:
            DF[t] = np.nan
        else:
            DF[t] = 1
    
    for t in range(1,base_year_index+1):
        print(t)
        DF[base_year_index-t] = DF[base_year_index - t + 1]*(1.0 + r_s[base_year_index - t])
           
    return DF
discount_ramsey_relative=discount_factors_ramsey_relative(N-n, n,base_year_index-n,rho/100, eta, growth_consumption)

#scc
scc=np.sum(delta_D[n:]*discount_ramsey[:])
#scc relative to the first year
#there is no GDP growth on the first year with GDP
scc_pv=np.sum(delta_D[n+1:]*discount_ramsey_relative[1:])
