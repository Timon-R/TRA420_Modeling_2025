# Air Pollution Mortality Methodology

## General Methodology

The mortality due to air pollution can be modelled using a **concentration–response function**, which describes how a change in the concentration of a specific air pollutant affects the likelihood of death due to exposure to that pollutant. This function can be combined with the current mortality attributable to that pollutant to estimate how mortality changes with a change in pollution concentration.

Let

- \(D_{\text{new}}\): resulting number of deaths after a change in air pollution  
- \(D_{\text{current}}\): current number of deaths  
- \(\Delta C\): change in air pollution concentration  
- \(\text{CRF}\): concentration–response function  

Then, schematically,

\[
D_{\text{new}} = D_{\text{current}} \times \text{CRF}^{\Delta C}.
\]

For this application, the concentration–response function is assumed to be **constant** over the range of interest (in reality, it varies with concentration). The constant values are taken from meta-analyses (Chen, Qi, Li, Duan) and set to:

- **PM\(_{2.5}\)**: CRF = **1.08**  
- **NO\(_2\)**: CRF = **1.03**

The change in air pollution concentration is chosen, for simplicity, to be **proportional to the change in emissions**:

\[
\Delta C \propto \Delta E,
\]

where \(E\) denotes the quantity of emissions.

---

## Data for Western Balkans

To estimate the change in mortality we need:

1. Current number of deaths due to air pollution in the Western Balkans  
2. Current concentration levels of air pollution  

These can then be combined with modelled changes in emissions.

A country-wise breakdown of air-pollution mortality from the **European Environmental Agency (EEA) Air Pollution Country Fact Sheets, 2024** is used. In total:

- **27,190** deaths from PM\(_{2.5}\)  
- **2,820** deaths from NO\(_2\)

### Baseline Mortality

| Country          | PM\(_{2.5}\) deaths | NO\(_2\) deaths | Total deaths |
|------------------|--------------------:|----------------:|-------------:|
| Albania          | 3,460               | 310             | 3,770        |
| Bosnia           | 6,210               | 460             | 6,670        |
| Kosovo           | 2,370               | 240             | 2,610        |
| Montenegro       | 710                 | 50              | 760          |
| North Macedonia  | 3,640               | 330             | 3,970        |
| Serbia           | 10,800              | 1,430           | 12,230       |
| **Total**        | **27,190**          | **2,820**       | **30,010**   |

The fact sheets also provide population-weighted mean concentrations for PM\(_{2.5}\) and NO\(_2\); these are used as baseline pollution levels.

### Baseline Concentrations (Population-Weighted Means)

| Country          | PM\(_{2.5}\) (µg/m³) | NO\(_2\) (µg/m³) |
|------------------|---------------------:|-----------------:|
| Albania          | 15.8                 | 12.4             |
| Bosnia           | 21.1                 | 12.9             |
| Kosovo           | 16.5                 | 13.6             |
| Montenegro       | 15.1                 | 11.3             |
| North Macedonia  | 22.8                 | 15.7             |
| Serbia           | 19.1                 | 16.2             |

These values must be adjusted because air pollution comes from multiple sources, not just electricity generation.

Currently, almost all fossil-based electricity in the Western Balkans is coal-based. Coal-related pollution is estimated to cause about **2,167** of the total **30,010** yearly deaths (Comply or Close), so electricity generation is roughly **7%** of total air pollution-mortality.

A second check is **“Every Breath We Take: Improving Air Quality in Europe”**, which estimates that across Europe:

- ~40% of PM\(_{2.5}\) comes from transportation  
- ~50% from households and businesses  

The remainder (≈10%) is associated with other sources, some of which are not domestic electricity generation (e.g. transboundary pollution). This is broadly consistent with the **7%** estimate for electricity in the Western Balkans.

Using the 7% share, we scale baseline deaths and pollution concentrations attributable to **electricity generation**.

### Adjusted Values Attributable to Electricity Generation

| Country          | PM\(_{2.5}\) deaths | NO\(_2\) deaths | PM\(_{2.5}\) conc. (µg/m³) | NO\(_2\) conc. (µg/m³) |
|------------------|--------------------:|----------------:|---------------------------:|-----------------------:|
| Albania          | 242                 | 22              | 1.11                       | 0.87                   |
| Bosnia           | 434                 | 32              | 1.48                       | 0.90                   |
| Kosovo           | 166                 | 17              | 1.16                       | 0.95                   |
| Montenegro       | 50                  | 4               | 1.06                       | 0.79                   |
| North Macedonia  | 255                 | 23              | 1.60                       | 1.10                   |
| Serbia           | 756                 | 100             | 1.34                       | 1.13                   |

The most precise way to use this data is:

1. For each **pollutant–country** pair, calculate the change in deaths given a change in emissions (via concentrations and CRFs).
2. Sum the changes in deaths across **all countries and both pollutants**.

---

## Value of a Statistical Life (VSL)

To monetise the value of lives lost or saved due to changes in air pollution, we use estimates for the **Value of a Statistical Life (VSL)**.

The **OECD report “Mortality Risk Valuation in Environment, Health and Transport Policies” (2012)** is used as the main source. It meta-analyses surveys of people in Europe regarding their willingness to pay to reduce mortality risk.

- Base VSL (OECD, Europe): **3,000,000 USD-2005**  
- Recommended range: **1,500,000–4,500,000 USD-2005**

Several factors in the report suggest using a **higher** VSL for this application:

- Deaths preceded by **suffering** (chronic illness rather than sudden death) → higher VSL  
- **Controllable / preventable** risks (air pollution can be reduced with known technologies) → higher VSL  
- Risks affecting a **public good** with altruistic components → higher VSL  

Given these, we pick the **upper bound**:

- Chosen base for this scenario: **4,500,000 USD-2005** per statistical life.

### Adjustments for PPP and Inflation

- **Purchasing Power Parity (PPP)**: OECD PPP indicators suggest an adjustment factor of about **0.5** for the Western Balkans relative to USD-2025.  
- **Inflation / price level**: Converting USD-2005 to USD-2025 uses a factor of **1.66**.

Combined:

\[
\text{VSL}_{\text{WB, 2025}} \approx 4{,}500{,}000 \times 1.66 \times 0.5
\approx 3{,}750{,}000\ \text{USD-2025 per life}.
\]

This is the value used to monetise lives lost due to air pollution in the Western Balkans.

---

## References

- Chen, X., Qi, L., Li, S., & Duan, X. (2024). *Long-term NO\(_2\) exposure and mortality: A comprehensive meta-analysis*. Retrieved from <https://www.sciencedirect.com/science/article/pii/S0269749123019735>  
- European Environment Agency (EEA). (2024). *Air Pollution Country Fact Sheets*. Retrieved from <https://www.eea.europa.eu/en/topics/in-depth/air-pollution/air-pollution-country-fact-sheets-2024>  
- CEE Bankwatch Network. (2021, September). *Comply or Close*. Retrieved from <https://www.complyorclose.org/wp-content/uploads/2021/09/En-COMPLY-OR-CLOSE-web.pdf>  
- European Environment Agency (EEA). (2013). *Every Breath We Take: Improving Air Quality in Europe*. Retrieved from <https://www.eea.europa.eu/en/analysis/publications/eea-signals-2013>  
- OECD. (2012). *Mortality Risk Valuation in Environment, Health and Transport Policies*. OECD Publishing. <http://dx.doi.org/10.1787/9789264130807-en>  
- OECD. (2023). *Purchasing Power Parities*. Retrieved from <https://www.oecd.org/en/data/indicators/purchasing-power-parities-ppp.html>