# Mass Balance Models

PyGEM computes the climatic mass balance for each elevation bin and timestep, estimates frontal ablation for marine-terminating glaciers at the end of each year (if this process is included), and updates the glacier geometry annually. The convention below follows [Cogley et al. (2011)](https://wgms.ch/downloads/Cogley_etal_2011.pdf). The total glacier-wide mass balance ($\Delta M$) is thus estimated as:

```{math}
\Delta M = B_{clim} + A_{f}/S
```

where $B_{clim}$ is the climatic mass balance in specific units, i.e. mass change per unit area (m w.e.), $A_{f}$ is frontal ablation, and $S$ is the glacier area. The basal mass balance is assumed to be zero.

The climatic mass balance for each elevation bin ($b_{clim}$) is computed according to:
```{math}
b_{clim} = a + c + R
```

where $a$ is the ablation, $c$ is accumulation, and $R$ is refreezing (all in units of m w.e.). Mass loss is negative and mass gain is positive. The glacier-wide specific climatic mass balance ($B_{clim}$) is thus calculated by:
```{math}
\sum_{i=1}^{nbins} b_{clim,i}
```

The model offers alternative methods for calculating the mass balance components and accounting for glacier geometry changes (i.e., representing glacier dynamics). These vary in level of complexity and computational expense. The current options for each component are described below:

```{toctree}
---
caption: Mass Balance Components:
maxdepth: 2
---

mb_ablation
mb_accumulation
mb_refreezing
mb_frontalablation
```

## Summary of model parameters
Below is a summary of some of the key mass balance model parameters, their symbols, units, and the values used in PyGEM. Note that some parameters are calculated, others are calibrated, and others may be specified by the user in the input file.

| Parameter | Symbol | Unit | Value |
| :--- | :--- | :--- | :--- |
| Ablation | $a$ | m w.e. | calculated |
| Accumulation | $c$ | m w.e. | calculated |
| Refreeze | $R$ | m w.e. | calculated |
| Frontal ablation | $A_{f}$ | m w.e. | calculated |
| Degree-day factor of snow | $f_{snow}$ | mm w.e. d$^{-1}$ K$^{-1}$ | calibrated |
| Degree-day factor of ice | $f_{ice}$ | mm w.e. d$^{-1}$ K$^{-1}$ | $f_{snow}$/0.7 <br>(user-specified) |
| Degree-day factor of firn | $f_{firn}$ | mm w.e. d$^{-1}$ K$^{-1}$ | $\frac{f_{snow}+f_{ice}}{2}$ |
| Degree-day factor of debris | $f_{debris}$ | mm w.e. d$^{-1}$ K$^{-1}$ | $E_{d} \cdot f_{ice}$ |
| Sub-debris melt enhancement factor | $E_{d}$ | - | 1 if no debris; <br> otherwise from [Rounce et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091311) |
| Temperature bias correction | $T_{bias}$ | K | calibrated |
| Threshold temperature (rain/snow) | $T_{snow}$ | $^{\circ}$C | 1 <br> (user-specified) |
| Precipitation correction factor | $k_{p}$ | - | calibrated |
| Precipitation gradient | $d_{prec}$ | m$^{-1}$ | 0.0001 <br> (user-specified) |
| Frontal ablation scaling parameter | $k$ | yr$^{-1}$ | calibrated |
