# Mass Balance
PyGEM computes the climatic mass balance for each elevation bin and timestep, estimates frontal ablation for marine-terminating glaciers at the end of each year (if this process is included), and updates the glacier geometry annually. The convention below follows [Cogley et al. (2011)](https://wgms.ch/downloads/Cogley_etal_2011.pdf). The total glacier-wide mass balance ($\Delta M$) is thus estimated as:
$$\Delta M = B_{clim} + A_{f}/S $$

where $B_{clim}$ is the climatic mass balance in specific units, i.e. mass change per unit area (m w.e.), $A_{f}$ is frontal ablation, and $S$ is the glacier area. The basal mass balance is assumed to be zero.

The climatic mass balance for each elevation bin ($b_{clim}$) is computed according to:
$$b_{clim} = a + c + R$$

where $a$ is the ablation, $c$ is accumulation, and $R$ is refreezing (all in units of m w.e.). Mass loss is negative and mass gain is positive. The glacier-wide specific climatic mass balance ($B_{clim}$) is thus calculated by:
$$\sum_{i=1}^{nbins} b_{clim,i} $$

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

