# MASS BALANCE PARAMETERIZATIONS

PyGEM computes the climatic mass balance for each elevation bin and timestep, estimates frontal ablation for marine-terminating glaciers at the end of each year (if this process is included), and updates the glacier geometry annually. The convention below follows Cogley et al. (2011). The total glacier-wide mass balance (ΔM) is thus estimated as:

$$\Delta w_{t+1} = w_{t+1} = \frac{1}{2} (1 + r_{t+1}) s(w_t) + y_{t+1}$$

where $B_{clim}$ is the climatic mass balance in specific units, i.e. mass change per unit area (m w.e.), $A_{f}$ is frontal ablation, and $S$ is the glacier area. The basal mass balance is assumed to be zero.

The climatic mass balance is computed for each elevation bin according to:

$$b_{clim} = a + c + R$$

where $a$ is the ablation, $c$ is accumulation, and $R$ is refreezing (all in units of m w.e.). Mass loss is negative and mass gain is positive. Glacier-wide specific climatic mass balance ($B_{clim}$) is then calculated by:

$$\sum_{i=1}^{nbins} b_{clim,i} $$

The model offers alternative methods for calculating the mass balance components and accounting for glacier geometry changes (i.e., representing glacier dynamics). These vary in level of complexity and computational expense. The current options for each component are described below:

.. toctree::
   :maxdepth: 2
   :caption: Mass Balance Components:

   mb_ablation
   mb_accumulation
   mb_refreezing
   mb_frontalablation