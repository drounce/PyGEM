# Glacier Dynamics Models
Glacier dynamics in large-scale glacier evolution models typically rely on geometry changes like volume-area-length scaling (e.g., [Radić and Hock, 2011](https://www.nature.com/articles/ngeo1052)), mass redistribution using empirical equations (e.g., [Huss and Hock, 2015](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full)), or simplified glacier dynamics (e.g., [Maussion et al., 2019](https://gmd.copernicus.org/articles/12/909/2019/)). [Zekollari et al. (2022)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021RG000754) provide a comprehensive review of ice dynamics for mountain glaciers. These methods all allow the glacier to evolve over time in response to the total glacier-wide mass balance. The benefit of volume-area-length scaling and mass redistribution is they are computationally inexpensive compared to simplified glacier dynamics methods. The two options available in PyGEM are OGGM’s flowline model using the shallow ice approximation ([Maussion et al. 2019](https://gmd.copernicus.org/articles/12/909/2019/)) and mass redistribution curves ([Rounce et al. 2020](https://www.frontiersin.org/articles/10.3389/feart.2019.00331/full)).

```{toctree}
---
caption: Mass Balance Components:
maxdepth: 2
---

dynamics_oggm
dynamics_massredistributioncurves
```
