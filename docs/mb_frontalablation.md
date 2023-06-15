## Frontal Ablation
For marine-terminating glaciers, frontal ablation is modeled using a frontal ablation parameterization coupled to the ice dynamical model (i.e., the glacier dynamics parameterization). Given the coupling to the dynamical model, frontal ablation is accounted for on an annual timestep and the code for the frontal ablation parameterization is located with the dynamical model. OGGM provides a nice overview of the frontal ablation parameterization in one of their advanced tutorials:

https://oggm.org/tutorials/stable/notebooks/kcalving_parameterization.html

The same parameterization is included for mass redistribution curves in PyGEM.

Frontal ablation ($A_{f}$) computes the mass that is removed at the glacier front when the bedrock is below sea level using an empirical formula following [Oerlemans and Nick (2005)](https://www.cambridge.org/core/journals/annals-of-glaciology/article/minimal-model-of-a-tidewater-glacier/C6B72F547D8C44CDAAAD337E1F2FC97F):
```{math}
A_{f} = k \cdot d \cdot H_{f} \cdot w
```
where $k$ is the frontal ablation scaling parameter (yr$^{-1}$), $d$ is the water depth at the calving front (m), $H_{f}$ is the ice thickness at the calving front, and $w$ is the glacier width at the calving front. Over the next century, many marine-terminating glaciers are projected to retreat onto land based on present-day frontal ablation rates ([Rounce et al. 2023](https://www.science.org/doi/10.1126/science.abo1324)); hence, the maximum frontal ablation rate is constrained by the mass of ice where the bed elevation of the bin is located below sea level. The user is also able to specify the water level (default is 0), which supports the application of the parameterization to lake-terminating glaciers in the future.