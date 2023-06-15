# Glacier Runoff
Following [Huss and Hock (2018)](https://www.nature.com/articles/s41558-017-0049-x), glacier runoff ($Q$) is defined as all water that leaves the initial glacierized area, which includes rain ($P_{liquid}$), ablation ($a$), and refreezing ($R$) as follows:

```{math}
Q = P_{liquid} + a - R
```

In the case of glacier retreat, rain, snow melt, and refreezing are computed for the non-glaciated portion of the initial glacier area and this runoff is referred to as “off-glacier” runoff. No other processes, e.g., evapotranspiration or groundwater recharge, are accounted for in these deglaciated areas. 

```{warning}
In the case of glacier advance, runoff is computed over the current year’s glacier area, which may exceed the initial glacierized area. Given that most glaciers are retreating, the increase in glacier runoff due to the additional glacier area is considered to be negligible.
```

```{note}
The model will also compute runoff assuming a moving-gauge, i.e., glacier runoff that is only computed from the glacierized areas. For some users who may want to use their models to compute off-glacier runoff, this is preferable.
```


## Excess meltwater
Excess meltwater is defined as the runoff caused by glacier mass loss that the glacier does not regain over the duration of the entire simulated period (**Figure 1**). For example, a glacier that melts completely contributes its entire mass as excess meltwater, while a glacier in equilibrium produces no excess meltwater. Since interannual glacier mass change is highly variable, i.e., a glacier can lose, gain, and then lose mass again, excess meltwater is computed retroactively as the last time that the glacier mass is lost.

```{figure} _static/excess_meltwater_diagram.png
---
width: 100%
---
```

**Figure 1.** Diagram exemplifying how excess meltwater (blue dashed line) is calculated retroactively based on annual glacier mass balance (black line) over time. Cumulative (top subplot) and annual (bottom subplot) mass balance and excess meltwater are shown. Years refer to mass-balance years and values of cumulative mass balances and excess meltwater refer to the end of each mass-balance year. The total excess meltwater is equivalent to the total mass loss over the entire period, and therefore is not equal to the absolute sum of all annual negative mass balances if positive mass balances have occurred in the period. Excess meltwater is distributed retroactively over all mass-balance years that are negative and where the lost mass is not regained in the future. For example, annual mass balances in year 6, 7 and 9 are negative, but all mass loss lost between year 6 and 10 is regained by the end of year 10; thus, excess meltwater is zero in years 6 to 10 despite negative annual mass balances. Excess meltwater for any year with negative mass balance cannot exceed the annual net mass loss of that year. The figure is copied from [Rounce et al. (2020)](https://www.frontiersin.org/articles/10.3389/feart.2019.00331/full).
