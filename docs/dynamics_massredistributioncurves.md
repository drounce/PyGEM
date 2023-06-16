(mass_redistribution_curves_target)=
## Mass Redistribution Curves
The mass redistribution curves in PyGEM follow those developed by [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full) based on [Huss et al. (2010)](https://hess.copernicus.org/articles/14/815/2010/hess-14-815-2010.html) but explicitly solve for area and ice thickness changes simultaneously to conserve mass. The approach is only applied to glaciers that have at least three elevation bins. Each year the glacier-wide mass balance is computed (see Mass Balance Section) and the mass is redistributed over the glacier using empirical equations that set the normalized surface elevation change ($\Delta h$) as a function of the glacier’s elevation bins:
```{math}
\Delta h = (h_{n} + a_{HH2015})^{\gamma} + b_{HH2015} \cdot (h_{n} + a_{HH2015}) + c_{HH2015}
```
where $h_{n}$ is the normalized elevation according to $\frac{z_{max} - z_{bin}}{z_{max} - z_{min}}$ and $a_{HH2015)$, $b_{HH2015)$, $c_{HH2015)$, and $\gamma$ are all calibrated coefficients based on 34 glaciers in the Swiss Alps. These coefficients vary depending on the size of the glacier ([Huss et al., 2010]((https://hess.copernicus.org/articles/14/815/2010/hess-14-815-2010.html))). In order to ensure that mass is conserved, i.e., the integration of the elevation change and glacier area (A) of each bin over all the elevation bins ($nbins$) is equal to the glacier-wide volume change ($\Delta V$), an ice thickness scaling factor ($f_{s,HH2015}$) must be computed:
```{math}
f_{s,HH2015} = \frac{\Delta V}{\sum_{i=0}^{nbins} A_{i} \cdot \Delta h_{i}}
```
The volume change in each elevation bin ($\Delta V_{bin}$) is computed as:
```{math}
\Delta V_{bin} = f_{s,HH2015} \cdot \Delta h_{bin} \cdot A_{bin}
```
Depending on the bed shape (parabolic, triangular or rectangular) of the glacier, the resulting area, ice thickness ($H$), and width ($W$) can be solved for explicitly based on mass conservation and the use of similar shapes:
```{math}
H_{bin,t+1} \cdot A_{bin,t+1} = H_{bin,t} \cdot A_{bin,t} + \Delta V_{bin}
```
```{math}
\frac{H_{bin,t+1}}{H_{bin,t}} \alpha \frac{A_{bin,t+1}}{A_bin,t}
```
This is a marked improvement over previous studies that have not explicitly solved for the area and ice thickness changes simultaneously, which can lead to mass loss or gain that is then corrected ([Huss and Hock, 2015](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full)).

### Glacier retreat
Glacier retreat occurs when the volume change in an elevation bin ($\Delta V_{bin}$) causes the ice thickness for the next time step to be less than zero. In this case, the ice thickness is set to zero and the remaining volume change is redistributed over the entire glacier according to the mass redistribution described above. 

### Glacier advance
Glacier advance occurs when the ice thickness change exceeds the ice thickness advance threshold (default: 5 m; [Huss and Hock, 2015](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full)). When this occurs, the ice thickness change is set to 5 m, the area and width of the bin are calculated accordingly, and the excess volume is recorded. The model then calculates the average area and thickness associated with the bins located in the glacier’s terminus, which is defined by the terminus percentage (default: 20%). However, this calculation excludes the bin actually located at the terminus because prior to adding a new elevation bin, the model checks that the bin located at the terminus is “full”. Specifically, the area and ice thickness of the lowermost bin are compared to the terminus’ average and if the area and ice thickness is less than the average, then the lowermost bin is first filled until it reaches the terminus average. This ensures that the lowermost bin is “full” and prevents adding new bins to a glacier that may only have a relatively small excess volume in consecutive years. In other words, if this criterion did not exist, then it would be possible to add new bins over multiple years that had small areas, which would appear as though the glacier was moving down a steep slope.

If there is still excess volume remaining after filling the lowermost bin to the terminus average, then a new bin is added below the terminus. The ice thickness in this new bin is set to be equal to the terminus average and the area is computed based on the excess volume. If the area of this bin would be greater than the average area of the terminus, this indicates that an additional bin needs to be added. However, prior to adding an additional bin the excess volume is redistributed over the glacier again. This allows the glacier’s area and thickness to increase and prevents the glacier from having a thin layer of ice that advances down-valley without thickening.

There are two exceptions for when a glacier is not allowed to advance to a particular bin. The first exception is if the added bin would be below sea-level, in which case the remaining excess volume is redistributed over the entire glacier. The second exception is if the bin is over a known discontinuous section of the glacier, which is determined based on the initial glacier area. For example, it is possible, albeit unlikely, that a glacier could retreat over a discontinuous section of a glacier and then advance in the future. This discontinuous area is assumed to be a steep vertical drop, hence why a glacier currently does not exist, so a glacier is not allowed to form there in the future. The glacier instead skips over this discontinuous bin and a new bin is added below it.