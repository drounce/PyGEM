## Ablation
There are currently two model options for ablation. Both model options use a degree-day model ($f$). The first calculates ablation ($a$) using the mean monthly temperature:

$$a=f_{snow/ice/firn/debris} \cdot T_{m}^{+} \cdot n$$

where $f$ is the degree-day factor of snow, ice, firn, or debris (m w.e. d-1 °C$^{-1}$), $T_{m}^{+}$ is the positive monthly mean temperature (°C), and $n$ is the number of days per month. 

The second option incorporates the daily variance associated with the temperature for each month according to Huss and Hock (2015):

$$a=f_{snow/ice/firn/debris} \cdot \sum_{i=1}^{ndays} T_{d,i}^{+} $$

where $T_{d}$ is the daily positive mean air temperature and is estimated by superimposing random variability from the standard deviation of the daily temperature for each month.

