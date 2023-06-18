# Bias Corrections
Bias corrections can be applied to ensure the temperature and precipitation associated with the future climate data is roughly consistent with the reference climate data. 

The temperature bias corrections follow [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full) and use an additive factor to ensure the mean monthly temperature and interannual monthly variability are consistent between the reference and future climate data. 

There are two options for the precipitation bias correction:

**Option 2** follows [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full) and uses a multiplicative factor to ensure the monthly mean precipitation are consistent between the reference and future climate data. 

**Option 1** modifies the approach of [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full), since in dry regions, the adjusted precipitation was found to be unrealistically high when using multiplicative values. This option uses the interannual monthly variability and quality controls the bias corrected precipitation by replacing any values that exceed the monthly maximum with the monthly average adjusted for potentially wetter years in the future using the normalized annual variation. Note that [Marzeion et al. (2020)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019EF001470) suggests that [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full) have updated their approach to also adjust the precipitation bias correction to account for monthly interannual variability as well.

```{note}
These bias corrections significantly impact projections for all glacier evolution models. Current work is thus investigating different bias corrections and how they affect projections. We hope to add additional bias correction options in the near future.
```