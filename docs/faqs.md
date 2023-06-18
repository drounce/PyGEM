# FAQs
**Why does my simulation from run_simulation.py script run smoothly without any error, but the only output is an error folder with a failed information .txt file?**

The run_simulation.py script uses try/except statements to support large-scale applications, i.e., if a simulation for a single glacier fails (e.g., because it grows beyond the maximum size), we don’t want this error to stop all of the simulations.  Hence, this error is caught in the try/except statement and the except statement generates a failed .txt file to let the individual know that the simulation did not run.

Troubleshooting this failure needs to be improved now that there are many new users who will likely cause this to fail more frequently.  At present, the best workaround is to replace the try/except statement.  Specifically, there will be a commented line named “for batman in [0]:”, which I suggest you uncomment, and comment out the try statement above.  You then need to go to towards the end of the statement and uncomment “print(‘\nADD BACK IN EXCEPTION\n\n’)” and comment out the except statement that is about 8 lines of code.  When you run the simulation now, you should get whatever runtime error was causing the failure to begin with.

In the future, we will seek to catch these errors and put them in the text file to make debugging easier.


**Why is the mass balance that I calculate from the calibration not the same as the simulation?**
There are three potential causes for this, which are all dependent on the calibration options:
1. The calibration is performed assuming the glacier area is constant. This is primarily to save computational time, but also enables the calibration to not be linked to a specific glacier dynamics option. Tests were performed in Fall 2020 that showed the impacts of this over the calibration period (2000-2019) was fairly minor. If you run the model with a dynamical option, then you will get a different mass balance.
2. Did you use the emulator? The emulator is a statistical representation of the mass balance. Tests were performed in Fall 2020 that showed the emulator performed quite well (typically within 0.01 - 0.02 mwea of the observed mass balance), which was considered acceptable given the uncertainty associated with the geodetic mass balance data. Hence, the mass balance you get from the emulator will be slightly different than one that you get from running a simulation.
3. Did you use the MCMC option?  The MCMC calibration is performed using a certain number of steps.  The simulations are performed for a subset of model parameter combinations from those steps; hence, they will differ.


**How can I export a different variable (e.g., binned glacier runoff)?**
There are two primary steps: (1) calculate the new variable you want to export, and (2) add the variable to the proper dataset.  We recommend the following for this example:

Calculate the binned glacier runoff.  You'll see in the massbalance.py script that it automatically adds the glac_bin_melt, glac_bin_refreeze, and bin_prec.  You'll need to create a new variable for glac_bin_runoff.
Add the binned glacier runoff to be exported.  For this you need to add the glac_bin_runoff to the create_xrdataset_binned_stats() function in the run_simulation.py script.  This will add the glacier runoff as a binned result to the binned netcdf file that is exported.