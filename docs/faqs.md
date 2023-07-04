# FAQs
### Why does my simulation from run_simulation.py script run smoothly without any error, but the only output is an error folder with a failed information .txt file?
The run_simulation.py script uses try/except statements to support large-scale applications, i.e., if a simulation for a single glacier fails (e.g., because it grows beyond the maximum size), we don’t want this error to stop all of the simulations.  Hence, this error is caught in the try/except statement and the except statement generates a failed .txt file to let the individual know that the simulation did not run.

Troubleshooting this failure needs to be improved now that there are many new users who will likely cause this to fail more frequently.  At present, the best workaround is to replace the try/except statement.  Specifically, there will be a commented line named “for batman in [0]:”, which I suggest you uncomment, and comment out the try statement above.  You then need to go to towards the end of the statement and uncomment “print(‘\nADD BACK IN EXCEPTION\n\n’)” and comment out the except statement that is about 8 lines of code.  When you run the simulation now, you should get whatever runtime error was causing the failure to begin with.

In the future, we will seek to catch these errors and put them in the text file to make debugging easier.


### Why is the mass balance that I calculate from the calibration not the same as the simulation?
There are three potential causes for this, which are all dependent on the calibration options:
1. The calibration is performed assuming the glacier area is constant. This is primarily to save computational time, but also enables the calibration to not be linked to a specific glacier dynamics option. Tests were performed in Fall 2020 that showed the impacts of this over the calibration period (2000-2019) was fairly minor. If you run the model with a dynamical option, then you will get a different mass balance.
2. Did you use the emulator? The emulator is a statistical representation of the mass balance. Tests were performed in Fall 2020 that showed the emulator performed quite well (typically within 0.01 - 0.02 mwea of the observed mass balance), which was considered acceptable given the uncertainty associated with the geodetic mass balance data. Hence, the mass balance you get from the emulator will be slightly different than one that you get from running a simulation.
3. Did you use the MCMC option?  The MCMC calibration is performed using a certain number of steps.  The simulations are performed for a subset of model parameter combinations from those steps; hence, they will differ.


### How can I export a different variable (e.g., binned glacier runoff)?
There are two primary steps: (1) calculate the new variable you want to export, and (2) add the variable to the proper dataset.  We recommend the following for this example:

Calculate the binned glacier runoff.  You'll see in the massbalance.py script that it automatically adds the glac_bin_melt, glac_bin_refreeze, and bin_prec.  You'll need to create a new variable for glac_bin_runoff.
Add the binned glacier runoff to be exported.  For this you need to add the glac_bin_runoff to the create_xrdataset_binned_stats() function in the run_simulation.py script.  This will add the glacier runoff as a binned result to the binned netcdf file that is exported.


### Why am I getting a 'has_internet'=False error?
Part of the beauty of our use of OGGM is access to the OGGM Shop. When we initialize our glacier directories in PyGEM, we are downloading them from OGGM shop.  OGGM has a cfg.PARAMS['has_internet'] variable that must be set to True in order to download; otherwise, it will throw this error.  If you skipped over running OGGM's test when you downloaded OGGM, you may not have downloaded the sample datasets that OGGM requires to run tests and it'll likely throw an error.  To correct this error, you have two options: (1) set <em>has_internet=True</em> in pygem_input.py or (2) manually modify your code to set <em>cfg.PARAMS['has_internet']=True</em> likely somewhere in your oggm_compat.py file, which will be located somewhere in your conda environment if you used pip install.


### The error message and line of code appears to be associated with code from OGGM. How do I troubleshoot OGGM's code from within PyGEM?
While we're doing everything we can to minimize these issues, and OGGM developers are excellent at supporting this as well, from time to time errors may come up based on OGGM. This often occurs during changes between versions as it's challenging to document every tiny change. Nonetheless, here's a guide for troubleshooting your errors to at least identify the problem. We'll use a recent example where for some reason with the new update, we couldn't get tidewater glaciers to invert for ice thickness. Here's what we did to solve the issue:
* Identify the source code associated with the error.
  - In our case, we couldn't get the inversion to work, so we knew it was associated with OGGM's core/inversion.py. If you're having trouble with this step, try copying a function from the error message and finding where that function exists in OGGM's source code on github.
* Find where OGGM’s code is installed in your environment on your local computer.
  - A good way to do this is to go to your directory and find where "inversion.py" exists. Then open this file and show the enclosing directory. Note that this can be a bit of a pain, but once you know where your conda environments are stored it makes it a lot easier. This is essentially where the packages are stored on your computer. Cool!
* Find where the error is coming from by putting print statements within the inversion.py script.
  - This is very basic troubleshooting. There are likely faster ways of doing so, but I prefer to dive into the code. In our case, the model_flowlines wasn’t being generated, so I started going through the find_inversion_calving_from_any_mb fxn. This identified that it was not even getting pass the first if statement, which is a clever thing that OGGM does - OGGM has built-in "off-ramps" within their functions such that if the input structure isn’t ideal (in this case if it’s not a tidewater glacier or if you don’t set cfg.PARAMS[‘use_kcalving_for_inversion’]=True, then it returns out of the function without an error.
* Lastly, come up with a solution!
  - This is where our help breaks down a bit as you'll need to figure out what the fix is for your situation. However, if you're able to get to this point, stay at it! If you are trying hard and can't figure out what's going on (perhaps you've tried for a couple hours and are now getting super frustrated - it happens), this is where using the "support" Channel on OGGM's Slack is a great option. In your message, make sure to include all the details above. If it's something with PyGEM, then the PyGEM folks on OGGM's Slack will do our best to help. If it's something with OGGM, then you'll find plenty of support as well.