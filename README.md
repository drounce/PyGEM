# PyGEM

All python code are written using a maximum line length of 120 characters to improve readability particularly with
respect to longer equations that were difficult to read with the PEP-8 suggestion of 80 characters.

========== MODEL RUN DETAILS ================================================
The model is run through a series of steps:
  > Step 01: Region/Glaciers Selection
             The user needs to define the region/glaciers that will be used in
             the model run.  The user has the option of choosing the standard
             RGI regions or defining their own.
  > Step 02: Model Time Frame
             The user should consider the time step and duration of the model
             run along with any calibration product and/or model spinup that
             may be included as well.
  > Step 03: Climate Data
             The user has the option to choose the type of climate data being
             used in the model run, and how that data will be downscaled to
             the glacier and bins.
  > Step 04: Glacier Evolution
             The heart of the model is the glacier evolution, which includes
             calculating the specific mass balance, the surface type, and any
             changes to the size of the glacier (glacier dynamics). The user
             has many options for how this aspect of the model is run.
  > Others: model output? model input?

========== LIST OF MODEL VARIABLES (alphabetical) ===========================
Prefixes and Suffixes:
  _annual: dataframe containing information with an annual time step as
          opposed to the time step specified by the model (daily or monthly).
  _bin_:  dataframe containing information related to each elevation bin/band
          on the glacier. These dataframes are indexed such that the main
          index (rows) are elevation bins and the columns are the time series.
  _glac_: dataframe containing information related to a glacier. When used by
          itself (e.g., main_glac_rgi or gcm_glac_temp) it refers to each row
          being a specific glacier. When used as a prefix followed by a
          descriptor (e.g., glac_bin_temp), the entire dataframe is for one
          glacier and the descriptor provides information as to the rows
  gcm_:   meteorological data from the global climate model or reanalysis
          dataset
  main_:  dataframe containing important information related to all the
          glaciers in the study, where each row represents a glacier (ex.
          main_glac_rgi).
  series_: series containing information for a given time step with respect to
          all the glacier bins. This is needed when cycling through each
          time step to calculate the mass balance since snow accumulates and
          alters the surface type.

Variables:
  dates_table: dataframe of the dates, month, year, and number of days in the
          month for each date.
          Rows = dates, Cols = attributes
> end_date: end date of model run
          (MAY NOT BE NEEDED - CHECK WHERE IT'S USED)
  gcm_glac_elev: Series of elevation data associated with the global climate
          model temperature data
  gcm_glac_prec: dataframe of the global climate model precipitation data,
          typically based on the nearest neighbor.
  gcm_glac_temp: dataframe of the global climate model temperature data,
          typically based on the nearest neighbor.
  glac_bin_temp: dataframe of temperature for each bin for each time step on
          the glacier
  glac_bin_prec: dataframe of precipitation (liquid) for each bin for each
          time step on the glacier
  glac_bin_precsnow: dataframe of the total precipitation (liquid and solid)
          for each bin for each time step on the glacier
  glac_bin_snow: dataframe of snow for each bin for each time step on
          the glacier
  glac_bin_snowonsurface: dataframe of the snow that has accumulated on the
          surface of the glacier for each bin for each time step
  glac_bin_surftype: datframe of the surface type for each bin for each time
          step on the glacier
  main_glac_rgi: dataframe containing generic attributes (RGIId, Area, etc.)
          from the Randolph Glacier Inventory for each glacier.
          Rows = glaciers, Cols = attributes
  main_glac_parameters: dataframe containing the calibrated parameters for
          each glacier
  main_glac_surftypeinit: dataframe containing in the initial surface type for
          each glacier
> start_date: start date of model run
          (MAY NOT BE NEEDED - CHECK WHERE IT'S USED)
