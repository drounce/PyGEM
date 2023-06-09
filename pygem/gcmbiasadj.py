"""Run bias adjustments a given climate dataset"""

# External libraries
import numpy as np
from scipy.ndimage import uniform_filter


#%% FUNCTIONS
def annual_avg_2darray(x):
    """
    Annual average of dataset, where columns are a monthly timeseries (temperature)
    """
    return x.reshape(-1,12).mean(1).reshape(x.shape[0],int(x.shape[1]/12))


def annual_sum_2darray(x):
    """
    Annual sum of dataset, where columns are a monthly timeseries (precipitation)
    """
    return x.reshape(-1,12).sum(1).reshape(x.shape[0],int(x.shape[1]/12))


def monthly_avg_2darray(x):
    """
    Monthly average for a given 2d dataset where columns are monthly timeseries
    """
    return x.reshape(-1,12).transpose().reshape(-1,int(x.shape[1]/12)).mean(1).reshape(12,-1).transpose()


def monthly_std_2darray(x):
    """
    Monthly standard deviation for a given 2d dataset where columns are monthly timeseries
    """
    return x.reshape(-1,12).transpose().reshape(-1,int(x.shape[1]/12)).std(1).reshape(12,-1).transpose()

    
def temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, dates_table_ref, dates_table, 
                        ref_spinupyears=0, gcm_spinupyears=0):
    """
    Huss and Hock (2015) temperature bias correction based on mean and interannual variability
    
    Note: the mean over the reference period will only equal the mean of the gcm for the same time period when the GCM
    time series is run for the same period, i.e., due to the 25-year moving average, the mean gcm temps from 2000-2019
    will differ if using a reference period of 2000-2020 to bias adjust gcm temps from 2000-2100.
    
    Parameters
    ----------
    ref_temp : np.array
        time series of reference temperature
    gcm_temp : np.array
        time series of GCM temperature
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_temp_biasadj : np.array
        GCM temperature bias corrected to the reference climate dataset according to Huss and Hock (2015)
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_temp_subset = gcm_temp[:,gcm_subset_idx_start:gcm_subset_idx_end+1]

    # Remove spinup years, so adjustment performed over calibration period
    ref_temp_nospinup = ref_temp[:,ref_spinupyears*12:]
    gcm_temp_nospinup = gcm_temp_subset[:,gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    if roll_amt == -12:
        roll_amt = 0 
    
    # Mean monthly temperature
    ref_temp_monthly_avg = np.roll(monthly_avg_2darray(ref_temp_nospinup), roll_amt, axis=1)
    gcm_temp_monthly_avg = np.roll(monthly_avg_2darray(gcm_temp_nospinup), roll_amt, axis=1)
    # Standard deviation monthly temperature
    ref_temp_monthly_std = np.roll(monthly_std_2darray(ref_temp_nospinup), roll_amt, axis=1)
    gcm_temp_monthly_std = np.roll(monthly_std_2darray(gcm_temp_nospinup), roll_amt, axis=1)

    # Monthly bias adjustment (additive)
    gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg

    # Monthly variability
    variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
    
    # Monthly temperature bias adjusted according to monthly average
    t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
    
    # Mean monthly temperature bias adjusted according to monthly average
    #  t_m25avg is the avg monthly temp in a 25-year period around the given year
    N = 25
    t_m_Navg = np.zeros(t_mt.shape)
    for month in range(0,12):
        t_m_subset = t_mt[:,month::12]
        # Uniform filter computes running average and uses 'reflects' values at borders
        t_m_Navg_subset = uniform_filter(t_m_subset,size=(1,N))
        t_m_Navg[:,month::12] = t_m_Navg_subset

    gcm_temp_biasadj = t_m_Navg + (t_mt - t_m_Navg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assert that mean temperatures for all the glaciers must be more-or-less equal
    gcm_temp_biasadj_subset = (
            gcm_temp_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,ref_spinupyears*12:])
    assert np.max(np.abs(np.mean(gcm_temp_biasadj_subset, axis=1) - 
                         np.mean(ref_temp[:,ref_spinupyears*12:], axis=1))) < 1, (
            'Error with gcm temperature bias adjustment: mean ref and gcm temps differ by more than 1 degree')
    
    return gcm_temp_biasadj, gcm_elev_biasadj


def prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec, dates_table_ref, dates_table,
                        ref_spinupyears=0, gcm_spinupyears=0):
    """
    Huss and Hock (2015) precipitation bias correction based on mean (multiplicative)
    
    Parameters
    ----------
    ref_prec : np.array
        time series of reference precipitation
    gcm_prec : np.array
        time series of GCM precipitation
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    
    Returns
    -------
    gcm_prec_biasadj : np.array
        GCM precipitation bias corrected to the reference climate dataset according to Huss and Hock (2015)
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,ref_spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = np.roll(monthly_avg_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_monthly_avg = np.roll(monthly_avg_2darray(gcm_prec_nospinup), roll_amt, axis=1)
    bias_adj_prec_monthly = ref_prec_monthly_avg / gcm_prec_monthly_avg
    # Bias adjusted precipitation accounting for differences in monthly mean
    gcm_prec_biasadj = gcm_prec * np.tile(bias_adj_prec_monthly, int(gcm_prec.shape[1]/12))
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assertion that bias adjustment does not drastically modify the precipitation and are reasonable
    gcm_prec_biasadj_subset = (
            gcm_prec_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,gcm_spinupyears*12:])
    gcm_prec_biasadj_frac = gcm_prec_biasadj_subset.sum(axis=1) / ref_prec_nospinup.sum(axis=1)
    assert np.min(gcm_prec_biasadj_frac) > 0.5 and np.max(gcm_prec_biasadj_frac) < 2, (
            'Error with gcm precipitation bias adjustment: total ref and gcm prec differ by more than factor of 2')
    assert gcm_prec_biasadj.max() <= 10, 'gcm_prec_adj (precipitation bias adjustment) too high, needs to be modified'
    assert gcm_prec_biasadj.min() >= 0, 'gcm_prec_adj is producing a negative precipitation value'   
    
    return gcm_prec_biasadj, gcm_elev_biasadj


def prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec, dates_table_ref, dates_table,
                        ref_spinupyears=0, gcm_spinupyears=0):
    """
    Precipitation bias correction based on mean with limited maximum
    
    Parameters
    ----------
    ref_prec : np.array
        time series of reference precipitation
    gcm_prec : np.array
        time series of GCM precipitation
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_prec_biasadj : np.array
        GCM precipitation bias corrected to the reference climate dataset according to Huss and Hock (2015)
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,ref_spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = np.roll(monthly_avg_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_monthly_avg = np.roll(monthly_avg_2darray(gcm_prec_nospinup), roll_amt, axis=1)
    bias_adj_prec_monthly = ref_prec_monthly_avg / gcm_prec_monthly_avg
    # Bias adjusted precipitation accounting for differences in monthly mean
    gcm_prec_biasadj_raw = gcm_prec * np.tile(bias_adj_prec_monthly, int(gcm_prec.shape[1]/12))
    
    # Adjust variance based on zscore and reference standard deviation
    ref_prec_monthly_std = np.roll(monthly_std_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_biasadj_raw_monthly_avg = monthly_avg_2darray(gcm_prec_biasadj_raw[:,0:ref_prec.shape[1]])
    gcm_prec_biasadj_raw_monthly_std = monthly_std_2darray(gcm_prec_biasadj_raw[:,0:ref_prec.shape[1]])
    # Calculate value compared to mean and standard deviation
    gcm_prec_biasadj_zscore = (
            (gcm_prec_biasadj_raw - np.tile(gcm_prec_biasadj_raw_monthly_avg, int(gcm_prec.shape[1]/12))) / 
             np.tile(gcm_prec_biasadj_raw_monthly_std, int(gcm_prec.shape[1]/12)))
    gcm_prec_biasadj = (
            np.tile(gcm_prec_biasadj_raw_monthly_avg, int(gcm_prec.shape[1]/12)) +
            gcm_prec_biasadj_zscore * np.tile(ref_prec_monthly_std, int(gcm_prec.shape[1]/12)))
    gcm_prec_biasadj[gcm_prec_biasadj < 0] = 0
    
    # Identify outliers using reference's monthly maximum adjusted for future increases
    ref_prec_monthly_max = np.roll((ref_prec_nospinup.reshape(-1,12).transpose()
                                    .reshape(-1,int(ref_prec_nospinup.shape[1]/12)).max(1).reshape(12,-1).transpose()), 
                                   roll_amt, axis=1)
    gcm_prec_max_check = np.tile(ref_prec_monthly_max, int(gcm_prec_biasadj.shape[1]/12))        
    # For wetter years in future, adjust monthly max by the annual increase in precipitation
    gcm_prec_annual = annual_sum_2darray(gcm_prec)
    gcm_prec_annual_norm = gcm_prec_annual / gcm_prec_annual.mean(1)[:,np.newaxis]
    gcm_prec_annual_norm_repeated = np.repeat(gcm_prec_annual_norm, 12).reshape(gcm_prec_biasadj.shape)
    gcm_prec_max_check_adj = gcm_prec_max_check * gcm_prec_annual_norm_repeated
    gcm_prec_max_check_adj[gcm_prec_max_check_adj < gcm_prec_max_check] = (
            gcm_prec_max_check[gcm_prec_max_check_adj < gcm_prec_max_check])
    
    # Replace outliers with monthly mean adjusted for the normalized annual variation
    outlier_replacement = (gcm_prec_annual_norm_repeated * 
                           np.tile(ref_prec_monthly_avg, int(gcm_prec_biasadj.shape[1]/12)))
    gcm_prec_biasadj[gcm_prec_biasadj > gcm_prec_max_check_adj] = (
            outlier_replacement[gcm_prec_biasadj > gcm_prec_max_check_adj])
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assertion that bias adjustment does not drastically modify the precipitation and are reasonable
    gcm_prec_biasadj_subset = (
            gcm_prec_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,gcm_spinupyears*12:])
    gcm_prec_biasadj_frac = gcm_prec_biasadj_subset.sum(axis=1) / ref_prec_nospinup.sum(axis=1)
    assert np.min(gcm_prec_biasadj_frac) > 0.5 and np.max(gcm_prec_biasadj_frac) < 2, (
            'Error with gcm precipitation bias adjustment: total ref and gcm prec differ by more than factor of 2')
    assert gcm_prec_biasadj.max() <= 10, 'gcm_prec_adj (precipitation bias adjustment) too high, needs to be modified'
    assert gcm_prec_biasadj.min() >= 0, 'gcm_prec_adj is producing a negative precipitation value'   
    
    return gcm_prec_biasadj, gcm_elev_biasadj

    
def monthly_avg_array_rolled(ref_array, dates_table_ref, dates_table):
    """ Monthly average array from reference data rolled to ensure proper months 
    
    Parameters
    ----------
    ref_array : np.array
        time series of reference lapse rates
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_array : np.array
        gcm climate data based on monthly average of reference data
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    ref_array_monthly_avg = np.roll(monthly_avg_2darray(ref_array), roll_amt, axis=1)
    gcm_array = np.tile(ref_array_monthly_avg, int(dates_table.shape[0]/12))
    return gcm_array