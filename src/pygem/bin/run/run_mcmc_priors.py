""" Export regional priors """

import argparse
import os
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# pygem imports
import pygem.setup.config as config
# check for config
config.ensure_config()
# read the config
pygem_prms = config.read_config()
import pygem.pygem_modelsetup as modelsetup

# Region dictionary for titles
reg_dict = {1:'Alaska',
            2:'W CA/USA',
            3:'Arctic CA N',
            4:'Arctic CA S',
            5:'Greenland',
            6:'Iceland',
            7:'Svalbard',
            8:'Scandinavia',
            9:'Russian Arctic',
            10:'N Asia',
            11:'C Europe',
            12:'Caucasus/Middle East',
            13:'C Asia',
            14:'S Asia W',
            15:'S Asia E',
            16:'Low Latitudes',
            17:'S Andes',
            18:'New Zealand',
            19:'Antarctica'}

# FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-rgi_region01', type=int, default=pygem_prms['setup']['rgi_region01'],
                        help='Randoph Glacier Inventory region (can take multiple, e.g. `-run_region01 1 2 3`)', nargs='+')

    # flags
    parser.add_argument('-v', '--debug', action='store_true',
                        help='Flag for debugging')
    return parser


def run(reg, debug=False):

    # Calibration filepath
    modelprms_fp = pygem_prms.output_filepath + 'calibration/' + str(reg).zfill(2) + '/'
    
    # Load glaciers
    glac_list = [x.split('-')[0] for x in os.listdir(modelprms_fp) if x.endswith('-modelprms_dict.json')]
    glac_list = sorted(glac_list)

    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_list)
    
    # Add model parameters to main dataframe
    main_glac_rgi['kp'] = np.nan
    main_glac_rgi['tbias'] = np.nan
    main_glac_rgi['ddfsnow'] = np.nan
    main_glac_rgi['mb_mwea'] = np.nan
    main_glac_rgi['kp_em'] = np.nan
    main_glac_rgi['tbias_em'] = np.nan
    main_glac_rgi['ddfsnow_em'] = np.nan
    main_glac_rgi['mb_mwea_em'] = np.nan
    for nglac, rgino_str in enumerate(list(main_glac_rgi.rgino_str.values)):
#    for nglac, rgino_str in enumerate(list(main_glac_rgi.rgino_str.values)[0:1]):
        
        glac_str = str(int(rgino_str.split('.')[0])) + '.' + rgino_str.split('.')[1]
        
        # Load model parameters
        modelprms_fn = glac_str + '-modelprms_dict.json'
        with open(modelprms_fp + modelprms_fn, 'r') as f:
            modelprms_dict = json.load(f)
        modelprms = modelprms_dict[pygem_prms.option_calibration]
        
##        modelprms_fp_em = pygem_prms.output_filepath + 'calibration/' + str(reg).zfill(2) + '-emulator/'
#        modelprms_fp_em = '/Users/drounce/Documents/HiMAT/spc_backup/calibration/' + str(reg).zfill(2) + '-emulator/'
#        with open(modelprms_fp_em + modelprms_fn, 'rb') as f:
#            modelprms_dict_em = pickle.load(f)
#            modelprms_em = modelprms_dict_em['emulator']
        
        
        main_glac_rgi.loc[nglac, 'kp'] = modelprms['kp'][0]
        main_glac_rgi.loc[nglac, 'tbias'] = modelprms['tbias'][0]
        main_glac_rgi.loc[nglac, 'ddfsnow'] = modelprms['ddfsnow'][0]
        main_glac_rgi.loc[nglac, 'mb_mwea'] = modelprms['mb_mwea'][0]
        
#        main_glac_rgi.loc[nglac, 'kp_em'] = modelprms_em['kp'][0]
#        main_glac_rgi.loc[nglac, 'tbias_em'] = modelprms_em['tbias'][0]
#        main_glac_rgi.loc[nglac, 'ddfsnow_em'] = modelprms_em['ddfsnow'][0]
#        main_glac_rgi.loc[nglac, 'mb_mwea_em'] = modelprms_em['mb_mwea'][0]
        
        main_glac_rgi.loc[nglac, 'mb_obs_mwea'] = modelprms['mb_obs_mwea'][0]
        
    main_glac_rgi['mb_dif_obs_cal'] = main_glac_rgi['mb_obs_mwea'] - main_glac_rgi['mb_mwea']
#    main_glac_rgi['mb_dif_obs_em'] = main_glac_rgi['mb_obs_mwea'] - main_glac_rgi['mb_mwea_em']
       
    # Priors for each subregion
    if reg not in [19]:
        rgi_regionsO2 = np.unique(main_glac_rgi.O2Region.values)
        for regO2 in rgi_regionsO2:
            main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi['O2Region'] == regO2, :]
        
            # Histograms and record model parameter statistics
            fig, ax = plt.subplots(1,2, figsize=(6,4), gridspec_kw = {'wspace':0.3, 'hspace':0.3})
            
            labelsize = 12
            
            fig.text(0.5,0.9, 'Region ' + str(reg) + ' (subregion: ' + str(regO2) + ')', ha='center', size=14)
            
            nbins = 50
            ax[0].hist(main_glac_rgi_subset['kp'], bins=nbins, color='grey')
            ax[0].set_xlabel('kp (-)')
            ax[0].set_ylabel('Count (glaciers)')
            
            ax[1].hist(main_glac_rgi_subset['tbias'], bins=50, color='grey')
            ax[1].set_xlabel('tbias (degC)')
            
            # Precipitation factors
            kp_mean = np.mean(main_glac_rgi_subset['kp'])
            kp_std = np.std(main_glac_rgi_subset['kp'])
            kp_med = np.median(main_glac_rgi_subset['kp'])
            kp_min = np.min(main_glac_rgi_subset['kp'])
            kp_max = np.max(main_glac_rgi_subset['kp'])
        
            # Small regions may all have the same values (e.g., 16-4 has 5 glaciers)
            if kp_std == 0:
                kp_std = 0.5
        
            kp_beta = kp_mean / kp_std
            kp_alpha = kp_mean * kp_beta
            
            print('\n', reg, '(' + str(regO2) + ')')
            print('kp (mean/std/med/min/max):', np.round(kp_mean,2), np.round(kp_std,2),
                  np.round(kp_med,2), np.round(kp_min,2), np.round(kp_max,2))
            print('  alpha/beta:', np.round(kp_alpha,2), np.round(kp_beta,2))
            
            # Temperature bias
            tbias_mean = main_glac_rgi_subset['tbias'].mean()
            tbias_std = main_glac_rgi_subset['tbias'].std()
            tbias_med = np.median(main_glac_rgi_subset['tbias'])
            tbias_min = np.min(main_glac_rgi_subset['tbias'])
            tbias_max = np.max(main_glac_rgi_subset['tbias'])
            
            # tbias_std of 1 is reasonable for most subregions
            if tbias_std == 0:
                tbias_std = 1
            
            print('tbias (mean/std/med/min/max):', np.round(tbias_mean,2), np.round(tbias_std,2),
                  np.round(tbias_med,2), np.round(tbias_min,2), np.round(tbias_max,2))
        
            # Save figure
            fig_fp = pygem_prms.output_filepath + 'calibration/figs/'
            if not os.path.exists(fig_fp):
                os.mkdir(fig_fp)
            fig_fn = str(reg) + '-' + str(regO2) + '_hist_mcmc_priors.png'
            fig.savefig(fig_fp + fig_fn, pad_inches=0, dpi=150)
                    
            # EXPORT PRIORS
            priors_cn = ['O1Region', 'O2Region', 'count',
                         'kp_mean', 'kp_std', 'kp_med', 'kp_min', 'kp_max', 'kp_alpha', 'kp_beta', 
                         'tbias_mean', 'tbias_std', 'tbias_med', 'tbias_min', 'tbias_max']
            priors_df_single = pd.DataFrame(np.zeros((1,len(priors_cn))), columns=priors_cn)
            priors_df_single.loc[0,:] = (
                    [reg, regO2, main_glac_rgi_subset.shape[0],
                     kp_mean, kp_std, kp_med, kp_min, kp_max, kp_alpha, kp_beta, 
                     tbias_mean, tbias_std, tbias_med, tbias_min, tbias_max])
            if os.path.exists(pygem_prms.priors_reg_fullfn):
                priors_df = pd.read_csv(pygem_prms.priors_reg_fullfn)
                
                # Add or overwrite existing priors
                priors_idx = np.where((priors_df.O1Region == reg) & (priors_df.O2Region == regO2))[0]
                if len(priors_idx) > 0:
                    priors_df.loc[priors_idx,:] = priors_df_single.values
                else:
                    priors_df = pd.concat([priors_df, priors_df_single], axis=0)
                    
            else:
                priors_df = priors_df_single
                
            priors_df = priors_df.sort_values(['O1Region', 'O2Region'], ascending=[True, True])
            priors_df.reset_index(inplace=True, drop=True)
            priors_df.to_csv(pygem_prms.priors_reg_fullfn, index=False)
    # Use the entire region for the prior (sometimes subregions make no sense; e.g., 24 regions in Antarctica)
    else:
        rgi_regionsO2 = np.unique(main_glac_rgi.O2Region.values)
        main_glac_rgi_subset = main_glac_rgi.copy()
        
        # Histograms and record model parameter statistics
        fig, ax = plt.subplots(1,2, figsize=(6,4), gridspec_kw = {'wspace':0.3, 'hspace':0.3})
        
        labelsize = 12
        
        fig.text(0.5,0.9, 'Region ' + str(reg) + ' (all subregions)', ha='center', size=14)
        
        nbins = 50
        ax[0].hist(main_glac_rgi_subset['kp'], bins=nbins, color='grey')
        ax[0].set_xlabel('kp (-)')
        ax[0].set_ylabel('Count (glaciers)')
        
        ax[1].hist(main_glac_rgi_subset['tbias'], bins=50, color='grey')
        ax[1].set_xlabel('tbias (degC)')
        
        # Precipitation factors
        kp_mean = np.mean(main_glac_rgi_subset['kp'])
        kp_std = np.std(main_glac_rgi_subset['kp'])
        kp_med = np.median(main_glac_rgi_subset['kp'])
        kp_min = np.min(main_glac_rgi_subset['kp'])
        kp_max = np.max(main_glac_rgi_subset['kp'])
    
        # Small regions may all have the same values (e.g., 16-4 has 5 glaciers)
        if kp_std == 0:
            kp_std = 0.5
    
        kp_beta = kp_mean / kp_std
        kp_alpha = kp_mean * kp_beta
        
        print('\n', reg, '(all subregions)')
        print('kp (mean/std/med/min/max):', np.round(kp_mean,2), np.round(kp_std,2),
              np.round(kp_med,2), np.round(kp_min,2), np.round(kp_max,2))
        print('  alpha/beta:', np.round(kp_alpha,2), np.round(kp_beta,2))
        
        # Temperature bias
        tbias_mean = main_glac_rgi_subset['tbias'].mean()
        tbias_std = main_glac_rgi_subset['tbias'].std()
        tbias_med = np.median(main_glac_rgi_subset['tbias'])
        tbias_min = np.min(main_glac_rgi_subset['tbias'])
        tbias_max = np.max(main_glac_rgi_subset['tbias'])
        
        # tbias_std of 1 is reasonable for most subregions
        if tbias_std == 0:
            tbias_std = 1
        
        print('tbias (mean/std/med/min/max):', np.round(tbias_mean,2), np.round(tbias_std,2),
              np.round(tbias_med,2), np.round(tbias_min,2), np.round(tbias_max,2))
    
        # Save figure
        fig_fp = pygem_prms.output_filepath + 'calibration/figs/'
        if not os.path.exists(fig_fp):
            os.mkdir(fig_fp)
        fig_fn = str(reg) + '_hist_mcmc_priors.png'
        fig.savefig(fig_fp + fig_fn, pad_inches=0, dpi=150)
        
        
        for regO2 in rgi_regionsO2:    
            # EXPORT PRIORS
            priors_cn = ['O1Region', 'O2Region', 'count',
                         'kp_mean', 'kp_std', 'kp_med', 'kp_min', 'kp_max', 'kp_alpha', 'kp_beta', 
                         'tbias_mean', 'tbias_std', 'tbias_med', 'tbias_min', 'tbias_max']
            priors_df_single = pd.DataFrame(np.zeros((1,len(priors_cn))), columns=priors_cn)
            priors_df_single.loc[0,:] = (
                    [reg, regO2, main_glac_rgi_subset.shape[0],
                     kp_mean, kp_std, kp_med, kp_min, kp_max, kp_alpha, kp_beta, 
                     tbias_mean, tbias_std, tbias_med, tbias_min, tbias_max])
            if os.path.exists(pygem_prms.priors_reg_fullfn):
                priors_df = pd.read_csv(pygem_prms.priors_reg_fullfn)
                
                # Add or overwrite existing priors
                priors_idx = np.where((priors_df.O1Region == reg) & (priors_df.O2Region == regO2))[0]
                if len(priors_idx) > 0:
                    priors_df.loc[priors_idx,:] = priors_df_single.values
                else:
                    priors_df = pd.concat([priors_df, priors_df_single], axis=0)
                    
            else:
                priors_df = priors_df_single
                
            priors_df = priors_df.sort_values(['O1Region', 'O2Region'], ascending=[True, True])
            priors_df.reset_index(inplace=True, drop=True)
            priors_df.to_csv(pygem_prms.priors_reg_fullfn, index=False)

    
    def plot_priors():
        pass

        # ===== REGIONAL PRIOR: PRECIPITATION FACTOR ======
        nbins = 50    
        ncols = 3
        nrows = int(np.ceil(len(rgi_regionsO2)/ncols))
        priors_df_regO1 = priors_df.loc[priors_df['O1Region'] == reg]
        
        fig, ax = plt.subplots(nrows, ncols, squeeze=False, gridspec_kw={'wspace':0.5, 'hspace':0.5})
        
        nrow = 0
        ncol = 0
        for nreg, regO2 in enumerate(rgi_regionsO2):
            priors_df_regO2 = priors_df_regO1.loc[priors_df['O2Region'] == regO2]
            kp_values = main_glac_rgi.loc[main_glac_rgi['O2Region'] == regO2, 'kp'].values
            nglaciers = kp_values.shape[0]

            # Plot histogram
            counts, bins, patches = ax[nrow,ncol].hist(kp_values, facecolor='grey', edgecolor='grey', 
                                                    linewidth=0.1, bins=nbins, density=True)
            
            # Plot gamma distribution
            alpha = priors_df_regO2.kp_alpha.values[0]
            beta = priors_df_regO2.kp_beta.values[0]
            rv = stats.gamma(alpha, scale=1/beta)
            ax[nrow,ncol].plot(bins, rv.pdf(bins), color='k')
            # add alpha and beta as text
            gammatext = (r'$\alpha$=' + str(np.round(alpha,2)) + '\n' + r'$\beta$=' + str(np.round(beta,2))
                        + '\n$n$=' + str(nglaciers))
            ax[nrow,ncol].text(0.98, 0.95, gammatext, size=10, horizontalalignment='right', 
                            verticalalignment='top', transform=ax[nrow,ncol].transAxes)
            
            # Subplot title
            title_str = reg_dict[reg] + ' (' + str(regO2) + ')'
            ax[nrow,ncol].text(0.5, 1.01, title_str, size=10, horizontalalignment='center', 
                            verticalalignment='bottom', transform=ax[nrow,ncol].transAxes)

            # Adjust row and column
            ncol += 1
            if ncol == ncols:
                nrow += 1
                ncol = 0

        # Remove extra plots
        if len(rgi_regionsO2)%ncols > 0:
            n_extras = ncols-len(rgi_regionsO2)%ncols
            if n_extras > 0:
                for nextra in np.arange(0,n_extras):
                    ax[nrow,ncol].axis('off')
                    ncol += 1
                
        # Labels
        fig.text(0.04, 0.5, 'Probability Density', va='center', ha='center', rotation='vertical', size=12)
        fig.text(0.5, 0.04, '$k_{p}$ (-)', va='center', ha='center', size=12)
            
        # Save figure
        if os.path.exists(fig_fp) == False:
            os.makedirs(fig_fp)    
        fig.set_size_inches(6, 6)
        fig.savefig(fig_fp + 'priors_kp_O2Regions-' + str(reg) + '.png', bbox_inches='tight', dpi=300)

        # ===== REGIONAL PRIOR: TEMPERATURE BIAS ======
        fig, ax = plt.subplots(nrows, ncols, squeeze=False, gridspec_kw={'wspace':0.3, 'hspace':0.3})
        
        nrow = 0
        ncol = 0
        for nreg, regO2 in enumerate(rgi_regionsO2):
            
            priors_df_regO2 = priors_df_regO1.loc[priors_df['O2Region'] == regO2]
            tbias_values = main_glac_rgi.loc[main_glac_rgi['O2Region'] == regO2, 'tbias'].values
            nglaciers = tbias_values.shape[0]
            
            # Plot histogram
            counts, bins, patches = ax[nrow,ncol].hist(tbias_values, facecolor='grey', edgecolor='grey', 
                                                    linewidth=0.1, bins=nbins, density=True)
            
            # Plot gamma distribution
            mu = priors_df_regO2.tbias_mean.values[0]
            sigma = priors_df_regO2.tbias_std.values[0]
            rv = stats.norm(loc=mu, scale=sigma)
            ax[nrow,ncol].plot(bins, rv.pdf(bins), color='k')
            # add alpha and beta as text
            normtext = (r'$\mu$=' + str(np.round(mu,2)) + '\n' + r'$\sigma$=' + str(np.round(sigma,2))
                        + '\n$n$=' + str(nglaciers))
            ax[nrow,ncol].text(0.98, 0.95, normtext, size=10, horizontalalignment='right', 
                            verticalalignment='top', transform=ax[nrow,ncol].transAxes)
            
            # Title
            title_str = reg_dict[reg] + ' (' + str(regO2) + ')'
            ax[nrow,ncol].text(0.5, 1.01, title_str, size=10, horizontalalignment='center', 
                            verticalalignment='bottom', transform=ax[nrow,ncol].transAxes)
            
            # Adjust row and column
            ncol += 1
            if ncol == ncols:
                nrow += 1
                ncol = 0

        # Remove extra plots
        if len(rgi_regionsO2)%ncols > 0:
            n_extras = ncols-len(rgi_regionsO2)%ncols
            if n_extras > 0:
                for nextra in np.arange(0,n_extras):
                    ax[nrow,ncol].axis('off')
                    ncol += 1
                
        # Labels
        fig.text(0.04, 0.5, 'Probability Density', va='center', ha='center', rotation='vertical', size=12)
        fig.text(0.5, 0.04, '$T_{bias}$ ($^\circ$C)', va='center', ha='center', size=12)
            
        # Save figure
        if os.path.exists(fig_fp) == False:
            os.makedirs(fig_fp)    
        fig.set_size_inches(6, 6)
        fig.savefig(fig_fp + 'priors_tbias_O2Regions-' + str(reg) + '.png', bbox_inches='tight', dpi=300)
    
def main():
    parser = getparser()
    args = parser.parse_args()
    time_start = time.time()

    # get priors for each region
    for reg in args.rgi_region01:
        if args.debug:
            print('Region:', reg)
        run(reg, args.debug)

    print('\n\n------\nTotal processing time:', time.time()-time_start, 's')

if __name__ == "__main__":
    main()    