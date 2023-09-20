import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

glac_props = {'01.00570':{'name':'Gulkana',
                            'AWS_fn':'gulkana1725_hourly.csv'},
            '01.01104':{'name':'Lemon Creek',
                            'AWS_fn':'LemonCreek1285_hourly.csv'},
            '01.16195':{'name':'South',
                            'AWS_fn':'Preprocessed/south/south2280_hourly_2008_wNR.csv'},
            '08.00213':{'name':'Storglaciaren',
                            'AWS_fn':'Storglaciaren/SITES_MET_TRS_SGL_dates_15MIN.csv'},
            '11.03674':{'name':'Saint-Sorlin',
                            'AWS_fn':'Preprocessed/saintsorlin/saintsorlin_hourly.csv'},
            '16.02444':{'name':'Artesonraju',
                            'AWS_fn':'Preprocessed/artesonraju/Artesonraju_hourly.csv'}}
vartype = {'surftemp':'Temperature','airtemp':'Temperature',
           'melt':'MB','runoff':'MB','accum':'MB','refreeze':'MB',
           'meltenergy':'Flux','SWin':'Flux','SWout':'Flux',
           'LWin':'Flux','LWout':'Flux','SWnet':'Flux','LWnet':'Flux',
           'NetRad':'Flux','sensible':'Flux','latent':'Flux','rain':'Flux',
           'layertemp':'Layers','layerdensity':'Layers','layerwater':'Layers',
           'layerheight':'Layers','snowdepth':'Snow depth','albedo':'Albedo'}
varprops = {'surftemp':{'Temperature'},'airtemp':'Temperature',
           'melt':'MB','runoff':'MB','accum':'MB','refreeze':'MB',
           'meltenergy':'Flux','SWin':'Flux','SWout':'Flux',
           'LWin':'Flux','LWout':'Flux','SWnet':'Flux','LWnet':'Flux',
           'NetRad':'Flux','sensible':'Flux','latent':'Flux','rain':'Flux',
           'layertemp':{'label':'Temperature (C)'},'layerdensity':{'label':'Density (kg m-3)'},'layerwater':{'label':'Water Content (kg m-2)'},
           'layerheight':'Layers','snowdepth':'Snow depth','albedo':'Albedo'}

def simple_plot(ds,time,vars,colors,res='d',t=''):
    fig,axes = plt.subplots(len(vars),1,figsize=(7,1.5*len(vars)),sharex=True)

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
    ds = ds.sel(time=time)
    ds = ds.resample(time=res).mean(dim='time')
    for i,v in enumerate(vars):
        ic = i-6 if i>5 else i
        vararray = np.array(v)
        for var in vararray:
            if len(vars) > 1:
                axis = axes[i]
            else:
                axis = axes
            if var in ['melt','runoff','accum','refreeze']:
                axis.plot(ds.coords['time'],ds[var].cumsum(),color=colors[ic],label=var)
            else:
                axis.plot(ds.coords['time'],ds[var],color=colors[ic],label=var)
            ic+=1
        axis.legend()
        try:
            axis.set_ylabel(f'{vartype[var]} [{ds[var].attrs["units"]}]')
        except:
            axis.set_ylabel(vartype[var])
    date_form = mpl.dates.DateFormatter('%d %b')
    axis.xaxis.set_major_formatter(date_form)
    fig.suptitle(t)
        
def compare_runs(ds1,ds2,time,var,colors,res='d',t=''):
    fig,ax = plt.subplots(figsize=(8,6))
    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq=res)
    ds1 = ds1.resample(time=res).mean(dim='time')
    ds2 = ds2.resample(time=res).mean(dim='time')
    ax.plot(time,ds1[var].sel(time=time),label='ds1')
    ax.plot(time,ds2[var].sel(time=time),label='ds2')
    date_form = mpl.dates.DateFormatter('%d %b')
    ax.xaxis.set_major_formatter(date_form)
    ax.set_ylabel(var)
    ax.legend()
    fig.suptitle(t)
    plt.show()

    return

def stacked_eb_barplot(ds,time,colors,res='d',t=''):
    fig,ax = plt.subplots(figsize=(10,2.5))
    vars = ['latent','NetRad','sensible','rain'] #'SWnet','LWnet'

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
    ds = ds.resample(time=res).mean(dim='time')
    bottom=0
    for i,var in enumerate(np.array(vars)):
        vardata = ds[var].to_numpy().T[0]
        if i==0:
            ax.bar(ds.coords['time'],vardata,label=var)
        else:
            bottom = ds[vars[i-1]].to_numpy().T[0]+bottom
            bottom[np.where(bottom<0)] = 0
            ax.bar(ds.coords['time'],vardata,bottom=bottom,label=var)
    # ax.plot(ds.coords['time'],ds['sum'],label='sum',color='black',linewidth=.6)
    ax.legend()
    date_form = mpl.dates.DateFormatter('%d %b')
    ax.xaxis.set_major_formatter(date_form)
    ax.set_ylabel('Fluxes (W/m2)')
    fig.suptitle(t)
    plt.show()

def plot_avgs(file,nyr,title=False):
    """
    Plots heat fluxes, surface/air temperature, and mass balance terms, averaged monthly and then interannually.
    """
    months = np.arange(1,13)

    fig,axes = plt.subplots(3,3,sharex=True,sharey='row',figsize=(12,6))

    ds = xr.open_dataset(file)
    varnames_idx = ['SWin','SWout','LWin','LWout','sensible','latent','rain','meltenergy','surftemp','melt','runoff','refreeze','accum']
    varnames = ['SWnet','LWnet','sensible','latent','rain','meltenergy','surftemp','melt','runoff','refreeze','accum']
    heat = ['SWnet','LWnet','sensible','latent','rain','meltenergy']
    temp = ['surftemp']
    mb = ['melt','runoff','refreeze','accum','MB']

    airtemp = np.zeros((3,12))
    for b in range(3):
        climateds = xr.open_dataset('/home/claire/research/Output/EB/climateds.nc').isel(bin=b).to_pandas()
        monthly = climateds['bin_temp'].resample('M').mean()
        monthly_avg = np.mean(monthly[:(nyr*12)].values.reshape((nyr,12)),axis=0)
        airtemp[b,:] = monthly_avg

    for bin_no in [0,1,2]:
        df = ds[varnames_idx].isel(bin=bin_no).to_pandas()
        df['SWnet'] = df['SWin'] + df['SWout']
        df['LWnet'] = df['LWin'] + df['LWout']
        df['melt'] = df['melt'] * -1
        
        for var in varnames:
            if var in ['melt','runoff','refreeze','accum','MB']:
                monthly = df[var].resample('M').sum()
            else:
                monthly = df[var].resample('M').mean()
            monthly_avg = np.mean(monthly[:(nyr*12)].values.reshape((nyr,12)),axis=0)
            run_start_month = pd.Timestamp(ds.coords['time'].values[0]).month
            if run_start_month > 1:
                monthly_avg_jan = np.append(monthly_avg[13-run_start_month:],monthly_avg[:13-run_start_month],)
            else:
                monthly_avg_jan = monthly_avg

            axis = np.piecewise(var,[var in heat, var in temp, var in mb],[0,1,2])
            lw = 1 if var in ['meltenergy','surftemp'] else 0.5
            axes[int(axis),bin_no].plot(months,monthly_avg_jan,label=var,linewidth=lw)
        axes[1,bin_no].plot(months,airtemp[bin_no,:],linewidth=0.5,label='air temp')
        axes[1,bin_no].axhline(0,lw=0.3,color='gray')
        axes[bin_no,2].set_xticks(months)
    axes[0,0].set_ylabel('Energy Flux ($W / m^2$)')
    axes[1,0].set_ylabel('Temperature (C)')
    axes[2,0].set_ylabel('Mass balance (m w.e.)')

    binname = ['Lower','Middle','Upper']
    for i in range(3):
        axes[0,i].set_title(binname[i])
    axes[0,0].legend(loc='upper right',bbox_to_anchor=(1.25, 1.0))
    axes[1,0].legend()
    axes[2,0].legend()
    if not title:
        fig.suptitle(f'Gulkana Glacier (ERA5-Hourly)\nMonthly Averages Over {nyr}-yr Run')
    else:
        fig.suptitle(title)
    fig.supxlabel('Months')
    #plt.gcf().autofmt_xdate()
    plt.show()
    return

def plot_yrs(file,bin,nyr):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig,axes = plt.subplots(3,nyr,sharey='row',sharex='col',figsize=(14,8))

    ds = xr.open_dataset(file)
    varnames_idx = ['SWin','SWout','LWin','LWout','sensible','latent','rain','meltenergy','surftemp','melt','runoff','refreeze','accum','snowdepth']
    varnames = ['SWnet','LWnet','sensible','latent','rain','meltenergy','melt','runoff','refreeze','accum','snowdepth']
    heat = ['SWnet','LWnet','sensible','latent','rain','meltenergy']
    temp = ['snowdepth']
    mb = ['melt','runoff','refreeze','accum','MB']

    df = ds[varnames_idx].isel(bin=bin).to_pandas()
    df['SWnet'] = df['SWin'] + df['SWout']
    df['LWnet'] = df['LWin'] + df['LWout']
    df['MB'] = df['accum']+df['refreeze']-df['melt']

    # Loop through variables to get monthly averages and plot them
    for var in varnames:
        if var in ['melt','runoff','refreeze','accum','MB']:
            monthly = df[var].resample('M').sum()
        else:
            monthly = df[var].resample('M').mean()
        monthly_avg = monthly[:(nyr*12)].values.reshape((nyr,12))
        
        axis = np.piecewise(var,[var in heat, var in temp, var in mb],[0,1,2])
        for yr in range(nyr):
            axes[int(axis),yr].plot(months,monthly_avg[yr,:],label=var)
            axes[int(axis),yr].set_xlabel(str(pd.to_datetime(ds.coords['time'].values[0]).year+yr))
            axes[int(axis),yr].set_ylabel(var)
            axes[int(axis),0].legend()
            axes[1,0].set_xlim(0,6)
    return

def plot_AWS(df,vars,time,t=''):
    df = df.set_index(pd.to_datetime(df.index))
    start = pd.to_datetime(time[0])
    end = pd.to_datetime(time[-1])
    df = df.loc[start:end]
    days = df.resample('d').mean().index
    hours = np.arange(0,24)
    d,h = np.meshgrid(days,hours)

    fig,axs = plt.subplots(len(vars),sharex=True)
    for i,var in enumerate(vars):
        vardata = df[var].to_numpy().reshape(np.shape(d))
        pc = axs[i].pcolormesh(d,h,vardata.T, cmap='RdBu_r')
        fig.colorbar(pc,ax=axs[i])
        axs[i].set_title(var)
        axs[i].set_ylabel('Hour')
        yticks = mpl.ticker.MultipleLocator(6)
        axs[i].yaxis.set_major_locator(yticks)
    date_form = mpl.dates.DateFormatter('%d %b')
    axs[i].xaxis.set_major_formatter(date_form)
    fig.suptitle(t)
    plt.show()

def plot_avg_layers(file,bin,nyr):
    """
    Plots layer temperature, density, water content and layer height averaged first across all layers,
    then averaged between years. 
    """
    ds = xr.open_dataset(file)
    fig,axes = plt.subplots(2,2,sharex=True,figsize=(8,6)) #,sharex=True,sharey='row',figsize=(12,5)
    idxs = [[0,0],[0,1],[1,0],[1,1]]
    days =  np.arange(365)
    for ax,var in enumerate(['layertemp','layerdensity','layerwater','snowdepth']):
        snow = ds[var].sel(bin=bin).to_pandas()
        loop = True
        i=19
        while loop:
            if np.isnan(snow.iloc[i]).all and var not in ['snowdepth']:
                snow.drop(i,axis=1)
            i -=1
            if i == 0:
                break

        if var in ['layertemp','layerdensity']:
            snow = snow.mean(axis=1)
        elif var in ['snowdepth']:
            pass
        else:
            snow = snow.sum(axis=1)
        if var in ['layerwater']:
            snow = snow/1000 # to m w.e.
        snowdaily = snow.resample('d').mean()
        snowdaily = np.mean(snow[:nyr*365].values.reshape((nyr,365)),axis=0)

        idx = idxs[ax]
        axes[idx[0],idx[1]].plot(days,snowdaily,label=var)
        axes[idx[0],idx[1]].set_title(var+'   '+ds[var].attrs['units'])
    axes[1,0].set_title('water content    m w .e')
    plt.gcf().autofmt_xdate()
    # plt.savefig('/home/claire/research/Output/EB/subsurfplot.png')
    plt.show()
    return

def plot_layers(ds,var,dates):
    fig,axes = plt.subplots(1,len(dates),sharey=True,sharex=True,figsize=(8,4)) #,sharex=True,sharey='row'
    for i,date in enumerate(dates):
        for bin in ds.coords['bin'].values:
            lheight = ds.sel(time=date,bin=bin)['layerheight'].to_numpy()
            full_bins = np.array([not y for y in np.isnan(lheight)])
            lheight = lheight[full_bins]
            lprop = ds.sel(time=date,bin=bin)[var].to_numpy()[full_bins]
            ldepth = -1*np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
            if var in ['layerwater']:
                lprop = lprop[:-1]
                ldepth = ldepth[:-1]
            axes[i].plot(lprop,ldepth,label='bin '+str(bin))
            axes[i].set_title(str(date)[:10])
    axes[0].legend()
    fig.supxlabel(varprops[var]['label'])
    axes[0].set_ylabel('Depth (m)')
    return

def plot_monthly_layer_avgs(file,var,dates_to_plot,colors):
    ds = xr.open_dataset(file)
    fig,axes = plt.subplots(1,len(dates_to_plot),sharey=True,sharex=True,figsize=(8,4)) #,sharex=True,sharey='row'
    df = ds['snowdepth'].to_pandas()
    snowdepth_monthly = df.resample('M').mean()
    # for each month, find average snow depth
    # interpolate the variable of interest at new depths, with max depth being the average snow depth of the month
    # take mean of resampled variables
    # plot new depths vs. monthly means
    ax=0
    for month in range(12):
        snowdepth = snowdepth_monthly.iloc[month]
        for bin_no in [0,1,2]:
            var_data = ds[var].isel(bin=bin_no).to_numpy()
            depth_data = ds['layerheight'].isel(bin=bin_no).to_numpy()
            new_depths = np.linspace(0,snowdepth.iloc[bin_no],20)

            zeros = np.zeros((len(ds.coords['time'].values),20))
            var_interp = zeros.copy()
            for i,t in enumerate(ds.coords['time'].values):
                var_interp[i,:] = np.interp(new_depths,depth_data[i,:],var_data[i,:])
            da = xr.DataArray(data=var_interp,
                    coords=dict(
                        time=(['time'],ds.coords['time'].values),
                        depth=(['depth'],new_depths)
                        ))
            da_monthly = da.resample(time='M').mean()
            if month in months_to_plot:
                axes[ax].plot(da.isel(time=month).data,-1*da.coords['depth'].values,label='Bin '+str(bin_no))
        if month in months_to_plot: 
            axes[ax].set_xlabel(var)
            axes[0].set_ylabel('Depth (m)')
            month_dict = {'0':'Jan','1':'Feb','2':'Mar','3':'Apr','4':'May','5':'Jun',
                        '6':'Jul','7':'Aug','8':'Sept','9':'Oct','10':'Nov','11':'Dec'}
            axes[ax].set_title(month_dict[str(month)])
            axes[ax].axhline(-snowdepth[0],color=colors[0],linestyle='--')
            axes[ax].axhline(-snowdepth[1],color=colors[1],linestyle='--')
            axes[ax].axhline(-snowdepth[2],color=colors[2],linestyle='--')
            ax += 1
    axes[len(months_to_plot)-1].legend()
    return