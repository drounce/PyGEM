import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.stats import linregress
import cartopy.crs as ccrs
import cartopy as car

#========== IMPORT INPUT AND FUNCTIONS FROM MODULES ===================================================================
import pygem_input as input
import pygemfxns_postprocessing as post


#%% DATA EXTRACTION 

regionO1_number = str(input.rgi_regionsO1[0])

NETfullfilename=(input.output_filepath + ('PyGEM_R'+regionO1_number+ '_ERA-Interim_'+ str((input.startyear)) + '_' +str(input.endyear) + '_1' ) )

# non-elevation dependant temp = 
glac_temp1=pd.read_csv(input.main_directory + '/../ERAInt_Sim_Selection/' + 'RGI_0' 
                      + str(input.rgi_regionsO1[0]) + '_ERA_Int_Glacier_Temp.csv')
glac_prec1=pd.read_csv(input.main_directory + '/../ERAInt_Sim_Selection/' + 'RGI_0' 
                      + str(input.rgi_regionsO1[0]) + '_ERA_Int_Glacier_PPT.csv')

output = nc.Dataset(NETfullfilename +'.nc')

#year ragne 
year= range((input.startyear-1),input.endyear)

## Select relevant data
glacier_data = pd.DataFrame(output['glacier_table'][:])
glacier_data_columns = output['glacier_table_header'][:]
lats = glacier_data[2].values.astype(float)
lons = glacier_data[1].values.astype(float)
massbal_monthly = output['massbaltotal_glac_monthly'][:]
volume=output['volume_glac_annual'][:]
RGI=output['glac_idx'][:]
temp=output['temp_glac_monthly'][:]
prec=output['prec_glac_monthly'][:]
glac_elev=glacier_data[17][:].astype(float).values
glac_area=glacier_data[5][:].astype(float).values



#%% Mass balance total over time period, mass balance averaged/yr, rate of change: for rate of change 
# do yearly average plot for entire glacier 
#do area weighted mass balance for each reigon (how to do that?)

#for future SIM: do time slices and do total mass balance/yearly average + everything above for each 
#slice 

# do temp and ppt plots: for both the temp and ppt over that period as well as the glacier temp and ppt
# that is adjusted for hypsometry to show the difference 

#volume change.... area weighted 

#%% MASS BALANCE TOTAL AND AVERAGE OVER OBVS PERIOD: MAPS

# total mass balance of each glacier for period of observation (mwe)
massbal_total= massbal_monthly.sum(axis=1)/(massbal_monthly.shape[1]/12)
# total mass balance of each glacier averaged over years of observation (mwea)
massbal_averaged=massbal_total/len(year)


#land definition for plotting 
land_50m = car.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='none')
# lat/long definition for plot
east = int(round(lons.min())) - 1
west = int(round(lons.max())) + 1
south = int(round(lats.min())) - 1
north = int(round(lats.max())) + 1
xtick = 1
ytick = 1

# define title 
title=('R'+str(regionO1_number) + ' ' + str(input.startyear) +'-' + str(input.endyear))

proj_crs = ccrs.PlateCarree()
projection = ccrs.RotatedPole(pole_longitude=40, pole_latitude=37.5)
geo_axes = plt.axes(projection=projection)

# total mass balance map 
plt.figure(1)
plt.figure(figsize=(10,10))
ax = plt.axes(projection=projection)
ax.add_feature(land_50m)
scat=ax.scatter(lons, lats, c=massbal_total,transform=proj_crs, cmap='seismic_r', vmin=-4, vmax=4, edgecolors='black')
cbar=plt.colorbar(scat, fraction=0.02, pad=0.04)
plt.axes(projection=projection)
cbar.set_label('Mass Balance mwe')
plt.title('Total Mass Balance '+ title)



# annual mass balance average map 
plt.figure(2)
plt.figure(figsize=(10,10))
ax = plt.axes(projection=projection)
ax.add_feature(land_50m)
scat=ax.scatter(lons, lats, c=massbal_averaged,transform=proj_crs, cmap='seismic_r', vmin=-0.15, vmax=0.15) #edgecoors='black' ??? 
cbar=plt.colorbar(scat, fraction=0.02, pad=0.04)
plt.axes(projection=projection)
cbar.set_label('Mass Balance mwea')
plt.title('Average Mass Balance '+ title)



#should i make plots of JUST positive and JUST negative? would need edge colors then 



#%% LINEAR REGRESSION/SLOPE AND YEARLY PLOTS

# annual mass balance for each glacier for period of observation 
massbal_annual=(np.sum(massbal_monthly.reshape(-1,12),axis=1)).reshape(len(massbal_total),len(year))
# annual mass balance for entire region 
massbal_annual_total=np.sum(massbal_annual, axis=0)

# linear regression trends for entire period 

linreg_info=[('RGIId','slope','incercept','r_val','p_val','std_err')]
slopes=[]


# removes slopes that have sig below 95%
for i in range(0,len(massbal_total)):
    slope, intercept, r_value, p_value, std_err = linregress(year, massbal_annual[i])
    RGI=glacier_data[0][i]
    if p_value > 0.05:
        slope=float('nan')
    if glacier_data[13][i] != 0:
        appends=(RGI, 'nan','nan','nan','nan','nan')
    else:
        appends=(RGI, slope, intercept,r_value, p_value, std_err) 
    linreg_info.append(appends)
    slopes.append(slope)


# plot glacier wide mass balance over time period 
plt.figure(3)
plt.plot(year,massbal_annual_total)
plt.xlabel('Year')
plt.ylabel('Mass Balance mwe')
plt.title('Region Total Mass Balance' + title)


# plot glacier slopes map for entire period 
plt.figure(4)
plt.figure(figsize=(10,10))
ax = plt.axes(projection=projection)
ax.add_feature(land_50m)
ax.scatter(lons,lats,c=[0.8,0.8,0.8], transform=proj_crs) #,edgecolors='black')
scat=ax.scatter(lons, lats, c=slopes,transform=proj_crs, cmap='seismic_r', vmin=-0.2,vmax=0.2)#, edgecolors='black') 
cbar=plt.colorbar(scat, fraction=0.02, pad=0.04)
plt.axes(projection=projection)
cbar.set_label('Mass Balance mwea')
plt.title('Rate of Mass Balance Change '+ title)


#%% TEMP & PREC FOR THE REGION, ELEVATION ADJUSTED AND NOT

#convert to float
glac_temp=glac_temp1.iloc[:,1:].astype(float).values
glac_prec=glac_prec1.iloc[:,1:].astype(float).values

#get average temp/prec for region 
annual_temp=(np.sum(glac_temp.reshape(-1,12),axis=1).reshape(len(lats),len(year)))/12
annual_prec=np.sum(glac_prec.reshape(-1,12),axis=1).reshape(len(lats),len(year)) #total prec 
average_temp=(annual_temp.sum(axis=0))/(len(lats))
average_prec=(annual_prec.sum(axis=0))/(len(lats))
# average temp/prec adjusted for glacier elevation 
glac_annual_temp=(np.sum(temp.reshape(-1,12),axis=1).reshape(len(lats),len(year)))/12
glac_annual_prec=np.sum(prec.reshape(-1,12),axis=1).reshape(len(lats),len(year)) #total prec
glac_average_temp=(glac_annual_temp.sum(axis=0))/(len(lats))
glac_average_prec=(glac_annual_prec.sum(axis=0))/(len(lats))

# check offset/diference 
temp_diff=average_temp-glac_average_temp
ppt_diff=average_prec-glac_average_prec
# check if offset is apropriate (approx 3.5 degC w/ 1000m and 24.5% ppt decrease) 
elev_ave=np.sum(glac_elev)/len(lats)

temp_offset=(elev_ave/1000)*3.5
ppt_offset=(((elev_ave/1000)*24.5)/100)*(np.sum(average_prec, axis=0)/len(average_prec))

plt.figure(5)
plt.plot(year,average_temp)
plt.plot(year, glac_average_temp)
plt.plot(year, temp_diff, color=[0.5,0.5,0.5])
plt.xlabel('Year')
plt.ylabel('Tempearture (C)')
plt.title('Region-Average Temperature ' + title)
plt.legend(['Not Adjusted','Adjusted', 'Difference'])

plt.figure(6)
plt.plot(year,average_prec)
plt.plot(year, glac_average_prec)
plt.plot(year, ppt_diff, color=[0.5,0.5,0.5])
plt.xlabel('Year')
plt.ylabel('Precipitation (m)')
plt.title('Region-Average Annual Precipitation ' + title)
plt.legend(['Not Adjusted','Adjusted','Difference'])

print('expected temp offset =' + str(temp_offset) + 'C')
print('expected ppt offset= ' + str(ppt_offset) + 'm')

#this will probably change with future stuff because elevation will not be constant 

#%% AREA WEIGHTED AVERAGE 

# total glacier area 
area_total=np.sum(glac_area, axis=0)
# area percentage for each glacier
area_prcnt=[x/area_total for x in glac_area]
# area weighted glacier average
glac_area_w=np.sum(massbal_total*area_prcnt, axis=0)
# not weighted glacier average 
glac_area_nw=(np.sum(massbal_total, axis=0))/len(lats)

year_weight=[]

# area weighted annual values 
for y in range(0,len(year)): 
    yeararea=pd.Series(massbal_annual[:,y]*area_prcnt)
    year_weight.append(yeararea)

df_area=(pd.DataFrame(year_weight))
areaweight_annual=df_area.T

# final area weighted annual values 
areaweight_total=np.sum(areaweight_annual, axis=0)

plt.figure(7)
plt.plot(year, areaweight_total)
plt.xlabel('Year')
plt.ylabel('Mass Balance (mwe)')
plt.title('Area Weighted Mass Balance ' + title) 


#%% VOLUME CHANGE ANALYSIS, INCLUDING AREA-WEIGHTED 

#years needed for volume
year2= range((input.startyear-2),input.endyear)

# total volume change 
volume_total=volume.sum(axis=0)
volume_perglac=volume_total/len(lats)

# volume percentage change 
volume_first=volume_total[0]
volume_prcnt=volume_total/volume_first 

# area weighted volume change 
vol_weight=[]

for y in range(0,(len(year)+1)): 
    volarea=pd.Series(volume[:,y]*area_prcnt)
    vol_weight.append(volarea)

df_vol=(pd.DataFrame(vol_weight))
volweight_annual=df_vol.T

volweight_total=np.sum(volweight_annual, axis=0)


for i in range(5,len(year2)): #this code really needs to be improved... need to incorporate another for loop 
    test=i-1
    test2=i-2
    test3=i-3
    test4=i-4
    test5=i-5
    volchange1=abs(volume_prcnt[test]-volume_prcnt[i])
    volchange2=abs(volume_prcnt[test2]-volume_prcnt[test])
    volchange3=abs(volume_prcnt[test3]-volume_prcnt[test2])
    volchange4=abs(volume_prcnt[test4]-volume_prcnt[test3])
    volchange5=abs(volume_prcnt[test5]-volume_prcnt[test4])
    if volchange1 and volchange2 and volchange3 and volchange4 and volchange5 <= 0.0001:
        print (year[i])


volume_change=[]

for x in range(0,len(year2)-1):
    yr=x+1
    change=(volume_total[yr]-volume_total[x])
    volume_change.append(change)




plt.figure(8)
plt.plot(year2,volume_total)
plt.title('Total Glacier Volume ' + title)
plt.xlabel('Year')
plt.ylabel('Volume km3')

plt.figure(9)
plt.plot(year2,volume_prcnt)
plt.title('% Glacier Volume Change ' + title)
plt.xlabel('Year')
plt.ylabel('Percent (%)')

plt.figure(10)
plt.plot(year,volume_change)
plt.title('Total Volume Change ' + title)
plt.xlabel('Year')
plt.ylabel('Volume km3')

plt.figure(11)
plt.plot(year2, volweight_total)
plt.title('Area Weighted Volume ' + title)
plt.xlabel('Year')
plt.ylabel('Volume km3')


#%% FUTURE SIMULATION SLICE ANALYSIS (total, annual average, slope and area-weighted reg.total) 
#28 year slices if 2017-2100: 2017-2044, 2045-2072, 2073-2100 (inclusive)

# create three slices 
yearslice1=range(2017,2045)
yearslice2=range(2045,2073)
yearslice3=range(2073,2101)

rangeslice1=range(0,len(yearslice1)) 
rangeslice2=range(len(yearslice1),(len(yearslice1)+len(yearslice2)))
rangeslice3=range((len(yearslice1)+len(yearslice2)),(len(yearslice1)+len(yearslice2)+ len(yearslice3)))

# annual glacier mass balance for each time slice 
massbal_annual1=massbal_annual[:,rangeslice1]
massbal_annual2=massbal_annual[:,rangeslice2]
massbal_annual3=massbal_annual[:,rangeslice3]

# total glacier mass balance for each glacier for each time slice 
massbal_total1=np.sum(massbal_annual1, axis=1)
massbal_total2=np.sum(massbal_annual2, axis=1)
massbal_total3=np.sum(massbal_annual3, axis=1)

title2= ('R'+str(regionO1_number) + ' ')

# area weighted total mass balance 
glac_area_w1=np.sum(massbal_total1*area_prcnt, axis=0)
glac_area_w2=np.sum(massbal_total2*area_prcnt, axis=0)
glac_area_w3=np.sum(massbal_total3*area_prcnt, axis=0)


# area weighted annual mass balance for each glacier

year_weight1=[]
year_weight2=[]
year_weight3=[]

# area weighted annual values 
for y in range(0,len(yearslice1)): 
    yeararea1=pd.Series(massbal_annual1[:,y]*area_prcnt)
    year_weight1.append(yeararea1)
    yeararea2=pd.Series(massbal_annual2[:,y]*area_prcnt)
    year_weight2.append(yeararea2)   
    yeararea3=pd.Series(massbal_annual3[:,y]*area_prcnt)
    year_weight3.append(yeararea3)

df_area1=(pd.DataFrame(year_weight1))
areaweight_annual1=df_area1.T
df_area2=(pd.DataFrame(year_weight2))
areaweight_annual2=df_area2.T
df_area3=(pd.DataFrame(year_weight3))
areaweight_annual3=df_area3.T

# final area weighted annual values 
areaweight_total=np.sum(areaweight_annual, axis=0)


plt.figure(12)
plt.figure(figsize=(10,10))
ax = plt.axes(projection=projection)
ax.add_feature(land_50m)
scat=ax.scatter(lons, lats, c=massbal_total1,transform=proj_crs, cmap='seismic_r', vmin=-4, vmax=4, edgecolors='black')
cbar=plt.colorbar(scat, fraction=0.02, pad=0.04)
plt.axes(projection=projection)
cbar.set_label('Mass Balance mwe')
plt.title('Total Mass Balance '+ title2 + '2017-2044')

plt.figure(13)
plt.figure(figsize=(10,10))
ax = plt.axes(projection=projection)
ax.add_feature(land_50m)
scat=ax.scatter(lons, lats, c=massbal_total2,transform=proj_crs, cmap='seismic_r', vmin=-4, vmax=4, edgecolors='black')
cbar=plt.colorbar(scat, fraction=0.02, pad=0.04)
plt.axes(projection=projection)
cbar.set_label('Mass Balance mwe')
plt.title('Total Mass Balance '+ title2 + '2045-2072')

plt.figure(14)
plt.figure(figsize=(10,10))
ax = plt.axes(projection=projection)
ax.add_feature(land_50m)
scat=ax.scatter(lons, lats, c=massbal_total3,transform=proj_crs, cmap='seismic_r', vmin=-4, vmax=4, edgecolors='black')
cbar=plt.colorbar(scat, fraction=0.02, pad=0.04)
plt.axes(projection=projection)
cbar.set_label('Mass Balance mwe')
plt.title('Total Mass Balance '+ title2 + '2073-2100')


#do total mass balance and annual average mass balance for each glacier 
# and total area weighted 

#%% OLD CODE 




#%%Linear regressions for total dataset

#want linear regressions(with scipy) and then remove slopes based on high p values 

year2=list(range(2015,2100))
massbal_annual=(np.sum(massbal_total.reshape(-1,12),axis=1)).reshape(len(lats),len(year))

linreg_info=[('RGIId','slope','incercept','r_val','p_val','std_err')]
slopes=[]


# removes slopes that have a statsig below 95%
for i in range(0,len(lats)):
    slope, intercept, r_value, p_value, std_err = linregress(year2, massbal_annual[i])
    RGI=glacier_data[0][i]
    if p_value > 0.05:
        slope=float('nan')
    if glacier_data[13][i] != 0:
        appends=(RGI, 'nan','nan','nan','nan','nan')
    else:
        appends=(RGI, slope, intercept,r_value, p_value, std_err) 
    linreg_info.append(appends)
    slopes.append(slope)


#%% slope plotting 
    
   
east = int(round(lons.min())) - 1
west = int(round(lons.max())) + 1
south = int(round(lats.min())) - 1
north = int(round(lats.max())) + 1
xtick = 1
ytick = 1

g=plt.figure(1)
# Plot regional maps
post.plot_latlonvar(lons, lats, slopes,-4.5, 1.5, 'modelled linear rates of SMB Change', 'longitude [deg]', 
               'latitude [deg]', 'jet_r', east, west, south, north, xtick, ytick)



land_50m = car.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='none')
#pp=plt.figure(1)
plt.figure(figsize=(10,10))
ax = plt.axes(projection=car.crs.PlateCarree())
#ax.set_global()
#ax.coastlines()
ax.add_feature(land_50m)
scat1=ax.scatter(lons,lats, c='none', edgecolors='black')
scat=ax.scatter(lons, lats, c=slopes, cmap='winter_r', edgecolors='black')
#scat.set_clim(-2.5,2.5)
cbar=plt.colorbar(scat, fraction=0.02, pad=0.04)
cbar.set_label('Rate of Chagne mwea')
plt.title('Mass Change/Yr, 1985-2015')

# NEED TO CONSIDER WHAT COLORPLOTS TO USE; DO I WANT TO DISTINGUISH BETWEEN LOW CHANGE AND NAN? 

#%%

east = int(round(lons.min())) - 1
west = int(round(lons.max())) + 1
south = int(round(lats.min())) - 1
north = int(round(lats.max())) + 1
xtick = 1
ytick = 1

g=plt.figure(1)
# Plot regional maps


land_50m = car.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='none')
#pp=plt.figure(1)
plt.figure(figsize=(10,10))
ax = plt.axes(projection=car.crs.PlateCarree())
#ax.set_global()
#ax.coastlines()
ax.add_feature(land_50m)
scat1=ax.scatter(lons,lats, c='none', edgecolors='black')
scat=ax.scatter(lons, lats, c=massbal_total_mwea, cmap='jet_r', edgecolors='black')
#scat.set_clim(-2.5,2.5)
cbar=plt.colorbar(scat, fraction=0.02, pad=0.04)
plt.title('Total Mass Balance Change 1985-2015')
cbar.set_label('Mass change mwe')



#%% Analysis of volume data 




volume_total=volume.sum(axis=0)
volume_first=volume_total[0]
volume_prcnt=volume_total/volume_first 

plt.plot(year,volume_total)
plt.title('Total Volume 2016-2100')
plt.xlabel('Year')
plt.ylabel('Volume m3')


plt.plot(year,volume_prcnt)
plt.title('Volume % Change 2016-2100')
plt.xlabel('Year')
plt.ylabel('%')


for i in range(5,85): #this code really needs to be improved... need to incorporate another for loop 
    test=i-1
    test2=i-2
    test3=i-3
    test4=i-4
    test5=i-5
    volchange1=abs(volume_prcnt[test]-volume_prcnt[i])
    volchange2=abs(volume_prcnt[test2]-volume_prcnt[test])
    volchange3=abs(volume_prcnt[test3]-volume_prcnt[test2])
    volchange4=abs(volume_prcnt[test4]-volume_prcnt[test3])
    volchange5=abs(volume_prcnt[test5]-volume_prcnt[test4])
    if volchange1 and volchange2 and volchange3 and volchange4 and volchange5 <= 0.0001:
        print (year[i])


volume_change=[]

for x in range(0,85):
    yr=x+1
    change=(volume_total[yr]-volume_total[x])
    volume_change.append(change)


plt.plot(year2,volume_change)
plt.title('Total Volume Change 2016-2100')
plt.xlabel('Year')
plt.ylabel('Volume m3')

#%%Analysis of climate data 

temp_annual=((np.sum(temp.reshape(-1,12),axis=1)).reshape(568,85))/12
temp_ave=((temp_annual.sum(axis=0)))/568

plt.plot(year2,temp_ave)
plt.title('Era-Int Temp Simulation 2016-2100 [Years:1990-2000]')
plt.xlabel('Year')
plt.ylabel('T(degC)')

prec_annual=((np.sum(prec.reshape(-1,12),axis=1)).reshape(568,85))/12
prec_ave=((prec_annual.sum(axis=0)))/568

plt.plot(year2, prec_ave)
plt.title('Era-Int Prec Simulation 2016-2100 [Years:1990-2000]')
plt.xlabel('Year')
plt.ylabel('Prec(mm)')

#%% For Plotting 

# Set extent
east = int(round(lons.min())) - 1
west = int(round(lons.max())) + 1
south = int(round(lats.min())) - 1
north = int(round(lats.max())) + 1
xtick = 1
ytick = 1

g=plt.figure(1)
# Plot regional maps
post.plot_latlonvar(lons, lats, massbal_total_mwea, -4.5, 1.5, 'Modeled mass balance [mwea]', 'longitude [deg]', 
               'latitude [deg]', 'jet_r', east, west, south, north, xtick, ytick)
#plt.savefig(input.output_filepath+'/../../CH_1/iceland_2', dpi=100)
