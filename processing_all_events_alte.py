import xarray as xr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utide import solve, reconstruct
import xarray.plot as xplt  # Import xarray.plot module
import mat73
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Circle




# Define the RGB values for your custom color palette
colors_rgb = [
    (255, 255, 255), (214, 226, 255), (181, 201, 255), (142, 178, 255),
    (127, 150, 255), (99, 112, 247), (50, 141, 208), (0, 170, 170), 
    (0, 150, 150), (0, 198, 51), (99, 255, 0), (150, 255, 0), 
    (198, 255, 51), (255, 255, 0), (255, 198, 0), (255, 160, 0), 
    (255, 124, 0), (255, 102, 0), (255, 25, 0)
]

# Convert RGB tuples to normalized RGB values (values between 0 and 1)
colors_normalized = [(r / 255., g / 255., b / 255.) for (r, g, b) in colors_rgb]

# Create a colormap using the normalized RGB values
cmap = mcolors.ListedColormap(colors_normalized)

def read_first_available(dataset, var_names):
    """
    Try to read the first available variable from a list of possible variable names in the dataset.

    Parameters:
    dataset (xarray.Dataset): The dataset to read from.
    var_names (list of str): A list of possible variable names.

    Returns:
    numpy.array: The variable data if found, otherwise raises a ValueError.
    """
    for var in var_names:
        if var in dataset.columns:
            return dataset[var].values[0]
    raise ValueError("None of the specified variable names found in the dataset.")

# Define the haversine function
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1_tmp = np.deg2rad(lon1)
    lon2_tmp = np.deg2rad(lon2)
    lat1_tmp = np.deg2rad(lat1)
    lat2_tmp = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2_tmp - lon1_tmp
    dlat = lat2_tmp - lat1_tmp
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

TG_path='/home/amores/SWOT/A_data/A_TGs/'
SWOT_path = '/home/dvega/anaconda3/work/SWOT_STORM/'
era5_path = '/home/dvega/anaconda3/work/SWOT_STORM/ERA5/'
model_path = '/home/amores/SWOT/A_data/C_modelo/'


# PROCCESSING SET DATA -----------------------------------------------------------------------------------------------

# ------------------------- TIDE GAUGE -----------------------------------------

# Processing tide gauge
kiel_tg = xr.open_dataset(f'{TG_path}NO_TS_TG_AlteWeserTG.nc')

kiel_tg = kiel_tg[['TIME', 'SLEV', 'LATITUDE', 'LONGITUDE']]

kiel_tg = kiel_tg.to_dataframe().reset_index(level='DEPTH', drop=True)

TG_data = [kiel_tg]


lonNames=['lon','longitude','LONGITUDE']
latNames=['lat','latitude','LATITUDE']

latTGs = []
lonTGs = []

for i in range(len(TG_data)):
    tg = TG_data[i]
    # Read the first available longitude and latitude variables
    lonTG = read_first_available(tg, lonNames)
    latTG = read_first_available(tg, latNames)
    latTGs.append(np.unique(latTG).astype(float))
    lonTGs.append(np.unique(lonTG).astype(float))
    print(latTG,lonTG)

df_tg_k = kiel_tg.reset_index()
df_tg_k = df_tg_k[['TIME', 'SLEV']]
df_tg_k['TIME'] = pd.to_datetime(df_tg_k['TIME'])
df_tg_k.set_index('TIME', inplace=True)
# df_tg_k = df_tg_k[df_tg_k.index > pd.to_datetime('2020-01-01')]

df_tg_k['SLEV_demean'] = df_tg_k['SLEV'] - df_tg_k['SLEV'].mean()



# Detide
coef_k = solve(df_tg_k.index, df_tg_k['SLEV_demean'],
             lat=latTGs[0],
             nodal=False,
             trend=False,
             method='ols',
             conf_int='linear',
             verbose=False,)


tide_k = reconstruct(df_tg_k.index, coef_k, verbose=False)

df_tg_k['detided'] = df_tg_k['SLEV_demean']-tide_k.h

# CALCULATE EXTREME EVENTS PERCENTILES 95 AND 5
tg_perc_95_k = np.percentile(df_tg_k['detided'], 95)
tg_perc_5_k = np.percentile(df_tg_k['detided'], 5)


# ------------------------- SWOT time series 50km -----------------------------------------

df = pd.read_csv(f'{SWOT_path}df_Kiel_Alteweser_SWOT_series_50km.csv')

df_kiel= df[df['latitude'] == latTGs[0][0]]
df_kiel['ssha_demean'] = df_kiel['ssha_dac']-df_kiel['ssha_dac'].mean()



#  --------------------- READ SCHISM MODEL DATA ------------------------------------
# --------- Define triangulation data
mat_data = mat73.loadmat(f'{model_path}tri_simu.mat')
tri = mat_data['tri'].astype(int) - 1

model23 = xr.open_dataset(f'{model_path}merged_elevation_2023.nc')
model24 = xr.open_dataset(f'{model_path}merged_elevation_2024.nc')

model = xr.concat([model23, model24], dim='time')
model_df = model.to_dataframe().reset_index()


# Filter data for the specific region
model_df_cut_kiel = model_df[(model_df['SCHISM_hgrid_node_y'] > 53.5) & (model_df['SCHISM_hgrid_node_y'] < 55) &
                    (model_df['SCHISM_hgrid_node_x'] > 7) & (model_df['SCHISM_hgrid_node_x'] < 9.5)]


# OBTAING THE TIME SERIES OF THE SCHISM MODEL DATA ------------------------------------

# Calculate the distance from the target location to all points in the dataframe
model_df_cut_kiel['distance'] = model_df_cut_kiel.apply(lambda row: haversine(lonTGs[0], latTGs[0], row['SCHISM_hgrid_node_x'], row['SCHISM_hgrid_node_y']), axis=1)

# Filter the points within 20 km radius
radius = 50  # km
df_filtered_kiel = model_df_cut_kiel[model_df_cut_kiel['distance'] <= radius]

# Group by time and calculate the average sea level anomaly
ts_kiel_SC = df_filtered_kiel.groupby('time')['elevation'].mean().reset_index()

# Detide time series of SCHISM

# Detide
coef_k_SC = solve(ts_kiel_SC['time'], ts_kiel_SC['elevation'],
             lat=latTGs[0],
             nodal=False,
             trend=False,
             method='ols',
             conf_int='linear',
             verbose=False,)


tide_k_SC = reconstruct(ts_kiel_SC['time'], coef_k_SC, verbose=False)

ts_kiel_SC['detided'] = ts_kiel_SC['elevation'] - tide_k_SC.h


# PROCESS TIME SERIES OF SWOT, TIDE GAUGE AND SCHISM MODEL DATA ------------------------------------

# Convert time to datetime
df_kiel['time'] = pd.to_datetime(df_kiel['time'])

# Set same period for all TSs
time_margin = pd.Timedelta(days=7)  # 7 days margin

# Calculate initial and final times for Kiel
initial_time_kiel = np.max([np.min(df_tg_k.index), np.min(ts_kiel_SC['time']), np.min(df_kiel['time'])]) - time_margin
final_time_kiel = np.min([np.max(df_tg_k.index), np.max(ts_kiel_SC['time']), np.max(df_kiel['time'])]) + time_margin

# Set the same period for all time series with the margin
df_tg_k = df_tg_k[(df_tg_k.index >= initial_time_kiel) & (df_tg_k.index <= final_time_kiel)]

ts_kiel_SC = ts_kiel_SC[(ts_kiel_SC['time'] >= initial_time_kiel) & (ts_kiel_SC['time'] <= final_time_kiel)]

df_kiel = df_kiel[(df_kiel['time'] >= initial_time_kiel) & (df_kiel['time'] <= final_time_kiel)]

# SELECT TARGET DATE POINTS OF SWOT

df_kiel_max = df_kiel.sort_values(by='ssha_demean', ascending=False)[:5]

df_kiel_min = df_kiel.sort_values(by='ssha_demean', ascending=True)[:5]

# Set the time columns to hourly values for calculating the RMSD
df_kiel['time'] = df_kiel['time'].dt.round('H')

# set time index for calculating rmsd
df_kiel_1 = df_kiel.set_index('time')

ts_kiel_SC_1 = ts_kiel_SC.set_index('time').rename(columns={'detided': 'detided_SC'})

# Convert to same units
ts_kiel_SC_1['detided_SC'] = ts_kiel_SC_1['detided_SC']*100

df_tg_k_1 = df_tg_k['detided']*100

merged_df_k = pd.concat([df_kiel_1, ts_kiel_SC_1, df_tg_k_1], axis=1, join='inner')

merged_df_k = merged_df_k[['ssha_demean', 'detided_SC', 'detided']]

# Define a function to calculate RMSD between two columns
def calculate_rmsd(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

# For merged_df_k (You can repeat this for merged_df_a)
rmsd_ssha_detided_SC_k = calculate_rmsd(merged_df_k['ssha_demean'], merged_df_k['detided_SC'])
rmsd_ssha_detided_k = calculate_rmsd(merged_df_k['ssha_demean'], merged_df_k['detided'])
rmsd_detided_SC_detided_k = calculate_rmsd(merged_df_k['detided_SC'], merged_df_k['detided'])



# EVENT Nº1 -----------------------------------------------------------------------------------------------

# ---------------------- ERA5 REANALYSIS MODEL -------------------------------------

# ds_era5 = xr.open_dataset(f'{era5_path}msl_wind_alte_191223_231223.nc') # 
ds_era5 = xr.open_dataset(f'{era5_path}msl_wind_alte_131023_221023.nc') # Consecutive events

msl1 = ds_era5['msl']
u101 = ds_era5['u10']
v101 = ds_era5['v10']

# Select speciphic date
# time_swot1 = '2023-12-21 16:00:00' # Most extreme event
time_swot1 = '2023-10-14 17:00:00'


time_short1 = 20231221

u101 = u101.sel(valid_time = time_swot1)
v101 = v101.sel(valid_time = time_swot1)
msl1 = msl1.sel(valid_time = time_swot1)

# Calculate wind speed and direction
wind_speed1 = np.sqrt(u101**2 + v101**2)
wind_dir1 = np.arctan2(v101, u101) * (180 / np.pi)

msl_hpa1 = msl1 / 100.0


# ---------------------- SCHISM -------------------------------------

model_df_kiel_time1 = model_df[model_df['time'] == pd.to_datetime(time_swot1)] # Most extreme event

# Extract data for create a map of speciphic date
latSC1 = model_df_kiel_time1['SCHISM_hgrid_node_y'].values
lonSC1 = model_df_kiel_time1['SCHISM_hgrid_node_x'].values
sshSC1 = model_df_kiel_time1['elevation'].values
time1 = model_df_kiel_time1['time'].values

# ------------------------READ SWOT DATA ---------------------------------------------

ds_swot_kiel1 = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_008_208_20231221T161853_20231221T171019_v1.0.nc') # Most extreme event

ssha_dac_kiel1 = ds_swot_kiel1.ssha+ds_swot_kiel1.dac


# EVENT Nº2 -----------------------------------------------------------------------------------------------

# ---------------------- ERA5 REANALYSIS MODEL -------------------------------------

# ds_era5 = xr.open_dataset(f'{era5_path}msl_wind_alte_121023_161023.nc') 

msl2 = ds_era5['msl']
u102 = ds_era5['u10']
v102 = ds_era5['v10']

# Select speciphic date
# time_swot2 = '2023-10-14 17:00:00'
time_swot2 = '2023-10-19 02:00:00' 

time_short2 = 20230505

u102 = u102.sel(valid_time = time_swot2)
v102 = v102.sel(valid_time = time_swot2)
msl2 = msl2.sel(valid_time = time_swot2)

# Calculate wind speed and direction
wind_speed2 = np.sqrt(u102**2 + v102**2)
wind_dir2 = np.arctan2(v102, u102) * (180 / np.pi)

msl_hpa2 = msl2 / 100.0


# ---------------------- SCHISM -------------------------------------

model_df_kiel_time = model_df[model_df['time'] == pd.to_datetime(time_swot2)] 

# Extract data for create a map of speciphic date
latSC2 = model_df_kiel_time['SCHISM_hgrid_node_y'].values
lonSC2 = model_df_kiel_time['SCHISM_hgrid_node_x'].values
sshSC2 = model_df_kiel_time['elevation'].values
time2 = model_df_kiel_time['time'].values


# ------------------------READ SWOT DATA ---------------------------------------------

ds_swot_kiel2 = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_005_057_20231014T163508_20231014T172634_v1.0.nc') 
ds_swot_kiel2 = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_005_180_20231019T020307_20231019T025433_v1.0.nc') 
ssha_dac_kiel2 = ds_swot_kiel2.ssha+ds_swot_kiel2.dac


# EVENT Nº3 -----------------------------------------------------------------------------------------------

# ---------------------- ERA5 REANALYSIS MODEL -------------------------------------

# ds_era5 = xr.open_dataset(f'{era5_path}msl_wind_alte_010224_030224.nc') 

msl3 = ds_era5['msl']
u103 = ds_era5['u10']
v103 = ds_era5['v10']

# Select speciphic date
# time_swot3 = '2024-02-01 10:00:00' 
time_swot3 = '2023-10-20 02:00:00' 

time_short3 = 20240201

u103 = u103.sel(valid_time = time_swot3)
v103 = v103.sel(valid_time = time_swot3)
msl3 = msl3.sel(valid_time = time_swot3)

# Calculate wind speed and direction
wind_speed3 = np.sqrt(u103**2 + v103**2)
wind_dir3 = np.arctan2(v103, u103) * (180 / np.pi)

msl_hpa3 = msl3 / 100.0


# ---------------------- SCHISM -------------------------------------

model_df_kiel_time = model_df[model_df['time'] == pd.to_datetime(time_swot3)] 

# Extract data for create a map of speciphic date
latSC3 = model_df_kiel_time['SCHISM_hgrid_node_y'].values
lonSC3 = model_df_kiel_time['SCHISM_hgrid_node_x'].values
sshSC3 = model_df_kiel_time['elevation'].values
time3 = model_df_kiel_time['time'].values


# ------------------------READ SWOT DATA ---------------------------------------------

ds_swot_kiel3 = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_010_208_20240201T094904_20240201T104031_v1.0.nc') 
ds_swot_kiel3 = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_005_208_20231020T020338_20231020T025504_v1.0.nc') 

ssha_dac_kiel3 = ds_swot_kiel3.ssha+ds_swot_kiel3.dac


# PLOT FIGURES -----------------------------------------------------------------------------------------------

# Updated bounding boxes
lolabox_era5 = [9, 22, 53.5, 63]
lolabox = [9.5, 12.5, 53.5, 58]
lolabox = [6, 9, 53.5, 58] # Alte Weser

# Define the radius in kilometers for the circle around the tide gauge
radius_km = 50

# Create a circle patch with the radius in degrees (convert kilometers to degrees)
radius_deg = radius_km / 111  # Rough conversion factor: 1 degree ≈ 111 km

# Define the location of the tide gauge
center_lat = df_kiel['latitude'].iloc[0]
center_lon = df_kiel['longitude'].iloc[0]

# Add the circle to the plot
def add_circle(ax, center_lon, center_lat, radius, **kwargs):
    circle = Circle((center_lon, center_lat), radius, transform=ccrs.PlateCarree(), **kwargs)
    ax.add_patch(circle)



fig = plt.figure(figsize=(18, 32.5))  # Increased the height for better clarity
gs = plt.GridSpec(5, 3, height_ratios=[0.4, 0.4, 1, 1, 1])  # 5 rows, adjusted height ratios for better balance

# Time series plot (occupies the entire first row)
ax0 = fig.add_subplot(gs[0, :])  # Spans all columns in the first row
ax0.plot(df_tg_k.index, df_tg_k['detided']*100, zorder=0, c='black', linewidth=0.8, label=r'Tide Gauge')  # TG time series
ax0.plot(ts_kiel_SC['time'], ts_kiel_SC['detided']*100, zorder=1, c='blue', linewidth=0.6, label=r'SCHISM model')  # TG time series
ax0.scatter(df_kiel['time'], df_kiel['ssha_demean'], c='r', zorder=2, s=30, label=r'SWOT Data')
ax0.scatter(df_kiel_max['time'].iloc[1], df_kiel_max['ssha_demean'].iloc[1], marker='*', c='green', zorder=3, s=380, label=time_swot3)
ax0.scatter(df_kiel_min['time'].iloc[0], df_kiel_min['ssha_demean'].iloc[0], marker='s', c='green', zorder=3, s=200, label=time_swot2)
ax0.scatter(df_kiel_min['time'].iloc[1], df_kiel_min['ssha_demean'].iloc[1], marker='^', c='green', zorder=3, s=300, label=time_swot1)
textstr1 = f'RMSD SWOT vs TG: {rmsd_ssha_detided_k:.2f} cm'
ax0.text(0.8, 0.935, textstr1, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
textstr2 = f'RMSD SCHISM vs TG: {rmsd_detided_SC_detided_k:.2f} cm'
ax0.text(0.8, 0.86, textstr2, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
ax0.set_ylabel('Storm surge height (cm)', fontsize=18)
ax0.grid(True, alpha=0.3)
ax0.legend(fontsize=12, loc='lower right', ncol=6)
ax0.tick_params(axis='both', which='major', labelsize=18)  # For both x and y axes, set major tick label size

ax0.set_title('ALTE WESER TG', fontsize=18)
# Adding vertical dashed lines at specific dates
ax0.axvline(x=pd.to_datetime(time_swot1), color='purple', linestyle='--', linewidth=1, label='Event 1')
ax0.axvline(x=pd.to_datetime(time_swot2), color='orange', linestyle='--', linewidth=1, label='Event 2')
ax0.axvline(x=pd.to_datetime(time_swot3), color='cyan', linestyle='--', linewidth=1, label='Event 3')

# Fifth row (New row - can be used for any additional information or plot)
ax_middle = fig.add_subplot(gs[1, :])  # Spans all columns in the second row (now row 1)
ax_middle.axis('off')  # Hide axes if this row is just for display or textual content

# EVENTO 1 --------------------------------------------------------------------------------

# Map 1: SWOT (occupies the first column in the third row, was second row)
ax1 = fig.add_subplot(gs[2, 2], projection=ccrs.PlateCarree())
ax1.set_extent(lolabox, crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.LAND)
ax1.add_feature(cfeature.COASTLINE)
ssha_plot = ax1.pcolormesh(ssha_dac_kiel1.longitude, ssha_dac_kiel1.latitude, ssha_dac_kiel1.values, vmin=-1.2, vmax=1.5, cmap=cmap, transform=ccrs.PlateCarree())
ax1.set_title('SWOT SLA', fontsize=18)
gl1 = ax1.gridlines(draw_labels=False, alpha=0.5)
gl1.top_labels = False
gl1.right_labels = False
ax1.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)
add_circle(ax1, center_lon, center_lat, radius=radius_deg, color='green', fill=False) # Add the circle to the plot

# Map 2: SCHISM (occupies the second column in the third row)
ax2 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
ax2.set_extent(lolabox, crs=ccrs.PlateCarree())
triang = Triangulation(lonSC1, latSC1, tri)
ax2.add_feature(cfeature.LAND)
ax2.add_feature(cfeature.COASTLINE)
collection = ax2.tripcolor(triang, sshSC1, shading='flat', cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree())
ax2.set_title('SCHISM SLA', fontsize=18)
gl2 = ax2.gridlines(draw_labels=False, alpha=0.5)
gl2.top_labels = False
gl2.right_labels = False
ax2.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)
add_circle(ax2, center_lon, center_lat, radius=radius_deg, color='green', fill=False) # Add the circle to the plot

# Map 3: ERA5 (occupies the third column in the third row)
ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
ax3.set_extent(lolabox, crs=ccrs.PlateCarree())
ax3.add_feature(cfeature.LAND)
ax3.add_feature(cfeature.COASTLINE)
min_value3 = np.floor(msl_hpa1.min().item())
max_value3 = np.ceil(msl_hpa1.max().item())
levels3 = np.arange(min_value3, max_value3 + 1, 2)

# Plot mean sea level pressure as contours
msl_contour3 = ax3.contour(msl1.longitude, msl1.latitude, msl_hpa1, levels=levels3, colors='#4d4d4d', linewidths=1, transform=ccrs.PlateCarree())
ax3.clabel(msl_contour3, inline=True, inline_spacing=2, fontsize=20, fmt='%1.0f', rightside_up=True)

# Set the colorbar range from 0 to 20 m/s for wind speed
vmin, vmax = 0, 20
wind_speed_levels1 = np.linspace(vmin, vmax, 21)
wind_speed_plot1 = ax3.contourf(u101.longitude, u101.latitude, wind_speed1, levels=wind_speed_levels1, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

# Add features and wind vectors
stride = 1  # Adjust this value to reduce the number of arrows
u10_reduced1 = u101[::stride, ::stride]
v10_reduced1 = v101[::stride, ::stride]
lon_reduced1 = u101.longitude[::stride]
lat_reduced1 = u101.latitude[::stride]

ax3.quiver(lon_reduced1, lat_reduced1, u10_reduced1, v10_reduced1, scale=250, width=0.005, transform=ccrs.PlateCarree())
gl3 = ax3.gridlines(draw_labels=True, alpha=0.5)
gl3.top_labels = False
gl3.right_labels = False
gl3.bottom_labels = False
gl3.ylabel_style = {'size': 18}  # Change the y-axis label font size

ax3.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)
ax3.set_title('ERA5 MSLP and Wind Speed', fontsize=18)


# EVENTO 2 --------------------------------------------------------------------------------

# Map 1: SWOT (occupies the first column in the second row)
ax4 = fig.add_subplot(gs[3, 2], projection=ccrs.PlateCarree())
ax4.set_extent(lolabox, crs=ccrs.PlateCarree())
ax4.add_feature(cfeature.LAND)
ax4.add_feature(cfeature.COASTLINE)
ssha_plot4 = ax4.pcolormesh(ssha_dac_kiel2.longitude, ssha_dac_kiel2.latitude, ssha_dac_kiel2.values, vmin=-1.2, vmax=1.5, cmap=cmap, transform=ccrs.PlateCarree())
gl4 = ax4.gridlines(draw_labels=False, alpha=0.5)
gl4.top_labels = False
gl4.right_labels = False
ax4.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)
add_circle(ax4, center_lon, center_lat, radius=radius_deg, color='green', fill=False) # Add the circle to the plot

# Map 2: SCHISM (occupies the second column in the second row)
ax5 = fig.add_subplot(gs[3, 1], projection=ccrs.PlateCarree())
ax5.set_extent(lolabox, crs=ccrs.PlateCarree())
triang = Triangulation(lonSC2, latSC2, tri)
ax5.add_feature(cfeature.LAND)
ax5.add_feature(cfeature.COASTLINE)
collection5 = ax5.tripcolor(triang, sshSC2, shading='flat', cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree())
gl5 = ax5.gridlines(draw_labels=False, alpha=0.5)
gl5.top_labels = False
gl5.right_labels = False
ax5.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)
add_circle(ax5, center_lon, center_lat, radius=radius_deg, color='green', fill=False) # Add the circle to the plot

# Map 3: ERA5 (occupies the third column in the second row)
ax6 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
ax6.set_extent(lolabox, crs=ccrs.PlateCarree())
ax6.add_feature(cfeature.LAND)
ax6.add_feature(cfeature.COASTLINE)
min_value6 = np.floor(msl_hpa2.min().item())
max_value6 = np.ceil(msl_hpa2.max().item())
levels6 = np.arange(min_value6, max_value6 + 1, 2)

# Plot mean sea level pressure as contours
msl_contour6 = ax6.contour(msl2.longitude, msl2.latitude, msl_hpa2, levels=levels6, colors='#4d4d4d', linewidths=1, transform=ccrs.PlateCarree())
ax6.clabel(msl_contour6, inline=True, inline_spacing=2, fontsize=20, fmt='%1.0f', rightside_up=True)

# Set the colorbar range from 0 to 20 m/s for wind speed
vmin, vmax = 0, 20
wind_speed_levels = np.linspace(vmin, vmax, 21)
wind_speed_plot6 = ax6.contourf(u102.longitude, u102.latitude, wind_speed2, levels=wind_speed_levels, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)


# Add features and wind vectors
stride = 1  # Adjust this value to reduce the number of arrows
u10_reduced2 = u102[::stride, ::stride]
v10_reduced2 = v102[::stride, ::stride]
lon_reduced2 = u102.longitude[::stride]
lat_reduced2 = u102.latitude[::stride]

ax6.quiver(lon_reduced2, lat_reduced2, u10_reduced2, v10_reduced2, scale=250, width=0.005, transform=ccrs.PlateCarree())
gl6 = ax6.gridlines(draw_labels=True, alpha=0.5)
gl6.top_labels = False
gl6.right_labels = False
gl6.bottom_labels = False
gl6.ylabel_style = {'size': 18}  # Change the y-axis label font size


ax6.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)


# EVENTO 3 --------------------------------------------------------------------------------

# Map 1: SWOT (occupies the first column in the second row)
ax7 = fig.add_subplot(gs[4, 2], projection=ccrs.PlateCarree())
ax7.set_extent(lolabox, crs=ccrs.PlateCarree())
ax7.add_feature(cfeature.LAND)
ax7.add_feature(cfeature.COASTLINE)
ssha_plot7 = ax7.pcolormesh(ssha_dac_kiel3.longitude, ssha_dac_kiel3.latitude, ssha_dac_kiel3.values, vmin=-1.5, vmax=1.5, cmap=cmap, transform=ccrs.PlateCarree())
# cbar7 = plt.colorbar(ssha_plot, ax=ax7, orientation='horizontal', pad=0.06, aspect=30)
gl7 = ax7.gridlines(draw_labels=True, alpha=0.5)
gl7.top_labels = False
gl7.right_labels = False
gl7.left_labels=False
gl7.xlabel_style = {'size': 18}  # Change the x-axis label font size
ax7.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)
add_circle(ax7, center_lon, center_lat, radius=radius_deg, color='green', fill=False) # Add the circle to the plot


# Map 2: SCHISM (occupies the second column in the second row)
ax8 = fig.add_subplot(gs[4, 1], projection=ccrs.PlateCarree())
ax8.set_extent(lolabox, crs=ccrs.PlateCarree())
triang = Triangulation(lonSC3, latSC3, tri)
ax8.add_feature(cfeature.LAND)
ax8.add_feature(cfeature.COASTLINE)
collection8 = ax8.tripcolor(triang, sshSC3, shading='flat', cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree())
# cbar8 = plt.colorbar(collection8, ax=ax8, orientation='horizontal', pad=0.02)
gl8 = ax8.gridlines(draw_labels=True, alpha=0.5)
gl8.top_labels = False
gl8.right_labels = False
gl8.left_labels=False
gl8.xlabel_style = {'size': 18}  # Change the x-axis label font size
ax8.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)
add_circle(ax8, center_lon, center_lat, radius=radius_deg, color='green', fill=False) # Add the circle to the plot


# Map 3: ERA5 (occupies the third column in the second row)
ax9 = fig.add_subplot(gs[4, 0], projection=ccrs.PlateCarree())
ax9.set_extent(lolabox, crs=ccrs.PlateCarree())
ax9.add_feature(cfeature.LAND)
ax9.add_feature(cfeature.COASTLINE)
min_value9 = np.floor(msl_hpa3.min().item())
max_value9 = np.ceil(msl_hpa3.max().item())
levels9 = np.arange(min_value9, max_value9 + 1, 2)

# Plot mean sea level pressure as contours
msl_contour = ax9.contour(msl3.longitude, msl3.latitude, msl_hpa3, levels=levels9, colors='#4d4d4d', linewidths=1, transform=ccrs.PlateCarree())
ax9.clabel(msl_contour, inline=True, inline_spacing=2, fontsize=18, fmt='%1.0f', rightside_up=True)

# Set the colorbar range from 0 to 20 m/s for wind speed
vmin, vmax = 0, 20
wind_speed_levels9 = np.linspace(vmin, vmax, 21)
wind_speed_plot9 = ax9.contourf(u103.longitude, u103.latitude, wind_speed3, levels=wind_speed_levels, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

# Add features and wind vectors
stride = 1  # Adjust this value to reduce the number of arrows
u10_reduced3 = u103[::stride, ::stride]
v10_reduced3 = v103[::stride, ::stride]
lon_reduced3 = u103.longitude[::stride]
lat_reduced3 = u103.latitude[::stride]

ax9.quiver(lon_reduced3, lat_reduced3, u10_reduced3, v10_reduced3, scale=250, width=0.005, transform=ccrs.PlateCarree())
gl9 = ax9.gridlines(draw_labels=True, alpha=0.5)
gl9.top_labels = False
gl9.right_labels = False
gl9.ylabel_style = {'size': 18}  # Change the y-axis label font size
gl9.xlabel_style = {'size': 18}  # Change the x-axis label font size

ax9.scatter(df_kiel['longitude'].iloc[0], df_kiel['latitude'].iloc[0], c='black', s=200, zorder=3)

# Shared colorbar for SWOT and SCHISM (spanning columns 2 and 3)
cax9 = fig.add_axes([0.39, 0.05, 0.56, 0.015])  # Adjusted for third event row
cb9 = fig.colorbar(ssha_plot7, cax=cax9, orientation='horizontal', label='SLA (m)')
cb9.ax.tick_params(labelsize=18)
cb9.set_label('SLA (m)', fontsize=18)  # Set colorbar label size to 14

#colorbar for ERA5
cax10 = fig.add_axes([0.1, 0.05, 0.27, 0.015])  # Adjusted for third event row
cb10 = fig.colorbar(wind_speed_plot9, cax=cax10, orientation='horizontal', label='Wind Speed (m/s)')
cb10.ax.tick_params(labelsize=18)
cb10.set_label('Wind Speed (m/s)', fontsize=18)  # Set colorbar label size to 14


# Use fig.text to add vertical text (rotated) next to each row
fig.text(0.05, 0.68, time_swot1, va='center', ha='center', rotation=90, fontsize=18)  # Row 1
fig.text(0.05, 0.45, time_swot2, va='center', ha='center', rotation=90, fontsize=18)  # Row 2
fig.text(0.05, 0.20, time_swot3, va='center', ha='center', rotation=90, fontsize=18)  # Row 3


# Adjust layout for better spacing
# plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.02, hspace=-0.28)
plt.show()