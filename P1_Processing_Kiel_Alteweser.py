import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utide import solve, reconstruct
import xarray.plot as xplt  # Import xarray.plot module
import mat73
from matplotlib.tri import Triangulation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

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
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Define the RGB values for your custom color palette
colors_rgb = [
    (255, 255, 255), (214, 226, 255), (181, 201, 255), (142, 178, 255),
    (127, 150, 255), (99, 112, 247), (0, 99, 255), (0, 102, 102), 
    (0, 150, 150), (0, 198, 51), (99, 255, 0), (150, 255, 0), 
    (198, 255, 51), (255, 255, 0), (255, 198, 0), (255, 160, 0), 
    (255, 124, 0), (255, 102, 0), (255, 25, 0)
]

# Convert RGB tuples to normalized RGB values (values between 0 and 1)
colors_normalized = [(r / 255., g / 255., b / 255.) for (r, g, b) in colors_rgb]

# Create a colormap using the normalized RGB values
cmap = mcolors.ListedColormap(colors_normalized)


TG_path='/home/amores/SWOT/A_data/A_TGs/'
SWOT_path = '/home/dvega/anaconda3/work/SWOT_STORM/'
era5_path = '/home/dvega/anaconda3/work/SWOT_STORM/ERA5/'
model_path = '/home/amores/SWOT/A_data/C_modelo/'


# ------------------------- TIDE GAUGE -----------------------------------------

# Processing tide gauge
kiel_tg = xr.open_dataset(f'{TG_path}NO_TS_TG_KielTG.nc')
Alte_tg = xr.open_dataset(f'{TG_path}NO_TS_TG_AlteWeserTG.nc')

kiel_tg = kiel_tg[['TIME', 'SLEV', 'LATITUDE', 'LONGITUDE']]
Alte_tg = Alte_tg[['TIME', 'SLEV', 'LATITUDE', 'LONGITUDE']]

kiel_tg = kiel_tg.to_dataframe().reset_index(level='DEPTH', drop=True)
Alte_tg = Alte_tg.to_dataframe().reset_index(level='DEPTH', drop=True)

TG_data = [kiel_tg, Alte_tg]


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
df_tg_k = df_tg_k[df_tg_k.index > pd.to_datetime('2020-01-01')]

df_tg_a = Alte_tg.reset_index()
df_tg_a = df_tg_a[['TIME', 'SLEV']]
df_tg_a['TIME'] = pd.to_datetime(df_tg_a['TIME'])
df_tg_a.set_index('TIME', inplace=True)
df_tg_a = df_tg_a[df_tg_a.index > pd.to_datetime('2020-01-01')]

df_tg_k['SLEV_demean'] = df_tg_k['SLEV'] - df_tg_k['SLEV'].mean()
df_tg_a['SLEV_demean'] = df_tg_a['SLEV'] - df_tg_a['SLEV'].mean()


# Detide
coef_k = solve(df_tg_k.index, df_tg_k['SLEV_demean'],
             lat=latTGs[0],
             nodal=False,
             trend=False,
             method='ols',
             conf_int='linear',
             verbose=False,)

coef_a = solve(df_tg_a.index, df_tg_a['SLEV_demean'],
             lat=latTGs[1],
             nodal=False,
             trend=False,
             method='ols',
             conf_int='linear',
             verbose=False)

tide_k = reconstruct(df_tg_k.index, coef_k, verbose=False)
tide_a = reconstruct(df_tg_a.index, coef_a, verbose=False)

df_tg_k['detided'] = df_tg_k['SLEV_demean']-tide_k.h
df_tg_a['detided'] = df_tg_a['SLEV_demean']-tide_a.h


# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# plt.plot(df_tg_k.index[-70000:], (df_tg_k['Detided'][-70000:]*100, zorder=0, c='black', linewidth=0.8, label=r'Sea level')
# plt.scatter(df_kiel['time'], df_kiel['ssha_demean'], c='r', zorder = 1, s=8, label=r'SWOT Data')
# plt.xticks(rotation=45)
# plt.xlabel(r'Time (days)', labelpad=15)
# plt.ylabel(r'Sea level anomaly (cm)')
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=7)
# plt.title(r'Kiel TG')
# plt.show()

# plt.plot(df_tg_a.index[-50000:], (df_tg_a['SLEV_demean']-tide_a.h)[-50000:]*100, zorder=0, c='black', linewidth=0.8, label=r'Sea level')
# plt.scatter(df_alte['time'], df_alte['ssha_demean'], c='r', zorder = 1, s=8, label=r'SWOT Data')
# plt.xticks(rotation=45)
# plt.grid(True, alpha=0.3)
# plt.xlabel(r'Time (days)', labelpad=15)
# plt.ylabel(r'Sea level anomaly (cm)')
# plt.legend(fontsize=7)
# plt.title(r'Alteweser TG')
# plt.show()


# ------------ READ SWOT TIME SERIE FOR KIEL AND ALTEWESER------------------------

df = pd.read_csv(f'{SWOT_path}df_Kiel_Alteweser_SWOT_series_50km.csv')

df_kiel= df[df['latitude'] == latTGs[0][0]]
df_kiel['ssha_demean'] = df_kiel['ssha_dac']-df_kiel['ssha_dac'].mean()

df_alte = df[df['latitude'] == latTGs[1][0]]
df_alte['ssha_demean'] = df_alte['ssha_dac']-df_alte['ssha_dac'].mean()

# Convert time to datetime
df_kiel['time'] = pd.to_datetime(df_kiel['time'])
df_alte['time'] = pd.to_datetime(df_alte['time'])


# ---------------------- ERA5 REANALYSIS MODEL -------------------------------------

ds_era5 = xr.open_dataset(f'{era5_path}wind_mean_sea_level_pres_kiel.nc')

time_swot = '2023-04-02 00:00:00' 

msl = ds_era5['msl'].sel(time = time_swot)
u10 = ds_era5['u10'].sel(time = time_swot)
v10 = ds_era5['v10'].sel(time = time_swot)


# Calculate wind speed and direction (optional)
wind_speed = np.sqrt(u10**2 + v10**2)
wind_dir = np.arctan2(v10, u10) * (180 / np.pi)


#  --------------------- READ SCHISM MODEL DATA ------------------------------------

model23 = xr.open_dataset(f'{model_path}merged_elevation_2023.nc')
model24 = xr.open_dataset(f'{model_path}merged_elevation_2024.nc')

model = xr.concat([model23, model24], dim='time')
model_df = model.to_dataframe().reset_index()


# Filter data for the specific date
model_df_cut_kiel = model_df[(model_df['SCHISM_hgrid_node_y'] > 54) & (model_df['SCHISM_hgrid_node_y'] < 55) &
                    (model_df['SCHISM_hgrid_node_x'] > 9.8) & (model_df['SCHISM_hgrid_node_x'] < 10.6)]

model_df_cut_alte = model_df[(model_df['SCHISM_hgrid_node_y'] > 53.4) & (model_df['SCHISM_hgrid_node_y'] < 54.2) &
                    (model_df['SCHISM_hgrid_node_x'] > 7.6) & (model_df['SCHISM_hgrid_node_x'] < 8.7)]

model_df_kiel_time = model_df[model_df['time'] == pd.to_datetime('2023-04-02 00:00:00')]
model_df_alte_time = model_df[model_df['time'] == pd.to_datetime('2023-10-14 17:00:00')]

# Extract data for create a map of speciphic date
latSC = model_df_kiel_time['SCHISM_hgrid_node_y'].values
lonSC = model_df_kiel_time['SCHISM_hgrid_node_x'].values
sshSC = model_df_kiel_time['elevation'].values
time = model_df_kiel_time['time'].values

# --------- Define plot bounds and load triangulation data
latmin, latmax = 53.5, 57
lonmin, lonmax = 9, 13
lonTG, latTG = np.array([10]), np.array([55])
peak_date = '2023-04-02 00:00:00'
mat_data = mat73.loadmat(f'{model_path}tri_simu.mat')
tri = mat_data['tri'].astype(int) - 1

# # ------------- EXTRACT TIME SERIES FROM SCHISM (KIEL AND ALTEWESER LOCATIONS) -----------------------------

# Calculate the distance from the target location to all points in the dataframe
model_df_cut_kiel['distance'] = model_df_cut_kiel.apply(lambda row: haversine(lonTGs[0], latTGs[0], row['SCHISM_hgrid_node_x'], row['SCHISM_hgrid_node_y']), axis=1)
model_df_cut_alte['distance'] = model_df_cut_alte.apply(lambda row: haversine(lonTGs[1], latTGs[1], row['SCHISM_hgrid_node_x'], row['SCHISM_hgrid_node_y']), axis=1)

# Filter the points within 20 km radius
radius = 50  # km
df_filtered_kiel = model_df_cut_kiel[model_df_cut_kiel['distance'] <= radius]
df_filtered_alte = model_df_cut_alte[model_df_cut_alte['distance'] <= radius]

# Group by time and calculate the average sea level anomaly
ts_kiel_SC = df_filtered_kiel.groupby('time')['elevation'].mean().reset_index()
ts_alte_SC = df_filtered_alte.groupby('time')['elevation'].mean().reset_index()

# Detide time series of SCHISM

# Detide
coef_k_SC = solve(ts_kiel_SC['time'], ts_kiel_SC['elevation'],
             lat=latTGs[0],
             nodal=False,
             trend=False,
             method='ols',
             conf_int='linear',
             verbose=False,)

coef_a_SC = solve(ts_alte_SC['time'], ts_alte_SC['elevation'],
             lat=latTGs[1],
             nodal=False,
             trend=False,
             method='ols',
             conf_int='linear',
             verbose=False)

tide_k_SC = reconstruct(ts_kiel_SC['time'], coef_k_SC, verbose=False)
tide_a_SC = reconstruct(ts_alte_SC['time'], coef_a_SC, verbose=False)

ts_kiel_SC['detided'] = ts_kiel_SC['elevation'] - tide_k_SC.h
ts_alte_SC['detided'] = ts_alte_SC['elevation'] - tide_a_SC.h

# --------------------- READ SWOT PASS ------------------------------------------------

ds_swot_kiel = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_478_003_20230401T230647_20230401T235752_v1.0.nc')
ssha_dac_kiel = ds_swot_kiel.ssha+ds_swot_kiel.dac

ds_swot_alte = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_005_057_20231014T163508_20231014T172634_v1.0.nc')
ssha_dac_alte = ds_swot_alte.ssha+ds_swot_alte.dac


# ------------------- CREATE FIGURE FOR ALL KIEL DATA -----------------------------------

# Set same period for all TSs
# Margin of 1 week
time_margin = pd.Timedelta(days=7)  # Example margin of 1 day

# Calculate initial and final times for Kiel
initial_time_kiel = np.max([np.min(df_tg_k.index), np.min(ts_kiel_SC['time']), np.min(df_kiel['time'])]) - time_margin
final_time_kiel = np.min([np.max(df_tg_k.index), np.max(ts_kiel_SC['time']), np.max(df_kiel['time'])]) + time_margin

# Calculate initial and final times for Alte
initial_time_alte = np.max([np.min(df_tg_a.index), np.min(ts_alte_SC['time']), np.min(df_alte['time'])]) - time_margin
final_time_alte = np.min([np.max(df_tg_a.index), np.max(ts_alte_SC['time']), np.max(df_alte['time'])]) + time_margin

# Set the same period for all time series with the margin
df_tg_k = df_tg_k[(df_tg_k.index >= initial_time_kiel) & (df_tg_k.index <= final_time_kiel)]
df_tg_a = df_tg_a[(df_tg_a.index >= initial_time_alte) & (df_tg_a.index <= final_time_alte)]

ts_kiel_SC = ts_kiel_SC[(ts_kiel_SC['time'] >= initial_time_kiel) & (ts_kiel_SC['time'] <= final_time_kiel)]
ts_alte_SC = ts_alte_SC[(ts_alte_SC['time'] >= initial_time_alte) & (ts_alte_SC['time'] <= final_time_alte)]

df_kiel = df_kiel[(df_kiel['time'] >= initial_time_kiel) & (df_kiel['time'] <= final_time_kiel)]
df_alte = df_alte[(df_alte['time'] >= initial_time_alte) & (df_alte['time'] <= final_time_alte)]

# SELECT TARGET DATE POINTS OF SWOT

df_kiel_max = df_kiel.sort_values(by='ssha_demean', ascending=False)[:5]
df_alte_max = df_alte.sort_values(by='ssha_demean', ascending=False)[:3]
df_alte_min = df_alte.sort_values(by='ssha_demean', ascending=True)[:2]

lolabox = [9.5, 12.5, 53.5, 58]
fig = plt.figure(figsize=(18, 6))
gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

# Time series plot ----------------------------------------
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(df_tg_k.index, df_tg_k['detided']*100, zorder=0, c='black', linewidth=0.8, label=r'Tide Gauge')  # TG time series
ax0.plot(ts_kiel_SC['time'], ts_kiel_SC['detided']*100, zorder=1, c='blue', linewidth=0.6, label=r'SCHISM model')  # TG time series
ax0.scatter(df_kiel['time'], df_kiel['ssha_demean'], c='r', zorder=2, s=8, label=r'SWOT Data')
ax0.scatter(df_kiel_max['time'], df_kiel_max['ssha_demean'], c='green', zorder=3, s=50, label=r'Extremes SWOT Data')

ax0.set_xlabel('Time (days)', labelpad=15)
ax0.set_ylabel('Sea level anomaly (cm)')
ax0.grid(True, alpha=0.3)
ax0.legend(fontsize=7)
ax0.set_title('Kiel TG')

# Map 1: SWOT pass ---------------------------------------
ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax1.set_extent(lolabox, crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.LAND)
ax1.add_feature(cfeature.COASTLINE)
ssha_plot = ax1.pcolormesh(ssha_dac_kiel.longitude, ssha_dac_kiel.latitude, ssha_dac_kiel, cmap=cmap, transform=ccrs.PlateCarree())
cbar1 = plt.colorbar(ssha_plot, ax=ax1, orientation='vertical', pad=0.06, aspect=30)
ax1.set_title('SWOT SLA')
gl1 = ax1.gridlines(draw_labels=True, alpha=0.5)
gl1.top_labels = False
gl1.right_labels = False

# Map 2: ERA5 -------------------------------------------
wind_speed_levels = np.arange(-0.5, 21.5, 1)  # Adjusted to include up to 20.5 m/s

ax2 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax2.set_extent(lolabox, crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.LAND)
ax2.add_feature(cfeature.COASTLINE)
msl_hpa = msl / 100.0

# Define custom levels for contours
min_value = np.floor(msl_hpa.min().item())
max_value = np.ceil(msl_hpa.max().item())
levels = np.arange(min_value, max_value + 5, 5)

# Plot mean sea level pressure as contours with thicker grey lines
msl_contour = ax2.contour(msl.longitude, msl.latitude, msl_hpa, levels=levels, colors='#4d4d4d', linewidths=1, transform=ccrs.PlateCarree())
plt.clabel(msl_contour, inline=True, fontsize=8, fmt='%1.0f')

# Set the colorbar range from 0 to 21 m/s for wind speed
vmin, vmax = 0, 20

# Add colorbar for wind speed with discrete levels using the custom colormap
wind_speed_plot = ax2.contourf(u10.longitude, u10.latitude, wind_speed, levels=wind_speed_levels, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
cbar = plt.colorbar(wind_speed_plot, orientation='vertical', pad=0.02, aspect=30, ticks=range(0, 21, 1))
cbar.set_label('Wind Speed (m/s)', labelpad=10)

stride = 15  # Adjust this value to reduce the number of arrows
u10_reduced = u10[::stride, ::stride]
v10_reduced = v10[::stride, ::stride]
lon_reduced = u10.longitude[::stride]
lat_reduced = u10.latitude[::stride]

# Plot wind vectors with larger arrows
plt.quiver(lon_reduced, lat_reduced, u10_reduced, v10_reduced, scale=250, width=0.005, transform=ccrs.PlateCarree())

gl2 = ax2.gridlines(draw_labels=True, alpha=0.5)
gl2.top_labels = False
gl2.right_labels = False
ax2.quiver(u10.longitude, u10.latitude, u10, v10, scale=100, transform=ccrs.PlateCarree())

# Map 3: SCHISM ----------------------------------------
ax3 = fig.add_subplot(gs[0, 3], projection=ccrs.PlateCarree())
ax3.set_extent(lolabox, crs=ccrs.PlateCarree())
ax3.add_feature(cfeature.LAND)
ax3.add_feature(cfeature.COASTLINE)
triang = Triangulation(lonSC, latSC, tri)
collection = ax3.tripcolor(triang, sshSC, shading='flat', cmap=cmap)
cbar3 = plt.colorbar(collection, ax=ax3, orientation='vertical', pad=0.02)
ax3.set_title('SCHISM Model')
ax3.plot(lonTG[0], latTG[0], 'pk', markerfacecolor='k', markersize=10, transform=ccrs.PlateCarree())
gl3 = ax3.gridlines(draw_labels=True, alpha=0.5)
gl3.top_labels = False
gl3.right_labels = False

# Adjust layout
plt.tight_layout()
plt.show()
