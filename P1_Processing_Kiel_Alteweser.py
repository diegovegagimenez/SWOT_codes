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


# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# plt.plot(df_tg_k.index[-70000:], (df_tg_k['SLEV_demean']-tide_k.h)[-70000:]*100, zorder=0, c='black', linewidth=0.8, label=r'Sea level')
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



# ---------------------- ERA5 REANALYSIS MODEL -------------------------------------

ds_era5 = xr.open_dataset(f'{era5_path}wind_mean_sea_level_pres_kiel.nc')

msl = ds_era5['msl']
u10 = ds_era5['u10']
v10 = ds_era5['v10']

# Select speciphic date
time_swot = '2023-04-02 00:00:00' 
u10 = u10.sel(time = time_swot)
v10 = v10.sel(time = time_swot)
msl = msl.sel(time = time_swot)

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



# --------------------- READ SWOT PASS ------------------------------------------------

ds_swot_kiel = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_478_003_20230401T230647_20230401T235752_v1.0.nc')
ssha_dac_kiel = ds_swot_kiel.ssha+ds_swot_kiel.dac

ds_swot_alte = xr.open_dataset(f'{SWOT_path}SWOT_data/SWOT_L3_LR_SSH_Expert_005_057_20231014T163508_20231014T172634_v1.0.nc')
ssha_dac_alte = ds_swot_alte.ssha+ds_swot_alte.dac


# ------------------- CREATE FIGURE FOR ALL KIEL DATA -----------------------------------

lolabox = [9.5, 12.5, 53.5, 58]
fig = plt.figure(figsize=(18, 6))
gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.5])

# Time series plot ----------------------------------------
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(df_tg_k.index[-70000:], (df_tg_k['SLEV_demean']-tide_k.h)[-70000:]*100, zorder=0, c='black', linewidth=0.8, label=r'Sea level')  # TG time series
ax0.plot(ts_kiel_SC['time'], (ts_kiel_SC['elevation']-tide_k_SC.h)*100, zorder=1, c='blue', linewidth=0.6, label=r'Sea level SCHISM')  # TG time series
ax0.scatter(df_kiel['time'], df_kiel['ssha_demean'], c='r', zorder=2, s=8, label=r'SWOT Data')
ax0.set_xlabel('Time (days)', labelpad=15)
ax0.set_ylabel('Sea level anomaly (cm)')
ax0.grid(True, alpha=0.3)
ax0.legend(fontsize=7)

# Map 1: SWOT pass ---------------------------------------
ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax1.set_extent(lolabox, crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.LAND)
ax1.add_feature(cfeature.COASTLINE)
ssha_plot = ax1.pcolormesh(ssha_dac_kiel.longitude, ssha_dac_kiel.latitude, ssha_dac_kiel.values, cmap='coolwarm', transform=ccrs.PlateCarree())
cbar1 = plt.colorbar(ssha_plot, ax=ax1, orientation='vertical', pad=0.06, aspect=30)
ax1.set_title('SWOT SLA')
gl1 = ax1.gridlines(draw_labels=True, alpha=0.5)
gl1.top_labels = False
gl1.right_labels = False

# Map 2: ERA5 -------------------------------------------
ax2 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax2.set_extent(lolabox, crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.LAND)
ax2.add_feature(cfeature.COASTLINE)
msl_hpa = msl / 100.0
msl_plot = ax2.pcolormesh(msl.longitude, msl.latitude, msl_hpa, cmap='coolwarm', transform=ccrs.PlateCarree())
cbar2 = plt.colorbar(msl_plot, ax=ax2, orientation='vertical', pad=0.02, aspect=30)
ax2.set_title('MSLP and Wind vectors')
gl2 = ax2.gridlines(draw_labels=True, alpha=0.5)
gl2.top_labels = False
gl2.right_labels = False
ax2.quiver(u10.longitude, u10.latitude, u10, v10, scale=500, transform=ccrs.PlateCarree())

# Map 3: SCHISM ----------------------------------------
ax3 = fig.add_subplot(gs[0, 3], projection=ccrs.PlateCarree())
ax3.set_extent(lolabox, crs=ccrs.PlateCarree())
ax3.add_feature(cfeature.LAND)
ax3.add_feature(cfeature.COASTLINE)
triang = Triangulation(lonSC, latSC, tri)
collection = ax3.tripcolor(triang, sshSC, shading='flat', cmap='coolwarm')
cbar3 = plt.colorbar(collection, ax=ax3, orientation='vertical', pad=0.02)
ax3.set_title('SCHISM Model')
gl3 = ax3.gridlines(draw_labels=True, alpha=0.5)
gl3.top_labels = False
gl3.right_labels = False

# Adjust layout
plt.tight_layout()
plt.show()


plt.plot()
