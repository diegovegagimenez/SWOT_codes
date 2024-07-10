import xarray as xr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import utide
import cartopy.crs as ccrs
import xarray.plot as xplt  # Import xarray.plot module


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
        if var in dataset.variables:
            return dataset[var].values
    raise ValueError("None of the specified variable names found in the dataset.")

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
    r = 6371
    return c * r

dmin = 50  # km de radio
thres = 20*dmin/100

TG_path='/home/amores/SWOT/A_data/A_TGs/'
# TG_path = ['/home/amores/SWOT/A_data/A_TGs/TG_CMEMS/', '/home/amores/SWOT/A_data/A_TGs/TG_SOEST/']
SWOT_path = '/home/dvega/anaconda3/work/SWOT_STORM/SWOT_data_202406_07/'

# SWOTfiles = [f for f in os.listdir(SWOT_path) if f.endswith('.nc')]
# TGfiles = [f for f in os.listdir(TG_path) if f.endswith('.nc')]

kiel_tg = xr.open_dataset(f'{TG_path}NO_TS_TG_KielTG.nc')
Alte_tg = xr.open_dataset(f'{TG_path}NO_TS_TG_AlteWeserTG.nc')

TG_data = [kiel_tg, Alte_tg]


lonNames=['lon','longitude','LONGITUDE']
latNames=['lat','latitude','LATITUDE']

latTGs = []
lonTGs = []

latTGs = 19.3126333
lonTGs = -87.4460833

# for i in range(len(TG_data)):
#     tg = TG_data[i]
#     # Read the first available longitude and latitude variables
#     lonTG = read_first_available(tg, lonNames)
#     latTG = read_first_available(tg, latNames)
#     latTGs.append(latTG)
#     lonTGs.append(lonTG)
#     print(latTG,lonTG)

# time_data = []
# ssh_data = []
# npuntos_data = []

all_altimetry_timeseries = []

fileSWOT = [os.path.join(SWOT_path, f) for f in os.listdir(SWOT_path) if f.endswith('.nc')]
# for filename in tqdm(fileSWOT, desc='SWOT files'):

#     file_path = os.path.join(SWOT_path, filename)
#     ds = xr.open_dataset(file_path)

#     # Extract data from variables
#     lon = ds['longitude'].values.flatten()
#     lat = ds['latitude'].values.flatten()
#     ssh = ds['ssha_noiseless'].values.flatten()
#     # ssh = ds['ssha'].values.flatten()

#     time_values = ds['time'].values  # Adding a new
#     time = np.tile(time_values[:, np.newaxis], (1, 69)).flatten()  # Not efficient

#     # Find indices of non-NaN values
#     valid_indices = np.where(~np.isnan(ssh))
#     lonSWOT = lon[valid_indices]
#     latSWOT = lat[valid_indices]
#     timeSWOT = time[valid_indices]
#     ssh = ssh[valid_indices]

#     # Loop through each tide gauge location
#     for idx, (lon_tg, lat_tg) in enumerate(zip(lonTGs, latTGs)):

#         # d = np.sqrt((lonSWOT - lon_tg) ** 2 + (latSWOT - lat_tg) ** 2)

#         # if np.min(d) > thres:
#         #     continue

#         # # Mask distances greater than threshold
#         # mask = d <= thres
#         # alon = np.where(mask, lonSWOT, np.nan)
#         # alat = np.where(mask, latSWOT, np.nan)

#         # Calculate distance for each data point
#         distances = haversine(lonSWOT, latSWOT, lon_tg, lat_tg)

#         in_radius = distances <= dmin

#         # Average nearby SSH values (if any)
#         if np.any(in_radius):
#             print(f'there is data')

#             ssh_tmp = np.nanmean(ssh[in_radius])
#             ssh_serie = ssh_tmp * 100  # Convert to centimeters (cm)
#             time_serie = timeSWOT[in_radius][~np.isnan(timeSWOT[in_radius])][0]  # Picking the first value of time within the radius

#             # Store the latitudes and longitudes of SWOT within the radius
#             lat_within_radius = latSWOT[in_radius]
#             lon_within_radius = lonSWOT[in_radius]

#             # Store closest  distance and number of points used for the average
#             min_distance_point = distances[in_radius].min()
#             n_idx = sum(in_radius)  # How many values are used for compute the mean value

#         else:
#             # print(f'there is no data')
#             ssh_serie = np.nan  # No data within radius (remains NaN)
#             time_serie = timeSWOT[in_radius][~np.isnan(timeSWOT[in_radius])]
#             n_idx = np.nan  # Number of points for the average within the radius
#             min_distance_point = np.nan

#             # If there's no SWOT data within the radius, set latitudes and longitudes to None
#             lat_within_radius = None
#             lon_within_radius = None
#             # print(f"No SWOT data within {dmedia} km radius of tide gauge {sorted_names[idx]}")

#         # Create a dictionary to store tide gauge and SWOT data
#         selected_data = {
#             # "station": sorted_names[idx],  # Access station name
#             "longitude": lon_tg,  # Longitude of tide gauge
#             "latitude": lat_tg,  # Latitude of tide gauge
#             "ssha": ssh_serie,  # Retrieved SSH value
#             # "ssha_raw": ssh[in_radius],  # Raw SSH values within the radius
#             "time": time_serie,
#             "n_val": n_idx,  # Number of points for the average within the radius
#             "lat_within_radius": lat_within_radius,  # Latitudes of SWOT within the radius
#             "lon_within_radius": lon_within_radius,   # Longitudes of SWOT within the radius
#             "min_distance": min_distance_point,  # Closest distance within the radius
#             }
#         all_altimetry_timeseries.append(selected_data)

# df = pd.DataFrame(all_altimetry_timeseries)
# df.dropna(subset='ssha',inplace=True)
# df.sort_values(by='time', inplace=True)

# df_kiel= df[df['latitude'] == latTGs[0]]
# df_kiel_mean=df_kiel['ssha'].mean()
# df_kiel['ssha_demean'] = df_kiel['ssha']-df_kiel_mean

# df_alte = df[df['latitude'] == latTGs[1]]

# df_tg_k = kiel_tg.to_dataframe()
# df_tg_k = df_tg_k.reset_index()
# df_tg_k = df_tg_k[['TIME', 'SLEV']]
# df_tg_k['TIME'] = pd.to_datetime(df_tg_k['TIME'])
# df_tg_k.set_index('TIME', inplace=True)
# df_tg_k = df_tg_k[df_tg_k.index > pd.to_datetime('2020-01-01')]

# df_tg_k_mean = df_tg_k['SLEV'].mean()
# df_tg_k['SLEV_demean'] = df_tg_k['SLEV']-df_tg_k_mean


# # Eliminar mareas del mareografo
# coef_kiel = utide.solve(df_tg_k.index, df_tg_k['SLEV_demean'].values, lat=latTGs[0], method='robust')

# # Reconstruct the tidal signal
# tide_kiel = utide.reconstruct(df_tg_k.index, coef_kiel)

# # Extract tidal components
# tidal_signal = tide_kiel.h 


# plt.plot(df_tg_k.index, tidal_signal*100, zorder=0)
# plt.scatter(df_kiel['time'], df_kiel['ssha_demean'], c='r', zorder = 1)
# plt.xticks(rotation=45)



for filename in tqdm(fileSWOT, desc='SWOT files'):
    file_path = os.path.join(SWOT_path, filename)
    ds = xr.open_dataset(file_path)

    ds = ds.drop_dims('num_nadir')
    ds = ds[['time', 'ssha', 'latitude', 'longitude', 'dac']]
    
    ssh_dac = ds.ssha+ds.dac

    # Extract data from variables
    lon = ds['longitude'].values.flatten()
    lat = ds['latitude'].values.flatten()
    ssh = ds['ssha'].values.flatten()
    ssh_dac = ssh_dac.values.flatten()
    # ssh = ds['ssha'].values.flatten()

    time_values = ds['time'].values  # Adding a new
    time = np.tile(time_values[:, np.newaxis], (1, 69)).flatten()  # Not efficient

    # Find indices of non-NaN values
    valid_indices = np.where(~np.isnan(ssh))
    lonSWOT = lon[valid_indices]
    latSWOT = lat[valid_indices]
    timeSWOT = time[valid_indices]
    ssh = ssh[valid_indices]
    ssh_dac = ssh_dac[valid_indices]

    # d = np.sqrt((lonSWOT - lon_tg) ** 2 + (latSWOT - lat_tg) ** 2)

    # if np.min(d) > thres:
    #     continue

    # # Mask distances greater than threshold
    # mask = d <= thres
    # alon = np.where(mask, lonSWOT, np.nan)
    # alat = np.where(mask, latSWOT, np.nan)

    # Calculate distance for each data point
    distances = haversine(lonSWOT, latSWOT, lonTGs, latTGs)

    in_radius = distances <= dmin

    # Average nearby SSH values (if any)
    if np.any(in_radius):
        print(f'there is data for file {filename}')

        ssh_tmp = np.nanmean(ssh[in_radius])
        ssh_serie = ssh_tmp * 100  # Convert to centimeters (cm)
        time_serie = timeSWOT[in_radius][~np.isnan(timeSWOT[in_radius])][0]  # Picking the first value of time within the radius
        ssh_dac_serie = np.nanmean(ssh_dac[in_radius])*100  # To cm
        
        # Store the latitudes and longitudes of SWOT within the radius
        lat_within_radius = latSWOT[in_radius]
        lon_within_radius = lonSWOT[in_radius]

        # Store closest  distance and number of points used for the average
        min_distance_point = distances[in_radius].min()
        n_idx = sum(in_radius)  # How many values are used for compute the mean value

    else:
        # print(f'there is no data')
        ssh_serie = np.nan  # No data within radius (remains NaN)
        time_serie = timeSWOT[in_radius][~np.isnan(timeSWOT[in_radius])]
        n_idx = np.nan  # Number of points for the average within the radius
        min_distance_point = np.nan
        ssh_dac_serie = np.nan

        # If there's no SWOT data within the radius, set latitudes and longitudes to None
        lat_within_radius = None
        lon_within_radius = None
        # print(f"No SWOT data within {dmedia} km radius of tide gauge {sorted_names[idx]}")

    # Create a dictionary to store tide gauge and SWOT data
    selected_data = {
        # "station": sorted_names[idx],  # Access station name
        "longitude": lonTGs,  # Longitude of tide gauge
        "latitude": latTGs,  # Latitude of tide gauge
        "ssha": ssh_serie, # Retrieved SSH value
        "ssha_dac": ssh_dac_serie,
        # "ssha_raw": ssh[in_radius],  # Raw SSH values within the radius
        "time": time_serie,
        "n_val": n_idx,  # Number of points for the average within the radius
        "lat_within_radius": lat_within_radius,  # Latitudes of SWOT within the radius
        "lon_within_radius": lon_within_radius,   # Longitudes of SWOT within the radius
        "min_distance": min_distance_point,  # Closest distance within the radius
        }
    all_altimetry_timeseries.append(selected_data)

df = pd.DataFrame(all_altimetry_timeseries)
df.dropna(subset='ssha',inplace=True)
df['time'] = pd.to_datetime(df['time'])
df.sort_values(by='time', inplace=True)
df.to_excel('df_SWOT_Carrie_Bow_time_serie_50.xlsx')


# Read Sian Kaan Tide gauge data

df_tg = pd.read_csv('/home/dvega/anaconda3/work/SWOT_STORM/datos_Sian_Kaan_TG.csv', delimiter=";")
df_tg['Time'] = pd.to_datetime(df_tg['Time'])
df_tg.dropna(inplace=True)

coef_sian = utide.solve(df_tg['Time'], df_tg['rad'].values, lat=19.31, method='robust')
tide_sian = utide.reconstruct(df_tg['Time'], coef_sian)

# Extract tidal components
tidal_signal = tide_sian.h 


# Standarize droping the mean
df_tg['detided'] = df_tg['rad'] -  df['rad'].mean()


ds = xr.open_dataset('/home/dvega/anaconda3/work/SWOT_STORM/SWOT_data_202406_07/SWOT_L3_LR_SSH_Expert_017_453_20240705T050906_20240705T060032_v1.0.nc')
ds = ds.drop_dims('num_nadir')

ds_a = ds.ssha+ds.dac

# df = ds.to_dataframe()
# df = df[['ssha_noiseless', 'ssha', 'latitude', 'longitude', 'time', 'dac']]
# df['ssha_a'] = df['ssha']+df['dac']
# data=df['ssha_a'].values



lolabox = [-94, -81, 11, 27]
fig, ax = plt.subplots(figsize=(10.5, 11), subplot_kw=dict(projection=ccrs.PlateCarree()))

# Set the extent to focus on the defined lon-lat box
ax.set_extent(lolabox, crs=ccrs.PlateCarree())



# Add scatter plot for specific locations
plot_kwargs = dict(
    x="longitude",
    y="latitude",
    cmap="Spectral_r",
    # vmin=-0.01,  # For checking the noise
    # vmax=0.01,
    vmin=-0.2,
    vmax=0.2,
    cbar_kwargs={"shrink": 0.7, "pad": 0.1},)

# SWOT SLA plots
# ds_2.ssha.plot.pcolormesh(ax=ax1, **plot_kwargs)
xplt.pcolormesh(ds_a, ax=ax, **plot_kwargs)
# ssh_diff = ds.ssha-ds.ssha_noiseless
# ssh_diff.plot.pcolormesh(ax=ax2, **plot_kwargs)

ax.coastlines()
ax.gridlines(draw_labels=True)

plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

plt.show()
