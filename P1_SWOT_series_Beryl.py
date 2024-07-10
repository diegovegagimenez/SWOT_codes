import xarray as xr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray.plot as xplt  # Import xarray.plot module
from utide import solve, reconstruct

def detide(time_series, ssha_series, latitude):
    """
    Extracts tidal and non-tidal components from a tide gauge time series using utide.

    Parameters:
    time_series (pd.Series): Pandas Series with datetime values.
    water_levels (pd.Series): Pandas Series with corresponding water level values.
    latitude (float): Latitude of the tide gauge location (default is 0).

    Returns:
    pd.DataFrame: DataFrame with columns 'time', 'water_level', 'tidal_signal', 'non_tidal_signal'.
    """
    # Ensure the data is sorted by time
    data = pd.DataFrame({'time': time_series, 'ssha_series': ssha_series}).sort_values(by='time')
    
    # Extract the time and water level
    time = data['time'].values
    ssha = data['ssha_series'].values
    
    # Convert time to decimal days since the first observation
    t0 = time[0]
    time_in_days = (time - t0) / np.timedelta64(1, 'D')
    
    # Fit the tidal model to the data
    coef = solve(time_in_days, ssha, lat=latitude)
    
    # Predict the tidal components
    reconstruction = reconstruct(time_in_days, coef)
    
    # Extract the tidal signal
    tidal_signal = reconstruction.h
    
    # Compute the non-tidal component by subtracting the tidal signal from the original water level
    ssha_detided = ssha - tidal_signal
    
    # Create a DataFrame to return the results
    result = pd.DataFrame({
        'time': time,
        'ssha_series': ssha_series,
        'tidal_signal': tidal_signal,
        'ssha_detided': ssha_detided
    })
    
    return result


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

TG_path='/home/dvega/anaconda3/work/SWOT_STORM/datos_TG_Beryl_STORM/'
# TG_path = ['/home/amores/SWOT/A_data/A_TGs/TG_CMEMS/', '/home/amores/SWOT/A_data/A_TGs/TG_SOEST/']
SWOT_path = '/home/dvega/anaconda3/work/SWOT_STORM/SWOT_data_202406_07/'

names_tg = ['Progreso', 'Puerto_Barrios', 'Carrie_Bow_Cay', 'Isla_Mujeres', 'Sian_Kaan']

loc_Prog = [21.303333, -89.666667]
loc_isla_mujeres = [21.2546667, -86.7460833]
loc_Sian_Kaan = [19.3126333, -87.4460833]
loc_Carrie_Bow_Cay = [16.80283, -88.08202]
loc_Puerto_Barrios = [15.694618, -88.622018]

latTGs = []
lonTGs = []

# Create a list of all locations
locations = [loc_Prog, loc_Puerto_Barrios, loc_Carrie_Bow_Cay, loc_isla_mujeres, loc_Sian_Kaan]

# Iterate over the locations and extract latitudes and longitudes
for loc in locations:
    latTGs.append(loc[0])
    lonTGs.append(loc[1])


all_altimetry_timeseries = []

fileSWOT = [os.path.join(SWOT_path, f) for f in os.listdir(SWOT_path) if f.endswith('.nc')]

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

    # Loop through each tide gauge location
    for idx, (lon_tg, lat_tg) in enumerate(zip(lonTGs, latTGs)):
        # d = np.sqrt((lonSWOT - lon_tg) ** 2 + (latSWOT - lat_tg) ** 2)

        # if np.min(d) > thres:
        #     continue

        # # Mask distances greater than threshold
        # mask = d <= thres
        # alon = np.where(mask, lonSWOT, np.nan)
        # alat = np.where(mask, latSWOT, np.nan)

        # Calculate distance for each data point
        distances = haversine(lonSWOT, latSWOT, lon_tg, lat_tg)

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
            "station": idx,  # Access station name
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
df.to_excel('df_SWOT_Beryl_time_serie_50km.xlsx')


# Tide gauges processing
TG_files = [os.path.join(TG_path, f) for f in os.listdir(TG_path) if f.endswith('.csv')]

for idx, tg_file in enumerate(TG_files):
    tg_path = os.path.join(TG_path, tg_file)
    df_tg = pd.read_csv(tg_path, delimiter=";", encoding='utf-8')
    df_tg['Time'] = pd.to_datetime(df_tg['Time'], format='%d/%m/%Y %H:%M')
    df_tg.sort_values(by='Time', inplace=True)
    df_tg['ssha_series'] = df_tg['rad']*100  # Convert to cm
    df_tg['station'] = tg_file

    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(df_tg['time'], df_tg['ssha_series'], label='Original data')
    # ax.plot(df_tg['time'], df_tg['tidal_signal'], label='Tidal signal')
    ax.plot(df_tg['Time'], df_tg['ssha_series'], label='')
    ax.set_title(f'tide gauge data for {names_tg[idx]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('SSH (cm)')
    ax.legend()

    # Extract the latitude of the tide gauge
    latitude = latTGs[idx]

    # Detide the tide gauge data
    detided_data = detide(df_tg['Time'], df_tg['ssha_series'], latitude)
    df_tg['tidal_signal'] = detided_data['tidal_signal']
    df_tg['ssha_detided'] = detided_data['ssha_detided']

    # Save the detided data to a new CSV file
    # df_tg.to_csv(f'{names_tg[idx]}_detided.csv', index=False)

    # Plot the detided data
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(df_tg['time'], df_tg['ssha_series'], label='Original data')
    # ax.plot(df_tg['time'], df_tg['tidal_signal'], label='Tidal signal')
    ax.plot(df_tg['Time'], df_tg['ssha_detided'], label='Detided data')
    ax.set_title(f'Detided tide gauge data for {names_tg[idx]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('SSH (cm)')
    ax.legend()
    # plt.savefig(f'{names_tg[idx]}_detided.png')
    # plt.close()

