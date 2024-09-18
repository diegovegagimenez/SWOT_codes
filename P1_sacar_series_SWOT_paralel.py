import xarray as xr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utide import solve, reconstruct
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def detide(time_series, ssha_series, latitude):
    """
    Extracts tidal and non-tidal components from a tide gauge time series using utide.
    """
    data = pd.DataFrame({'time': time_series, 'ssha_series': ssha_series}).sort_values(by='time')
    time = data['time'].values
    ssha = data['ssha_series'].values
    t0 = time[0]
    time_in_days = (time - t0) / np.timedelta64(1, 'D')
    coef = solve(time_in_days, ssha, lat=latitude)
    reconstruction = reconstruct(time_in_days, coef)
    tidal_signal = reconstruction.h
    ssha_detided = ssha - tidal_signal
    result = pd.DataFrame({
        'time': time,
        'ssha_series': ssha_series,
        'tidal_signal': tidal_signal,
        'ssha_detided': ssha_detided
    })
    return result

def read_first_available(dataset, var_names):
    for var in var_names:
        if var in dataset.columns:
            return dataset[var].values[0]
    raise ValueError("None of the specified variable names found in the dataset.")

def haversine(lon1, lat1, lon2, lat2):
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def process_swot_file(filename, SWOT_path, lonTGs, latTGs, dmin, thres):
    ds = xr.open_dataset(filename)
    ds = ds.drop_dims('num_nadir')
    ds = ds[['time', 'ssha', 'latitude', 'longitude', 'dac']]
    ssh_dac = ds.ssha + ds.dac

    lon = ds['longitude'].values.flatten()
    lat = ds['latitude'].values.flatten()
    ssh = ds['ssha'].values.flatten()
    ssh_dac = ssh_dac.values.flatten()
    time_values = ds['time'].values
    time = np.tile(time_values[:, np.newaxis], (1, 69)).flatten()

    valid_indices = np.where(~np.isnan(ssh))
    lonSWOT = lon[valid_indices]
    latSWOT = lat[valid_indices]
    timeSWOT = time[valid_indices]
    ssh = ssh[valid_indices]
    ssh_dac = ssh_dac[valid_indices]

    result = []

    for idx, (lon_tg, lat_tg) in enumerate(zip(lonTGs, latTGs)):
        distances = haversine(lonSWOT, latSWOT, lon_tg, lat_tg)
        in_radius = distances <= dmin

        if np.any(in_radius):
            ssh_tmp = np.nanmean(ssh[in_radius])
            ssh_serie = ssh_tmp * 100
            time_serie = timeSWOT[in_radius][~np.isnan(timeSWOT[in_radius])][0]
            ssh_dac_serie = np.nanmean(ssh_dac[in_radius]) * 100
            lat_within_radius = latSWOT[in_radius]
            lon_within_radius = lonSWOT[in_radius]
            min_distance_point = distances[in_radius].min()
            n_idx = sum(in_radius)
        else:
            ssh_serie = np.nan
            time_serie = np.nan
            n_idx = np.nan
            min_distance_point = np.nan
            ssh_dac_serie = np.nan
            lat_within_radius = None
            lon_within_radius = None

        selected_data = {
            "station": filename,
            "longitude": lon_tg,
            "latitude": lat_tg,
            "ssha": ssh_serie,
            "ssha_dac": ssh_dac_serie,
            "time": time_serie,
            "n_val": n_idx,
            "lat_within_radius": lat_within_radius,
            "lon_within_radius": lon_within_radius,
            "min_distance": min_distance_point,
        }
        result.append(selected_data)

    return result

TG_path = '/home/dvega/anaconda3/work/SWOT_STORM/datos_tg_cmems/datos_TG_med_CMEMS/'
TG_path = '/home/amores/SWOT/A_data/A_TGs/'

SWOT_path = '/home/dvega/anaconda3/work/SWOT_STORM/SWOT_data_1_0_2/'
dmin = 50
thres = 20 * dmin / 100

fileTG = [os.path.join(TG_path, f) for f in os.listdir(TG_path) if f.endswith('.nc')]
lonNames = ['lon', 'longitude', 'LONGITUDE']
latNames = ['lat', 'latitude', 'LATITUDE']

latTGs = [37.64045]
lonTGs = [21.319233]


fileSWOT = [os.path.join(SWOT_path, f) for f in os.listdir(SWOT_path) if f.endswith('.nc')]

# Define a partial function to pass fixed arguments
partial_process_swot_file = partial(process_swot_file, SWOT_path=SWOT_path, lonTGs=lonTGs, latTGs=latTGs, dmin=dmin, thres=thres)

# Specify the number of workers
num_workers = 50  # You can adjust this number based on your CPU cores and desired parallelism

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    results = list(tqdm(executor.map(partial_process_swot_file, fileSWOT), total=len(fileSWOT), desc='Processing SWOT files'))

# Flatten the results
all_altimetry_timeseries = [item for sublist in results for item in sublist]

df = pd.DataFrame(all_altimetry_timeseries)
df.dropna(subset=['ssha'], inplace=True)
df.sort_values(by='time', inplace=True)

df.to_csv('df_SWOT_1_0_2_CMEMS_TGs_50km.csv')
