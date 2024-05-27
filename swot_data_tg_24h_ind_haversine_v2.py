import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cartopy.crs as ccrs
from shapely.geometry import LineString
import cartopy.feature as cfeature
import statsmodels.api as sm  # for LOWESS filter
import loess_smooth_handmade as loess  # for LOESS filter
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")


# Function for calculating the Haversine distance between two points
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


path ='/home/dvega/anaconda3/work/SWOT/'

folder_path = (f'{path}swot_basic_1day/003_016_pass/')  # Define SWOT passes folders
                
data_tg = np.load(f'{path}mareografos/TGresiduals1d_2023_European_Seas_SWOT_FSP.npz')
names_tg = pd.read_csv(f'{path}mareografos/GLOBAL_TGstations_CMEMS_SWOT_FSP_Feb2024', header=None)
# Change format of names
names_tg_short_sorted = pd.DataFrame({'Stations': ['Porquerolles', 'La_Capte', 'Les_Oursinieres', 'Saint_Louis_Mourillon',
                                            'Toulon', 'Baie_Du_Lazaret', 'Port_De_StElme', 'Tamaris', 'Bregaillon',
                                            'Le_Brusc', 'La_Ciotat', 'Cassis', 'Marseille', 'Port_De_La_Redonne',
                                            'Port_De_Carro', 'Fos_Sur_Mer', 'Mahon', 'Porto_Cristo',
                                            'Palma_de_Mallorca', 'Barcelona', 'Tarragona']})

names_tg_short = pd.DataFrame({'Stations': ['Baie_Du_Lazaret', 'Bregaillon', 'Cassis', 'Fos_Sur_Mer', 'La_Capte',
                                            'La_Ciotat', 'Le_Brusc', 'Les_Oursinieres', 'Marseille', 'Porquerolles',
                                            'Port_De_Carro', 'Port_De_La_Redonne', 'Port_De_StElme',
                                            'Saint_Louis_Mourillon', 'Tamaris', 'Toulon', 'Barcelona', 'Mahon',
                                            'Palma_de_Mallorca', 'Tarragona', 'Porto_Cristo']})


# ---------------------------------- READING TIDE GAUGE DATA --------------------------------------------------------

sla = data_tg.get('resdacTOTday') * 100  # Convert to centimeters (cm)
time = data_tg.get('timeTOTday')
lat_tg = data_tg.get('latTOT')
lon_tg = data_tg.get('lonTOT')

# Convert datenum into datetime
epoch_start = pd.Timestamp('1970-01-01')
time_days = np.array(time, dtype='timedelta64[D]')  # Convert 'time' to a timedelta array
time_datetime = epoch_start.to_numpy() + time_days

data_arrays = []

for i in range(lat_tg.size):  # Assuming lat_tg and lon_tg are 1D arrays with length 21
    # Create a DataArray for each station with its corresponding time and SLA values

    station_name = names_tg[0][i]

    da = xr.DataArray(
        data=sla[i, :],  # SLA data for the i-th station
        dims=["time"],
        coords={
            "time": time_datetime[i, :],  # Time data for the i-th station
            "latitude": lat_tg[i],
            "longitude": lon_tg[i]
        },
        name=f"station_{station_name}"
    )
    data_arrays.append(da)

    # Order by longitude from east to west:
    # Pair each station name with its corresponding longitude
    name_longitude_pairs = [(da.name, da.longitude.item()) for da in data_arrays]

    # Sort the pairs by longitude in descending order
    sorted_pairs = sorted(name_longitude_pairs, key=lambda x: x[1], reverse=True)

    # Extract the sorted names
    sorted_names = [name for name, _ in sorted_pairs]


# Setting the names_tg per longitude order from east to west
# Create a new list to store ordered data arrays
ordered_data_arrays = []
ordered_lat = []
ordered_lon = []

# Loop through the sorted station names and get the corresponding DataArray
for name in sorted_names:
  for da in data_arrays:
    if da.name == name:
      ordered_data_arrays.append(da)
      ordered_lat.append(np.float64(da['latitude'].values))
      ordered_lon.append(np.float64(da['longitude'].values))
      break  # Exit the inner loop once the matching DataArray is found

# Replace the original data_arrays list with the ordered version
data_arrays = ordered_data_arrays


# ------------------------ PROCESSING SWOT DATA AROUND TG LOCATIONS --------------------------------------------------
# Choose strategy for handling missing data (average nearby points or closest non-NaN)
# 0: Average nearby points (within radius)
# 1: Retrieve closest non-NaN value (within radius)
strategy = 0

# Radius in km for averaging nearby points
dmedia = 6  # km

# List to store all SWOT time series (one list per tide gauge)
all_swot_timeseries = []

# Loop through all netCDF files in the folder

nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]

for filename in nc_files:
    file_path = os.path.join(folder_path, filename)
    ds = xr.open_dataset(file_path)

    # Extract data from variables
    lon = ds['longitude'].values.flatten()
    lat = ds['latitude'].values.flatten()
    ssh = ds['ssha_noiseless'].values.flatten()
    # ssh = ds['ssha'].values.flatten()

    time_values = ds['time'].values  # Adding a new
    time = np.tile(time_values[:, np.newaxis], (1, 69)).flatten()  # Not efficient

    # Find indices of non-NaN values
    valid_indices = np.where(~np.isnan(ssh))
    lon = lon[valid_indices]
    lat = lat[valid_indices]
    time = time[valid_indices]
    ssh = ssh[valid_indices]

    # Loop through each tide gauge location
    for idx, (gauge_lon, gauge_lat) in enumerate(zip(ordered_lon, ordered_lat)):

        if strategy == 0:  # ------------------------------------------------------------------------------------------
            # Calculate distance for each data point
            distances = haversine(lon, lat, gauge_lon, gauge_lat)

            # Find indices within the specified radius
            in_radius = distances <= dmedia

            # Average nearby SSH values (if any)
            if np.any(in_radius):
                n_idx = sum(in_radius)  # How many values are used for compute the mean value
                ssh_tmp = np.nanmean(ssh[in_radius])
                ssh_serie = ssh_tmp * 100  # Convert to centimeters (cm)
                time_serie = time[in_radius][~np.isnan(time[in_radius])][0]  # Picking the first value of time within the radius

                # Store the latitudes and longitudes of SWOT within the radius
                swot_lat_within_radius = lat[in_radius]
                swot_lon_within_radius = lon[in_radius]

                # Store closes  distance
                min_distance_point = distances[in_radius].min()

            else:
                ssh_serie = np.nan  # No data within radius (remains NaN)
                time_serie = time[in_radius][~np.isnan(time[in_radius])]
                n_idx = np.nan  # 0 points for the average within the radius

                # If there's no SWOT data within the radius, set latitudes and longitudes to None
                swot_lat_within_radius = None
                swot_lon_within_radius = None
                min_distance_point = None

                # print(f"No SWOT data within {dmedia} km radius of tide gauge {sorted_names[idx]}")

            # Create a dictionary to store tide gauge and SWOT data
            selected_data = {
                "station_name": sorted_names[idx],  # Access station name
                "longitude": gauge_lon,  # Longitude of tide gauge
                "latitude": gauge_lat,  # Latitude of tide gauge
                "ssha": ssh_serie,  # Retrieved SSH value
                "ssha_raw": ssh[in_radius],  # Raw SSH values within the radius
                "time": time_serie,
                "n_val": n_idx,  # Number of points for the average within the radius
                "swot_lat_within_radius": swot_lat_within_radius,  # Latitudes of SWOT within the radius
                "swot_lon_within_radius": swot_lon_within_radius,   # Longitudes of SWOT within the radius
                "min_distance": min_distance_point  # Closest distance within the radius
            }

        else:  # ----------------------------------------------------------------------------------------------------
            # Calculate distance for each data point
            distances = haversine(lon, lat, gauge_lon, gauge_lat)

            # Find indices within the specified radius
            in_radius = distances <= dmedia

            # Select closest SSH value within the radius (if any)
            if np.any(in_radius):
                # Find the closest non-NaN index
                closest_idx = np.nanargmin(distances[in_radius])
                distance_point = distances[in_radius][closest_idx]  # Obtain the closest distance
                ssh_serie = ssh[closest_idx] * 100  # Retrieve closest non-NaN SSH and convert to centimeters (cm)
                time_serie = time[closest_idx]

                # Create a dictionary to store selected SWOT data
                selected_data = {
                    "station_name": sorted_names[idx],  # Access station name
                    "longitude": lon[closest_idx],  # Longitude of selected SWOT point
                    "latitude": lat[closest_idx],  # Latitude of selected SWOT point
                    "ssha": ssh_serie,  # Retrieved SSH value
                    "time": time_serie,
                    "n_val": distance_point
                }
        # Append the list of SSH values for all gauges for this file
        all_swot_timeseries.append(selected_data)

# CONVERT TO DATAFRAME for easier managing
df = pd.DataFrame(all_swot_timeseries).dropna(how='any')  # Convert to DataFrame


# Dataframe to store time series data for each station (key: station_name, value[list of SSH, lat, lon and time values])
station_time_series = []

for i in range(0, len(all_swot_timeseries)):  # Loop through each file's data
    file_data = all_swot_timeseries[i]

    station_name = file_data["station_name"]
    ssha_value = file_data["ssha"]
    lon = file_data['longitude'],  # Longitude of selected SWOT point
    lat = file_data['latitude'],  # Latitude of selected SWOT point
    time = file_data['time']
    n_val = file_data['n_val']  # Number of points for the average within the radius or distance of the closest point
    ssh_raw = file_data['ssha_raw']  # Raw SSH values within the radius
    swot_lat_within_radius = file_data['swot_lat_within_radius']  # Latitudes of SWOT within the radius
    swot_lon_within_radius = file_data['swot_lon_within_radius']  # Longitudes of SWOT within the radius
    min_distance = file_data['min_distance']  # Closest distance within the radius
    
    # Append the SSH and time s to the station's time series
    station_time_series.append({'station_name': station_name,
                                'ssha': ssha_value,
                                'latitude': lat,
                                'longitude': lon,
                                'time': time,
                                'num_swot_points': n_val,
                                'raw_ssha': ssh_raw,
                                'swot_lat_within_radius': swot_lat_within_radius,
                                'swot_lon_within_radius': swot_lon_within_radius,
                                'min_distance': min_distance})


df2 = pd.DataFrame(station_time_series)

# ------------------ OBTAINING STATISTICS COMPARISON ---------------------------------------------------

correlations = []
rmsds = []
consistencys = []
var_tg = []
var_SWOT = []
var_diff = []
days_used_per_gauge = []
n_nans_tg = []
min_distances = []


# Convert from Series of ndarrays containing dates to Series of timestamps
# df['time'] = df['time'].apply(lambda x: x[0])

df_tg = []

# Iterate over each DataArray
for da in data_arrays:
    # Extracting values from the DataArray
    time_values = da["time"].values
    sla_values = da.values
    station_name = da.name  # Extracting station name from DataArray name
    latitude = da.latitude.values  
    longitude = da.longitude.values

    # Create a DataFrame for the current DataArray
    tg_data = pd.DataFrame({
        'time': time_values,
        'ssha': sla_values,
        'station': [station_name] * len(time_values),
        'latitude': latitude,
        'longitude': longitude
    })

    # Append the DataFrame to the list
    df_tg.append(tg_data)

# Concatenate all DataFrames into a single DataFrame
df_tg = pd.concat(df_tg, ignore_index=True).dropna(how='any')



# ---------------- MANAGING COMPARISON BETWEEN TG AND SWO ------------------------------------------------

empty_stations = []  # List to store empty stations indexes
lolabox = [1, 8, 35, 45]

# Define the column name for the demeaned values according if the data is filtered or not
# demean = 'demean'
demean = 'demean_filtered'

idx_tg = np.arange(len(sorted_names))
for station in idx_tg:
    try:

        # ssh_swot_station = df[df['station_name'] == sorted_names[station]]
        ssh_swot_station = df[df['station_name'] == sorted_names[station]].copy()  # Corrected warnings
        print(len(ssh_swot_station))
        if ssh_swot_station.empty:
            empty_stations.append(station)
            print(f"No SWOT data found for station {sorted_names[station]}")
            continue

        else:
            ssh_swot_station.sort_values(by='time', inplace=True)

            # tg_station = closest_tg_times[station].dropna(dim='time')
            tg_station = df_tg[df_tg['station'] == sorted_names[station]].copy()

            # Convert time column to numpy datetime64
            ssh_swot_station['time'] = pd.to_datetime(ssh_swot_station['time'])

            tg_station['time'] = pd.to_datetime(tg_station['time'])

            # Cropping data (starting and ending of matching time series)
            tg_start_date = np.datetime64(tg_station['time'].min())  # Start value of TG
        
            swot_end_date = np.datetime64(ssh_swot_station['time'].max())  # End value of SWOT

            swot_ts = ssh_swot_station[(ssh_swot_station['time'] > tg_start_date) & (ssh_swot_station['time'] < swot_end_date)]

            tg_ts = tg_station[(tg_station['time'] > tg_start_date) & (tg_station['time'] < swot_end_date)]


            # SUBSTRACT THE MEAN VALUE OF EACH TIME SERIE FOR COMPARING
            tg_mean = tg_ts['ssha'].mean()
            swot_mean = swot_ts['ssha'].mean()

            tg_ts_demean = tg_ts['ssha'] - tg_mean
            tg_ts['demean'] = tg_ts_demean
            swot_ts_demean = swot_ts['ssha'] - swot_mean
            swot_ts['demean'] = swot_ts_demean

            #  Create a Dataframe with both variables tg_ts and swot_ts
            tg_ts.reset_index(inplace=True)
            swot_ts.reset_index(inplace=True)

            # Filter noise from SWOT data using LOESS filter
            # frac_lowess = 10 / len(swot_ts)  #  10 days window
            frac_loess = 1 / 14  #  7 days window
            # filt_lowess = sm.nonparametric.lowess(swot_ts['demean'], swot_ts['time'], frac=frac_lowess, return_sorted=False)
            filt_loess = loess.loess_smooth_handmade(swot_ts['demean'].values, frac_loess)
            swot_ts['demean_filtered'] = filt_loess

            # Calculate correlation between swot and tg
            correlation = swot_ts[demean].corr(tg_ts['demean'])

            # Calculate RMSD between swot and tg
            rmsd = np.sqrt(np.mean((swot_ts[demean] - tg_ts['demean']) ** 2))

            # Calculate variances of swot and tg
            var_swot_df = swot_ts[demean].var()
            var_tg_df = tg_ts['demean'].var()

            # Calculate the variance of the difference between swot and tg
            var_diff_df = (swot_ts[demean] - tg_ts['demean']).var()

            rmsds.append(rmsd)
            correlations.append(correlation)
            var_tg.append(var_tg_df)
            var_SWOT.append(var_swot_df)
            var_diff.append(var_diff_df)

            # Num days used
            days_used_per_gauge.append(len(swot_ts))

            # Average min distances
            min_distances.append(swot_ts['min_distance'].min())

            # PLOT SERIES TEMPORALES INCLUYENDO GAPS!
            plt.figure(figsize=(10, 6))
            plt.plot(swot_ts['time'], swot_ts[demean], label='SWOT data')
            plt.scatter(swot_ts['time'], swot_ts[demean], label='SWOT data')
            plt.plot(tg_ts['time'], tg_ts['demean'], label='Tide Gauge Data')
            plt.title(f'Station {sorted_names[station]} using radius of {dmedia} km')
            plt.legend()
            plt.xticks(rotation=20)
            plt.yticks(np.arange(-15, 18, 3))
            # plt.xlabel('time')
            plt.grid(True, alpha=0.2)
            plt.ylabel('SSHA (cm)')
            plt.tick_params(axis='both', which='major', labelsize=11)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # Use '%m-%d' for MM-DD format
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))

            # PLOT MAP OF DATA OBTAINED FROM EACH GAUGE!
            fig, ax = plt.subplots(figsize=(10.5, 11), subplot_kw=dict(projection=ccrs.PlateCarree()))

            # Set the extent to focus on the defined lon-lat box
            ax.set_extent(lolabox, crs=ccrs.PlateCarree())

            # Add scatter plot for specific locations
            ax.scatter(tg_ts['longitude'][0], tg_ts['latitude'][0], c='black', marker='o', s=50, transform=ccrs.Geodetic(), label='Tide Gauge')
            ax.scatter(swot_ts['swot_lon_within_radius'][0], swot_ts['swot_lat_within_radius'][0], c='blue', marker='o', s=50, transform=ccrs.Geodetic(), label='SWOT data')

            # Add coastlines and gridlines
            ax.coastlines()
            ax.gridlines(draw_labels=True)

            ax.legend(loc="upper left")
            # ax.title(f'Station {sorted_names[station]} taking radius of {dmedia} km')


    except (KeyError, ValueError):  # Handle cases where station might not exist in df or time has NaNs
        # Print message or log the issue and save the station index for dropping the empty stations
        empty_stations.append(station)
        print(f"Station {sorted_names[station]} not found in SWOT data or time series has NaNs")
        continue  # Skip to the next iteration


n_val = []  # List to store average number of SWOT values per station

# Loop through each station name
for station_name in sorted_names:
    # Filter data for the current station
    station_data = df2[df2['station_name'] == station_name]

    # Calculate the average number of SWOT values used within the radius (if data exists)
    if not station_data.empty:  # Check if DataFrame is empty
        n_val_avg = np.mean(station_data['num_swot_points'])  # Access 'n_val' column directly
    else:
        n_val_avg = np.nan  # Assign NaN for missing data

    n_val.append(round(n_val_avg, 2))  # Round n_val_avg to 2 decimals


# Drop stations variables with no SWOT data for matching with the table
sorted_names_mod = [x for i, x in enumerate(sorted_names) if i not in empty_stations]
ordered_lat_mod = [x for i, x in enumerate(ordered_lat) if i not in empty_stations]
ordered_lon_mod = [x for i, x in enumerate(ordered_lon) if i not in empty_stations] 
n_val = [x for i, x in enumerate(n_val) if i not in empty_stations]

# Create a DataFrame to store all the statistics
table_all = pd.DataFrame({'station': sorted_names_mod,
                          'correlation': correlations,
                          'rmsd': rmsds,
                          'var_TG': var_tg,
                          'var_SWOT': var_SWOT,
                          'var_diff': var_diff,
                          'n_val': n_val,
                          'n_days': days_used_per_gauge,
                          # 'n_nans': [int(x) for x in n_nans_tg],
                          # '%_Gaps': [round(num, 2) for num in percent_nans],
                          'latitude': ordered_lat_mod,
                          'longitude': ordered_lon_mod,
                            'min_distance': min_distances
                          })

    # Dropping wrong tide gauges (errors in tide gauge raw data) or tide gauges with no SWOT data
drop_tg_names = ['station_GL_TS_TG_TamarisTG',
                    'station_GL_TS_TG_BaieDuLazaretTG',
                    'station_GL_TS_TG_PortDeCarroTG',
                    'station_GL_TS_TG_CassisTG',
                    'station_MO_TS_PORTO_CRISTO']
    
table = table_all[~table_all['station'].isin(drop_tg_names)].reset_index(drop=True)# table.to_excel(f'{path}SWOT-TG_comparisons/comparison_swot_tg_{dmedia}km.xlsx', index=False)
