
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cartopy.crs as ccrs
from shapely.geometry import LineString
import cartopy.feature as cfeature
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

cmems_path = f'{path}CMEMS_data/SEALEVEL_EUR_PHY_L4_NRT_008_060/daily'
# cmems_path = f'{path}CMEMS_data/cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D'


data_tg = np.load(f'{path}mareografos/TGresiduals1d_2023_European_Seas_SWOT_FSP.npz')
names_tg = pd.read_csv(f'{path}mareografos/GLOBAL_TGstations_CMEMS_SWOT_FSP_Feb2024', header=None)

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

# ------------------------ PROCESSING CMEMS DATA AROUND TG LOCATIONS --------------------------------------------------

lolabox = [-2, 8, 35, 45]

strategy = 0

# Maximum distance from each CMEMS point to the tide gauge location
# dmedia = 10 # Km
dmedia = np.arange(5, 20, 5)

# Loop through all netCDF files in the folder
nc_files = [f for f in os.listdir(cmems_path) if f.endswith('.nc')]

# List to store distances between CMEMS and tide gauge locations
all_distances = []

# Loop through all netCDF files in the folder
nc_files = [f for f in os.listdir(cmems_path) if f.endswith('.nc')]

results_rad_comparison = []  # List to store results for each radius


for rad in dmedia:
    print(f'Beggining processing radius {rad} km')
    print(f'{sorted_names}')
    # List to store all CMEMS time series (one list per tide gauge)
    all_cmems_timeseries = []

    for filename in nc_files:
        file_path = os.path.join(cmems_path, filename)
        ds_all = xr.open_dataset(file_path)

        # Select all latitudes and the last 370 longitudes
        # ds_subset = ds.sel(longitude=slice(-370, None))
        ds = ds_all.sel(latitude=slice(lolabox[2], lolabox[3]), longitude=slice(lolabox[0], lolabox[1]))

        time = ds['time'].values
        ssh = ds['sla'][0, :, :]
        lat = ds['latitude']
        lon = ds['longitude']

        # Loop through each tide gauge location
        for tg_idx, (tg_lon, tg_lat) in enumerate(zip(ordered_lon, ordered_lat)):

            if strategy == 0:  # ------------------------------------------------------------------------------------------

                # Calculate distances
                distances = haversine(lon, lat, tg_lon, tg_lat)

                mask = distances <= rad

                # Convert the mask to a numpy array to use it for indexing
                mask_array = np.array(mask)

                # Apply the mask to the ssh array
                ssh_serie_within_rad = ssh.values[mask_array]
                mean_distance = np.mean(distances.values[mask_array])

                # Apply the mask to get boolean indexes for all dimensions
                ssh_indexes = np.where(mask_array)

                # Use the boolean indexes to subset ssh, lat, and lon
                ssh_series = ssh[ssh_indexes]
                cmems_lat_within_radius = lat.values[ssh_indexes[0]]
                cmems_lon_within_radius = lon.values[ssh_indexes[1]]

                # Average nearby SSH values (if any)
                if len(ssh_serie_within_rad) > 0:
                    n_idx = len(ssh_serie_within_rad)  # How many values are used for computing the mean value
                    ssh_serie = np.nanmean(ssh_serie_within_rad) * 100  # Convert to centimeters (cm)
                else:
                    ssh_serie = np.nan  # No data within radius (remains NaN)

                # Create a dictionary to store tide gauge and CMEMS  data
                selected_data = {
                    "station_name": sorted_names[tg_idx],  # Access station name
                    "longitude": tg_lon,  # Longitude of tide gauge
                    "latitude": tg_lat,  # Latitude of tide gauge
                    "num_cmems_points": np.sum(mask),  # Number of CMEMS points within the radius
                    "ssha": ssh_serie,  # Retrieved SSH valueu
                    "time": time,
                    "mean_distance": mean_distance,  # Mean distance between CMEMS and tide gauge
                    "cmems_lat_within_radius": cmems_lat_within_radius,
                    "cmems_lon_within_radius": cmems_lon_within_radius
                    }
                
            else:  # ----------------------------------------------------------------------------------------------------
                # Calculate distance for each data point
                distances = haversine(lon, lat, tg_lon, tg_lat)

                # Find indices within the specified radius
                mask = distances <= rad

                # Select closest SSH value within the radius (if any)
                if np.any(mask):
                    # Find the closest non-NaN index
                    closest_idx = np.nanargmin(distances[mask])
                    distance_point = distances[mask][closest_idx]  # Obtain the closest distance
                    ssh_serie = ssh[closest_idx] * 100  # Retrieve closest non-NaN SSH and convert to centimeters (cm)
                    time_serie = time[closest_idx]

                    # Create a dictionary to store selected CMEMS data
                    selected_data = {
                        "station_name": sorted_names[tg_idx],  # Access station name
                        "longitude": lon[closest_idx],  # Longitude of selected CMEMS point
                        "latitude": lat[closest_idx],  # Latitude of selected CMEMS point
                        "ssha": ssh_serie,  # Retrieved SSH value
                        "time": time_serie,
                        "mean_distance": distance_point
                    }

            # Append the selected data to the list
            all_cmems_timeseries.append(selected_data)

    # Dataframe to store time series data for each station
    station_time_series = []

    for i in range(0, len(all_cmems_timeseries)):  # Loop through each file's data
        file_data = all_cmems_timeseries[i]

        station_name = file_data["station_name"]
        ssha_value = file_data["ssha"]
        lon = file_data['longitude']  # Longitude of selected CMEMS point
        lat = file_data['latitude']  # Latitude of selected CMEMS point
        time = file_data['time']
        num_cmems_points = file_data['num_cmems_points']  # Number of points for the average within the radius or distance of the closest point
        cmems_lat_within_radius = file_data['cmems_lat_within_radius']  # Latitude of CMEMS points within the radius
        cmems_lon_within_radius = file_data['cmems_lon_within_radius']  # Longitude of CMEMS points within the radius

        # Append the SSH and time s to the station's time series
        station_time_series.append({'station_name': station_name,
                                    'ssha': ssha_value,
                                    'latitude': lat,
                                    'longitude': lon,
                                    'time': time,
                                    'num_cmems_points': num_cmems_points,
                                    'cmems_lat_within_radius': cmems_lat_within_radius,
                                    'cmems_lon_within_radius': cmems_lon_within_radius})

    df2 = pd.DataFrame(station_time_series)

    # for i in range(len(sorted_names)):  # CHECKING HOW MANY NANS THERE ARE ACCORDING TO THE RADIUS SIZE
    #     print((df2[df2['station_name'] == sorted_names[i]]['ssha']).isna().sum())

    correlations = []
    rmsds = []
    consistencys = []
    var_tg = []
    var_CMEMS = []
    var_diff = []
    days_used_per_gauge = []
    n_nans_tg = []
            
    # CONVERT TO DATAFRAME for easier managing
    df = pd.DataFrame(all_cmems_timeseries).dropna(how='any')  # Convert to DataFrame

    # Convert from Series of ndarrays containing dates to Series of timestamps
    df['time'] = df['time'].apply(lambda x: x[0])

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
    empty_stations = []
    lolabox = [1, 8, 35, 45]

    idx_tg = np.arange(len(sorted_names))
    for station in idx_tg:
        try:
            # ssh_cmems_station = df[df['station_name'] == sorted_names[station]]
            ssh_cmems_station = df[df['station_name'] == sorted_names[station]].copy()  # Corrected warnings
            ssh_cmems_station.sort_values(by='time', inplace=True) # Sort values by time

            # tg_station = closest_tg_times[station].dropna(dim='time')
            tg_station = df_tg[df_tg['station'] == sorted_names[station]].copy()

            # Convert time column to numpy datetime64
            # ssh_cmems_station['time'] = pd.to_datetime(ssh_cmems_station['time'])

            tg_station['time'] = pd.to_datetime(tg_station['time'])

            # Cropping data (starting and ending of matching time series)
            tg_start_date = np.datetime64(tg_station['time'].min())  # Start value of TG
            tg_end_date = np.datetime64(tg_station['time'].max())  # End value of CMEMS

            cmems_ts = ssh_cmems_station[
                (ssh_cmems_station['time'] > tg_start_date) & (ssh_cmems_station['time'] < tg_end_date)]

            tg_ts = tg_station[(tg_station['time'] > tg_start_date) & (tg_station['time'] < tg_end_date)]

            # SUBSTRACT THE MEAN VALUE OF EACH TIME SERIE FOR COMPARING
            tg_mean = tg_ts['ssha'].mean()
            cmems_mean = cmems_ts['ssha'].mean()

            tg_ts_demean = tg_ts['ssha'] - tg_mean
            tg_ts['demean'] = tg_ts_demean
            cmems_ts_demean = cmems_ts['ssha'] - cmems_mean
            cmems_ts['demean'] = cmems_ts_demean

            #  Create a Dataframe with both variables tg_ts and cmems_ts
            tg_ts.reset_index(inplace=True)
            cmems_ts.reset_index(inplace=True)

            # Calculate correlation between cmems and tg
            correlation = cmems_ts['demean'].corr(tg_ts['demean'])

            # Calculate RMSD between cmems and tg
            rmsd = np.sqrt(np.mean((cmems_ts['demean'] - tg_ts['demean']) ** 2))

            # Calculate variances of cmems and tg
            var_cmems_df = cmems_ts['demean'].var()
            var_tg_df = tg_ts['demean'].var()

            # Calculate the variance of the difference between cmems and tg
            var_diff_df = (cmems_ts['demean'] - tg_ts['demean']).var()

            rmsds.append(rmsd)
            correlations.append(correlation)
            var_tg.append(var_tg_df)
            var_CMEMS.append(var_cmems_df)
            var_diff.append(var_diff_df)

            # Num days used
            days_used_per_gauge.append(len(cmems_ts))

            # # PLOTTING TIME SERIES
            # plt.figure(figsize=(10, 6))
            # plt.plot(cmems_ts['time'], cmems_ts['demean'], label='CMEMS data')
            # plt.scatter(cmems_ts['time'], cmems_ts['demean'])
            # plt.plot(tg_ts['time'], tg_ts['demean'], label='Tide Gauge Data')
            # plt.scatter(tg_ts['time'], tg_ts['demean'])
            # plt.title(f'Station {sorted_names[station]} using {dmedia} km radius')
            # plt.legend()
            # plt.xticks(rotation=20)
            # # plt.xlabel('time')
            # plt.grid(True, alpha=0.2)
            # plt.ylabel('SSHA (m)')
            # plt.tick_params(axis='both', which='major', labelsize=11)

            # # MAP PLOT OF CMEMS LOCATIONS OBTAINED FROM EACH GAUGE!
            # fig, ax = plt.subplots(figsize=(10.5, 11), subplot_kw=dict(projection=ccrs.PlateCarree()))

            # # Set the extent to focus on the defined lon-lat box
            # ax.set_extent(lolabox, crs=ccrs.PlateCarree())

            # # Add scatter plot for specific locations
            # ax.scatter(tg_ts['longitude'][0], tg_ts['latitude'][0], c='black', marker='o', s=50, transform=ccrs.Geodetic(), label='Tide Gauge')
            # ax.scatter(cmems_ts['cmems_lon_within_radius'][0], cmems_ts['cmems_lat_within_radius'][0], c='blue', marker='o', s=50, transform=ccrs.Geodetic(), label='CMEMS data')

            # # Add coastlines and gridlines
            # ax.coastlines()
            # ax.gridlines(draw_labels=True)

            # ax.legend(loc="upper left")
            # # ax.title(f'Station {sorted_names[station]}')

        except (KeyError, ValueError):  # Handle cases where station might not exist in df or time has NaNs
            # Print message or log the issue and save the station index for dropping the empty stations
            empty_stations.append(station)
            print(f"Station {sorted_names[station]} not found in CMEMS data or time series has NaNs")
            continue  # Skip to the next iteration


    n_val = []  # List to store average number of CMEMS values per station

    # Loop through each station name
    for station_name in sorted_names:
        # Filter data for the current station
        station_data = df2[df2['station_name'] == station_name]

        # Calculate the average number of CMEMS values used within the radius (if data exists)
        if not station_data.empty:  # Check if DataFrame is empty
            n_val_avg = np.mean(station_data['num_cmems_points'])  # Access 'n_val' column directly
        else:
            n_val_avg = np.nan  # Assign NaN for missing data

        n_val.append(round(n_val_avg, 2))  # Round n_val_avg to 2 decimals


    table_all = pd.DataFrame({'station': sorted_names,
                            'correlation': correlations,
                            'rmds': rmsds,
                            'var_TG': var_tg,
                            'var_CMEMS': var_CMEMS,
                            'var_diff': var_diff,
                            'num_cmems_points': n_val,
                            'n_days': days_used_per_gauge,
                            # 'n_nans': [int(x) for x in n_nans_tg],
                            # '%_Gaps': [round(num, 2) for num in percent_nans],
                            'latitude': ordered_lat,
                            'longitude': ordered_lon
                            })

    # TG with problems
    drop_tg = [5, 7, 11, 14]

    table = table_all.drop(drop_tg)
    table.reset_index(inplace=True)
    # table.to_excel(f'{path}Figures/CMEMS/tablas_CMEMS_reprocessed/comparison_cmems_tg_{dmedia}rad_nrt.xlsx', index=False)
    results_rad_comparison.append({'radius': rad, 'rmsd': table['rmds'].mean(), 'n_tg_used': len(table.dropna(subset=['rmds']))})
    print(f'results for radius {rad} km added to the list')
    print(table)

results_df = pd.DataFrame(results_rad_comparison)
results_df

