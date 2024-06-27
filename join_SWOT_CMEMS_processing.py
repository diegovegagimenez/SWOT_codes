import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cartopy.crs as ccrs
from shapely.geometry import LineString
import warnings
import cartopy.feature as cfeature
import statsmodels.api as sm  # for LOWESS filter
# Set the warning filter to "ignore" to suppress all warnings
warnings.filterwarnings("ignore")
import loess_smooth_handmade as loess  # for LOESS filter
import math

# ----------------------------------PARAMETERS------------------------------------------------------------------------
# Strategy for selecting CMEMS data points around tide gauge locations
# 0: Average nearby SSH values within the radius
# 1: Select the closest SSH value within the radius
strategy = 0

# Maximum distance from each CMEMS point to the tide gauge location
dmedia = np.arange(10, 15, 5)  # Array of distances from 5 to 110 km in 5 km increments

# Window size for the LOESS filter (in days)
day_window = 7

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

# Function for calculating the Z-scores of a series and removing outliers
def calculate_z_scores(series):
    mean = series.mean()
    std = series.std()
    return (series - mean) / std

threshold_outliers = 3  # Threshold for Z-scores to remove outliers

def process_ts(time_series, product_n):
    """
    Process the given time series by standarizing, removing outliers and smooth the data.
    """
    ts_mean = time_series[product_n].mean()
    ts_demean = time_series[product_n] - ts_mean
    time_series['demean'] = ts_demean
    z_scores = calculate_z_scores(time_series['demean'])
    time_series = time_series[np.abs(z_scores) <= threshold_outliers]

    # Apply a LOESS filter to the time series
    frac_loess = 1 / day_window

    time_series['demean_filtered'] = loess.loess_smooth_handmade(time_series['demean'].values, frac_loess)
    time_series.reset_index(inplace=True)

    return time_series

# Path to the general SWOT data folder -----------------------------------------------------------------------------------
path ='/home/dvega/anaconda3/work/SWOT/'

# ALTIMETRY DATA PRODUCTS PATHS ----------------------------------------------------------------------------------------
# Define the products to process
products = [
    {   # DUACS (SWOT L4) GLOBAL
        'folder_path': f'{path}SWOT_L4/', 
        'plot_path': f'{path}figures/radius_comparisons_SWOT_L4/',
        'product_name': 'DUACS (SWOT_L4)'
    },
    {   # Near real time (NRT) EUROPE
        'folder_path': f'{path}CMEMS_data/SEALEVEL_EUR_PHY_L4_NRT_008_060/daily',
        'plot_path': f'{path}figures/radius_comparisons_CMEMS_EUR_NRT/',
        'product_name': 'CMEMS_NRT_EUR'
    },
    {   # Near real time (NRT) GLOBAL
        'folder_path': f'{path}CMEMS_data/SEALEVEL_GLO_PHY_L4_NRT_008_046',
        'plot_path': f'{path}figures/radius_comparisons_CMEMS_GLO_NRT/',
        'product_name': 'CMEMS_NRT_GLO'
    },
    {   # SWOT L3
        'folder_path' : f'{path}swot_basic_1day/003_016_passv1.0/',  # Folder of 016 and 003 passes
        'plot_path': f'{path}figures/radius_comparisons_rmsdCorrected_7dLoess_SWOT/',
        'product_name': 'SWOT L3'
    }
]
products_names = [product['product_name'] for product in products]

# tide gauge data paths ---------------------------------------------------
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

# Create a DataFrame to store tide gauge data
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
df_tg = pd.concat(df_tg, ignore_index=True)  # Keep NaNs for calculating the percentage of NaNs per station

# ------------------------ PROCESSING CMEMS DATA AROUND TG LOCATIONS --------------------------------------------------

lolabox = [-2, 8, 35, 45]

results_rad_comparison = []  # List to store results for each radius

# Loop through each radius size
for rad in dmedia:
    print(f'Beggining processing radius {rad} km')

    products_timeseries = []  # List to store time series for each product

    # Processing each product for each radius size
    for product in products:
        print(f'Processing product {product["product_name"]}')

        all_altimetry_timeseries = [] # List to store all CMEMS time series (one list per radius size)

        folder_path = product['folder_path']
        plot_path = product['plot_path']
        product_name = product['product_name']

        # Loop through all netCDF files in the product folder
        nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]

        for filename in nc_files:
            file_path = os.path.join(folder_path, filename)

            if product_name == 'SWOT L3':  # ------------------------------------------------------------------------------------------

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

                    # Calculate distance for each data point
                    distances = haversine(lon, lat, gauge_lon, gauge_lat)

                    # Find indices within the specified radius
                    in_radius = distances <= rad

                    # Average nearby SSH values (if any)
                    if np.any(in_radius):

                        ssh_tmp = np.nanmean(ssh[in_radius])
                        ssh_serie = ssh_tmp * 100  # Convert to centimeters (cm)
                        time_serie = time[in_radius][~np.isnan(time[in_radius])][0]  # Picking the first value of time within the radius

                        # Store the latitudes and longitudes of SWOT within the radius
                        lat_within_radius = lat[in_radius]
                        lon_within_radius = lon[in_radius]

                        # Store closest  distance and number of points used for the average
                        min_distance_point = distances[in_radius].min()
                        n_idx = sum(in_radius)  # How many values are used for compute the mean value

                    else:
                        ssh_serie = np.nan  # No data within radius (remains NaN)
                        time_serie = time[in_radius][~np.isnan(time[in_radius])]
                        n_idx = np.nan  # Number of points for the average within the radius
                        min_distance_point = np.nan

                        # If there's no SWOT data within the radius, set latitudes and longitudes to None
                        lat_within_radius = None
                        lon_within_radius = None
                        # print(f"No SWOT data within {dmedia} km radius of tide gauge {sorted_names[idx]}")
        
                    # Create a dictionary to store tide gauge and SWOT data
                    selected_data = {
                        "station": sorted_names[idx],  # Access station name
                        "longitude": gauge_lon,  # Longitude of tide gauge
                        "latitude": gauge_lat,  # Latitude of tide gauge
                        "ssha": ssh_serie,  # Retrieved SSH value
                        "ssha_raw": ssh[in_radius],  # Raw SSH values within the radius
                        "time": time_serie,
                        "n_val": n_idx,  # Number of points for the average within the radius
                        "lat_within_radius": lat_within_radius,  # Latitudes of SWOT within the radius
                        "lon_within_radius": lon_within_radius,   # Longitudes of SWOT within the radius
                        "min_distance": min_distance_point,  # Closest distance within the radius
                        "product": product_name  # Product name 
                        }
                    all_altimetry_timeseries.append(selected_data)
            else:

                # DUACS PRODUCTS ----------------------------------------------------------------------------------------------------
                ds_all = xr.open_dataset(file_path)

                # Select all latitudes and the last 370 longitudes
                ds = ds_all.sel(latitude=slice(lolabox[2], lolabox[3]), longitude=slice(lolabox[0], lolabox[1]))

                time = ds['time'].values[0]
                ssh = ds['sla'][0, :, :]
                lat = ds['latitude']
                lon = ds['longitude']

                # Loop through each tide gauge location
                for tg_idx, (tg_lon, tg_lat) in enumerate(zip(ordered_lon, ordered_lat)):

                    # Calculate distances
                    distances = haversine(lon, lat, tg_lon, tg_lat)

                    mask = distances <= rad  # Create a mask to select points within the radius

                    # Convert the mask to a numpy array to use it for indexing
                    in_radius = np.array(mask)

                    # Apply the mask to the ssh array
                    ssh_serie_within_rad = ssh.values[in_radius]
    
                    # Average nearby SSH values (if any)
                    if np.any(in_radius):

                        # Apply the mask to get boolean indexes for all dimensions
                        ssh_indexes = np.where(in_radius)

                        # Use the boolean indexes to subset ssh, lat, and lon
                        ssh_series = ssh[ssh_indexes].values
                        ssh_serie = np.nanmean(ssh_serie_within_rad) * 100  # Convert to centimeters (cm)

                        lat_within_radius = lat.values[ssh_indexes[0]]
                        lon_within_radius = lon.values[ssh_indexes[1]]

                        n_idx = np.sum(in_radius)  # How many values are used for computing the mean value
                        min_distance = np.min(distances.values[in_radius])  # Store minimum distance within the radius

                    
                    else:
                        ssh_serie = np.nan  # No data within radius (remains NaN)

                    # Create a dictionary to store tide gauge and CMEMS  data
                    selected_data = {
                        "station": sorted_names[tg_idx],  # Access station name
                        "longitude": tg_lon,  # Longitude of tide gauge
                        "latitude": tg_lat,  # Latitude of tide gauge
                        "ssha": ssh_serie,  # Retrieved SSH values
                        "ssha_raw": ssh_series.flatten(),  # Raw SSH values within the radius
                        "time": time,
                        "n_val": n_idx,  # Number of points within the radius
                        "lat_within_radius": lat_within_radius,
                        "lon_within_radius": lon_within_radius,                        
                        "min_distance": min_distance,  # Mean distance between CMEMS and tide gauge
                        "product": product_name  # Product name
                        }
                    # Append the selected data to the list
                    all_altimetry_timeseries.append(selected_data)

        products_timeseries.extend(all_altimetry_timeseries)

    # CONVERT TO DATAFRAME for easier managing 
    prod_df = pd.DataFrame(products_timeseries)
    prod_df['time'] = prod_df['time'].apply(lambda x: pd.NaT if isinstance(x, np.ndarray) and x.size == 0 else x) # Convert empty arrays to NaT

    prod_df_dropna = pd.DataFrame(products_timeseries).dropna(how='any')

    # for i in range(len(sorted_names)):  # CHECKING HOW MANY NANS THERE ARE ACCORDING TO THE RADIUS SIZE
    #     print((df2[df2['station_name'] == sorted_names[i]]['ssha']).isna().sum())


    # # Dropping wrong tide gauges (errors in tide gauge raw data)
    drop_tg_names = ['station_GL_TS_TG_TamarisTG',
                    'station_GL_TS_TG_BaieDuLazaretTG',
                    'station_GL_TS_TG_PortDeCarroTG',
                    'station_GL_TS_TG_CassisTG',
                    'station_MO_TS_TG_PORTO-CRISTO']
    
    prod_df_dropna = prod_df_dropna[~prod_df_dropna['station'].isin(drop_tg_names)].reset_index(drop=True)
    prod_df = prod_df[~prod_df['station'].isin(drop_tg_names)].reset_index(drop=True)
    
    # # Convert from Series of ndarrays containing dates to Series of timestamps
    # prod_df_dropna['time'] = prod_df_dropna['time'].apply(lambda x: x[0])

    # Drop wrong stations from the tide gauge data
    df_tg = df_tg[~df_tg['station'].isin(drop_tg_names)].reset_index(drop=True)
    df_tg.set_index('time', inplace=True)
    df_tg.sort_index(inplace=True)
    df_tg_dropna = df_tg.dropna(how='any')


    # Obtain the time overlapping time index that is not NaNs

    # Droped NaNs----------------------------------------------------------------------------------------------------------------
    prod_df_dropna['time'] = pd.to_datetime(prod_df_dropna['time'])
    prod_df_dropna.sort_values(by='time', inplace=True)
    # Round SWOT time to nearest day
    prod_df_dropna['time'] = prod_df_dropna['time'].dt.floor('d')
    prod_df_dropna.set_index('time', inplace=True)

    # Crop the time series to the overlapping period
    DUACS_SWOT_L4_dropna = prod_df_dropna[prod_df_dropna['product']==products_names[0]]
    CMEMS_NRT_EUR_dropna = prod_df_dropna[prod_df_dropna['product']==products_names[1]]
    CMEMS_NRT_GLO_dropna = prod_df_dropna[prod_df_dropna['product']==products_names[2]]
    SWOT_L3_dropna = prod_df_dropna[prod_df_dropna['product']==products_names[3]]

    min_time = max(DUACS_SWOT_L4_dropna.index.min(), CMEMS_NRT_EUR_dropna.index.min(), CMEMS_NRT_GLO_dropna.index.min(), SWOT_L3_dropna.index.min(), df_tg_dropna.index.min())
    max_time = min(DUACS_SWOT_L4_dropna.index.max(), CMEMS_NRT_EUR_dropna.index.max(), CMEMS_NRT_GLO_dropna.index.max(), SWOT_L3_dropna.index.max(), df_tg_dropna.index.max())
    
    DUACS_SWOT_L4_dropna = DUACS_SWOT_L4_dropna[['station', 'ssha']]
    CMEMS_NRT_EUR_dropna = CMEMS_NRT_EUR_dropna[['station', 'ssha']]
    CMEMS_NRT_GLO_dropna = CMEMS_NRT_GLO_dropna[['station', 'ssha']]
    SWOT_L3_dropna = SWOT_L3_dropna[['station', 'ssha']] 
    df_tg_dropna = df_tg_dropna[['station', 'ssha']]

    DUACS_SWOT_L4_dropna = DUACS_SWOT_L4_dropna.loc[min_time:max_time]
    CMEMS_NRT_EUR_dropna = CMEMS_NRT_EUR_dropna.loc[min_time:max_time]
    CMEMS_NRT_GLO_dropna = CMEMS_NRT_GLO_dropna.loc[min_time:max_time]
    SWOT_L3_dropna = SWOT_L3_dropna.loc[min_time:max_time]
    df_tg_dropna = df_tg_dropna[(df_tg_dropna.index >= min_time) & (df_tg_dropna.index <= max_time)]

    DUACS_SWOT_L4_dropna = DUACS_SWOT_L4_dropna.rename(columns={'ssha': 'DUACS (SWOT_L4)'})
    CMEMS_NRT_EUR_dropna = CMEMS_NRT_EUR_dropna.rename(columns={'ssha': 'CMEMS_NRT_EUR'})
    CMEMS_NRT_GLO_dropna = CMEMS_NRT_GLO_dropna.rename(columns={'ssha': 'CMEMS_NRT_GLO'})
    SWOT_L3_dropna = SWOT_L3_dropna.rename(columns={'ssha': 'SWOT L3'})
    df_tg_dropna = df_tg_dropna.rename(columns={'ssha': 'TG'})

    # Containing NaNs-----------------------------------------------------------------------------------------------------------
    prod_df['time'] = pd.to_datetime(prod_df['time'])
    prod_df.sort_values(by='time', inplace=True)
    # Round SWOT time to nearest day
    prod_df['time'] = prod_df['time'].dt.floor('d')
    prod_df.set_index('time', inplace=True)

    # Crop the time series to the overlapping period
    DUACS_SWOT_L4 = prod_df[prod_df['product']==products_names[0]]
    CMEMS_NRT_EUR = prod_df[prod_df['product']==products_names[1]]
    CMEMS_NRT_GLO = prod_df[prod_df['product']==products_names[2]]
    SWOT_L3 = prod_df[prod_df['product']==products_names[3]]

    DUACS_SWOT_L4 = DUACS_SWOT_L4[['station', 'ssha']]
    CMEMS_NRT_EUR = CMEMS_NRT_EUR[['station', 'ssha']]
    CMEMS_NRT_GLO = CMEMS_NRT_GLO[['station', 'ssha']]
    SWOT_L3 = SWOT_L3[['station', 'ssha']] 
    df_tg = df_tg[['station', 'ssha']]

    DUACS_SWOT_L4 = DUACS_SWOT_L4.loc[min_time:max_time]
    CMEMS_NRT_EUR = CMEMS_NRT_EUR.loc[min_time:max_time]
    CMEMS_NRT_GLO = CMEMS_NRT_GLO.loc[min_time:max_time]
    SWOT_L3 = SWOT_L3.loc[min_time:max_time]
    df_tg = df_tg[(df_tg.index >= min_time) & (df_tg.index <= max_time)]

    DUACS_SWOT_L4 = DUACS_SWOT_L4.rename(columns={'ssha': 'DUACS (SWOT_L4)'})
    CMEMS_NRT_EUR = CMEMS_NRT_EUR.rename(columns={'ssha': 'CMEMS_NRT_EUR'})
    CMEMS_NRT_GLO = CMEMS_NRT_GLO.rename(columns={'ssha': 'CMEMS_NRT_GLO'})
    SWOT_L3 = SWOT_L3.rename(columns={'ssha': 'SWOT L3'})
    df_tg = df_tg.rename(columns={'ssha': 'TG'})

    # MERGE ALL DATAFRAMES-----------------------------------------------------------------------------------------------------
    # Reset index
    df_tg_reset = df_tg.reset_index()
    SWOT_L3_reset = SWOT_L3.reset_index()
    CMEMS_NRT_EUR_reset = CMEMS_NRT_EUR.reset_index()
    CMEMS_NRT_GLO_reset = CMEMS_NRT_GLO.reset_index()
    DUACS_SWOT_L4_reset = DUACS_SWOT_L4.reset_index()

    matched_df = df_tg_reset.merge(SWOT_L3_reset, left_on=['time', 'station'], right_on=['time', 'station'])
    matched_df = matched_df.merge(CMEMS_NRT_EUR_reset, left_on=['time', 'station'], right_on=['time', 'station'])
    matched_df = matched_df.merge(CMEMS_NRT_GLO_reset, left_on=['time', 'station'], right_on=['time', 'station'])
    matched_df = matched_df.merge(DUACS_SWOT_L4_reset, left_on=['time', 'station'], right_on=['time', 'station'])

    # ---------------- MANAGING COMPARISON BETWEEN TG AND SWO ------------------------------------------------
    empty_stations = []
    correlations = []
    rmsds = []
    consistencys = []
    var_tg = []
    var_CMEMS = []
    var_diff = []
    days_used_per_gauge = []
    n_nans_tg = []
    min_distances = []
    nans_percentage = []
    lolabox = [1, 8, 35, 45]

    # Define the column name for the demeaned values according if the data is filtered or not
    # demean = 'demean'
    demean = 'demean_filtered'

    idx_tg = np.arange(len(sorted_names))
    for station in idx_tg:
        try:
            ssh_station = matched_df[matched_df['station'] == sorted_names[station]].copy()  # Corrected warnings
            
            # if ssh_station.empty:  # Check if the DataFrame is empty                
            #     empty_stations.append(station)
            #         # print(f"No CMEMS data found for station {sorted_names[station]}")
            #     continue

            # else:
                # ssh_station.sort_values(by='time', inplace=True) # Sort values by time

                # # tg_station = closest_tg_times[station].dropna(dim='time')
            ssh_station.dropna(how='any', inplace=True)  # Drop NaNs in time and ssha

            # SUBSTRACT THE MEAN VALUE OF EACH TIME SERIE FOR COMPARING
            tg_ts = process_ts(ssh_station[['time', 'TG']], 'TG')
            cmems_eur = process_ts(ssh_station[['time', 'CMEMS_NRT_EUR']], 'CMEMS_NRT_EUR')
            cmems_glo = process_ts(ssh_station[['time', 'CMEMS_NRT_GLO']], 'CMEMS_NRT_GLO')
            swot_l3 = process_ts(ssh_station[['time', 'SWOT L3']], 'SWOT L3')
            duacs_swot_l4 = process_ts(ssh_station[['time', 'DUACS (SWOT_L4)']], 'DUACS (SWOT_L4)')

            # # Drop the stations where the TG stations have more than 20% of NaNs for the period
            # tg_ts_nans = df_tg[df_tg['station'] == sorted_names[station]].copy()
            # tg_ts_nans['time'] = pd.to_datetime(tg_ts_nans['time']) # Convert time to datetime
            # tg_ts_nans.set_index('time', inplace=True)
            # tg_ts_nans = tg_ts_nans[start_date:end_date]
            # tg_ts_nans = tg_ts_nans['ssha'].isna().sum() / len(tg_ts) * 100
            
            # if len(alt_ts) != 0 and tg_ts_nans < 20:

            #     # frac_lowess = day_window / len(cmems_ts)  #  10 days window
            #     frac_loess = 1 / day_window #  7 days window    fc = 1/Scale

            #     # CMEMS
            #     # filt_lowess = sm.nonparametric.lowess(cmems_ts['demean'], cmems_ts['time'], frac=frac_lowess, return_sorted=False)
            #     filt_loess_cmems = loess.loess_smooth_handmade(alt_ts['demean'].values, frac_loess)
            #     alt_ts['demean_filtered'] = filt_loess_cmems

            #     # TGs
            #     # filt_lowess = sm.nonparametric.lowess(tg_ts['demean'], tg_ts['time'], frac=frac_lowess, return_sorted=False)
            #     filt_loess_tg = loess.loess_smooth_handmade(tg_ts['demean'].values, frac_loess)
            #     tg_ts['demean_filtered'] = filt_loess_tg

            #     # Add the stations with less than 20% of NaNs
            #     nans_percentage.append(tg_ts_nans)

            # else:

            #     empty_stations.append(station)
            #     print(f"Station {sorted_names[station]} has no CMEMS data")
            #     print(f"Station {sorted_names[station]} has more than 20% of NaNs")
            #     continue

            # Calculate correlation between cmems and tg
            corr_swot_l3 = swot_l3[demean].corr(tg_ts[demean])
            corr_cmems_eur = cmems_eur[demean].corr(tg_ts[demean])
            corr_cmems_glo = cmems_glo[demean].corr(tg_ts[demean])
            corr_duacs_swot_l4 = duacs_swot_l4[demean].corr(tg_ts[demean])

            # Calculate RMSD between cmems and tg
            rmsd_swot_l3 = np.sqrt(np.mean((swot_l3[demean] - tg_ts[demean]) ** 2))
            rmsd_cmems_eur = np.sqrt(np.mean((cmems_eur[demean] - tg_ts[demean]) ** 2))
            rmsd_cmems_glo = np.sqrt(np.mean((cmems_glo[demean] - tg_ts[demean]) ** 2))
            rmsd_duacs_swot_l4 = np.sqrt(np.mean((duacs_swot_l4[demean] - tg_ts[demean]) ** 2))

            # Calculate variances of products and tg
            var_swot_l3 = swot_l3[demean].var()
            var_cmems_eur = cmems_eur[demean].var()
            var_cmems_glo = cmems_glo[demean].var()
            var_duacs_swot_l4 = duacs_swot_l4[demean].var()
            var_tg = tg_ts[demean].var()

            # Calculate the variance of the difference between cmems and tg
            var_diff_swot_l3 = (swot_l3[demean] - tg_ts[demean]).var()
            var_diff_cmems_eur = (cmems_eur[demean] - tg_ts[demean]).var()
            var_diff_cmems_glo = (cmems_glo[demean] - tg_ts[demean]).var()
            var_diff_duacs_swot_l4 = (duacs_swot_l4[demean] - tg_ts[demean]).var()


            rmsds.append(rmsd)
            correlations.append(correlation)
            var_tg.append(var_tg_df)
            var_CMEMS.append(var_cmems_df)
            var_diff.append(var_diff_df)

            # Num days used
            days_used_per_gauge.append(len(alt_ts))

            # Average min distances
            min_distances.append(alt_ts['min_distance'].min())

            # PLOTTING TIME SERIES
            # plt.figure(figsize=(10, 6))
            # plt.plot(alt_ts['time'], cmems_ts[demean], label=f'{product_name}',  linewidth=3, c='b')
            # plt.plot(cmems_ts['time'], cmems_ts['demean'], label=f'{product_name} unfiltered', linestyle='--', c='b', alpha=0.5)

            # # plt.scatter(cmems_ts['time'], cmems_ts[demean])
            # plt.plot(tg_ts['time'], tg_ts[demean], label='TGs',  linewidth=3, c='g')
            # plt.plot(tg_ts['time'], tg_ts['demean'], label='TGs unfiltered', linestyle='--', c='g', alpha=0.5)

            # # plt.scatter(tg_ts['time'], tg_ts[demean])
            # plt.title(f'{sorted_names[station]}, {rad}km_radius, {day_window}dLoess, {product_name}')
            # plt.legend()
            # plt.xticks(rotation=20)
            # plt.yticks(np.arange(-15, 18, 3))

            # # plt.xlabel('time')
            # plt.grid(True, alpha=0.2)
            # plt.ylabel('SSHA (cm)')
            # plt.tick_params(axis='both', which='major', labelsize=11)
            # plt.text(0.95, 0.1, f'RMSD: {rmsd:.2f} cm', fontsize=12, color='black', 
            #          transform=plt.gca().transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.5))
            # plt.text(0.95, 0.2, f'CORRELATION: {correlation:.2f}', fontsize=12, color='black', 
            #          transform=plt.gca().transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.5))


            # plt.savefig(f'{plot_path}{sorted_names[station]}_{rad}km_{day_window}dLoess_{product_name}.png')


            # MAP PLOT OF CMEMS LOCATIONS OBTAINED FROM EACH GAUGE!
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
            # ax.title(f'Station {sorted_names[station]}')

        except (KeyError, ValueError):  # Handle cases where station might not exist in df or time has NaNs
            # Print message or log the issue and save the station index for dropping the empty stations
            print(f'empty station {station} 3')

            # empty_stations.append(station)
            # print(f"Station {sorted_names[station]} not found in CMEMS data or time series has NaNs")
            continue  # Skip to the next iteration


    n_val = []  # List to store average number of altimetry values per station

    # Loop through each station name
    for station_name in sorted_names:
        # Filter data for the current station
        station_data = prod_df[prod_df['station_name'] == station_name]

        # Calculate the average number of CMEMS values used within the radius (if data exists)
        if not station_data.empty:  # Check if DataFrame is empty
            n_val_avg = np.mean(station_data['n_val'])  # Access 'n_val' column directly
        else:
            n_val_avg = np.nan  # Assign NaN for missing data

        n_val.append(round(n_val_avg, 2))  # Round n_val_avg to 2 decimals


    # Drop stations variables with no CMEMS data for matching with the table
    sorted_names_mod = [x for i, x in enumerate(sorted_names) if i not in empty_stations]
    ordered_lat_mod = [x for i, x in enumerate(ordered_lat) if i not in empty_stations]
    ordered_lon_mod = [x for i, x in enumerate(ordered_lon) if i not in empty_stations] 
    n_val = [x for i, x in enumerate(n_val) if i not in empty_stations]

    table_all = pd.DataFrame({'station': sorted_names_mod,
                            'correlation': correlations,
                            'rmsd': rmsds,
                            'var_TG': var_tg,
                            'var_CMEMS': var_CMEMS,
                            'var_diff': var_diff,
                            'num_cmems_points': n_val,
                            'n_days': days_used_per_gauge,
                            'latitude': ordered_lat_mod,
                            'longitude': ordered_lon_mod,
                            'min_distance': min_distances,
                            'nans_percentage': nans_percentage
                            })


    # Average RMSD values taking in account the non-linear behaviour of the RMSD
    # Delete the wrong rows/tgs from rmsds
    threshold = 10  # RMSD > 5 out

    # Step 1: Square each RMSD value and filter by threshold
    squared_rmsd = [x**2 for x in rmsds if x < threshold]
    
    # Step 2: Compute the mean of the squared RMSD values
    mean_squared_rmsd = sum(squared_rmsd)/ len(rmsds)
    
    # Step 3: Take the square root of the mean
    combined_rmsd = math.sqrt(mean_squared_rmsd)

    results_rad_comparison.append({'radius': rad,
                                'rmsd': combined_rmsd,
                                'n_tg_used': len(table_all),
                                'avg_days_used':np.mean(days_used_per_gauge),
                                'avg_tg_nans': np.mean(nans_percentage),
                                'correlation': np.mean(correlations),
                                'var_tg': np.mean(var_tg),
                                'var_cmems': np.mean(var_CMEMS),
                                'var_diff': np.mean(var_diff),
                                'min_distance': np.mean(min_distances),
                                'n_stations': len(sorted_names_mod)})

results_df = pd.DataFrame(results_rad_comparison)

results_df

