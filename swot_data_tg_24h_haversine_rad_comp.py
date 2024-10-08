import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import os
import cartopy.crs as ccrs
import warnings
import cartopy.feature as cfeature
from tqdm import tqdm
import statsmodels.api as sm  # for LOWESS filter
# Set the warning filter to "ignore" to suppress all warnings
warnings.filterwarnings("ignore")
import loess_smooth_handmade as loess  # for LOESS filter
import math
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------PARAMETERS------------------------------------------------------------------------
# Strategy for selecting CMEMS data points around tide gauge locations
# 0: Average nearby SSH values within the radius
# 1: Select the closest SSH value within the radius
strategy = 0

# Maximum distance from each CMEMS point to the tide gauge location
dmedia = np.arange(100, 101, 1)  # Array of distances from 5 to 110 km in 5 km increments

# Window size for the LOESS filter (in days)
day_window = 7

# % max of NaNs to drop stations 90-(90*0.2) = 72
min_valid_vals = 70

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

threshold_outliers = 5  # Threshold for Z-scores to remove outliers

def process_ts(time_series, product_n):
    """
    Process the given time series by standarizing, removing outliers and smooth the data.
    """
    ts_mean = time_series[product_n].mean()
    ts_demean = time_series[product_n] - ts_mean
    time_series['demean'] = ts_demean
    # z_scores = calculate_z_scores(time_series['demean'])
    # time_series = time_series[np.abs(z_scores) <= threshold_outliers]

    # Apply a LOESS filter to the time series
    frac_loess = 1 / day_window

    time_series['demean_filtered'] = loess.loess_smooth_handmade(time_series['demean'].values, frac_loess)

    if len(time_series) < min_valid_vals:
        empty_stations.append(station)
        print(f"Station {sorted_names[station]} has more than 20% of NaNs")
    
    # Drop time series which have more than 20% of NaNs
    return time_series if len(time_series) >= min_valid_vals else None

def compute_combined_rmsd(rmsds, threshold):

    """Calculate the mean of the RMSD's values that are below a threshold 
    taking in account the non-linearity of the RMSD values.""" 

    # Step 1: Square each RMSD value and filter by threshold
    squared_rmsd = [x**2 for x in rmsds if x < threshold]
    
    # Step 2: Compute the mean of the squared RMSD values
    mean_squared_rmsd = sum(squared_rmsd)/ len(rmsds)
    
    # Step 3: Take the square root of the mean
    combined_rmsd = math.sqrt(mean_squared_rmsd)

    return combined_rmsd

# Function to plot the time series of a tide gauge and a altimetry products
def plot_time_series(alt_ts, tg_ts, rmsd, correlation, product_name, station, rad, plot_path):
    plt.figure(figsize=(8, 4))
    
    demean = 'demean_filtered'
    
    plt.plot(alt_ts['time'], alt_ts[demean], label=f'{product_name}', linewidth=3, color='b')
    plt.plot(alt_ts['time'], alt_ts['demean'], label=f'{product_name} unfiltered', linestyle='--', color='b', alpha=0.5)
    
    plt.plot(tg_ts['time'], tg_ts[demean], label='TGs', linewidth=3, color='g')
    plt.plot(tg_ts['time'], tg_ts['demean'], label='TGs unfiltered', linestyle='--', color='g', alpha=0.5)
    
    plt.title(f'{sorted_names[station]}, {rad}km_radius, {day_window}dLoess, {product_name}', fontsize=12)
    plt.legend(fontsize=8)
    plt.xticks(rotation=20)
    plt.yticks(np.arange(-15, 18, 3))
    plt.grid(True, alpha=0.2)
    plt.ylabel('SSHA (cm)', fontsize=12)

    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.text(0.95, 0.1, f'RMSD: {rmsd:.2f} cm', fontsize=12, color='black', 
             transform=plt.gca().transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.95, 0.2, f'CORRELATION: {correlation:.2f}', fontsize=12, color='black', 
             transform=plt.gca().transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(f'{plot_path}{sorted_names[station]}_{rad}km_{day_window}dLoess_{product_name}.png')
    plt.show()

# Function to calculate the errors of combined RMSD values
def bootstrap_rmsd(data, num_bootstrap=1000):
    bootstrap_samples = np.random.choice(data, (num_bootstrap, len(data)), replace=True)
    bootstrap_rmsd = np.mean(bootstrap_samples, axis=1)
    return bootstrap_rmsd


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
        'product_name': 'SWOT L3'},

    # {    # SWOT L3 UNSMOOTHED
    #     'folder_path' : f'{path}ftp_data/unsmoothed/',  # Folder of 016 and 003 unsmoothed passes
    #     'plot_path': f'{path}figures/radius_comparisons_rmsdCorrected_7dLoess_SWOT_unsmoothed/',
    #     'product_name': 'SWOT L3 unsmoothed'
    # }   
    ]
products_names = [product['product_name'] for product in products]

# tide gauge data paths ---------------------------------------------------
data_tg = np.load(f'{path}mareografos/TGresiduals1d_2023_European_Seas_SWOT_FSP.npz')
names_tg = pd.read_csv(f'{path}mareografos/GLOBAL_TGstations_CMEMS_SWOT_FSP_Feb2024', header=None)


# # Dropping wrong tide gauges (errors in tide gauge raw data)
drop_tg_names = ['station_GL_TS_TG_TamarisTG',
                'station_GL_TS_TG_BaieDuLazaretTG',
                'station_GL_TS_TG_PortDeCarroTG',
                'station_GL_TS_TG_CassisTG',
                'station_MO_TS_TG_PORTO-CRISTO']
    
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

sorted_names_short = [
    "Porquerolles",
    "La Capte",
    "Les Oursinieres",
    "Saint Louis Mourillon",
    "Port De St.Elme",
    "Bregaillon",
    "Le Brusc",
    "La Ciotat",
    "Port De La Redonne",
    "Mahon",
    "Palma de Mallorca",
    "Barcelona",
    "Tarragona"
]


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

# Drop wrong stations from the tide gauge data
df_tg = df_tg[~df_tg['station'].isin(drop_tg_names)].reset_index(drop=True)
df_tg.set_index('time', inplace=True)
df_tg.sort_index(inplace=True)
df_tg_dropna = df_tg.dropna(how='any')

# ------------------------ PROCESSING CMEMS DATA AROUND TG LOCATIONS --------------------------------------------------

lolabox = [-2, 8, 35, 45]

results_rad_comparison = []  # List to store results for each radius

# Create empty lists to store errors for each radius
errors_swot_l3 = []
errors_cmems_eur = []
errors_cmems_glo = []
errors_duacs_swot_l4 = [] 

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

        for filename in tqdm(nc_files):
            file_path = os.path.join(folder_path, filename)

            if product_name in ['SWOT L3', 'SWOT L3 unsmoothed']:  # ------------------------------------------------------------------------------------------

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


    prod_df_dropna = prod_df_dropna[~prod_df_dropna['station'].isin(drop_tg_names)].reset_index(drop=True)
    prod_df = prod_df[~prod_df['station'].isin(drop_tg_names)].reset_index(drop=True)
    
    # # Convert from Series of ndarrays containing dates to Series of timestamps
    # prod_df_dropna['time'] = prod_df_dropna['time'].apply(lambda x: x[0])


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
    # SWOT_L3_unsmoothed_dropna = prod_df_dropna[prod_df_dropna['product']==products_names[4]]

    # min_time = max(DUACS_SWOT_L4_dropna.index.min(), CMEMS_NRT_EUR_dropna.index.min(), CMEMS_NRT_GLO_dropna.index.min(), SWOT_L3_dropna.index.min(), SWOT_L3_unsmoothed_dropna.index.min(), df_tg_dropna.index.min())
    # max_time = min(DUACS_SWOT_L4_dropna.index.max(), CMEMS_NRT_EUR_dropna.index.max(), CMEMS_NRT_GLO_dropna.index.max(), SWOT_L3_dropna.index.max(), SWOT_L3_unsmoothed_dropna.index.max(), df_tg_dropna.index.max())
    
    min_time = max(DUACS_SWOT_L4_dropna.index.min(), CMEMS_NRT_EUR_dropna.index.min(), CMEMS_NRT_GLO_dropna.index.min(), SWOT_L3_dropna.index.min(), df_tg_dropna.index.min())
    max_time = min(DUACS_SWOT_L4_dropna.index.max(), CMEMS_NRT_EUR_dropna.index.max(), CMEMS_NRT_GLO_dropna.index.max(), SWOT_L3_dropna.index.max(), df_tg_dropna.index.max())
    
    # Containing NaNs-----------------------------------------------------------------------------------------------------------
    prod_df['time'] = pd.to_datetime(prod_df['time'])
    prod_df.sort_values(by='time', inplace=True)
    # Round SWOT time to nearest day
    prod_df['time'] = prod_df['time'].dt.floor('d')
    prod_df.set_index('time', inplace=True)

    # Crop the time series of each product to the overlapping period
    DUACS_SWOT_L4 = prod_df[prod_df['product']==products_names[0]]
    CMEMS_NRT_EUR = prod_df[prod_df['product']==products_names[1]]
    CMEMS_NRT_GLO = prod_df[prod_df['product']==products_names[2]]
    SWOT_L3 = prod_df[prod_df['product']==products_names[3]]
    # SWOT_L3_unsmoothed = prod_df[prod_df['product']==products_names[4]]

    DUACS_SWOT_L4 = DUACS_SWOT_L4[['station', 'ssha', 'min_distance', 'n_val']]
    CMEMS_NRT_EUR = CMEMS_NRT_EUR[['station', 'ssha', 'min_distance', 'n_val']]
    CMEMS_NRT_GLO = CMEMS_NRT_GLO[['station', 'ssha', 'min_distance', 'n_val']]
    SWOT_L3 = SWOT_L3[['station', 'ssha', 'min_distance', 'n_val']]
    # SWOT_L3_unsmoothed = SWOT_L3_unsmoothed[['station', 'ssha', 'min_distance', 'n_val']]
    df_tg_p1 = df_tg[['station', 'ssha']]

    DUACS_SWOT_L4 = DUACS_SWOT_L4[(DUACS_SWOT_L4.index >= min_time) & (DUACS_SWOT_L4.index <= max_time)]
    CMEMS_NRT_EUR = CMEMS_NRT_EUR[(CMEMS_NRT_EUR.index >= min_time) & (CMEMS_NRT_EUR.index <= max_time)]
    CMEMS_NRT_GLO = CMEMS_NRT_GLO[(CMEMS_NRT_GLO.index >= min_time) & (CMEMS_NRT_GLO.index <= max_time)]
    SWOT_L3 = SWOT_L3[(SWOT_L3.index >= min_time) & (SWOT_L3.index <= max_time)]
    # SWOT_L3_unsmoothed = SWOT_L3_unsmoothed[(SWOT_L3_unsmoothed.index >= min_time) & (SWOT_L3_unsmoothed.index <= max_time)]
    df_tg_p2 = df_tg_p1[(df_tg_p1.index >= min_time) & (df_tg_p1.index <= max_time)]

    DUACS_SWOT_L4 = DUACS_SWOT_L4.rename(columns={'ssha': 'DUACS (SWOT_L4)', 'min_distance':"min_distance_duacs_swot_l4", 'n_val': 'n_val_duacs_swot_l4'})
    CMEMS_NRT_EUR = CMEMS_NRT_EUR.rename(columns={'ssha': 'CMEMS_NRT_EUR', 'min_distance': 'min_distance_cmems_eur', 'n_val': 'n_val_cmems_eur'})
    CMEMS_NRT_GLO = CMEMS_NRT_GLO.rename(columns={'ssha': 'CMEMS_NRT_GLO', 'min_distance': 'min_distance_cmems_glo', 'n_val': 'n_val_cmems_glo'})
    SWOT_L3 = SWOT_L3.rename(columns={'ssha': 'SWOT L3', 'min_distance': 'min_distance_swot_l3', 'n_val': 'n_val_swot_l3'})
    # SWOT_L3_unsmoothed = SWOT_L3_unsmoothed.rename(columns={'ssha': 'SWOT L3 unsmoothed', 'min_distance': 'min_distance_swot_l3_unsmoothed', 'n_val': 'n_val_swot_l3_unsmoothed'})
    df_tg_p3 = df_tg_p2.rename(columns={'ssha': 'TG'})

    # MERGE ALL DATAFRAMES-----------------------------------------------------------------------------------------------------
    # Reset index
    df_tg_reset = df_tg_p3.reset_index()
    SWOT_L3_reset = SWOT_L3.reset_index()
    CMEMS_NRT_EUR_reset = CMEMS_NRT_EUR.reset_index()
    CMEMS_NRT_GLO_reset = CMEMS_NRT_GLO.reset_index()
    DUACS_SWOT_L4_reset = DUACS_SWOT_L4.reset_index()
    # SWOT_L3_unsmoothed_reset = SWOT_L3_unsmoothed.reset_index()

    matched_df = df_tg_reset.merge(SWOT_L3_reset, left_on=['time', 'station'], right_on=['time', 'station'])
    matched_df = matched_df.merge(CMEMS_NRT_EUR_reset, left_on=['time', 'station'], right_on=['time', 'station'])
    matched_df = matched_df.merge(CMEMS_NRT_GLO_reset, left_on=['time', 'station'], right_on=['time', 'station'])
    matched_df = matched_df.merge(DUACS_SWOT_L4_reset, left_on=['time', 'station'], right_on=['time', 'station'])
    # matched_df = matched_df.merge(SWOT_L3_unsmoothed_reset, left_on=['time', 'station'], right_on=['time', 'station'])

    # Drop rows which have more than 20% of valid values  90*0.2 = 18

    # ---------------- MANAGING COMPARISON BETWEEN TG AND SWO ------------------------------------------------
    empty_stations = []

    correlations_swot_l3 = []
    correlations_cmems_eur = []
    correlations_cmems_glo = []
    correlations_duacs_swot_l4 = []
    correlations_swot_l3_unsmoothed = []

    rmsds_swot_l3 = []
    rmsds_cmems_eur = []
    rmsds_cmems_glo = []
    rmsds_duacs_swot_l4 = []
    rmsds_swot_l3_unsmoothed = []

    variances_tg = []
    variances_swot_l3 = []
    variances_cmems_eur = []
    variances_cmems_glo = []
    variances_duacs_swot_l4 = []
    variances_swot_l3_unsmoothed = []

    variances_diff_swot_l3 = []
    variances_diff_cmems_eur = []
    variances_diff_cmems_glo = []
    variances_diff_duacs_swot_l4 = []
    variances_diff_swot_l3_unsmoothed = []
    
    days_used_per_gauge_swot_l3 = []
    days_used_per_gauge_cmems_eur = []
    days_used_per_gauge_cmems_glo = []
    days_used_per_gauge_duacs_swot_l4 = []
    days_used_per_gauge_swot_l3_unsmoothed = []

    min_distances_swot_l3 = []
    min_distances_cmems_eur = []
    min_distances_cmems_glo = []
    min_distances_duacs_swot_l4 = []
    min_distances_swot_l3_unsmoothed = []

    n_val_swot_l3 = []
    n_val_cmems_eur = []
    n_val_cmems_glo = []
    n_val_duacs_swot_l4 = []
    n_val_swot_l3_unsmoothed = []

    lolabox = [1, 8, 35, 45]

    # Define the column name for the demeaned values according if the data is filtered or not
    # demean = 'demean'
    demean = 'demean_filtered'

    idx_tg = np.arange(len(sorted_names))
    for station in idx_tg:
        try:
            ssh_station = matched_df[matched_df['station'] == sorted_names[station]].copy()  # Corrected warnings
            
            if ssh_station.empty:  # Check if the DataFrame is empty                
                empty_stations.append(station)
                print(f"No CMEMS data found for station {sorted_names[station]}")
                continue

            else:
                    # ssh_station.sort_values(by='time', inplace=True) # Sort values by time

                    # # tg_station = closest_tg_times[station].dropna(dim='time')
                ssh_station.dropna(how='any', inplace=True)  # Drop NaNs in time and ssha

                # SUBSTRACT THE MEAN VALUE OF EACH TIME SERIE FOR COMPARING
                tg_ts = process_ts(ssh_station[['time', 'TG']], 'TG')
                cmems_eur = process_ts(ssh_station[['time', 'CMEMS_NRT_EUR', 'min_distance_cmems_eur', 'n_val_cmems_eur']], 'CMEMS_NRT_EUR')
                cmems_glo = process_ts(ssh_station[['time', 'CMEMS_NRT_GLO', 'min_distance_cmems_glo', 'n_val_cmems_glo']], 'CMEMS_NRT_GLO')
                swot_l3 = process_ts(ssh_station[['time', 'SWOT L3', 'min_distance_swot_l3', 'n_val_swot_l3']], 'SWOT L3')
                duacs_swot_l4 = process_ts(ssh_station[['time', 'DUACS (SWOT_L4)', 'min_distance_duacs_swot_l4', 'n_val_duacs_swot_l4']], 'DUACS (SWOT_L4)')
                # swot_l3_unsmoothed = process_ts(ssh_station[['time', 'SWOT L3 unsmoothed', 'min_distance_swot_l3_unsmoothed', 'n_val_swot_l3_unsmoothed']], 'SWOT L3 unsmoothed')

                if tg_ts is None or cmems_eur is None or cmems_glo is None or swot_l3 is None or duacs_swot_l4 is None:
                # if tg_ts is None or cmems_eur is None or cmems_glo is None or swot_l3 is None or duacs_swot_l4 is None or swot_l3_unsmoothed is None:
                    empty_stations.append(station_name)
                    print(f"One or more products missing for station {station_name}")
                    continue

                #     empty_stations.append(station)
                #     print(f"Station {sorted_names[station]} has no CMEMS data")
                #     print(f"Station {sorted_names[station]} has more than 20% of NaNs")
                #     continue

                # Calculate correlation between cmems and tg
                corr_swot_l3 = swot_l3[demean].corr(tg_ts[demean])
                corr_cmems_eur = cmems_eur[demean].corr(tg_ts[demean])
                corr_cmems_glo = cmems_glo[demean].corr(tg_ts[demean])
                corr_duacs_swot_l4 = duacs_swot_l4[demean].corr(tg_ts[demean])
                # corr_swot_l3_unsmoothed = swot_l3_unsmoothed[demean].corr(tg_ts[demean])

                # Calculate RMSD between cmems and tg
                rmsd_swot_l3 = np.sqrt(np.mean((swot_l3[demean] - tg_ts[demean]) ** 2))
                rmsd_cmems_eur = np.sqrt(np.mean((cmems_eur[demean] - tg_ts[demean]) ** 2))
                rmsd_cmems_glo = np.sqrt(np.mean((cmems_glo[demean] - tg_ts[demean]) ** 2))
                rmsd_duacs_swot_l4 = np.sqrt(np.mean((duacs_swot_l4[demean] - tg_ts[demean]) ** 2))
                # rmsd_swot_l3_unsmoothed = np.sqrt(np.mean((swot_l3_unsmoothed[demean] - tg_ts[demean]) ** 2))

                # Calculate variances of products and tg
                var_swot_l3 = swot_l3[demean].var()
                var_cmems_eur = cmems_eur[demean].var()
                var_cmems_glo = cmems_glo[demean].var()
                var_duacs_swot_l4 = duacs_swot_l4[demean].var()
                # var_swot_l3_unsmoothed = swot_l3_unsmoothed[demean].var()
                var_tg = tg_ts[demean].var()

                # Calculate the variance of the difference between cmems and tg
                var_diff_swot_l3 = (swot_l3[demean] - tg_ts[demean]).var()
                var_diff_cmems_eur = (cmems_eur[demean] - tg_ts[demean]).var()
                var_diff_cmems_glo = (cmems_glo[demean] - tg_ts[demean]).var()
                var_diff_duacs_swot_l4 = (duacs_swot_l4[demean] - tg_ts[demean]).var()
                # var_diff_swot_l3_unsmoothed = (swot_l3_unsmoothed[demean] - tg_ts[demean]).var()

                # Append the results to the lists
                rmsds_swot_l3.append(rmsd_swot_l3)
                rmsds_cmems_eur.append(rmsd_cmems_eur)
                rmsds_cmems_glo.append(rmsd_cmems_glo)
                rmsds_duacs_swot_l4.append(rmsd_duacs_swot_l4)
                # rmsds_swot_l3_unsmoothed.append(rmsd_swot_l3_unsmoothed)

                correlations_swot_l3.append(corr_swot_l3)
                correlations_cmems_eur.append(corr_cmems_eur)
                correlations_cmems_glo.append(corr_cmems_glo)
                correlations_duacs_swot_l4.append(corr_duacs_swot_l4)
                # correlations_swot_l3_unsmoothed.append(corr_swot_l3_unsmoothed)

                variances_tg.append(var_tg)
                variances_swot_l3.append(var_swot_l3)
                variances_cmems_eur.append(var_cmems_eur)
                variances_cmems_glo.append(var_cmems_glo)
                variances_duacs_swot_l4.append(var_duacs_swot_l4)
                # variances_swot_l3_unsmoothed.append(var_swot_l3_unsmoothed)

                variances_diff_swot_l3.append(var_diff_swot_l3)
                variances_diff_cmems_eur.append(var_diff_cmems_eur)
                variances_diff_cmems_glo.append(var_diff_cmems_glo)
                variances_diff_duacs_swot_l4.append(var_diff_duacs_swot_l4)
                # variances_diff_swot_l3_unsmoothed.append(var_diff_swot_l3_unsmoothed)

                # Num days used
                days_used_per_gauge_swot_l3.append(len(swot_l3))
                days_used_per_gauge_cmems_eur.append(len(cmems_eur))
                days_used_per_gauge_cmems_glo.append(len(cmems_glo))
                days_used_per_gauge_duacs_swot_l4.append(len(duacs_swot_l4))
                # days_used_per_gauge_swot_l3_unsmoothed.append(len(swot_l3_unsmoothed))

                # Average min distances
                min_distances_swot_l3.append(swot_l3['min_distance_swot_l3'].min())
                min_distances_cmems_eur.append(cmems_eur['min_distance_cmems_eur'].min())
                min_distances_cmems_glo.append(cmems_glo['min_distance_cmems_glo'].min())
                min_distances_duacs_swot_l4.append(duacs_swot_l4['min_distance_duacs_swot_l4'].min())
                # min_distances_swot_l3_unsmoothed.append(swot_l3_unsmoothed['min_distance_swot_l3_unsmoothed'].min())

                # Number of values used for the average
                n_val_swot_l3.append(swot_l3['n_val_swot_l3'].mean())
                n_val_cmems_eur.append(cmems_eur['n_val_cmems_eur'].mean())
                n_val_cmems_glo.append(cmems_glo['n_val_cmems_glo'].mean())
                n_val_duacs_swot_l4.append(duacs_swot_l4['n_val_duacs_swot_l4'].mean())
                # n_val_swot_l3_unsmoothed.append(swot_l3_unsmoothed['n_val_swot_l3_unsmoothed'].mean())

                # PLOTTING TIME SERIES
                plot_time_series(swot_l3, tg_ts, rmsd_swot_l3, corr_swot_l3, product_name='SWOT L3', station=station, rad=rad, plot_path=products[3]['plot_path'])
                plot_time_series(cmems_eur, tg_ts, rmsd_cmems_eur, corr_cmems_eur, product_name='CMEMS NRT EUR', station=station, rad=rad, plot_path=products[1]['plot_path'])
                plot_time_series(cmems_glo, tg_ts, rmsd_cmems_glo, corr_cmems_glo, product_name='CMEMS NRT GLO', station=station, rad=rad,  plot_path=products[2]['plot_path'])
                plot_time_series(duacs_swot_l4, tg_ts, rmsd_duacs_swot_l4, corr_duacs_swot_l4, product_name='DUACS SWOT L4', station=station, rad=rad, plot_path=products[0]['plot_path'])
                # plot_time_series(swot_l3_unsmoothed, tg_ts)

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
            print(f'empty station {station}')

            # empty_stations.append(station)
            # print(f"Station {sorted_names[station]} not found in CMEMS data or time series has NaNs")
            continue  # Skip to the next iteration

    # empty_stations = [5, 7, 11, 14, 17]
    # Drop stations variables with no CMEMS data for matching with the table
    sorted_names_mod = [x for i, x in enumerate(sorted_names) if i not in empty_stations]
    ordered_lat_mod = [x for i, x in enumerate(ordered_lat) if i not in empty_stations]
    ordered_lon_mod = [x for i, x in enumerate(ordered_lon) if i not in empty_stations] 
    # n_val = [x for i, x in enumerate(n_val) if i not in empty_stations]



    table_all_swot_l3 = pd.DataFrame({'station': sorted_names_mod,
                            'correlation_swot_l3': correlations_swot_l3,
                            'rmsd_swot_l3': rmsds_swot_l3,
                            'var_TG': variances_tg,
                            'var_swot_l3': variances_swot_l3,
                            'var_diff_swot_l3': variances_diff_swot_l3,
                            'num_points_swot_l3': n_val_swot_l3,
                            'n_days_swot_l3': days_used_per_gauge_swot_l3,
                            'latitude': ordered_lat_mod,
                            'longitude': ordered_lon_mod,
                            'min_distance_swot_l3': min_distances_swot_l3,
                            # 'nans_percentage': nans_percentage
                            })
    
    
    table_all_cmems_eur = pd.DataFrame({'station': sorted_names_mod,
                            'correlation_cmems_eur': correlations_cmems_eur,
                            'rmsd_cmems_eur': rmsds_cmems_eur,
                            'var_TG': variances_tg,
                            'var_cmems_eur': variances_cmems_eur,
                            'var_diff_cmems_eur': variances_diff_cmems_eur,
                            'num_points_cmems_eur': n_val_cmems_eur,
                            'n_days_cmems_eur': days_used_per_gauge_cmems_eur,
                            'latitude': ordered_lat_mod,
                            'longitude': ordered_lon_mod,
                            'min_distance_cmems_eur': min_distances_cmems_eur,
                            # 'nans_percentage': nans_percentage
                            })
    

    table_all_cmems_glo = pd.DataFrame({'station': sorted_names_mod,
                            'correlation_cmems_glo': correlations_cmems_glo,
                            'rmsd_cmems_glo': rmsds_cmems_glo,
                            'var_TG': variances_tg,
                            'var_cmems_glo': variances_cmems_glo,
                            'var_diff_cmems_glo': variances_diff_cmems_glo,
                            'num_points_cmems_glo': n_val_cmems_glo,
                            'n_days_cmems_glo': days_used_per_gauge_cmems_glo,
                            'latitude': ordered_lat_mod,
                            'longitude': ordered_lon_mod,
                            'min_distance_cmems_glo': min_distances_cmems_glo,
                            # 'nans_percentage': nans_percentage
                            })
    
    
    table_all_duacs_swot_l4 = pd.DataFrame({'station': sorted_names_mod,
                            'correlation_duacs_swot_l4': correlations_duacs_swot_l4,
                            'rmsd_duacs_swot_l4': rmsds_duacs_swot_l4,
                            'var_TG': variances_tg,
                            'var_CMEMS_duacs_swot_l4': variances_duacs_swot_l4,
                            'var_diff_duacs_swot_l4': variances_diff_duacs_swot_l4,
                            'num_points_duacs_swot_l4': n_val_duacs_swot_l4,
                            'n_days_duacs_swot_l4': days_used_per_gauge_duacs_swot_l4,
                            'latitude': ordered_lat_mod,
                            'longitude': ordered_lon_mod,
                            'min_distance_duacs_swot_l4': min_distances_duacs_swot_l4,
                            # 'nans_percentage': nans_percentage
                            })
    
    # table_all_swot_l3_unsmoothed = pd.DataFrame({'station': sorted_names_mod,
    #                         'correlation_swot_l3_unsmoothed': correlations_swot_l3_unsmoothed,
    #                         'rmsd_swot_l3_unsmoothed': rmsds_swot_l3_unsmoothed,
    #                         # 'var_TG': variances_tg,
    #                         'var_swot_l3_unsmoothed': variances_swot_l3_unsmoothed,
    #                         'var_diff_swot_l3_unsmoothed': variances_diff_swot_l3_unsmoothed,
    #                         'num_points_swot_l3_unsmoothed': n_val_swot_l3_unsmoothed,
    #                         'n_days_swot_l3_unsmoothed': days_used_per_gauge_swot_l3_unsmoothed,
    #                         'latitude': ordered_lat_mod,
    #                         'longitude': ordered_lon_mod,
    #                         'min_distance_swot_l3_unsmoothed': min_distances_swot_l3_unsmoothed,
    #                         # 'nans_percentage': nans_percentage
    #                         })

    # table_all_swot_l3.to_excel(f'{path}tables/table_all_swot_l3_{rad}.xlsx')
    # table_all_cmems_eur.to_excel(f'{path}tables/table_all_cmems_eur_{rad}.xlsx')
    # table_all_cmems_glo.to_excel(f'{path}tables/table_all_cmems_glo_{rad}.xlsx')
    # table_all_duacs_swot_l4.to_excel(f'{path}tables/table_all_duacs_swot_l4_{rad}.xlsx')
    # # table_all_swot_l3_unsmoothed.to_excel(f'table_all_swot_l3_unsmoothed_{rad}.xlsx')
    
    # PLOT HISTOGRAMS OF VARIANCES PER STATION TO CHECK THE DISTRIBUTION

    table_all_tg = table_all_swot_l3[['station', 'var_TG']]

    table_all_tg = table_all_duacs_swot_l4.rename(columns={'var_TG': 'var'})  # Use the same name for the TG variances
    table_all_swot_l3 = table_all_swot_l3.rename(columns={'var_swot_l3': 'var', 'rmsd_swot_l3': 'rmsd'})
    table_all_cmems_eur = table_all_cmems_eur.rename(columns={'var_cmems_eur': 'var', 'rmsd_cmems_eur': 'rmsd'})
    table_all_cmems_glo = table_all_cmems_glo.rename(columns={'var_cmems_glo': 'var', 'rmsd_cmems_glo': 'rmsd'})
    table_all_duacs_swot_l4 = table_all_duacs_swot_l4.rename(columns={'var_CMEMS_duacs_swot_l4': 'var', 'rmsd_duacs_swot_l4': 'rmsd'})
    
    table_all_tg['source'] = 'tg'
    table_all_swot_l3['source'] = 'table_all_swot_l3'
    table_all_cmems_eur['source'] = 'table_all_cmems_eur'
    table_all_cmems_glo['source'] = 'table_all_cmems_glo'
    table_all_duacs_swot_l4['source'] = 'table_all_duacs_swot_l4'
    # table_all_swot_l3_unsmoothed['source'] = 'table_all_swot_l3_unsmoothed'

    combined_df = pd.concat([table_all_tg, table_all_swot_l3, table_all_cmems_eur, table_all_cmems_glo, table_all_duacs_swot_l4])
    
    # Pivot the combined dataframe to have sources as columns
    pivot_df_var = combined_df.pivot(index='station', columns='source', values='var')
    pivot_df_var = pivot_df_var.reset_index()
    
    pivot_df_rmsd = combined_df.pivot(index='station', columns='source', values='rmsd')
    pivot_df_rmsd = pivot_df_rmsd.reset_index()

    # PLOTTING HISTOGRAM FOR VARIANCE
    stations = pivot_df_var['station']
    num_stations = len(stations)
    bar_width = 1.5
    space_between_groups = 2  # Adjust this to set the space between groups

# Create a new x array with extra space between groups
    x = np.arange(num_stations) * (5 * bar_width + space_between_groups)

    # Plotting
    fig, ax = plt.subplots()

    # ax.bar(x - 2 * bar_width, pivot_df_var['tg'], width=bar_width, label='TG', color="#9467bd")
    # ax.bar(x - 1 * bar_width, pivot_df_var['table_all_swot_l3'], width=bar_width, label='SWOT L3', color = '#1f77b4')
    # ax.bar(x - 0 * bar_width, pivot_df_var['table_all_duacs_swot_l4'], width=bar_width, label='DUACS_SWOT_L4', color = '#ff7f0e')    
    # ax.bar(x + 1 * bar_width, pivot_df_var['table_all_cmems_glo'], width=bar_width, label='NRT_GLO', color = '#2ca02c')
    # ax.bar(x + 2 * bar_width, pivot_df_var['table_all_cmems_eur'], width=bar_width, label='NRT_EUR', color = '#d62728')
    
    ax.bar(x - 3 * bar_width, pivot_df_var['table_all_swot_l3'], width=bar_width, label='SWOT L3', color='#1f77b4')
    ax.bar(x - 2 * bar_width, pivot_df_var['table_all_duacs_swot_l4'], width=bar_width, label='DUACS_SWOT_L4', color='#ff7f0e')
    ax.bar(x - 1 * bar_width, pivot_df_var['table_all_cmems_glo'], width=bar_width, label='NRT_GLO', color='#2ca02c')
    ax.bar(x - 0 * bar_width, pivot_df_var['table_all_cmems_eur'], width=bar_width, label='NRT_EUR', color='#d62728')
    ax.bar(x + 1 * bar_width, pivot_df_var['tg'], width=bar_width, label='TG', color="#9467bd")
        # Add labels and title
    ax.set_xlabel('Stations', fontsize=12)
    ax.set_ylabel('Variance (cm²)', fontsize=12)
    ax.set_title(f'Variance of altimetry products at {rad} km radius', fontsize=14)
    ax.set_xticks(x)
    # ax.set_xticklabels(stations, rotation=90, fontsize='small')
    ax.set_xticklabels(stations.index, fontsize=12)
    ax.set_yticklabels(stations.index, fontsize=12)

    ax.legend(fontsize='small')
    ax.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.show()

    # plt.savefig(f'{path}histograms/variances_{rad}.png')

    # PLOTTING HISTOGRAM FOR RMSD
    # Define the positions for the bars
    stations = pivot_df_rmsd['station']
    bar_width = 0.2
    x = np.arange(len(stations))
    # Plotting
    fig, ax = plt.subplots()

    ax.bar(x - 1.5 * bar_width, pivot_df_rmsd['table_all_swot_l3'], width=bar_width, label='SWOT L3')
    ax.bar(x - 0.5 * bar_width, pivot_df_rmsd['table_all_duacs_swot_l4'], width=bar_width, label='DUACS_SWOT_L4')    
    ax.bar(x + 0.5 * bar_width, pivot_df_rmsd['table_all_cmems_glo'], width=bar_width, label='NRT_GLO')
    ax.bar(x + 1.5 * bar_width, pivot_df_rmsd['table_all_cmems_eur'], width=bar_width, label='NRT_EUR')
    
    # Add labels and title
    ax.set_xlabel('Stations', fontsize=12)
    ax.set_ylabel('RMSD (cm²)', fontsize=12)
    ax.set_title(f'RMSD of altimetry products at {rad} km radius', fontsize=14)
    ax.set_xticks(x)
    # ax.set_xticklabels(stations, rotation=90, fontsize='small')
    ax.set_xticklabels(stations.index, fontsize=12)
    ax.set_yticklabels(stations.index, fontsize=12)

    ax.legend(fontsize=8)
    ax.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.show()

    # PERCENTAGE OF IMPROVEMENT
    # improvement_cmems_eur = (table_all_swot_l3['rmsd'] - table_all_cmems_eur['rmsd'])/table_all_swot_l3['rmsd']*100
    improvement_swot_l3 = (table_all_cmems_glo['rmsd'] - table_all_swot_l3['rmsd'])/table_all_cmems_glo['rmsd']*100
    improvement_duacs_swot_l4 = (table_all_cmems_glo['rmsd'] - table_all_duacs_swot_l4['rmsd'])/table_all_cmems_glo['rmsd']*100

# Example station names (replace with your actual station names if different)

    # Combine improvements into a DataFrame
    # improvement_df = pd.DataFrame({
    #     'Station': stations,
    #     # 'CMEMS EUR': improvement_cmems_eur.values.flatten(),
    #     'CMEMS GLO': improvement_cmems_glo.values.flatten(),
    #     'DUACS SWOT L4': improvement_duacs_swot_l4.values.flatten()
    # })

    improvement_df = pd.DataFrame({
    'Station': stations,
    # 'CMEMS EUR': improvement_cmems_eur.values.flatten(),
    'SWOT L3': improvement_swot_l3.values.flatten(),
    'DUACS SWOT L4': improvement_duacs_swot_l4.values.flatten()
    })



    # Set 'Station' as the index
    # improvement_df.set_index('Station', inplace=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))  # Adjust the figure size as needed

    # Define bar width and positions
    bar_width = 0.2
    x = np.arange(len(improvement_df))

    # Plot bars for each product
    # ax.bar(x - bar_width, improvement_df['CMEMS EUR'], width=bar_width, label='CMEMS EUR')
    ax.bar(x, improvement_df['SWOT L3'], width=bar_width, label='SWOT L3')
    ax.bar(x + bar_width, improvement_df['DUACS SWOT L4'], width=bar_width, label='DUACS SWOT L4')

    # Add labels and title
    ax.set_xlabel('Stations', fontsize=12)
    ax.set_ylabel('Percentage Improvement (%)', fontsize=12)
    ax.set_title('Percentage Improvement of Products Compared to CMEMS NRT GLO', fontsize=14)
    ax.set_xticks(x)
    # ax.set_xticklabels(improvement_df.index, rotation=90, fontsize='small')
    ax.set_xticklabels(improvement_df.index, fontsize=12)

    # Set fontsize for x-ticks and y-ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.legend(fontsize=12)
    ax.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()
    # Average RMSD values taking in account the non-linear behaviour of the RMSD
    # Delete the wrong rows/tgs from rmsds
    threshold = 10  # RMSD > 5 out

    combined_rmsd_swot_l3 = compute_combined_rmsd(rmsds_swot_l3, threshold)
    combined_rmsd_cmems_eur = compute_combined_rmsd(rmsds_cmems_eur, threshold)
    combined_rmsd_cmems_glo = compute_combined_rmsd(rmsds_cmems_glo, threshold)
    combined_rmsd_duacs_swot_l4 = compute_combined_rmsd(rmsds_duacs_swot_l4, threshold)
    
    # Apply bootstrap method to calculate the confidence interval of RMSD
    rmsds_error_swot_l3 = bootstrap_rmsd(rmsds_swot_l3)
    rmsds_error_cmems_eur = bootstrap_rmsd(rmsds_cmems_eur)
    rmsds_error_cmems_glo = bootstrap_rmsd(rmsds_cmems_glo)
    rmsds_error_duacs_swot_l4 = bootstrap_rmsd(rmsds_duacs_swot_l4)

    errors_swot_l3.append(rmsds_error_swot_l3)
    errors_cmems_eur.append(rmsds_error_cmems_eur)
    errors_cmems_glo.append(rmsds_error_cmems_glo)
    errors_duacs_swot_l4.append(rmsds_error_duacs_swot_l4)

    mean_rmsd_swot_l3 = np.mean(errors_swot_l3)
    mean_rmsd_cmems_eur = np.mean(errors_cmems_eur)
    mean_rmsd_cmems_glo = np.mean(errors_cmems_glo)
    mean_rmsd_duacs_swot_l4 = np.mean(errors_duacs_swot_l4)

    std_error_swot_l3 = np.std(errors_swot_l3)
    std_error_cmems_eur = np.std(errors_cmems_eur)
    std_error_cmems_glo = np.std(errors_cmems_glo)
    std_error_duacs_swot_l4 = np.std(errors_duacs_swot_l4)

    conf_interval_swot_l3 = np.percentile(errors_swot_l3, [2.5, 97.5])
    conf_interval_cmems_eur = np.percentile(errors_cmems_eur, [2.5, 97.5])
    conf_interval_cmems_glo = np.percentile(errors_cmems_glo, [2.5, 97.5])
    conf_interval_duacs_swot_l4 = np.percentile(errors_duacs_swot_l4, [2.5, 97.5])


    results_rad_comparison.append({'radius': rad,
                                'variances_tg': variances_tg,  
                                
                                'rmsd_swot_l3': combined_rmsd_swot_l3,
                                'rmsd_cmems_eur': combined_rmsd_cmems_eur,
                                'rmsd_cmems_glo': combined_rmsd_cmems_glo,
                                'rmsd_duacs_swot_l4': combined_rmsd_duacs_swot_l4,
                                # 'rmsd_swot_l3_unsmoothed': np.mean(rmsds_swot_l3_unsmoothed),

                                'rmsd_error_swot_l3': mean_rmsd_swot_l3,
                                'rmsd_error_cmems_eur': mean_rmsd_cmems_eur,
                                'rmsd_error_cmems_glo': mean_rmsd_cmems_glo,
                                'rmsd_error_duacs_swot_l4': mean_rmsd_duacs_swot_l4,

                                'std_error_swot_l3': std_error_swot_l3,
                                'std_error_cmems_eur': std_error_cmems_eur,
                                'std_error_cmems_glo': std_error_cmems_glo,
                                'std_error_duacs_swot_l4': std_error_duacs_swot_l4,

                                'conf_interval_swot_l3': conf_interval_swot_l3,
                                'conf_interval_cmems_eur': conf_interval_cmems_eur,
                                'conf_interval_cmems_glo': conf_interval_cmems_glo,
                                'conf_interval_duacs_swot_l4': conf_interval_duacs_swot_l4,

                                'n_tg_used_swot_l3': len(table_all_swot_l3),
                                'n_tg_used_cmems_eur': len(table_all_cmems_eur),
                                'n_tg_used_cmems_glo': len(table_all_cmems_glo),
                                'n_tg_used_duacs_swot_l4': len(table_all_duacs_swot_l4),
                                # 'n_tg_used_swot_l3_unsmoothed': len(table_all_swot_l3_unsmoothed),

                                'avg_days_used_swot_l3':np.mean(days_used_per_gauge_swot_l3),
                                'avg_days_used_cmems_eur':np.mean(days_used_per_gauge_cmems_eur),
                                'avg_days_used_cmems_glo':np.mean(days_used_per_gauge_cmems_glo),
                                'avg_days_used_duacs_swot_l4':np.mean(days_used_per_gauge_duacs_swot_l4),
                                'avg_days_used_swot_l3_unsmoothed':np.mean(days_used_per_gauge_swot_l3_unsmoothed),

                                'correlation_swot_l3': np.mean(correlations_swot_l3),
                                'correlation_cmems_eur': np.mean(correlations_cmems_eur),
                                'correlation_cmems_glo': np.mean(correlations_cmems_glo),
                                'correlation_duacs_swot_l4': np.mean(correlations_duacs_swot_l4),
                                # 'correlation_swot_l3_unsmoothed': np.mean(correlations_swot_l3_unsmoothed),

                                'var_diff_swot_l3': np.mean(var_diff_swot_l3),
                                'var_diff_cmems_eur': np.mean(var_diff_cmems_eur),
                                'var_diff_cmems_glo': np.mean(var_diff_cmems_glo),
                                'var_diff_duacs_swot_l4': np.mean(var_diff_duacs_swot_l4),
                                # 'var_diff_swot_l3_unsmoothed': np.mean(var_diff_swot_l3_unsmoothed),

                                'min_distance_swot': np.mean(min_distances_swot_l3),
                                'min_distance_cmems_eur': np.mean(min_distances_cmems_eur),
                                'min_distance_cmems_glo': np.mean(min_distances_cmems_glo),
                                'min_distance_duacs_swot_l4': np.mean(min_distances_duacs_swot_l4),
                                # 'min_distance_swot_l3_unsmoothed': np.mean(min_distances_swot_l3_unsmoothed),

                                'n_val_swot_l3': np.mean(n_val_swot_l3),
                                'n_val_cmems_eur': np.mean(n_val_cmems_eur),
                                'n_val_cmems_glo': np.mean(n_val_cmems_glo),
                                'n_val_duacs_swot_l4': np.mean(n_val_duacs_swot_l4),
                                # 'n_val_swot_l3_unsmoothed': np.mean(n_val_swot_l3_unsmoothed),
                                })

results_df = pd.DataFrame(results_rad_comparison)

results_df

# results_df[['radius', 'rmsd_swot_l3', 'rmsd_cmems_eur', 'rmsd_cmems_glo', 'rmsd_duacs_swot_l4',
#             'correlation_swot_l3', 'correlation_cmems_eur', 'correlation_cmems_glo', 'correlation_duacs_swot_l4' ]]

# results_df.to_excel('results_df_5_prod_n_vals.xlsx')

# plot the results with errors

# Calculate the lower and upper bounds of the confidence intervals
ci_lower_swot_l3 = [ci[0] for ci in results_df['conf_interval_swot_l3']]
ci_upper_swot_l3 = [ci[1] for ci in results_df['conf_interval_swot_l3']]
ci_lower_cmems_eur = [ci[0] for ci in results_df['conf_interval_cmems_eur']]
ci_upper_cmems_eur = [ci[1] for ci in results_df['conf_interval_cmems_eur']]
ci_lower_cmems_glo = [ci[0] for ci in results_df['conf_interval_cmems_glo']]
ci_upper_cmems_glo = [ci[1] for ci in results_df['conf_interval_cmems_glo']]
ci_lower_duacs_swot_l4 = [ci[0] for ci in results_df['conf_interval_duacs_swot_l4']]
ci_upper_duacs_swot_l4 = [ci[1] for ci in results_df['conf_interval_duacs_swot_l4']]

# Calculate the error bars as the difference between the RMSD values and the bounds of the confidence intervals
yerr_lower_swot_l3 = results_df['rmsd_swot_l3'] - np.array(ci_lower_swot_l3)
yerr_upper_swot_l3 = np.array(ci_upper_swot_l3) - results_df['rmsd_swot_l3']
yerr_lower_cmems_eur = results_df['rmsd_cmems_eur'] - np.array(ci_lower_cmems_eur)
yerr_upper_cmems_eur = np.array(ci_upper_cmems_eur) - results_df['rmsd_cmems_eur']
yerr_lower_cmems_glo = results_df['rmsd_cmems_glo'] - np.array(ci_lower_cmems_glo)
yerr_upper_cmems_glo = np.array(ci_upper_cmems_glo) - results_df['rmsd_cmems_glo']
yerr_lower_duacs_swot_l4 = results_df['rmsd_duacs_swot_l4'] - np.array(ci_lower_duacs_swot_l4)
yerr_upper_duacs_swot_l4 = np.array(ci_upper_duacs_swot_l4) - results_df['rmsd_duacs_swot_l4']


# Combine lower and upper errors into a list of pairs
yerr_swot_l3 = [yerr_lower_swot_l3, yerr_upper_swot_l3]
yerr_cmems_eur = [yerr_lower_cmems_eur, yerr_upper_cmems_eur]
yerr_cmems_glo = [yerr_lower_cmems_glo, yerr_upper_cmems_glo]
yerr_duacs_swot_l4 = [yerr_lower_duacs_swot_l4, yerr_upper_duacs_swot_l4]


plt.errorbar(results_df['radius'], results_df['rmsd_swot_l3'], yerr=yerr_swot_l3, fmt='-o', capsize=5,label='SWOT L3')
plt.scatter(results_df['radius'], results_df['rmsd_swot_l3'])

plt.errorbar(results_df['radius'], results_df['rmsd_cmems_eur'], yerr=yerr_cmems_eur, fmt='-o', capsize=5,label='CMEMS NRT EUR')
plt.scatter(results_df['radius'], results_df['rmsd_cmems_eur'])

plt.errorbar(results_df['radius'], results_df['rmsd_cmems_glo'], yerr=yerr_cmems_glo, fmt='-o', capsize=5,label='CMEMS NRT GLO')
plt.scatter(results_df['radius'], results_df['rmsd_cmems_glo'])

plt.errorbar(results_df['radius'], results_df['rmsd_duacs_swot_l4'], yerr=yerr_duacs_swot_l4, fmt='-o', capsize=5,label='DUACS SWOT L4')
plt.scatter(results_df['radius'], results_df['rmsd_duacs_swot_l4'])

plt.xlabel('Radius (km)')
plt.ylabel('RMSD (cm)')
plt.title('RMSD vs Radius')
plt.legend(fontsize=6)
plt.grid(alpha=0.2)
plt.show()
# plt.savefig('rmsd_vs_radius_4_prod_100km_errorbar.png')


# plt.savefig('rmsd_vs_radius_5_prod.png')


# plt.plot(results_df['radius'], results_df['rmsd_swot_l3'], label='SWOT L3')
# plt.scatter(results_df['radius'], results_df['rmsd_swot_l3'])

# plt.plot(results_df['radius'], results_df['rmsd_cmems_eur'], label='CMEMS NRT EUR')
# plt.scatter(results_df['radius'], results_df['rmsd_cmems_eur'])

# plt.plot(results_df['radius'], results_df['rmsd_cmems_glo'], label='CMEMS NRT GLO')
# plt.scatter(results_df['radius'], results_df['rmsd_cmems_glo'])

# plt.plot(results_df['radius'], results_df['rmsd_duacs_swot_l4'], label='DUACS SWOT L4')
# plt.scatter(results_df['radius'], results_df['rmsd_duacs_swot_l4'])

# # plt.plot(results_df['radius'], results_df['rmsd_swot_l3_unsmoothed'], label='SWOT L3 unsmoothed')
# # plt.scatter(results_df['radius'], results_df['rmsd_swot_l3_unsmoothed'])

# plt.xlabel('Radius (km)')
# plt.ylabel('RMSD (cm)')
# plt.title('RMSD vs Radius')
# plt.legend(fontsize='small')
# plt.grid(alpha=0.2)
# plt.show()

# plt.savefig('rmsd_vs_radius_5_prod.png')

# ----------------- CALCULATE PROPORTION BETWEEN RMSD AND VARIANCE --------------------------


# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the variances and RMSDs
variances = {
    'SWOT_L3': variances_swot_l3,
    'CMEMS_NRT_EUR': variances_cmems_eur,
    'CMEMS_NRT_GLO': variances_cmems_glo,
    'DUACS_SWOT_L4': variances_duacs_swot_l4
}

rmsds = {
    'SWOT_L3': rmsds_swot_l3,
    'CMEMS_NRT_EUR': rmsds_cmems_eur,
    'CMEMS_NRT_GLO': rmsds_cmems_glo,
    'DUACS_SWOT_L4': rmsds_duacs_swot_l4
}

# Convert to DataFrames for scaling
var_df = pd.DataFrame(variances)
rmsd_df = pd.DataFrame(rmsds)

# Normalize
var_df_normalized = pd.DataFrame(scaler.fit_transform(var_df), columns=var_df.columns)
rmsd_df_normalized = pd.DataFrame(scaler.fit_transform(rmsd_df), columns=rmsd_df.columns)

# Decide on weights based on the relative importance of variance and RMSD
weights = {
    'variance': 0.5,
    'rmsd': 0.5
}

# Initialize an empty dictionary to hold the composite scores
composite_scores = {}


# ------------------ MEDIA DE TODOS LAS ESTACIONES PARA CADA PRODUCTO ---------------------------------------------------
for product in variances.keys():
    # Weighted sum of normalized variance and RMSD
    composite_scores[product] = (
        weights['variance'] * var_df_normalized[product].mean() +
        weights['rmsd'] * rmsd_df_normalized[product].mean()
    )

# Convert to DataFrame for easy handling
composite_scores_df = pd.DataFrame(list(composite_scores.items()), columns=['Product', 'Composite Score'])


# ------------------ PROPORCION PARA CADA ESTACION Y PRODUCTO --------------------------------------------
# Inicializa un diccionario para almacenar las proporciones
proportions = {}


# Calcula la proporción entre varianza y RMSD para cada producto
for product in variances.keys():
    # Asegúrate de que ambos DataFrames tengan las mismas estaciones
    var_normalized = var_df_normalized[product]
    rmsd_normalized = rmsd_df_normalized[product]
    
    # Calcula la proporción y maneja posibles divisiones por cero
    proportion = np.divide(var_normalized, rmsd_normalized, out=np.zeros_like(var_normalized), where=rmsd_normalized != 0)
    
    # Guarda la proporción en el diccionario
    proportions[product] = proportion

# Convierte el diccionario de proporciones en un DataFrame
proportions_df = pd.DataFrame(proportions, index=var_df_normalized.index)

# ------------ INDICE COMBINADO DE DESEMPEÑO ---------------------------
# ------------ COMBINED PERFORMANCE INDEX --------------------------------------

# Definir los pesos para RMSD y Varianza (ajusta estos valores según tus necesidades)
alpha = 0.5  # Peso para RMSD
beta = 0.5   # Peso para Varianza (inverso)

# Invertir la varianza normalizada
var_df_inverted = 1 / var_df_normalized

# Calcular el Índice Combinado de Desempeño (ICD)
# ICD = alpha * RMSD_normalized + beta * 1/Varianza_normalized
icd_df = alpha * rmsd_df_normalized + beta * var_df_inverted

icd_df = 1/icd_df
# Opcional: Normalizar el ICD para que también esté en un rango entre 0 y 1
scaler_icd = MinMaxScaler()
icd_df_normalized = pd.DataFrame(scaler_icd.fit_transform(icd_df), columns=icd_df.columns)

# Mostrar el DataFrame con los ICD normalizados
print(icd_df_normalized)

import matplotlib.pyplot as plt
import seaborn as sns

# Calcular el ICD para el tamaño de los puntos
# Asumimos que icd_df_normalized ya está calculado
icd_sizes = icd_df_normalized * 1000  # Escalar para que los puntos sean visibles

# Crear un gráfico de dispersión con tamaños y colores diferenciados
plt.figure(figsize=(12, 8))

# Iterar sobre cada estación para añadir colores diferentes
for i, station in enumerate(var_df_normalized.index):
    plt.scatter(rmsd_df_normalized.iloc[i], var_df_normalized.iloc[i], 
                s=icd_sizes.iloc[i], label=station, alpha=0.6, edgecolor='k')

# Añadir líneas de referencia en las medianas
plt.axvline(rmsd_df_normalized.median().median(), color='grey', linestyle='--')
plt.axhline(var_df_normalized.median().median(), color='grey', linestyle='--')

# Etiquetas y título
plt.xlabel('RMSD Normalizado')
plt.ylabel('Varianza Normalizada')
plt.title('Gráfico de Dispersión de RMSD vs Varianza Normalizada por Estación y Producto')

# Leyenda y ajuste
plt.legend(title='Estación', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Mostrar el gráfico
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Suponiendo que var_df_normalized y rmsd_df_normalized ya están disponibles

# Unir los DataFrames normalizados en un solo DataFrame largo (long format) para usar en Seaborn
data = pd.DataFrame({
    'RMSD_Normalizado': rmsd_df_normalized.melt(var_name='Producto', value_name='RMSD_Normalizado')['RMSD_Normalizado'],
    'Varianza_Normalizada': var_df_normalized.melt(var_name='Producto', value_name='Varianza_Normalizada')['Varianza_Normalizada'],
    'Producto': rmsd_df_normalized.columns.repeat(rmsd_df_normalized.shape[0])
})

# Crear el gráfico de dispersión con líneas de regresión para cada producto
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='RMSD_Normalizado', y='Varianza_Normalizada', hue='Producto', s=100, edgecolor='k')

# Añadir líneas de regresión para cada producto
sns.lmplot(data=data, x='RMSD_Normalizado', y='Varianza_Normalizada', hue='Producto', markers='o', 
           aspect=1.5, height=6, ci=None)

# Etiquetas y título
plt.xlabel('RMSD Normalizado')
plt.ylabel('Varianza Normalizada')
plt.title('Gráfico de Dispersión con Líneas de Regresión por Producto')

# Mostrar el gráfico
plt.show()




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize the variances, RMSDs, and correlations
variances = {
    'swot_l3': variances_swot_l3,
    'cmems_eur': variances_cmems_eur,
    'cmems_glo': variances_cmems_glo,
    'duacs_swot_l4': variances_duacs_swot_l4
}

rmsds = {
    'swot_l3': rmsds_swot_l3,
    'cmems_eur': rmsds_cmems_eur,
    'cmems_glo': rmsds_cmems_glo,
    'duacs_swot_l4': rmsds_duacs_swot_l4
}

correlations = {
    'swot_l3': correlations_swot_l3,
    'cmems_eur': correlations_cmems_eur,
    'cmems_glo': correlations_cmems_glo,
    'duacs_swot_l4': correlations_duacs_swot_l4
}

# Convert to DataFrames for scaling
var_df = pd.DataFrame(variances)
rmsd_df = pd.DataFrame(rmsds)
corr_df = pd.DataFrame(correlations)

# Normalize
var_df_normalized = pd.DataFrame(scaler.fit_transform(var_df), columns=var_df.columns)
rmsd_df_normalized = pd.DataFrame(scaler.fit_transform(rmsd_df), columns=rmsd_df.columns)
corr_df_normalized = pd.DataFrame(scaler.fit_transform(corr_df), columns=corr_df.columns)

# Definir los pesos para RMSD, Varianza y Correlación (ajusta estos valores según tus necesidades)
alpha = 0.4  # Peso para RMSD
beta = 0.4   # Peso para Varianza (inverso)
gamma = 0.2  # Peso para Correlación

# Invertir la varianza normalizada
var_df_inverted = 1 / var_df_normalized

# Calcular el Índice Combinado de Desempeño (ICD) incluyendo la correlación
icd_df = alpha * rmsd_df_normalized + beta * var_df_inverted - gamma * corr_df_normalized

# Opcional: Invertir el ICD para que un valor más bajo signifique un mejor desempeño (si es necesario)
icd_df = 1 / icd_df

# Opcional: Normalizar el ICD para que también esté en un rango entre 0 y 1
scaler_icd = MinMaxScaler()
icd_df_normalized = pd.DataFrame(scaler_icd.fit_transform(icd_df), columns=icd_df.columns)

# Mostrar el DataFrame con los ICD normalizados
print(icd_df_normalized)


# Supongamos que icd_df_normalized, rmsd_df_normalized, var_df_normalized, y corr_df_normalized ya están disponibles

# Calcular el ICD para el tamaño de los puntos
icd_sizes = icd_df_normalized * 1000  # Escalar para que los puntos sean visibles

# Crear un gráfico de dispersión con tamaños y colores diferenciados
plt.figure(figsize=(12, 8))

# Iterar sobre cada estación para añadir colores diferentes
for i, station in enumerate(var_df_normalized.index):
    plt.scatter(rmsd_df_normalized.iloc[i], var_df_normalized.iloc[i], 
                s=icd_sizes.iloc[i], c=corr_df_normalized.iloc[i], cmap='viridis', 
                label=station, alpha=0.6, edgecolor='k')

# Añadir líneas de referencia en las medianas
plt.axvline(rmsd_df_normalized.median().median(), color='grey', linestyle='--')
plt.axhline(var_df_normalized.median().median(), color='grey', linestyle='--')

# Etiquetas y título
plt.xlabel('RMSD Normalizado')
plt.ylabel('Varianza Normalizada')
plt.title('Gráfico de Dispersión de RMSD vs Varianza Normalizada por Estación y Producto')

# Añadir barra de color para la correlación
cbar = plt.colorbar()
cbar.set_label('Correlación Normalizada')

# Leyenda y ajuste
plt.legend(title='Estación', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Mostrar el gráfico
plt.show()


# Example DataFrame indices and sizes (adjust according to your actual data)
# Suppose rmsd_df_normalized, var_df_normalized, and icd_sizes are already defined DataFrames

# Crear un gráfico de dispersión con tamaños y colores diferenciados
plt.figure(figsize=(12, 8))

# Iterar sobre cada estación para añadir colores diferentes y números dentro de los círculos
for i, station in enumerate(var_df_normalized.index):
    for product in rmsd_df_normalized.columns:
        x_value = rmsd_df_normalized.loc[station, product]
        y_value = var_df_normalized.loc[station, product]
        size_value = icd_sizes.loc[station, product]
        label_text = str(i + 1)  # Station number
        
        plt.scatter(x_value, y_value, s=size_value, label=station, alpha=0.6, edgecolor='k')

        # Añadir el número de la estación en el centro de cada círculo
        plt.text(x_value, y_value, label_text, color='black', ha='center', va='center', fontsize=9, weight='bold')

# Añadir líneas de referencia en las medianas
plt.axvline(rmsd_df_normalized.median().median(), color='grey', linestyle='--')
plt.axhline(var_df_normalized.median().median(), color='grey', linestyle='--')

# Etiquetas y título
plt.xlabel('RMSD Normalizado')
plt.ylabel('Varianza Normalizada')
plt.title('Gráfico de Dispersión de RMSD vs Varianza Normalizada por Estación y Producto')

# Leyenda y ajuste
plt.legend(title='Estación', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Mostrar el gráfico
plt.show()

# SCATTER PLOT DEL ICD PARA CADA PRODUCTO
product_index = np.arange(0,3,1)

for j in product_index:
    # Create an empty list to store legend handles
    legend_handles = []

    plt.figure(figsize=(12, 8))

    # Iterate over each station
    for i, station in enumerate(var_df_normalized.index):
        rmsd_value = float(rmsd_df_normalized.iloc[i, j])
        var_value = float(var_df_normalized.iloc[i, j])
        size_value = float(icd_sizes.iloc[i, j])

        # Scatter plot with station number inside
        scatter = plt.scatter(rmsd_value, var_value, 
                            s=size_value, edgecolor='k', label=station)

        # Add station number inside the circle
        plt.text(rmsd_value, var_value, 
                str(i), fontsize=8, ha='center', va='center', color='white')

        # Append the handle for the legend with a fixed size marker
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=scatter.get_facecolor()[0], 
                                        markeredgecolor='k', markersize=10, label=station))

    # Add reference lines at the medians
    plt.axvline(rmsd_df_normalized.iloc[:, j].median(), color='grey', linestyle='--')
    plt.axhline(var_df_normalized.iloc[:, j].median(), color='grey', linestyle='--')

    # Labels and title
    plt.xlabel('RMSD Normalized')
    plt.ylabel('Variance Normalized')
    plt.title(f'Scatter Plot of Normalized RMSD vs Variance for Product {rmsd_df_normalized.columns[j]}')

    # Customize the legend to have circles of the same size
    plt.legend(handles=legend_handles, title='Station', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True)

    # Show the plot
    plt.show()

print(yerr_cmems_glo[1].mean() + yerr_cmems_glo[0].mean(), yerr_cmems_eur[1].mean() + yerr_cmems_eur[0].mean(), yerr_swot_l3[1].mean()+yerr_swot_l3[0].mean(), yerr_duacs_swot_l4[1].mean() + yerr_duacs_swot_l4[0].mean())




# ALL STADISTICCS NORMALIZED TOGETHER FOR ALL THE STATIONS
# Assuming variances, rmsds, and correlations dictionaries are already defined
# Normalize the variances and RMSDs
variances = {
    'SWOT_L3': variances_swot_l3,
    'CMEMS_NRT_EUR': variances_cmems_eur,
    'CMEMS_NRT_GLO': variances_cmems_glo,
    'DUACS_SWOT_L4': variances_duacs_swot_l4
}

rmsds = {
    'SWOT_L3': rmsds_swot_l3,
    'CMEMS_NRT_EUR': rmsds_cmems_eur,
    'CMEMS_NRT_GLO': rmsds_cmems_glo,
    'DUACS_SWOT_L4': rmsds_duacs_swot_l4
}


correlations = {
    'swot_l3': correlations_swot_l3,
    'cmems_eur': correlations_cmems_eur,
    'cmems_glo': correlations_cmems_glo,
    'duacs_swot_l4': correlations_duacs_swot_l4
}

# Convert to DataFrames for scaling
var_df = pd.DataFrame(variances)
rmsd_df = pd.DataFrame(rmsds)
corr_df = pd.DataFrame(correlations)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize across all columns (all products) together for each statistic
var_df_normalized = pd.DataFrame(scaler.fit_transform(var_df.values.reshape(-1, 1)).reshape(var_df.shape),
                                 columns=var_df.columns, index=var_df.index)

rmsd_df_normalized = pd.DataFrame(scaler.fit_transform(rmsd_df.values.reshape(-1, 1)).reshape(rmsd_df.shape),
                                  columns=rmsd_df.columns, index=rmsd_df.index)

# Define weights for RMSD, Variance, and Correlation
alpha = 0.5  # Weight for RMSD
beta = 0.5   # Weight for Variance (inverted)

# Invert the normalized variance
var_df_inverted = 1 / var_df_normalized

# Calculate the Combined Performance Index (ICD) including correlation
icd_df = alpha * rmsd_df_normalized + beta * var_df_inverted 

# Optional: Invert ICD to make lower values better (if needed)
icd_df = 1 / icd_df

# Optional: Normalize the ICD to a range between 0 and 1
scaler_icd = MinMaxScaler()
icd_df_normalized = pd.DataFrame(scaler_icd.fit_transform(icd_df), columns=icd_df.columns, index=icd_df.index)

# Display the normalized ICD DataFrame
print(icd_df_normalized)

# Calculate ICD for point sizes
icd_sizes = icd_df_normalized * 1000  # Scale for visibility

# Create a scatter plot with differentiated sizes and colors
plt.figure(figsize=(12, 8))

# Iterate over each station to add different colors
for i, station in enumerate(var_df_normalized.index):
    plt.scatter(rmsd_df_normalized.iloc[i], var_df_normalized.iloc[i], 
                s=icd_sizes.iloc[i], c=corr_df_normalized.iloc[i], cmap='viridis', 
                label=station, alpha=0.6, edgecolor='k')

# Add reference lines at the medians
plt.axvline(rmsd_df_normalized.median().median(), color='grey', linestyle='--')
plt.axhline(var_df_normalized.median().median(), color='grey', linestyle='--')

# Labels and title
plt.xlabel('Normalized RMSD')
plt.ylabel('Normalized Variance')
plt.title('Scatter Plot of RMSD vs Normalized Variance by Station and Product')

# Add a colorbar for correlation
cbar = plt.colorbar()
cbar.set_label('Normalized Correlation')

# Legend and adjustments
plt.legend(title='Station', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Show the plot
plt.show()



from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Assuming variances, rmsds, and correlations dictionaries are already defined
# Normalize the variances and RMSDs
variances = {
    'SWOT_L3': variances_swot_l3,
    'CMEMS_NRT_EUR': variances_cmems_eur,
    'CMEMS_NRT_GLO': variances_cmems_glo,
    'DUACS_SWOT_L4': variances_duacs_swot_l4
}

rmsds = {
    'SWOT_L3': rmsds_swot_l3,
    'CMEMS_NRT_EUR': rmsds_cmems_eur,
    'CMEMS_NRT_GLO': rmsds_cmems_glo,
    'DUACS_SWOT_L4': rmsds_duacs_swot_l4
}


correlations = {
    'swot_l3': correlations_swot_l3,
    'cmems_eur': correlations_cmems_eur,
    'cmems_glo': correlations_cmems_glo,
    'duacs_swot_l4': correlations_duacs_swot_l4
}

# Convert to DataFrames for scaling
var_df = pd.DataFrame(variances)
rmsd_df = pd.DataFrame(rmsds)
corr_df = pd.DataFrame(correlations)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize across all columns (all products) together for each statistic
var_df_normalized = pd.DataFrame(scaler.fit_transform(var_df.values.reshape(-1, 1)).reshape(var_df.shape),
                                 columns=var_df.columns, index=var_df.index)

rmsd_df_normalized = pd.DataFrame(scaler.fit_transform(rmsd_df.values.reshape(-1, 1)).reshape(rmsd_df.shape),
                                  columns=rmsd_df.columns, index=rmsd_df.index)

# Define weights for RMSD and Variance
alpha = 0.5  # Weight for RMSD
beta = 0.5   # Weight for Variance (inverted)

# Invert the normalized variance
var_df_inverted = 1 / var_df_normalized

# Calculate the Combined Performance Index (ICD)
icd_df = alpha * rmsd_df_normalized + beta * var_df_inverted 

# Optional: Invert ICD to make lower values better (if needed)
icd_df = 1 / icd_df

# Optional: Normalize the ICD to a range between 0 and 1
scaler_icd = MinMaxScaler()
icd_df_normalized = pd.DataFrame(scaler_icd.fit_transform(icd_df), columns=icd_df.columns, index=icd_df.index)

# Display the normalized ICD DataFrame
print(icd_df_normalized)

# Calculate ICD for point sizes
icd_sizes = icd_df_normalized * 1000  # Scale for visibility

# Create a scatter plot with colors representing different products
plt.figure(figsize=(12, 8))

# List of colors (one for each product)
colors = plt.cm.tab10(range(len(icd_df_normalized.columns)))

# Iterate over each product (column) to add different colors
for i, product in enumerate(icd_df_normalized.columns):
    for station_idx in range(len(icd_df_normalized.index)):
        # Plot each station's point
        plt.scatter(rmsd_df_normalized[product][station_idx], var_df_normalized[product][station_idx], 
                    s=icd_sizes[product][station_idx], c=[colors[i]], label=product if station_idx == 0 else "", alpha=0.6, edgecolor='k')
        
        # Add station number at the center of each circle
        plt.text(rmsd_df_normalized[product][station_idx], var_df_normalized[product][station_idx], 
                 station_idx + 1, ha='center', va='center', fontsize=8, color='white')

# Add reference lines at the medians
plt.axvline(rmsd_df_normalized.median().median(), color='grey', linestyle='--')
plt.axhline(var_df_normalized.median().median(), color='grey', linestyle='--')

# Labels and title
plt.xlabel('Normalized RMSD')
plt.ylabel('Normalized Variance')
plt.title('Scatter Plot of RMSD vs Normalized Variance by Product')

# Create custom legend
legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, 
                                 label=product, markeredgecolor='k') for i, product in enumerate(icd_df_normalized.columns)]
plt.legend(handles=legend_elements, title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)

# Show the plot
plt.show()

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Assuming variances, rmsds, and correlations dictionaries are already defined
# Normalize the variances and RMSDs
variances = {
    'SWOT_L3': variances_swot_l3,
    'CMEMS_NRT_EUR': variances_cmems_eur,
    'CMEMS_NRT_GLO': variances_cmems_glo,
    'DUACS_SWOT_L4': variances_duacs_swot_l4
}

rmsds = {
    'SWOT_L3': rmsds_swot_l3,
    'CMEMS_NRT_EUR': rmsds_cmems_eur,
    'CMEMS_NRT_GLO': rmsds_cmems_glo,
    'DUACS_SWOT_L4': rmsds_duacs_swot_l4
}

correlations = {
    'swot_l3': correlations_swot_l3,
    'cmems_eur': correlations_cmems_eur,
    'cmems_glo': correlations_cmems_glo,
    'duacs_swot_l4': correlations_duacs_swot_l4
}

# Convert to DataFrames for scaling
var_df = pd.DataFrame(variances)
rmsd_df = pd.DataFrame(rmsds)
corr_df = pd.DataFrame(correlations)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize across all columns (all products) together for each statistic
var_df_normalized = pd.DataFrame(scaler.fit_transform(var_df.values.reshape(-1, 1)).reshape(var_df.shape),
                                 columns=var_df.columns, index=var_df.index)

rmsd_df_normalized = pd.DataFrame(scaler.fit_transform(rmsd_df.values.reshape(-1, 1)).reshape(rmsd_df.shape),
                                  columns=rmsd_df.columns, index=rmsd_df.index)

# Define weights for RMSD and Variance
alpha = 0.5  # Weight for RMSD
beta = 0.5   # Weight for Variance (inverted)

# Invert the normalized variance
var_df_inverted = 1 / var_df_normalized

# Calculate the Combined Performance Index (ICD)
icd_df = alpha * rmsd_df_normalized + beta * var_df_inverted 

# Optional: Invert ICD to make lower values better (if needed)
icd_df = 1 / icd_df

# Optional: Normalize the ICD to a range between 0 and 1
scaler_icd = MinMaxScaler()
icd_df_normalized = pd.DataFrame(scaler_icd.fit_transform(icd_df), columns=icd_df.columns, index=icd_df.index)

# Display the normalized ICD DataFrame
print(icd_df_normalized)

# Calculate ICD for point sizes
icd_sizes = icd_df_normalized * 1000  # Scale for visibility

# Get default color cycle from Matplotlib
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Map each product to a color from the default cycle
product_colors = {
    'SWOT_L3': color_cycle[0],
    'DUACS_SWOT_L4': color_cycle[1],
    'CMEMS_NRT_EUR': color_cycle[2],
    'CMEMS_NRT_GLO': color_cycle[3]
}



# Create a scatter plot with colors representing different products
plt.figure(figsize=(12, 8))

# Iterate over each product (column) to add different colors
for i, product in enumerate(icd_df_normalized.columns):
    for station_idx in range(len(icd_df_normalized.index)):
        # Plot each station's point with the specified color
        plt.scatter(rmsd_df_normalized[product][station_idx], var_df_normalized[product][station_idx], 
                    s=icd_sizes[product][station_idx], c=[product_colors[product]], 
                    label=product if station_idx == 0 else "", alpha=0.6, edgecolor='k')
        
        # Add station number at the center of each circle
        plt.text(rmsd_df_normalized[product][station_idx], var_df_normalized[product][station_idx], 
                 station_idx, ha='center', va='center', fontsize=12, color='white')

# Add reference lines at the means instead of medians
plt.axvline(rmsd_df_normalized.mean().mean(), color='grey', linestyle='--')
plt.axhline(var_df_normalized.mean().mean(), color='grey', linestyle='--')

# Labels and title
plt.xlabel('Normalized RMSD', fontsize=12)
plt.ylabel('Normalized Variance', fontsize=12)
plt.title('Combined Performance Index (RMSD vs Variance by Product)', fontsize=14)

# Set fontsize for x-ticks and y-ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Create custom legend with scatter points to match plot circles
scatter_legend = [plt.scatter([], [], s=100, color=product_colors[product], 
                              alpha=0.6, edgecolor='k', label=product) 
                  for product in icd_df_normalized.columns]

# Position the legend within the figure
plt.legend(handles=scatter_legend, title='PRODUCT', loc='upper right', fontsize=12, title_fontsize=12)

plt.grid(True)

# Show the plot
plt.show()


