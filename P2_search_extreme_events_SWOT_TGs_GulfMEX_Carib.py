import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
from utide import solve, reconstruct
from scipy.stats import pearsonr
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Paths to data
swot_path = '/home/dvega/anaconda3/work/SWOT_STORM/data_SWOT_series_results_P1/df_SWOT_1_0_2_GulfMEX_Carib_TG_50km.csv'
tg_folder = '/home/dvega/anaconda3/work/SWOT_STORM/datos_TG_IOC/datos_TGs_GulfMEX_Carib/download_prueba2/'

# Read SWOT dataframe
df = pd.read_csv(swot_path)

# Read tide gauge locations
fileTG = [os.path.join(tg_folder, f) for f in os.listdir(tg_folder) if f.endswith('.csv')]
fileNames = os.listdir(tg_folder)

column_names = ["flt(m)", "rad(m)", "prs(m)", "wls(m)", "pwl(m)"] # Potential column names to process

latTGs = []
lonTGs = []

for i in fileTG:
    tg = pd.read_csv(i)
    lonTG = tg['Longitude'].values[0].astype(float)
    latTG = tg['Latitude'].values[0].astype(float)
    latTGs.append(latTG)
    lonTGs.append(lonTG)


def process_station(station):
    try:
        name_tg = fileNames[station]
        swot_ts = df[(df['latitude'] == latTGs[station]) & (df['longitude'] == lonTGs[station])].copy()

        if swot_ts.empty:
            return None
        else:
            swot_ts.loc[:, 'time'] = pd.to_datetime(swot_ts['time'])
            swot_ts.sort_values(by='time', inplace=True)

            tg_ts = pd.read_csv(fileTG[station])

            tg_ts['Time (UTC)'] = pd.to_datetime(tg_ts['Time (UTC)'])

            # Find the first valid variable in tg_ts
            valid_variable = next((col for col in column_names if col in tg_ts.columns), None)

            if valid_variable is None:
                print(f"No valid variable found in tide gauge data for station {station}. Skipping...")
                return None

            tg_ts[valid_variable] = tg_ts[valid_variable]*100

            # Remove outliers based on 3 times standard deviation
            mean_val = tg_ts[valid_variable].mean()
            std_val = tg_ts[valid_variable].std()
            tg_ts = tg_ts[(tg_ts[valid_variable] >= mean_val - 3 * std_val) & (tg_ts[valid_variable] <= mean_val + 3 * std_val)]

            # Demean the tide gauge data using the found variable
            tg_ts['demean1'] = tg_ts[valid_variable] - tg_ts[valid_variable].mean()

            # Detide the entire tide gauge time series
            coef = solve(tg_ts['Time (UTC)'], tg_ts['demean1'],
                         lat=latTGs[station],
                         nodal=False,
                         trend=False,
                         method='ols',
                         conf_int='linear',
                         verbose=False)

            tide = reconstruct(tg_ts['Time (UTC)'], coef, verbose=False)
            tg_ts['detided'] = tg_ts['demean1'] - tide.h

            # Calculate the 95th and 5th percentiles from the demeaned tide gauge data
            perc_95 = np.percentile(tg_ts['detided'], 95)
            perc_5 = np.percentile(tg_ts['detided'], 5)

            # Filter to the overlapping time period
            min_time = np.max([np.min(tg_ts['Time (UTC)']), np.min(swot_ts['time'])])
            max_time = np.min([np.max(tg_ts['Time (UTC)']), np.max(swot_ts['time'])])

            tg_ts = tg_ts[(tg_ts['Time (UTC)'] > min_time) & (tg_ts['Time (UTC)'] < max_time)]
            swot_ts = swot_ts[(swot_ts['time'] > min_time) & (swot_ts['time'] < max_time)]

            # Demean the tide gauge time series for the overlapping period
            tg_ts['demean'] = tg_ts['detided'] - tg_ts['detided'].mean()

            # Demean the SWOT time series for the overlapping period
            swot_ts['demean'] = swot_ts['ssha_dac'] - np.mean(swot_ts['ssha_dac'])

            # Find closest tide gauge points for each SWOT time point
            closest_tg_values = []
            for time_point in swot_ts['time']:
                closest_idx = (tg_ts['Time (UTC)'] - time_point).abs().idxmin()
                closest_tg_values.append(tg_ts.loc[closest_idx, 'demean'])

            swot_ts['closest_tg'] = closest_tg_values

            # Calculate RMSD
            rmsd = np.sqrt(np.mean((swot_ts['closest_tg'] - swot_ts['demean'])**2))

            # Check if any value in the SWOT time series is above 95th percentile or below 5th percentile of the demeaned tide gauge data
            if (swot_ts['demean'] > perc_95).any() or (swot_ts['demean'] < perc_5).any():
                return (station, name_tg, tg_ts['Time (UTC)'], tg_ts['demean'], swot_ts['time'], swot_ts['demean'], perc_95, perc_5, rmsd)
    except Exception as e:
        print(f"Error processing station {station}: {e}")
    return None


# Specify the number of workers
num_workers = 40  # You can adjust this number based on your CPU cores and desired parallelism

# Use ProcessPoolExecutor for parallel processing
results = []
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    print('start_processing_paralelization')
    futures = [executor.submit(process_station, station) for station in range(len(fileTG))]
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f'Error processing station: {e}')


# Obtain tropical cyclone data

# Replace 'your_file.csv' with the path to your CSV file
cyclone_path = '/home/dvega/anaconda3/work/SWOT_STORM/cyclone_tracks/ibtracs.ALL.list.v04r01.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(cyclone_path)

# Filter the DataFrame to select rows where the SUBBASIN is "CS" or "GM"
filtered_df = df[df['SUBBASIN'].isin(['CS', 'GM'])]

# Convert the ISO_TIME column to datetime
filtered_df['ISO_TIME'] = pd.to_datetime(filtered_df['ISO_TIME'], errors='coerce')

# Filter the DataFrame to select rows where the ISO_TIME is after 01-01-2023
filtered_df = filtered_df[filtered_df['ISO_TIME'] > '2023-01-01']





# Plot results in the main process
for result in results:
    print("PLOTTING")
    station, name_tg, tg_time, tg_demean, swot_time, swot_demean, perc_95, perc_5, rmsd = result
    plt.plot(tg_time, tg_demean, label='Tide Gauge')
    plt.scatter(swot_time, swot_demean, c='r', s=20, label='SWOT_data', zorder=2)

    # plot cyclone periods
    for i in range(len(filtered_df)):
        plt.axvspan(filtered_df['ISO_TIME'].iloc[i], filtered_df['ISO_TIME'].iloc[i] + pd.Timedelta(hours=6), color='gray', alpha=0.5)


    plt.axhline(y=perc_95, color='g', linestyle='--', label='95th Percentile')
    plt.axhline(y=perc_5, color='b', linestyle='--', label='5th Percentile')

    # Add text with metrics
    textstr = f'RMSD: {rmsd:.2f}'
    plt.text(0.85, 0.05, textstr, transform=plt.gca().transAxes, fontsize=6,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.title(f'{name_tg} - {station}')
    plt.legend(fontsize=6, loc='upper right')
    plt.savefig(f'/home/dvega/anaconda3/work/SWOT_STORM/P2_search_extreme_events/Figures_search_extreme_events_P2/{name_tg}_{station}.png', dpi=300)
    plt.close()
    print(f"Station {station} processed and plotted successfully")

