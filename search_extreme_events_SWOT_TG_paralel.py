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
swot_path = '/home/dvega/anaconda3/work/SWOT_STORM/df_SWOT_series_Med_5km.csv'
TG_path = '/home/dvega/anaconda3/work/SWOT_STORM/datos_tg_cmems/datos_TG_med_CMEMS/'

# Read SWOT dataframe
df = pd.read_csv(swot_path)

# Read tide gauge locations
fileTG = [os.path.join(TG_path, f) for f in os.listdir(TG_path) if f.endswith('.nc')]
fileNames = os.listdir(TG_path)

latTGs = [37.64045]
lonTGs = [21.319233]
latTGs = []
lonTGs = []

for i in fileTG:
    tg = xr.open_dataset(i)
    lonTG = tg['LONGITUDE'].values.astype(float)
    latTG = tg['LATITUDE'].values.astype(float)
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

            tg_ts = xr.open_dataset(fileTG[station])
            tg_ts = tg_ts.to_dataframe()
            tg_ts.reset_index(inplace=True)

            tg_ts['SLEV'] = tg_ts['SLEV'] * 100

            # Demean the entire tide gauge time series
            tg_ts['demean1'] = tg_ts['SLEV'] - tg_ts['SLEV'].mean()

            # Detide the entire tide gauge time series
            coef = solve(tg_ts['TIME'], tg_ts['demean1'],
                         lat=latTGs[station],
                         nodal=False,
                         trend=False,
                         method='ols',
                         conf_int='linear',
                         verbose=False)

            tide = reconstruct(tg_ts['TIME'], coef, verbose=False)
            tg_ts['detided'] = tg_ts['demean1'] - tide.h

            # Calculate the 95th and 5th percentiles from the demeaned tide gauge data
            perc_95 = np.percentile(tg_ts['detided'], 95)
            perc_5 = np.percentile(tg_ts['detided'], 5)

            # Filter to the overlapping time period
            min_time = np.max([np.min(tg_ts['TIME']), np.min(swot_ts['time'])])
            max_time = np.min([np.max(tg_ts['TIME']), np.max(swot_ts['time'])])

            tg_ts = tg_ts[(tg_ts['TIME'] > min_time) & (tg_ts['TIME'] < max_time)]
            swot_ts = swot_ts[(swot_ts['time'] > min_time) & (swot_ts['time'] < max_time)]

            # Demean the entire tide gauge time series
            tg_ts['demean'] = tg_ts['detided'] - tg_ts['detided'].mean()

            # Demean the overlapping period of both time series
            swot_ts['demean'] = swot_ts['ssha_dac'] - np.mean(swot_ts['ssha_dac'])

            # Find closest tide gauge points for each SWOT time point
            closest_tg_values = []
            for time_point in swot_ts['time']:
                closest_idx = (tg_ts['TIME'] - time_point).abs().idxmin()
                closest_tg_values.append(tg_ts.loc[closest_idx, 'demean'])

            swot_ts['closest_tg'] = closest_tg_values

            # Calculate RMSD
            rmsd = np.sqrt(np.mean((swot_ts['closest_tg'] - swot_ts['demean'])**2))

            # Check if any value in the SWOT time series is above 95th percentile or below 5th percentile of the demeaned tide gauge data
            if (swot_ts['demean'] > perc_95).any() or (swot_ts['demean'] < perc_5).any():
                return (station, name_tg, tg_ts['TIME'], tg_ts['demean'], swot_ts['time'], swot_ts['demean'], perc_95, perc_5, rmsd)
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

# Plot results in the main process
for result in results:
    station, name_tg, tg_time, tg_demean, swot_time, swot_demean, perc_95, perc_5, rmsd = result
    plt.plot(tg_time, tg_demean, label='Tide Gauge')
    plt.scatter(swot_time, swot_demean, c='r', s=20, label='SWOT_data', zorder=2)
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
    plt.show()
