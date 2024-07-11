import xarray as xr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utide import solve, reconstruct
import cartopy.crs as ccrs
import xarray.plot as xplt  # Import xarray.plot module
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


TG_path='/home/amores/SWOT/A_data/A_TGs/'
SWOT_path = '/home/dvega/anaconda3/work/SWOT_STORM/'

df = pd.read_csv(f'{SWOT_path}df_Kiel_Alteweser_SWOT_series_50km.csv')

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


df_kiel= df[df['latitude'] == latTGs[0][0]]
df_kiel['ssha_demean'] = df_kiel['ssha_dac']-df_kiel['ssha_dac'].mean()

df_alte = df[df['latitude'] == latTGs[1][0]]
df_alte['ssha_demean'] = df_alte['ssha_dac']-df_alte['ssha_dac'].mean()


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


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(df_tg_k.index[-70000:], (df_tg_k['SLEV_demean']-tide_k.h)[-70000:]*100, zorder=0, c='black', linewidth=0.8, label=r'Sea level')
plt.scatter(df_kiel['time'], df_kiel['ssha_demean'], c='r', zorder = 1, s=8, label=r'SWOT Data')
plt.xticks(rotation=45)
plt.xlabel(r'Time (days)', labelpad=15)
plt.ylabel(r'Sea level anomaly (cm)')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=7)
plt.title(r'Kiel TG')
plt.show()

plt.plot(df_tg_a.index[-50000:], (df_tg_a['SLEV_demean']-tide_a.h)[-50000:]*100, zorder=0, c='black', linewidth=0.8, label=r'Sea level')
plt.scatter(df_alte['time'], df_alte['ssha_demean'], c='r', zorder = 1, s=8, label=r'SWOT Data')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.xlabel(r'Time (days)', labelpad=15)
plt.ylabel(r'Sea level anomaly (cm)')
plt.legend(fontsize=7)
plt.title(r'Alteweser TG')
plt.show()
# plt.title()

# plt.plot(df_tg_a.index[-70000:], (df_tg_a['SLEV_demean']-tide_a.h)[-70000:]*100, zorder=0, c='black', linewidth=0.8)
# plt.scatter(df_alte['time'], df_alte['ssha_demean'], c='r', zorder = 1, s=8)
# plt.xticks(rotation=45)
