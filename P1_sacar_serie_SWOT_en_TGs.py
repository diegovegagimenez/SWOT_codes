import xarray as xr
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

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

dmin = 2  # km de radio

TG_path='/home/amores/SWOT/A_data/A_TGs/'
# TG_path = ['/home/amores/SWOT/A_data/A_TGs/TG_CMEMS/', '/home/amores/SWOT/A_data/A_TGs/TG_SOEST/']
SWOT_path = '/home/amores/SWOT/A_data/B_SWOT/SWOT_data/'

# SWOTfiles = [f for f in os.listdir(SWOT_path) if f.endswith('.nc')]
# TGfiles = [f for f in os.listdir(TG_path) if f.endswith('.nc')]

kiel_tg = xr.open_dataset(f'{TG_path}NO_TS_TG_KielTG.nc')
Alte_tg = xr.open_dataset(f'{TG_path}NO_TS_TG_AlteWeserTG.nc')

TG_data = [kiel_tg, Alte_tg]

lonNames=['lon','longitude','LONGITUDE']
latNames=['lat','latitude','LATITUDE']

for i in range(len(TG_data)):
    tg = TG_data[i]
    # Read the first available longitude and latitude variables
    lonTG = read_first_available(tg, lonNames)
    latTG = read_first_available(tg, latNames)
    print(latTG,lonTG)


fileSWOT = [os.path.join(SWOT_path, f) for f in os.listdir(SWOT_path) if f.endswith('.nc')]

for filename in tqdm(fileSWOT[:5], desc='SWOT files'):

    file_path = os.path.join(SWOT_path, filename)
    ds = xr.open_dataset(file_path)

    latSWOT = read_first_available(ds, latNames)
    lonSWOT = read_first_available(ds, lonNames)

    ssh = ds['ssha'].values + ds['dac'].values

    time = ds['time']

    thres=(20*dmin/100)

    for lon_tg, lat_tg in zip(lonTG, latTG):
        d = np.sqrt((lonSWOT - lon_tg) ** 2 + (latSWOT - lat_tg) ** 2)
        if np.min(d) > thres:
            continue








# for filename in TGfiles:
#     print(filename)
#     # Read the TG
#     tg = xr.open_dataset(TG_path + filename)

    

