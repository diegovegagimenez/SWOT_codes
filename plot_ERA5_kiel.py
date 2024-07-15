import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import xarray as xr


path = '/home/dvega/anaconda3/work/SWOT_STORM/ERA5/'

ds = xr.open_dataset(f'{path}wind_mean_sea_level_pres_kiel.nc')

# lon = ds.sel(longitude=slice(9, 13), latitude=slice(53, 59), time='2023-04-02T00:00:00').values
# lat = ds.sel(longitude=slice(9, 13), latitude=slice(53, 59), time='2023-04-02T00:00:00').values
msl = ds['msl']
u10 = ds['u10']
v10 = ds['v10']

# Select speciphic date
time_swot = '2023-04-02 00:00:00' 
u10 = u10.sel(time = time_swot)
v10 = v10.sel(time = time_swot)
msl = msl.sel(time = time_swot)

# Calculate wind speed and direction
wind_speed = np.sqrt(u10**2 + v10**2)
wind_dir = np.arctan2(v10, u10) * (180 / np.pi)

lonlatbox = [9, 12, 53, 59]

fig, ax = plt.subplots(figsize=(10.5, 11), subplot_kw=dict(projection=ccrs.PlateCarree()))

ax.set_extent(lonlatbox)

msl_hpa = msl / 100.0
msl_plot = ax.pcolormesh(msl.longitude, msl.latitude, msl_hpa, cmap='viridis', transform=ccrs.PlateCarree())

# Add colorbar for msl plot
cbar = plt.colorbar(msl_plot, orientation='vertical', pad=0.06, aspect=50)
cbar.set_label('Mean Sea Level Pressure (hPa)')

# Plot wind vectors
plt.quiver(u10.longitude, u10.latitude, 
           u10, v10, 
           scale=250, transform=ccrs.PlateCarree())

# Add features
ax.add_feature(cfeature.LAND, zorder=0)
ax.add_feature(cfeature.COASTLINE, zorder=1, )
ax.add_feature(cfeature.BORDERS, linestyle=':')

ax.gridlines(draw_labels=True, linestyle='--')

# Add title and labels
plt.title('ERA5 Wind Vectors at 10m')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show the plot
plt.show()
