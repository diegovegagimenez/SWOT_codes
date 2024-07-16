import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import xarray as xr

# Load data
path = '/home/dvega/anaconda3/work/SWOT_STORM/ERA5/'
ds = xr.open_dataset(f'{path}msl_pressure_global_01-02042023.nc')
# ds = xr.open_dataset(f'{path}wind_mean_sea_level_pres_kiel.nc')
ds = xr.open_dataset(f'{path}msl_wind_kiel_v6.nc')
ds = xr.open_dataset(f'{path}msl_wind_alte_v1.nc')

# Select specific date
time_swot = '2023-04-02 00:00:00'
time_swot = '2023-10-14 17:00:00'

u10 = ds['u10'].sel(time=time_swot)
v10 = ds['v10'].sel(time=time_swot)
msl = ds['msl'].sel(time=time_swot)

# Calculate wind speed and direction
wind_speed = np.sqrt(u10**2 + v10**2)

# Normalize msl to hPa
msl_hpa = msl / 100.0

# lolabox = [-10, 20, 10, 70]

# Define the plot
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())

# Define custom levels for contours
min_value = np.floor(msl_hpa.min().item())
max_value = np.ceil(msl_hpa.max().item())
levels = np.arange(min_value, max_value + 5, 5)

# ax.set_extent(lolabox, crs=ccrs.PlateCarree())
# Plot mean sea level pressure as contours with thicker grey lines
msl_contour = ax.contour(msl.longitude, msl.latitude, msl_hpa, levels=levels, colors='#4d4d4d', linewidths=1.5, transform=ccrs.PlateCarree())
plt.clabel(msl_contour, inline=True, fontsize=8, fmt='%1.0f')

# # Add colorbar for wind speed
wind_speed_plot = ax.contourf(u10.longitude, u10.latitude, wind_speed, levels=1000, cmap='coolwarm', transform=ccrs.PlateCarree())
cbar = plt.colorbar(wind_speed_plot, orientation='vertical', pad=0.02, aspect=30)
cbar.set_label('Wind Speed (m/s)')

stride = 15  # Adjust this value to reduce the number of arrows
u10_reduced = u10[::stride, ::stride]
v10_reduced = v10[::stride, ::stride]
lon_reduced = u10.longitude[::stride]
lat_reduced = u10.latitude[::stride]

# Plot wind vectors with larger arrows
plt.quiver(lon_reduced, lat_reduced, u10_reduced, v10_reduced, scale=300, width=0.005, transform=ccrs.PlateCarree())


# Add features
ax.add_feature(cfeature.LAND, zorder=0)
ax.add_feature(cfeature.COASTLINE, zorder=1)
gl = ax.gridlines(draw_labels=True, alpha=0.5)
gl.top_labels = False
gl.right_labels = False

# Add title and labels
plt.title(f'{time_swot} ')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show the plot
plt.show()
