import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import xarray as xr
import matplotlib.colors as mcolors

# Define the RGB values for your custom color palette
colors_rgb = [
    (255, 255, 255), (214, 226, 255), (181, 201, 255), (142, 178, 255),
    (127, 150, 255), (99, 112, 247), (0, 99, 255), (0, 102, 102), 
    (0, 150, 150), (0, 198, 51), (99, 255, 0), (150, 255, 0), 
    (198, 255, 51), (255, 255, 0), (255, 198, 0), (255, 160, 0), 
    (255, 124, 0), (255, 102, 0), (255, 25, 0)
]

# Convert RGB tuples to normalized RGB values (values between 0 and 1)
colors_normalized = [(r / 255., g / 255., b / 255.) for (r, g, b) in colors_rgb]

# Create a colormap using the normalized RGB values
cmap = mcolors.ListedColormap(colors_normalized)

# Load data
path = '/home/dvega/anaconda3/work/SWOT_STORM/ERA5/'
files = ['msl_wind_kiel_v6.nc', 'msl_wind_alte_v1.nc', 'msl_wind_Beryl_v1.nc', 'msl_wind_Beryl_v3.nc']
time_swot_list = ['2023-04-02 00:00:00', '2023-10-14 17:00:00', '2024-07-05 07:00:00', '2024-07-05 07:00:00']

# Define discrete levels for the wind speed colorbar
wind_speed_levels = np.arange(-0.5, 21.5, 1)  # Adjusted to include up to 20.5 m/s

for file, time_swot in zip(files, time_swot_list):
    ds = xr.open_dataset(f'{path}{file}')

    u10 = ds['u10'].sel(time=time_swot)
    v10 = ds['v10'].sel(time=time_swot)
    msl = ds['msl'].sel(time=time_swot)

    # Calculate wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)

    # Normalize msl to hPa
    msl_hpa = msl / 100.0

    # Define the plot
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Define custom levels for contours
    min_value = np.floor(msl_hpa.min().item())
    max_value = np.ceil(msl_hpa.max().item())
    levels = np.arange(min_value, max_value + 5, 5)

    # Plot mean sea level pressure as contours with thicker grey lines
    msl_contour = ax.contour(msl.longitude, msl.latitude, msl_hpa, levels=levels, colors='#4d4d4d', linewidths=1, transform=ccrs.PlateCarree())
    plt.clabel(msl_contour, inline=True, fontsize=8, fmt='%1.0f')

    # Set the colorbar range from 0 to 21 m/s for wind speed
    vmin, vmax = 0, 20

    # Add colorbar for wind speed with discrete levels using the custom colormap
    wind_speed_plot = ax.contourf(u10.longitude, u10.latitude, wind_speed, levels=wind_speed_levels, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(wind_speed_plot, orientation='vertical', pad=0.02, aspect=30, ticks=range(0, 21, 1))
    cbar.set_label('Wind Speed (m/s)', labelpad=10)

    stride = 15  # Adjust this value to reduce the number of arrows
    u10_reduced = u10[::stride, ::stride]
    v10_reduced = v10[::stride, ::stride]
    lon_reduced = u10.longitude[::stride]
    lat_reduced = u10.latitude[::stride]

    # Plot wind vectors with larger arrows
    plt.quiver(lon_reduced, lat_reduced, u10_reduced, v10_reduced, scale=250, width=0.005, transform=ccrs.PlateCarree())

    # Add features
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.COASTLINE, zorder=1)
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Add title and labels
    plt.title(f'{time_swot}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show the plot
    plt.show()
