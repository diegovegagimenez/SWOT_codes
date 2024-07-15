import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.io import loadmat
from matplotlib.tri import Triangulation
import cmocean
import xarray as xr
import pandas as pd
import mat73

SWOT_path = '/home/dvega/anaconda3/work/SWOT_STORM/'
model_path = '/home/amores/SWOT/A_data/C_modelo/'
era5_path = '/home/dvega/anaconda3/work/SWOT_STORM/ERA5/'

model23 = xr.open_dataset(f'{model_path}merged_elevation_2023.nc')
model24 = xr.open_dataset(f'{model_path}merged_elevation_2024.nc')

model = xr.concat([model23, model24], dim='time')
model_df = model.to_dataframe().reset_index()

model_df = model_df[(model_df['time']>pd.to_datetime('2023-04-01'))&(model_df['time']<pd.to_datetime('2023-04-02'))]

model_df_kiel = model_df[model_df['time'] == pd.to_datetime('2023-04-01 23:00:00')]

# Plot 1 speciphic date
lonlatbox = [9, 13, 53.5, 57]

latSC = model_df_kiel['SCHISM_hgrid_node_y'].values
lonSC = model_df_kiel['SCHISM_hgrid_node_x'].values
sshSC = model_df_kiel['elevation'].values
time = model_df_kiel['time'].values

# Sample data (replace these with your actual data)
latmin, latmax = 53.5, 57
lonmin, lonmax = 9, 13
tri = np.random.randint(0, 100, (200, 3))
lonTG, latTG = np.array([10]), np.array([55])
peak_date = '2023-04-01 23:00:00'

mat_data = mat73.loadmat(f'{model_path}tri_simu.mat')
tri = mat_data['tri'].astype(int) - 1

# Create a figure and axis with Cartopy projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.LambertConformal(central_longitude=(lonmin + lonmax) / 2, 
                                                                                        central_latitude=(latmin + latmax) / 2)})
ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

# Convert lonSC, latSC to x, y coordinates for the projection
x, y = ax.projection.transform_points(ccrs.PlateCarree(), lonSC, latSC)[:, :2].T

# Create a Triangulation object
triang = Triangulation(x, y, tri)

# Plot the triangulated surface
ax.plot(x, y, 'o', markersize=1)  # Optionally, plot the mesh points
ax.tripcolor(triang, sshSC, shading='flat', cmap=cmocean.cm.balance)  # Use cmocean's 'balance' colormap

# Add geographic features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor=[.8, .8, .8])  # Land color

# Add grid lines
gl = ax.gridlines(draw_labels=True, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add colorbar
cbar = plt.colorbar(ax.collections[-1], ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Sea Level Elevation (m)')

# Plot additional markers
ax.plot(lonTG, latTG, 'pk', markerfacecolor='k', markersize=10, transform=ccrs.PlateCarree())

# Set color limits
plt.clim(-0.9, 0.9)

# Set colormap
plt.set_cmap(cmocean.cm.balance)

# Set title
plt.title(peak_date)

# Show plot
plt.show()
