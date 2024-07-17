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

# Define the haversine function
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
    r = 6371  # Radius of earth in kilometers
    return c * r

# Paths
SWOT_path = '/home/dvega/anaconda3/work/SWOT_STORM/'
model_path = '/home/amores/SWOT/A_data/C_modelo/'
era5_path = '/home/dvega/anaconda3/work/SWOT_STORM/ERA5/'

# Load data
model23 = xr.open_dataset(f'{model_path}merged_elevation_2023.nc')
model24 = xr.open_dataset(f'{model_path}merged_elevation_2024.nc')
model = xr.concat([model23, model24], dim='time')
model_df = model.to_dataframe().reset_index()

# Filter data for the specific date
model_df_cut_kiel = model_df[(model_df['SCHISM_hgrid_node_y'] > 54) & (model_df['SCHISM_hgrid_node_y'] < 55) &
                    (model_df['SCHISM_hgrid_node_x'] > 9.8) & (model_df['SCHISM_hgrid_node_x'] < 10.6)]

model_df_cut_alte = model_df[(model_df['SCHISM_hgrid_node_y'] > 53.4) & (model_df['SCHISM_hgrid_node_y'] < 54.2) &
                    (model_df['SCHISM_hgrid_node_x'] > 7.6) & (model_df['SCHISM_hgrid_node_x'] < 8.7)]

model_df_kiel_time = model_df[model_df['time'] == pd.to_datetime('2023-04-02 00:00:00')]
model_df_alte_time = model_df[model_df['time'] == pd.to_datetime('2023-10-14 17:00:00')]


# Extract data KIEL ----------------------------------------------------------------
latSC = model_df_kiel_time['SCHISM_hgrid_node_y'].values
lonSC = model_df_kiel_time['SCHISM_hgrid_node_x'].values
sshSC = model_df_kiel_time['elevation'].values
time = model_df_kiel_time['time'].values

# Define plot bounds and load triangulation data
latmin, latmax = 53.5, 57
lonmin, lonmax = 9, 13
lonTG, latTG = np.array([10]), np.array([55])
peak_date = '2023-04-02 00:00:00'
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
collection = ax.tripcolor(triang, sshSC, shading='flat', cmap=cmocean.cm.balance)  # Use cmocean's 'balance' colormap

# Add geographic features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor=[.8, .8, .8])  # Land color

# Add grid lines
gl = ax.gridlines(draw_labels=True, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add colorbar
cbar = plt.colorbar(collection, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Sea Level Elevation (m)')

# Plot additional markers
ax.plot(lonTG, latTG, 'pk', markerfacecolor='k', markersize=10, transform=ccrs.PlateCarree())

# Set color limits on the collection
collection.set_clim(-0.9, 0.9)

# Set title
plt.title(peak_date)

# Show plot
plt.show()


# Define the target location
kiel_lat = 54.50033   
kiel_lon = 10.275  
alte_lat = 53.8633
alte_lon = 8.1275

# Extract time series from Kiel and Alteweser locations -----------------------------

# Calculate the distance from the target location to all points in the dataframe
model_df_cut_kiel['distance'] = model_df_cut_kiel.apply(lambda row: haversine(kiel_lon, kiel_lat, row['SCHISM_hgrid_node_x'], row['SCHISM_hgrid_node_y']), axis=1)

# Filter the points within 20 km radius
radius = 20  # km
df_filtered_kiel = model_df_cut_kiel[model_df_cut_kiel['distance'] <= radius]

# Group by time and calculate the average sea level anomaly
ts_kiel = df_filtered_kiel.groupby('time')['elevation'].mean().reset_index()
plt.plot(ts_kiel['time'], ts_kiel['elevation'])
