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
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from geopy.distance import geodesic
from geopy.point import Point
import numpy as np


# Define the RGB values for your custom color palette
colors_rgb = [
    (255, 255, 255), (214, 226, 255), (181, 201, 255), (142, 178, 255),
    (127, 150, 255), (99, 112, 247), (50, 141, 208), (0, 170, 170), 
    (0, 150, 150), (0, 198, 51), (99, 255, 0), (150, 255, 0), 
    (198, 255, 51), (255, 255, 0), (255, 198, 0), (255, 160, 0), 
    (255, 124, 0), (255, 102, 0), (255, 25, 0)
]

# Convert RGB tuples to normalized RGB values (values between 0 and 1)
colors_normalized = [(r / 255., g / 255., b / 255.) for (r, g, b) in colors_rgb]

# Create a colormap using the normalized RGB values
cmap = mcolors.ListedColormap(colors_normalized)

# Define the target location
kiel_lat = 54.50033   
kiel_lon = 10.275  
alte_lat = 53.8633
alte_lon = 8.1275

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
model_path = '/storage/ada/alexandre/malla_ref_3000/oleaje23_24/'
era5_path = '/home/dvega/anaconda3/work/SWOT_STORM/ERA5/'
tri_file_path = '/home/amores/SWOT/A_data/C_modelo/'

# Load data
model23 = xr.open_dataset(f'{model_path}outputs_2023/merged_elevation_2023.nc')
model24 = xr.open_dataset(f'{model_path}outputs_2024/merged_elevation_2024.nc')
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
peak_date = '2023-04-02 00:00:00'
mat_data = mat73.loadmat(f'{tri_file_path}tri_simu.mat')
tri = mat_data['tri'].astype(int) - 1

# Create a figure and axis with Cartopy projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

# Convert lonSC, latSC to x, y coordinates for the projection
# x, y = ax.projection.transform_points(ccrs.PlateCarree(), lonSC, latSC)[:, :2].T

# Create a Triangulation object
triang = Triangulation(lonSC, latSC, tri)

# Plot the triangulated surface
collection = ax.tripcolor(triang, sshSC, shading='flat', cmap=cmap)  # Use cmocean's 'balance' colormap

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
ax.plot(kiel_lon, kiel_lat, 'pk', markerfacecolor='k', markersize=10, transform=ccrs.PlateCarree())

# Set color limits on the collection
collection.set_clim(-1, 12)

# Set title
plt.title(peak_date)

# Show plot
plt.show()


# ------------------ EXTRACT TIME SERIES FROM KIEL AND ALTEWESER LOCATIONS-----------------------------

# Calculate the distance from the target location to all points in the dataframe
model_df_cut_kiel['distance'] = model_df_cut_kiel.apply(lambda row: haversine(kiel_lon, kiel_lat, row['SCHISM_hgrid_node_x'], row['SCHISM_hgrid_node_y']), axis=1)
model_df_cut_alte['distance'] = model_df_cut_alte.apply(lambda row: haversine(alte_lon, alte_lat, row['SCHISM_hgrid_node_x'], row['SCHISM_hgrid_node_y']), axis=1)

# Filter the points within 20 km radius
radius = 50  # km
df_filtered_kiel = model_df_cut_kiel[model_df_cut_kiel['distance'] <= radius]
df_filtered_alte = model_df_cut_alte[model_df_cut_alte['distance'] <= radius]

# Group by time and calculate the average sea level anomaly
ts_kiel_SC = df_filtered_kiel.groupby('time')['elevation'].mean().reset_index()
ts_alte_SC = df_filtered_alte.groupby('time')['elevation'].mean().reset_index()


plt.plot(ts_kiel_SC['time'], ts_kiel_SC['elevation'])
plt.plot(ts_alte_SC['time'], ts_alte_SC['elevation'])



# OBTAIN PERPENDICULAR LINE TO THE COAST

# Regrid data

# Define the target grid resolution
lat_new = np.linspace(latmin, latmax, 1000)  # Increase number for finer resolution
lon_new = np.linspace(lonmin, lonmax, 1000)

# Create meshgrid for the target regular grid
lon_new, lat_new = np.meshgrid(lon_new, lat_new)

# Flatten the input data
points = np.column_stack((lonSC, latSC))
values = sshSC

# Perform interpolation
data_new = griddata(points, values, (lon_new, lat_new), method='linear')

# Handle NaN values if any
data_new = np.nan_to_num(data_new, nan=-9999)  # Replace NaNs with a fill value


# Create a new figure and axis with Cartopy projection
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

# Plot the regridded data
mesh = ax.pcolormesh(lon_new, lat_new, data_new, cmap=cmap, shading='auto', transform=ccrs.PlateCarree())


# Add grid lines
gl = ax.gridlines(draw_labels=True, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Sea Level Elevation (m)')

# Plot additional markers
ax.plot(kiel_lon, kiel_lat, 'pk', markerfacecolor='k', markersize=10, transform=ccrs.PlateCarree())
# Add geographic features
ax.add_feature(cfeature.COASTLINE, zorder=3)
ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
ax.add_feature(cfeature.LAND, facecolor=[.8, .8, .8], zorder=3)  # Land color

# Set color limits
mesh.set_clim(-1, 12)

# Set title
plt.title(peak_date)

# Show plot
plt.show()




def perpendicular_line(lat1, lon1, lat2, lon2, distance=10000):
    """
    Calculate the coordinates of a line segment perpendicular to the given line segment.
    
    Args:
    lat1, lon1: Coordinates of the first point of the coastline segment.
    lat2, lon2: Coordinates of the second point of the coastline segment.
    distance: The length of the perpendicular line to be plotted on each side in meters.
    
    Returns:
    (lat_start, lon_start, lat_end, lon_end): Coordinates of the start and end points of the perpendicular line.
    """
    # Calculate the midpoint of the original line segment
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    mid_point = Point(mid_lat, mid_lon)
    
    # Calculate the bearing of the original line segment
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    if delta_lon == 0:
        bearing = 90 if delta_lat > 0 else 270
    else:
        bearing = np.degrees(np.arctan2(delta_lon, delta_lat))
    
    # Calculate perpendicular bearings
    perp_bearing1 = (bearing + 90) % 360
    perp_bearing2 = (bearing - 90) % 360
    
    # Use geopy to find the points at the specified distance along the perpendicular bearings
    point1 = geodesic(meters=distance).destination(mid_point, perp_bearing1)
    point2 = geodesic(meters=distance).destination(mid_point, perp_bearing2)
    
    return point1.latitude, point1.longitude, point2.latitude, point2.longitude

# Example coastline segment coordinates
lat1, lon1 = 54.4825, 10.1406
lat2, lon2 = 54.4335, 10.3359

# Calculate the perpendicular line with a length of 10 km on each side
perp_lat1, perp_lon1, perp_lat2, perp_lon2 = perpendicular_line(lat1, lon1, lat2, lon2, distance=20000)

print(f"Perpendicular Line Endpoints:")
print(f"Point 1: Latitude: {perp_lat1}, Longitude: {perp_lon1}")
print(f"Point 2: Latitude: {perp_lat2}, Longitude: {perp_lon2}")


def create_transect_points(start_lat, start_lon, end_lat, end_lon, num_points=100):
    """
    Create points along a transect between two geographic coordinates.
    
    Args:
    start_lat, start_lon: Starting coordinates of the transect.
    end_lat, end_lon: Ending coordinates of the transect.
    num_points: Number of points along the transect.
    
    Returns:
    A tuple of lists (latitudes, longitudes) of the transect points.
    """
    lats = np.linspace(start_lat, end_lat, num_points)
    lons = np.linspace(start_lon, end_lon, num_points)
    return lats, lons

def extract_transect_values(data_array, lats, lons, lon_grid, lat_grid):
    """
    Extract values along a transect from a DataArray.
    
    Args:
    data_array: xarray DataArray with the data.
    lats, lons: Coordinates of the transect.
    lon_grid, lat_grid: 2D arrays of longitude and latitude coordinates of the grid.
    
    Returns:
    Values along the transect.
    """
    # Flatten the grid coordinates for interpolation
    grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
    data_values = data_array.ravel()
    
    # Interpolate the data values at the transect points
    transect_values = griddata(grid_points, data_values, (lons, lats), method='linear')
    
    return transect_values


transect_lats, transect_lons = create_transect_points(perp_lat1, perp_lon1, perp_lat2, perp_lon2)

transect_values = extract_transect_values(data_new, transect_lats, transect_lons, lon_new, lat_new)


plt.plot(np.arange(len(transect_values)), transect_values)
plt.xlabel('Point Index')
plt.ylabel('Data Value')
plt.title('Transect Values')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

# Plot the regridded data
mesh = ax.pcolormesh(lon_new, lat_new, data_new, cmap=cmap, shading='auto', transform=ccrs.PlateCarree())

# Add grid lines
gl = ax.gridlines(draw_labels=True, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Sea Level Elevation (m)')

# Plot additional markers
ax.plot(kiel_lon, kiel_lat, 'pk', markerfacecolor='k', markersize=10, transform=ccrs.PlateCarree())

# Add the transect points
ax.plot(transect_lons, transect_lats, 'r-', marker='o', markersize=5, transform=ccrs.PlateCarree(), label='Transect')

# Add geographic features
ax.add_feature(cfeature.COASTLINE, zorder=3)
ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
ax.add_feature(cfeature.LAND, facecolor=[.8, .8, .8], zorder=3)  # Land color

# Set color limits
mesh.set_clim(-1, 12)

# Set title and legend
plt.title(peak_date)
plt.legend()

# Show plot
plt.show()
