import cartopy.feature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd

lolabox = [112, 131, 20, 40] # Bebinca



# Read Bebinca track file and obtain hourly values --------------------------------------------------------------
df_bebinca = pd.read_csv('/home/dvega/anaconda3/work/SWOT_STORM/data_bebinca_track.csv')

# Create a new time index with hourly resolution
time_hourly = pd.date_range(start=df_bebinca['time'].min(), 
                            end=df_bebinca['time'].max(), 
                            freq='H')

# Convert the 'time' column to datetime format if it's not already
df_bebinca['time'] = pd.to_datetime(df_bebinca['time'])

# Sort by time in case the data isn't in chronological order
df_bebinca = df_bebinca.sort_values(by='time').reset_index(drop=True)

# Reindex the dataframe to this hourly time index
df_bebinca_hourly = df_bebinca.set_index('time').reindex(time_hourly)

# fill these NaNs using interpolation
df_bebinca_hourly[['latitude', 'longitude', 'vel']] = df_bebinca_hourly[['latitude', 'longitude', 'vel']].interpolate(method='linear')

df_bebinca_hourly = df_bebinca_hourly.reset_index().rename(columns={'index': 'time'})



# SWOT data -------------------------------------------------------------------------------------------
filename = '/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_021_172_20240916T151249_20240916T160416_v1.0.2.nc'
horas_pass = '20240916T151249_20240916T160416'
track_time = 112

# filename = '/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_021_159_20240916T040400_20240916T045527_v1.0.2.nc'
# horas_pass = '20240916T040400_20240916T045527'  
# track_time = 100

# filename = '/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_021_144_20240915T151218_20240915T160345_v1.0.2.nc'
# horas_pass = '20240915T151218_20240915T160345'
# track_time = 88

filename = '/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_021_131_20240915T040329_20240915T045455_v1.0.2.nc'
horas_pass = '20240915T040329_20240915T045455'
track_time = 76

# filename = '/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_021_116_20240914T151146_20240914T160313_v1.0.2.nc'
# horas_pass = '20240914T151146_20240914T160313'
# track_time = 63

# horas_pass4 = '20240916T151249_20240916T160416'
# horas_pass3 = '20240916T040400_20240916T045527'
# horas_pass2 = '20240915T151218_20240915T160345'
# horas_pass1 = '20240915T040329_20240915T045455'

ds = xr.open_dataset(filename)
ds_dac = ds.ssha_noiseless + ds.dac

ds_dac_1 = ds.dac


# Plotting -------------------------------------------------------------------------------------------
# fig, ax1 = plt.subplots(figsize=(21, 11), subplot_kw=dict(projection=ccrs.PlateCarree()))
# ax1.set_extent(lolabox)
# plot_kwargs = dict(
#     x="longitude",
#     y="latitude",
#     cmap="Spectral_r",
#     vmin=-0.2,
#     vmax=0.2,
#     cbar_kwargs={"shrink": 0.7, "pad": 0.05},)

# # SWOT SLA plots

# ds_dac.plot.pcolormesh(ax=ax1, **plot_kwargs)

# df_bebinca_hourly.plot(x='longitude', y='latitude', ax=ax1, color='black', marker='.', linewidth=3, zorder=2, label='Typhoon track')


# ax1.scatter(df_bebinca_hourly['longitude'].iloc[track_time], df_bebinca_hourly['latitude'].iloc[track_time], color='red', s=150, zorder=3)


# ax1.coastlines()
# ax1.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')

# # Adjust gridline label font sizes
# gl = ax1.gridlines(draw_labels=True)

# gl.left_labels = True  # Enable left labels
# gl.bottom_labels = True  # Enable bottom labels
# gl.right_labels = False  # Disable right labels
# gl.top_labels = False  # Disable top labels

# gl.xlabel_style = {'size': 16}  # Font size for x-axis grid labels
# gl.ylabel_style = {'size': 16}  # Font size for y-axis grid labels

# # Disable the automatic y-labels 
# ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: ''))  # Clear existing y-labels

# ax1.set_title(f'{horas_pass}')
# ax1.set_ylabel('')  # 'Latitude' is the label text
# ax1.set_xlabel('')  # 'Latitude' is the label text

# ax1.title.set_fontsize(20)  # Set title font size

# # plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

# plt.show()




# Plotting -------------------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 11), subplot_kw=dict(projection=ccrs.PlateCarree()))  # 1 row, 2 columns

# Typhoon track plot (left subplot)
ax1.set_extent(lolabox)
plot_kwargs = dict(
    x="longitude",
    y="latitude",
    cmap="Spectral_r",
    vmin=-0.2,
    vmax=0.2,
    cbar_kwargs={"shrink": 0.7, "pad": 0.05},
)

# SWOT SLA plots (left subplot)
ds_dac_1.plot.pcolormesh(ax=ax1, **plot_kwargs)

# Typhoon track (hourly interpolated)
df_bebinca_hourly.plot(x='longitude', y='latitude', ax=ax1, color='black', marker='.', linewidth=3, zorder=2, label='Typhoon track')

# Highlight specific track time with red marker
ax1.scatter(df_bebinca_hourly['longitude'].iloc[track_time], df_bebinca_hourly['latitude'].iloc[track_time], color='red', s=150, zorder=3)

# Coastlines and land features
ax1.coastlines()
ax1.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')

# Gridline label settings
gl = ax1.gridlines(draw_labels=True)
gl.left_labels = True
gl.bottom_labels = True
gl.right_labels = False
gl.top_labels = False
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: ''))  # Remove y-axis labels
ax1.set_title(f'{horas_pass}', fontsize=20)
ax1.set_ylabel('')
ax1.set_xlabel('')

ax2.set_extent(lolabox)
# Second plot (right subplot) - pcolormesh ds_dac
plot_kwargs_2 = dict(
    x="longitude", 
    y="latitude", 
    cmap="Spectral_r",  # Use the same colormap or change as needed
    vmin=-0.2, 
    vmax=0.5,
    cbar_kwargs={"shrink": 0.7, "pad": 0.05},
)

# ds_dac variable pcolormesh
ds_dac.plot.pcolormesh(ax=ax2, **plot_kwargs_2)

# Typhoon track (hourly interpolated)
df_bebinca_hourly.plot(x='longitude', y='latitude', ax=ax2, color='black', marker='.', linewidth=3, zorder=2, label='Typhoon track')

# Highlight specific track time with red marker
ax2.scatter(df_bebinca_hourly['longitude'].iloc[track_time], df_bebinca_hourly['latitude'].iloc[track_time], color='red', s=150, zorder=3)

# Coastlines and land features for the second plot
ax2.coastlines()
ax2.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='black')

# Gridline label settings for the second plot
gl2 = ax2.gridlines(draw_labels=True)
gl2.left_labels = False  # Disable left labels since it's on the right subplot
gl2.bottom_labels = True
gl2.right_labels = False
gl2.top_labels = False
gl2.xlabel_style = {'size': 16}
gl2.ylabel_style = {'size': 16}
ax2.set_title('Sea level anomaly', fontsize=20)
ax2.set_ylabel('')
ax2.set_xlabel('')

# Adjust layout
plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05, wspace=0.2)

plt.show()

