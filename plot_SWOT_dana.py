    # loop to to a pcolor of all the files opening them one by one in xarray
import glob
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
import numpy as np
import cartopy.crs           as ccrs
import matplotlib.gridspec   as gridspec
import matplotlib.ticker as mticker
import cartopy.feature as cf
from geopy.geocoders import Nominatim
import matplotlib.patches as patches


lonmin, lonmax, latmin, latmax = -1,3,36,41.5

ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_023_320_20241102T153709_20241102T162836_v1.0.2.nc')
ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_023_348_20241103T153741_20241103T162907_v1.0.2.nc')
ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_023_363_20241104T042923_20241104T052049_v1.0.2.nc')
ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_023_376_20241104T153812_20241104T162938_v1.0.2.nc')
# ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_023_391_20241105T042954_20241105T052120_v1.0.2.nc')  # demasiado al suroeste, no hay datos para valencia
ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_024_042_20241113T135958_20241113T145124_v1.0.2.nc')  # Just NADIR DATA
# ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_024_057_20241114T025140_20241114T034306_v1.0.2.nc')
# ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_024_070_20241114T140029_20241114T145155_v1.0.2.nc')
# ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_024_085_20241115T025211_20241115T034337_v1.0.2.nc')
# ds = xr.open_dataset('/home/dvega/anaconda3/SWOT_L3_LR_SSH_Expert_024_098_20241115T140100_20241115T145226_v1.0.2.nc') # doesn't work

# Change the longitude from 0-360 to -180-180
ds['longitude'] = xr.where(ds['longitude'] > 180, ds['longitude'] - 360, ds['longitude'])

# Filter the data to the area of interest
ds = ds.where((ds['longitude'] < lonmax) & (ds['longitude'] > lonmin) &
                        (ds['latitude'] < latmax) & (ds['latitude'] > latmin), drop=True)
 


# Calculate the time in the middle of the time range
time_swot= pd.to_datetime(ds.time)
# Drop NaTs (missing values)
cleaned_time = time_swot.dropna()

# Calculate the mean by converting to numeric timestamps (nanoseconds since epoch)
mean_time = np.mean(cleaned_time.astype(np.int64))

# Convert back to datetime and round to the nearest minute
mean_time = pd.to_datetime(mean_time).round('T')


address='Valencia'
geolocator = Nominatim(user_agent="Your_Name")
location = geolocator.geocode(address)
print(location.address)
print((location.latitude, location.longitude))

address2='Denia'
geolocator2 = Nominatim(user_agent="Your_Name")
location2 = geolocator2.geocode(address2)

nscale = 5 #30
nheadlength = 5 #5nheadwidth = 2

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])

ax = plt.subplot(gs[0], projection=ccrs.PlateCarree())
ax.set_extent([lonmin, lonmax, latmin, latmax])
# Plot the magnitude of the velocities
velocity_magnitude = np.sqrt(ds.ugos**2 + ds.vgos**2)

magnitude_plot = ax.pcolormesh(ds.longitude, ds.latitude, velocity_magnitude, cmap='inferno', transform=ccrs.PlateCarree(), vmin=0, vmax=1)
cbar = plt.colorbar(magnitude_plot, ax=ax, shrink=0.7, pad=0.01)
cbar.set_label('Velocity Magnitude (m/s)')
ds_big = ds.where(velocity_magnitude > 0.8, drop=True)


p = ax.quiver(ds.longitude[0:-1:5, 0:-1:5], ds.latitude[0:-1:5, 0:-1:5],
              ds.ugos[0:-1:5, 0:-1:5], ds.vgos[0:-1:5, 0:-1:5],
              scale=nscale, angles='xy', scale_units='xy', headlength=nheadlength, transform=ccrs.PlateCarree(),
              color='black')

q = ax.quiver(ds.longitude[0:-1:5, 0:-1:5], ds.latitude[0:-1:5, 0:-1:5],
              ds.ugos[0:-1:5, 0:-1:5], ds.vgos[0:-1:5, 0:-1:5],
              scale=nscale, angles='xy', scale_units='xy', headlength=nheadlength, transform=ccrs.PlateCarree(),
              color='white')


ax.quiver(ds_big.longitude[0:-1:5, 0:-1:5], ds_big.latitude[0:-1:5, 0:-1:5],
          ds_big.ugos[0:-1:5, 0:-1:5], ds_big.vgos[0:-1:5, 0:-1:5],
          scale=nscale/1.3, angles='xy', scale_units='xy', width=0.008, headlength=nheadlength, headwidth = 2.5, transform=ccrs.PlateCarree(),
          color='k')



# Add the quiver key using the black dummy quiver
ax.quiverkey(p, 1.05, 0.075, 0.5, r'$0.5 \, \frac{m}{s}$', labelpos='E', coordinates='axes', zorder=5)

ax.set_title(f'Surface currents from SWOT satellite observations\n {mean_time}', fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
#label size
ax.tick_params(axis='both', which='major', labelsize=10)

  
#gridlines
gl = ax.gridlines(draw_labels=True, linestyle='-',alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
land = cf.NaturalEarthFeature('physical', 'land', '10m', edgecolor='tan', facecolor='tan',alpha=0.3, zorder=1)
ax.add_feature(land)
coast = cf.GSHHSFeature(scale='full')
ax.add_feature(coast)
ax.scatter(location.longitude, location.latitude, s=25, transform=ccrs.PlateCarree(), c='red',zorder=100)
ax.text(location.longitude-0.25, location.latitude+0.05, 'València', transform=ccrs.PlateCarree(), fontsize=10)
ax.scatter(location2.longitude, location2.latitude, s=25, transform=ccrs.PlateCarree(), c='black',zorder=100)
ax.text(location2.longitude-0.13, location2.latitude+0.05, 'Dénia', transform=ccrs.PlateCarree(), fontsize=10)

# plt.savefig('SWOT_geo_vel_plot_pass363.png', dpi=300, bbox_inches='tight')


# # plot sigma0

# ax = plt.subplot(gs[0], projection=ccrs.PlateCarree())
# ax.set_extent([lonmin, lonmax, latmin, latmax])
# # Plot the magnitude of the velocities

# np.log10(ds.sigma0).plot.pcolormesh(ax=ax, x="longitude", y="latitude", cmap='Spectral_r', vmin=0.8, vmax=1.8)

# ax.quiver(ds.longitude[0:-1:5, 0:-1:5], ds.latitude[0:-1:5, 0:-1:5],
#            ds.ugos[0:-1:5, 0:-1:5], ds.vgos[0:-1:5, 0:-1:5],
#         scale=nscale, angles='xy', scale_units='xy', headlength=nheadlength, transform=ccrs.PlateCarree()
#     #, headwidth=nheadwidth#, scale=nscale, width=0.005#, scale=10, , alpha=.5
#     ,label='SWOT', color='white')
# ax.quiver(ds_big.longitude[0:-1:5, 0:-1:5], ds_big.latitude[0:-1:5, 0:-1:5],
#            ds_big.ugos[0:-1:5, 0:-1:5], ds_big.vgos[0:-1:5, 0:-1:5],
#         scale=nscale, angles='xy', scale_units='xy', headlength=nheadlength, transform=ccrs.PlateCarree()
#     #, headwidth=nheadwidth#, scale=nscale, width=0.005#, scale=10,
#     ,label='SWOT', color='k')
# ax.set_title('Sigma0 from SWOT 04/11/2024 ~ 04:23:00', fontsize=10)
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')

# ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
# ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
# #label size
# ax.tick_params(axis='both', which='major', labelsize=10)

  
# #gridlines
# gl = ax.gridlines(draw_labels=True, linestyle='-',alpha=0.5)
# gl.xlabels_top = False
# gl.ylabels_right = False
# land = cf.NaturalEarthFeature('physical', 'land', '10m', edgecolor='tan', facecolor='tan',alpha=0.3)
# ax.add_feature(land)
# coast = cf.GSHHSFeature(scale='full')
# ax.add_feature(coast)
# ax.scatter(location.longitude, location.latitude, s=25, transform=ccrs.PlateCarree(), c='red',zorder=100)
# ax.text(location.longitude-0.25, location.latitude+0.05, 'València', transform=ccrs.PlateCarree(), fontsize=10)
# ax.scatter(location2.longitude, location2.latitude, s=25, transform=ccrs.PlateCarree(), c='black',zorder=100)
# ax.text(location2.longitude-0.13, location2.latitude+0.05, 'Dénia', transform=ccrs.PlateCarree(), fontsize=10)

