import os
import netCDF4 as nc
import pdb
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature


def CMEMS_get_positions(folder_path):
    file_dict = {}
    
    #--- get all files names in the folder ending with .nc
    file_names = sorted([file_name for file_name in os.listdir(folder_path) if file_name.endswith('.nc')])

    #--- voy pasando por cada uno de los archivos+
    for nf, file_name in enumerate(file_names):
        file_path = os.path.join(folder_path, file_name)
        try:
            dataset = nc.Dataset(file_path)

            # I want to read the data from the varuiables lon, lat and station_name
            lon = dataset.variables['LONGITUDE'][:].data[0]
            lat = dataset.variables['LATITUDE'][:].data[0]
            time = dataset.variables['TIME'][:].data.min()
            name = file_name
            dataset.close()

            # I want to return the data in a dictionary
            file_dict[nf] = {
                    'lon': lon,
                    'lat': lat,
                    'name': name,
                    'time': time,
                    'filename': folder_path+file_name
                }
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
    
    #--- I want to merge all the lon, lat and name in a dictionary
    field_names = list(file_dict[0].keys())
    new_dict = {}
    for field_name in field_names:
        all_values = [data[field_name] for data in file_dict.values()]
        new_dict[field_name] = all_values
    
    return new_dict         

folder_path = 'C:/Users/Diego/INSITU_GLO_PHY_SSH_DISCRETE_MY_013_053/cmems_obs-ins_glo_phy-ssh_my_na_PT1H_202311/history/TG/'

data = CMEMS_get_positions(folder_path)
ref_time = pd.Timestamp('1950-01-01T00:00:000')
time = ref_time + pd.to_timedelta(data['time'], unit='D')
len(time[time>"2010-01-01"])

df = pd.DataFrame(data)

df_2010= df[time>"2010-01-01"]


lolabox = [-20, 40, 20, 60]

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))

# Set the extent to focus on the defined lon-lat box
ax.set_extent(lolabox, crs=ccrs.PlateCarree())
ax.gridlines(draw_labels=True)
scatter = ax.scatter(df_2010['lon'], df_2010['lat'], s=20, transform=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, color='lightgray')
# Add coastlines and gridlines
ax.coastlines()

