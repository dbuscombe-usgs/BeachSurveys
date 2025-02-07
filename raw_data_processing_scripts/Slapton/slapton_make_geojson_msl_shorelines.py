

from glob import glob 
import os 
import numpy as np
import geopandas as gpd
from scipy.interpolate import griddata
import pandas as pd

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
from shapely.geometry import LineString
from datetime import datetime, timedelta
from pymatreader import read_mat

# import datetime
from tqdm import tqdm
# import pandas as pd
# import glob
# import geopandas as gpd
# import os
# import numpy as np
import datetime
# from tqdm import tqdm
# from datetime import datetime
# from shapely.geometry import MultiLineString
from coastseg.common import convert_date_gdf, convert_points_to_linestrings
import shapely



def combine_geojson_files_with_dates(file_paths):
    """
    Combines multiple GeoJSON files into a single GeoDataFrame and assigns a date column 
    based on the filenames.

    Parameters:
    - file_paths (list of str): List of file paths to GeoJSON files.

    Returns:
    - GeoDataFrame: Combined GeoDataFrame containing all features from the input files, 
                    with an added date column.
    """
    gdfs = []
    for file in file_paths:
        gdf = gpd.read_file(file)
        if "z" in gdf.columns:
                gdf.drop(columns="z", inplace=True)
        date_str = "-".join(os.path.basename(file).split("_")[:-1])
        if len(date_str.split("-")[0]) == 4:  # year month day
            date_str = "-".join(date_str.split("-")[::-1])
        gdf["date"] = datetime.strptime(date_str, "%d-%m-%Y")
        gdfs.append(gdf)
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    return combined_gdf


def wgs84_to_utm_df(geo_df):
    """
    Converts gdf from wgs84 to UTM
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in wgs84
    outputs:
    geo_df_utm (geopandas  dataframe): a geopandas dataframe in utm
    """
    utm_crs = geo_df.estimate_utm_crs()
    gdf_utm = geo_df.to_crs(utm_crs)
    return gdf_utm

def cross_distance(start_x, start_y, end_x, end_y):
    """distance formula, sqrt((x_1-x_0)^2 + (y_1-y_0)^2)"""
    dist = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
    return dist

def utm_to_wgs84_df(geo_df):
    """
    Converts gdf from utm to wgs84
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in utm
    outputs:
    geo_df_wgs84 (geopandas  dataframe): a geopandas dataframe in wgs84
    """
    wgs84_crs = 'epsg:4326'
    gdf_wgs84 = geo_df.to_crs(wgs84_crs)
    return gdf_wgs84



#############################################################################################################################

### SDS data
sls = read_mat("/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Slapton/SLSdata.mat")

sls = sls['SLS']

profData = sls['profData']
sat = sls['sat']
transTimeSeries = sls['transTimeSeries']
# info = sls['info']
dnum = sls['dnum']
transects = sls['transects']


for ya,yb,xa,xb,name in zip(transects['startUTMlat'], transects['endUTMlat'], transects['startUTMlon'], transects['endUTMlon'], sorted(transTimeSeries.keys())):
    print(name)

    df = pd.DataFrame({'x':[xa,xb],'y':[ya,yb],'z':[0,0]})

    # df = pd.DataFrame({'y':[xa,xb],'x':[ya,yb],'z':[0,0]})

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:32630", #32630" , 27700
    )

    gdf2 = gdf.groupby(['z'])['geometry'].apply(lambda x: LineString(x.tolist()))
    gdf2 = gpd.GeoDataFrame(gdf2,  crs="EPSG:32630")
    gdf2.to_file(f"SLS_{name}_transect.json", driver="GeoJSON")



################################################################

data = pd.read_csv('/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Slapton/DS08_STB_Profiles/DS08_STB_Profiles.csv')
dates = [i.replace('-','_') for i in data['date_str']]

data_grid = []
for counter,k in enumerate(np.unique(dates)):
    
    tmp = data[data['date_str']==k]

    Y = tmp['N'].values
    X = tmp['E'].values
    Z = tmp['z'].values
    if counter==0:
        x = np.arange(np.min(X),np.max(X),1)
        y = np.arange(np.min(Y),np.max(Y),1)
        xv, yv = np.meshgrid(x,y, indexing='xy')
    grid_z0 = griddata((X,Y), Z, (xv, yv), method='linear')
    data_grid.append(grid_z0)


data_grid = np.dstack(data_grid)

dates = [i.replace('/','_') for i in data['date_str']]
dates = np.unique(dates)
################################################################

MLS = 0
MHWS = 4.3

plt.close('all')
# Xmsl = []
# Ymsl = []
# survey = []
for k in range(data_grid.shape[-1]):

    try:
        cs=plt.contour(xv,yv,data_grid[:,:,k],(-99,MLS,99), colors='k')
        # p = cs.collections[1].get_paths()[0]

        t = [len(cs.get_paths()[k]) for k in range(len(cs.get_paths()))]
        ind = np.argmax(t)
        p = cs.get_paths()[ind]

        v = p.vertices
        x = v[:,0]
        y = v[:,1]
        df = pd.DataFrame({'x':x,'y':y,'z':np.ones(len(x))})
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:27700"
        )

        gdf2 = gdf.groupby(['z'])['geometry'].apply(lambda x: LineString(x.tolist()))
        gdf2 = gpd.GeoDataFrame(gdf2,  crs="EPSG:27700")
        del gdf, df, cs, p, v, x, y

        cs=plt.contour(xv,yv,data_grid[:,:,k],(-99,MHWS,99), colors='k')
        # p = cs.collections[1].get_paths()[0]

        t = [len(cs.get_paths()[k]) for k in range(len(cs.get_paths()))]
        ind = np.argmax(t)
        p = cs.get_paths()[ind]

        v = p.vertices
        x = v[:,0]
        y = v[:,1]
        df = pd.DataFrame({'x':x,'y':y,'z':np.ones(len(x))})
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:27700"
        )

        gdf3 = gdf.groupby(['z'])['geometry'].apply(lambda x: LineString(x.tolist()))
        gdf3 = gpd.GeoDataFrame(gdf3,  crs="EPSG:27700")
        del gdf, df, cs, p, v, x, y


        gdf2.to_file(f"{dates[k]}_lineMSL.json", driver="GeoJSON")
        gdf3.to_file(f"{dates[k]}_lineMHWS.json", driver="GeoJSON")


    except:
        print(f"{k} failed")


################################################################
shoreline_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Slapton/SlaptonforZooSDSpaper_analysis/msl"
# this scripts assumes all transects are epsg 4326
transects_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Slapton/SlaptonforZooSDSpaper_analysis/"
# SLS_T32_transect_epsg4326.geojson

transect_files = glob(transects_dir + "/SLS_T*.geojson")
# print(transect_files)

# read each file into a single geodataframe
transects_gdf = gpd.GeoDataFrame(columns=["geometry", "id"])

for file in transect_files:
    gdf = gpd.read_file(file)
    # assign the filename based on the filename
    id = os.path.basename(file).split("_")[1]
    gdf["id"] = id
    # need to concatenate the geodataframes
    transects_gdf = gpd.GeoDataFrame(pd.concat([transects_gdf, gdf], ignore_index=True))


transects_gdf.to_crs(27700, inplace=True)
print(transects_gdf.crs)
print(f"transects gdf: {transects_gdf.head(2)}")

# transects_dict = get_transect_points_dict(transects_gdf)
# print(transects_dict)


shoreline_files = glob(shoreline_dir + "/*lineMSL.json")

shoreline_gdf = combine_geojson_files_with_dates(shoreline_files)
print(shoreline_gdf.head(2))
print(f"shoreline.crs: {shoreline_gdf.crs}")
# shorelines_dict = {}


# estimate the crs of the transects
crs = transects_gdf.estimate_utm_crs()
transects_gdf.to_crs(crs, inplace=True)



id_column_name = 'id'
transects_gdf = transects_gdf.rename(columns={id_column_name: 'transect_id'})

transects_gdf = transects_gdf.reset_index(drop=True)
transects_gdf['geometry_saved'] = transects_gdf['geometry']
coords = transects_gdf['geometry_saved'].get_coordinates()
coords = coords[~coords.index.duplicated(keep='first')]
transects_gdf['x_start'] = coords['x']
transects_gdf['y_start'] = coords['y']

# load shorelines, project to utm, smooth
shorelines_gdf = wgs84_to_utm_df(shoreline_gdf)

# join all the shorelines that occured on the same date together
shorelines_gdf = shorelines_gdf.dissolve(by='date')
shorelines_gdf = shorelines_gdf.reset_index()


# if the shorelines are multipoints convert them to linestrings because this function does not work well with multipoints
if 'MultiPoint' in [geom_type for geom_type in shorelines_gdf['geometry'].geom_type]:
    shorelines_gdf = convert_points_to_linestrings(shorelines_gdf, group_col='date', output_crs=crs)

# spatial join shorelines to transects
joined_gdf = gpd.sjoin(shorelines_gdf, transects_gdf, predicate='intersects')

# get points, keep highest cross distance point if multipoint (most seaward intersection)
joined_gdf['intersection_point'] = joined_gdf.geometry.intersection(joined_gdf['geometry_saved'])

for i in range(len(joined_gdf['intersection_point'])):
    point = joined_gdf['intersection_point'].iloc[i]
    start_x = joined_gdf['x_start'].iloc[i]
    start_y = joined_gdf['y_start'].iloc[i]
    if type(point) == shapely.MultiPoint:
        points = [shapely.Point(coord) for coord in point.geoms]
        points = gpd.GeoSeries(points, crs=crs)
        coords = points.get_coordinates()
        dists = [None]*len(coords)
        for j in range(len(coords)):
            dists[j] = cross_distance(start_x, start_y, coords['x'].iloc[j], coords['y'].iloc[j])
        max_dist_idx = np.argmax(dists)
        last_point = points[max_dist_idx]
        joined_gdf['intersection_point'].iloc[i] = last_point

# get x's and y's for intersections
intersection_coords = joined_gdf['intersection_point'].get_coordinates()
joined_gdf['shore_x'] = intersection_coords['x']
joined_gdf['shore_y'] = intersection_coords['y']

# get cross distance
joined_gdf['cross_distance'] = cross_distance(joined_gdf['x_start'], 
                                                joined_gdf['y_start'], 
                                                joined_gdf['shore_x'], 
                                                joined_gdf['shore_y'])
##clean up columns
joined_gdf = joined_gdf.rename(columns={'date':'dates'})
keep_columns = ['dates','satname','geoaccuracy','cloud_cover','transect_id',
                'shore_x','shore_y','cross_distance','x','y',]

# get start of each transect
transects_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(joined_gdf['x_start'], joined_gdf['y_start']),crs=crs)
transects_gdf = transects_gdf.to_crs('epsg:4326')

# convert the x and y intersection points to the final crs (4326) to match the rest of joined_df
points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(joined_gdf['shore_x'], joined_gdf['shore_y']),crs=crs)
points_gdf = points_gdf.to_crs('epsg:4326')

# you have to reset the index here otherwise the intersection point won't match the row correctly
# recall that the shorelines were group by dates and that changed the index
joined_gdf = joined_gdf.rename(columns={'date':'dates'}).reset_index(drop=True)

joined_gdf['shore_x'] = points_gdf.geometry.x
joined_gdf['shore_y'] = points_gdf.geometry.y
joined_gdf['x'] =  transects_gdf.geometry.x
joined_gdf['y'] =  transects_gdf.geometry.y

# convert the joined_df back to CRS 4326
joined_gdf = utm_to_wgs84_df(joined_gdf)


for col in joined_gdf.columns:
    if col not in keep_columns:
        joined_gdf = joined_gdf.drop(columns=[col])

joined_df = joined_gdf.reset_index(drop=True)

# convert the dates column from panddas dateime object to UTC with +00:00 timezone
joined_df['dates'] = pd.to_datetime(joined_df['dates'])
# sort by dates
joined_df = joined_df.sort_values(by='dates')

# check if the dates column is already in UTC
if joined_df['dates'].dt.tz is None:
    joined_df['dates'] = joined_df['dates'].dt.tz_localize('UTC')



plt.figure(figsize=(16,16))

plt.subplot(311)
id = np.unique(joined_df.transect_id.values)[5]
y = joined_df[joined_df.transect_id.values==id]['cross_distance']
x = joined_df[joined_df.transect_id.values==id]['dates']
plt.plot(x,y,'k-', alpha=0.5)
plt.plot(x,y,'k.')
plt.title(f'Transect {id}')

plt.subplot(312)
id = np.unique(joined_df.transect_id.values)[15]
y = joined_df[joined_df.transect_id.values==id]['cross_distance']
x = joined_df[joined_df.transect_id.values==id]['dates']
plt.plot(x,y,'k-', alpha=0.5)
plt.plot(x,y,'k.')
plt.title(f'Transect {id}')

plt.subplot(313)
id = np.unique(joined_df.transect_id.values)[-1]
y = joined_df[joined_df.transect_id.values==id]['cross_distance']
x = joined_df[joined_df.transect_id.values==id]['dates']
plt.plot(x,y,'k-', alpha=0.5)
plt.plot(x,y,'k.')
plt.title(f'Transect {id}')
plt.ylabel('Shoreline position (m)')

# plt.show()
plt.savefig('Slapton_MSL_timeseries_example.png', dpi=300, bbox_inches='tight')
plt.close()


joined_df.to_csv('Slapton_MSL_fieldsurvey_shoreline_chainage.csv')

joined_gdf = gpd.GeoDataFrame(
    joined_df, geometry=gpd.points_from_xy(joined_df.x, joined_df.y), crs="EPSG:4326"
)

joined_gdf.to_file("Slapton_MSL_fieldsurvey_shoreline_chainage.geojson", driver="GeoJSON") 




###############################################################################################
############################################
######### mhws
shoreline_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Slapton/SlaptonforZooSDSpaper_analysis/mhws"
shoreline_files = glob(shoreline_dir + "/*lineMHWS.json")

shoreline_gdf = combine_geojson_files_with_dates(shoreline_files)
print(shoreline_gdf.head(2))
print(f"shoreline.crs: {shoreline_gdf.crs}")
# shorelines_dict = {}



# read each file into a single geodataframe
transects_gdf = gpd.GeoDataFrame(columns=["geometry", "id"])

for file in transect_files:
    gdf = gpd.read_file(file)
    # assign the filename based on the filename
    id = os.path.basename(file).split("_")[1]
    gdf["id"] = id
    # need to concatenate the geodataframes
    transects_gdf = gpd.GeoDataFrame(pd.concat([transects_gdf, gdf], ignore_index=True))


transects_gdf.to_crs(27700, inplace=True)
print(transects_gdf.crs)
print(f"transects gdf: {transects_gdf.head(2)}")

# estimate the crs of the transects
crs = transects_gdf.estimate_utm_crs()
transects_gdf.to_crs(crs, inplace=True)



id_column_name = 'id'
transects_gdf = transects_gdf.rename(columns={id_column_name: 'transect_id'})

transects_gdf = transects_gdf.reset_index(drop=True)
transects_gdf['geometry_saved'] = transects_gdf['geometry']
coords = transects_gdf['geometry_saved'].get_coordinates()
coords = coords[~coords.index.duplicated(keep='first')]
transects_gdf['x_start'] = coords['x']
transects_gdf['y_start'] = coords['y']

# load shorelines, project to utm, smooth
shorelines_gdf = wgs84_to_utm_df(shoreline_gdf)

# join all the shorelines that occured on the same date together
shorelines_gdf = shorelines_gdf.dissolve(by='date')
shorelines_gdf = shorelines_gdf.reset_index()


# if the shorelines are multipoints convert them to linestrings because this function does not work well with multipoints
if 'MultiPoint' in [geom_type for geom_type in shorelines_gdf['geometry'].geom_type]:
    shorelines_gdf = convert_points_to_linestrings(shorelines_gdf, group_col='date', output_crs=crs)

# spatial join shorelines to transects
joined_gdf = gpd.sjoin(shorelines_gdf, transects_gdf, predicate='intersects')

# get points, keep highest cross distance point if multipoint (most seaward intersection)
joined_gdf['intersection_point'] = joined_gdf.geometry.intersection(joined_gdf['geometry_saved'])

for i in range(len(joined_gdf['intersection_point'])):
    point = joined_gdf['intersection_point'].iloc[i]
    start_x = joined_gdf['x_start'].iloc[i]
    start_y = joined_gdf['y_start'].iloc[i]
    if type(point) == shapely.MultiPoint:
        points = [shapely.Point(coord) for coord in point.geoms]
        points = gpd.GeoSeries(points, crs=crs)
        coords = points.get_coordinates()
        dists = [None]*len(coords)
        for j in range(len(coords)):
            dists[j] = cross_distance(start_x, start_y, coords['x'].iloc[j], coords['y'].iloc[j])
        max_dist_idx = np.argmax(dists)
        last_point = points[max_dist_idx]
        joined_gdf['intersection_point'].iloc[i] = last_point

# get x's and y's for intersections
intersection_coords = joined_gdf['intersection_point'].get_coordinates()
joined_gdf['shore_x'] = intersection_coords['x']
joined_gdf['shore_y'] = intersection_coords['y']

# get cross distance
joined_gdf['cross_distance'] = cross_distance(joined_gdf['x_start'], 
                                                joined_gdf['y_start'], 
                                                joined_gdf['shore_x'], 
                                                joined_gdf['shore_y'])
##clean up columns
joined_gdf = joined_gdf.rename(columns={'date':'dates'})
keep_columns = ['dates','satname','geoaccuracy','cloud_cover','transect_id',
                'shore_x','shore_y','cross_distance','x','y',]

# get start of each transect
transects_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(joined_gdf['x_start'], joined_gdf['y_start']),crs=crs)
transects_gdf = transects_gdf.to_crs('epsg:4326')

# convert the x and y intersection points to the final crs (4326) to match the rest of joined_df
points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(joined_gdf['shore_x'], joined_gdf['shore_y']),crs=crs)
points_gdf = points_gdf.to_crs('epsg:4326')

# you have to reset the index here otherwise the intersection point won't match the row correctly
# recall that the shorelines were group by dates and that changed the index
joined_gdf = joined_gdf.rename(columns={'date':'dates'}).reset_index(drop=True)

joined_gdf['shore_x'] = points_gdf.geometry.x
joined_gdf['shore_y'] = points_gdf.geometry.y
joined_gdf['x'] =  transects_gdf.geometry.x
joined_gdf['y'] =  transects_gdf.geometry.y

# convert the joined_df back to CRS 4326
joined_gdf = utm_to_wgs84_df(joined_gdf)


for col in joined_gdf.columns:
    if col not in keep_columns:
        joined_gdf = joined_gdf.drop(columns=[col])

joined_df = joined_gdf.reset_index(drop=True)

# convert the dates column from panddas dateime object to UTC with +00:00 timezone
joined_df['dates'] = pd.to_datetime(joined_df['dates'])
# sort by dates
joined_df = joined_df.sort_values(by='dates')

# check if the dates column is already in UTC
if joined_df['dates'].dt.tz is None:
    joined_df['dates'] = joined_df['dates'].dt.tz_localize('UTC')



plt.figure(figsize=(16,16))

plt.subplot(311)
id = np.unique(joined_df.transect_id.values)[5]
y = joined_df[joined_df.transect_id.values==id]['cross_distance']
x = joined_df[joined_df.transect_id.values==id]['dates']
plt.plot(x,y,'k-', alpha=0.5)
plt.plot(x,y,'k.')
plt.title(f'Transect {id}')

plt.subplot(312)
id = np.unique(joined_df.transect_id.values)[15]
y = joined_df[joined_df.transect_id.values==id]['cross_distance']
x = joined_df[joined_df.transect_id.values==id]['dates']
plt.plot(x,y,'k-', alpha=0.5)
plt.plot(x,y,'k.')
plt.title(f'Transect {id}')

plt.subplot(313)
id = np.unique(joined_df.transect_id.values)[-1]
y = joined_df[joined_df.transect_id.values==id]['cross_distance']
x = joined_df[joined_df.transect_id.values==id]['dates']
plt.plot(x,y,'k-', alpha=0.5)
plt.plot(x,y,'k.')
plt.title(f'Transect {id}')
plt.ylabel('Shoreline position (m)')

# plt.show()
plt.savefig('Slapton_MHWS_timeseries_example.png', dpi=300, bbox_inches='tight')
plt.close()


joined_df.to_csv('Slapton_MHWS_fieldsurvey_shoreline_chainage.csv')

joined_gdf = gpd.GeoDataFrame(
    joined_df, geometry=gpd.points_from_xy(joined_df.x, joined_df.y), crs="EPSG:4326"
)

joined_gdf.to_file("Slapton_MHWS_fieldsurvey_shoreline_chainage.geojson", driver="GeoJSON") 




#################################################################################
### SLOPE
joined_gdf_MSL = gpd.read_file("Slapton_MSL_fieldsurvey_shoreline_chainage.geojson")
joined_gdf_MHW = gpd.read_file("Slapton_MHWS_fieldsurvey_shoreline_chainage.geojson")

print(joined_gdf_MSL.head())
print(joined_gdf_MHW.head())


print(len(joined_gdf_MSL)) #17857
print(len(joined_gdf_MHW)) #17873

MSL_distances_by_time_and_transect = joined_gdf_MSL.pivot(index='dates',columns='transect_id', values='cross_distance').values

MHW_distances_by_time_and_transect = joined_gdf_MHW.pivot(index='dates',columns='transect_id', values='cross_distance').values


## get horizontal distance
hdist = MHW_distances_by_time_and_transect - MSL_distances_by_time_and_transect

dz = MHWS - MLS


slopes = np.tan(dz/hdist)

slopes[slopes>.4]=np.nan
slopes[slopes<.001]=np.nan

np.nanmedian(slopes)


df = pd.DataFrame(data=slopes,  index=np.unique(joined_gdf_MHW.dates.values), columns=np.unique(joined_gdf_MSL['transect_id']))

df.to_csv('Slapton_fieldsurvey_slopes.csv')


plt.figure(figsize=(16,8))

plt.subplot(121)
plt.plot(np.linspace(0,5185,28), df.mean(axis=0), label='mean')
plt.plot(np.linspace(0,5185,28), df.median(axis=0), label='median')
plt.xlabel('Alongshore distance (m)')
plt.ylabel('Slope (m/m)')
plt.legend()


plt.subplot(122)
plt.plot(df.mean(axis=1), label='mean')
plt.plot(df.median(axis=1), label='median')
# plt.xlabel('Alongshore distance (m)')
plt.ylabel('Slope (m/m)')
plt.legend()

plt.savefig('Slapton_slope_space_time.png',dpi=200, bbox_inches='tight')
plt.close()




df['Date'] = pd.to_datetime(df.index.values)
# Extract the month
df['Month'] = df['Date'].dt.month

M =[]; S =[]
# Group by January (month 1)
for i in range(1,13):
    data = df[df['Month'] == i].groupby('Month').mean() 
    data.drop('Date',axis=1, inplace=True)

    M.append(float(data.median(axis=1)))
    S.append(float(data.std(axis=1)))


plt.errorbar(np.arange(1,13),M,S,S)
plt.ylabel('Transect-averaged slope')
plt.xlabel('Month of Year')
plt.savefig('Slapton_Avslope_MonthOfYear.png',dpi=200, bbox_inches='tight')
plt.close()



MM = []
for k in df.columns[:-2]:
    M =[]; 
    # Group by January (month 1)
    for i in range(1,13):
        data = df[df['Month'] == i].groupby('Month').mean() 
        data.drop('Date',axis=1, inplace=True)

        M.append(float(data[k]))
    MM.append(M)


plt.figure(figsize=(8,16))
plt.subplot(121)
plt.pcolormesh(np.arange(0,12), np.arange(0,28), np.vstack(MM), vmin=0.05, vmax=0.2)
plt.colorbar(extend='both')
plt.xlabel('Month of Year')
plt.ylabel('Transect ID')
plt.gca().invert_yaxis()
plt.savefig('Slapton_spacetime_monthly.png',dpi=200, bbox_inches='tight')
plt.close()











# dates = []
# shorelines = []

# for row in shoreline_gdf.iterrows():
#     date_str = row[1].date.strftime("%d-%m-%Y")
#     shorelines_array = np.array(shoreline_gdf.iloc[row[0]].geometry.coords)
#     shorelines.append(shorelines_array)
#     dates.append(date_str)

# shorelines_dict['dates'] = dates
# shorelines_dict['shorelines'] = shorelines
# print(shorelines_dict)


# # # compute the intersection
# cross_dist = compute_intersection_QC(shorelines_dict, transects_dict)

# print(f"cross distance: {cross_dist}")

# v=[]
# for k in cross_dist.keys():
#     v.append(cross_dist[k])


# plt.pcolormesh(np.arange(len(dates)),np.arange(len(cross_dist)), np.vstack(v))
# plt.show()

# df = pd.DataFrame(data=np.vstack(v).T,  index=dates, columns=cross_dist.keys())
# df.to_csv('Slapton_fieldsurvey_shoreline_chainage.csv')





# shoreline_files2 = [s.replace('MSL','MHWS') for s in shoreline_files]


# #sorted(glob(shoreline_dir + "/*lineMHWS.json"))

# shoreline_gdf2 = combine_geojson_files_with_dates(shoreline_files2)
# print(shoreline_gdf2.head(2))
# print(f"shoreline.crs: {shoreline_gdf2.crs}")
# shorelines_dict2 = {}


# dates2 = []
# shorelines2 = []

# for row in shoreline_gdf2.iterrows():
#     date_str = row[1].date.strftime("%Y%m%d")
#     print(date_str)
#     shorelines_array2 = np.array(shoreline_gdf2.iloc[row[0]].geometry.coords)
#     shorelines2.append(shorelines_array2)
#     dates2.append(date_str)

# shorelines_dict2['dates'] = dates2
# shorelines_dict2['shorelines'] = shorelines2
# print(shorelines_dict2)


# # # compute the intersection
# cross_dist2 = compute_intersection_QC(shorelines_dict2, transects_dict)

# print(f"cross distance: {cross_dist2}")

# v2=[]
# for k in cross_dist2.keys():
#     v2.append(cross_dist2[k])





# vslope = np.vstack(v) - np.vstack(v2)

# dz = 4.3

# slopes = np.vstack(np.tan(dz/vslope))
# slopes[slopes<0] = np.nan
# slopes[slopes>.4] = np.nan


# np.nanmedian(slopes)


# df = pd.DataFrame(data=slopes,  index=cross_dist.keys(), columns=dates)
# df.to_csv('Slapton_fieldsurvey_slopes.csv')





# shoreline_chainage=[]
# for k in transTimeSeries.keys():
#     chainage = transTimeSeries[k][:,0] #only chainage values
#     shoreline_chainage.append(chainage)

# shoreline_chainage = np.vstack(shoreline_chainage)
# transect_names = transTimeSeries.keys()

# python_datetime = [datetime.fromordinal(int(d)) + timedelta(days=d%1) - timedelta(days = 366) for d in dnum]

# d = [ts.strftime('%Y_%m_%d') for ts in python_datetime]
# df = pd.DataFrame(data=shoreline_chainage.T,  index=d, columns=transect_names)
# df.to_csv('Slapton_survey_shoreline_chainage.csv')



# def compute_intersection_QC(output, 
#                             transects,
#                             along_dist = 25, 
#                             min_points = 3,
#                             max_std = 15,
#                             max_range = 30, 
#                             min_chainage = -100, 
#                             multiple_inter ="auto",
#                             prc_multiple=0.1, 
#                             use_progress_bar: bool = True):
#     """
    
#     More advanced function to compute the intersection between the 2D mapped shorelines
#     and the transects. Produces more quality-controlled time-series of shoreline change.

#     Arguments:
#     -----------
#         output: dict
#             contains the extracted shorelines and corresponding dates.
#         transects: dict
#             contains the X and Y coordinates of the transects (first and last point needed for each
#             transect).
#         along_dist: int (in metres)
#             alongshore distance to calculate the intersection (median of points
#             within this distance).
#         min_points: int
#             minimum number of shoreline points to calculate an intersection.
#         max_std: int (in metres)
#             maximum std for the shoreline points when calculating the median,
#             if above this value then NaN is returned for the intersection.
#         max_range: int (in metres)
#             maximum range for the shoreline points when calculating the median,
#             if above this value then NaN is returned for the intersection.
#         min_chainage: int (in metres)
#             furthest landward of the transect origin that an intersection is
#             accepted, beyond this point a NaN is returned.
#         multiple_inter: mode for removing outliers ('auto', 'nan', 'max').
#         prc_multiple: float, optional
#             percentage to use in 'auto' mode to switch from 'nan' to 'max'.
#         use_progress_bar(bool,optional). Defaults to True. If true uses tqdm to display the progress for iterating through transects.
#             False, means no progress bar is displayed.

#     Returns:
#     -----------
#         cross_dist: dict
#             time-series of cross-shore distance along each of the transects. These are not tidally
#             corrected.
#     """

#     # initialise dictionary with intersections for each transect
#     cross_dist = dict([])

#     shorelines = output["shorelines"]

#     # loop through each transect
#     transect_keys = transects.keys()
#     if use_progress_bar:
#         transect_keys = tqdm(
#             transect_keys, desc="Computing transect shoreline intersections"
#         )

#     for key in transect_keys:
#         # initialise variables
#         std_intersect = np.zeros(len(shorelines))
#         med_intersect = np.zeros(len(shorelines))
#         max_intersect = np.zeros(len(shorelines))
#         min_intersect = np.zeros(len(shorelines))
#         n_intersect = np.zeros(len(shorelines))

#         # loop through each shoreline
#         for i in range(len(shorelines)):
#             sl = shorelines[i]

#             # in case there are no shoreline points
#             if len(sl) == 0:
#                 std_intersect[i] = np.nan
#                 med_intersect[i] = np.nan
#                 max_intersect[i] = np.nan
#                 min_intersect[i] = np.nan
#                 n_intersect[i] = np.nan
#                 continue

#             # compute rotation matrix
#             X0 = transects[key][0, 0]
#             Y0 = transects[key][0, 1]
#             temp = np.array(transects[key][-1, :]) - np.array(transects[key][0, :])
#             phi = np.arctan2(temp[1], temp[0])
#             Mrot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

#             # calculate point to line distance between shoreline points and the transect
#             p1 = np.array([X0, Y0])
#             p2 = transects[key][-1, :]
#             d_line = np.abs(np.cross(p2 - p1, sl - p1) / np.linalg.norm(p2 - p1))
#             # calculate the distance between shoreline points and the origin of the transect
#             d_origin = np.array([np.linalg.norm(sl[k, :] - p1) for k in range(len(sl))])
#             # find the shoreline points that are close to the transects and to the origin
#             idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
#             idx_close = np.where(idx_dist)[0]

#             # in case there are no shoreline points close to the transect
#             if len(idx_close) == 0:
#                 std_intersect[i] = np.nan
#                 med_intersect[i] = np.nan
#                 max_intersect[i] = np.nan
#                 min_intersect[i] = np.nan
#                 n_intersect[i] = np.nan
#             else:
#                 # change of base to shore-normal coordinate system
#                 xy_close = np.array([sl[idx_close, 0], sl[idx_close, 1]]) - np.tile(
#                     np.array([[X0], [Y0]]), (1, len(sl[idx_close]))
#                 )
#                 xy_rot = np.matmul(Mrot, xy_close)
#                 # remove points that are too far landwards relative to the transect origin (i.e., negative chainage)
#                 xy_rot[0, xy_rot[0, :] < min_chainage] = np.nan

#                 # compute std, median, max, min of the intersections
#                 if not np.all(np.isnan(xy_rot[0, :])):
#                     std_intersect[i] = np.nanstd(xy_rot[0, :])
#                     med_intersect[i] = np.nanmedian(xy_rot[0, :])
#                     max_intersect[i] = np.nanmax(xy_rot[0, :])
#                     min_intersect[i] = np.nanmin(xy_rot[0, :])
#                     n_intersect[i] = len(xy_rot[0, :])
#                 else:
#                     std_intersect[i] = np.nan
#                     med_intersect[i] = np.nan
#                     max_intersect[i] = np.nan
#                     min_intersect[i] = np.nan
#                     n_intersect[i] = 0

#         # quality control the intersections using dispersion metrics (std and range)
#         condition1 = std_intersect <= max_std
#         condition2 = (max_intersect - min_intersect) <= max_range
#         condition3 = n_intersect >= min_points
#         idx_good = np.logical_and(np.logical_and(condition1, condition2), condition3)

#         # decide what to do with the intersections with high dispersion
#         if multiple_inter == "auto":
#             # compute the percentage of data points where the std is larger than the user-defined max
#             prc_over = np.sum(std_intersect > max_std) / len(std_intersect)
#             # if more than a certain percentage is above, use the maximum intersection
#             if prc_over > prc_multiple:
#                 med_intersect[~idx_good] = max_intersect[~idx_good]
#                 med_intersect[~condition3] = np.nan
#             # otherwise put a nan
#             else:
#                 med_intersect[~idx_good] = np.nan

#         elif multiple_inter == "max":
#             med_intersect[~idx_good] = max_intersect[~idx_good]
#             med_intersect[~condition3] = np.nan

#         elif multiple_inter == "nan":
#             med_intersect[~idx_good] = np.nan

#         else:
#             raise Exception(
#                 "The multiple_inter parameter can only be: nan, max, or auto."
#             )

#         # store in dict
#         cross_dist[key] = med_intersect

#     return cross_dist

# def get_transect_points_dict(feature: gpd.GeoDataFrame) -> dict:
#     """Returns dict of np.arrays of transect start and end points
#     Example
#     {
#         'usa_CA_0289-0055-NA1': array([[-13820440.53165404,   4995568.65036405],
#         [-13820940.93156407,   4995745.1518021 ]]),
#         'usa_CA_0289-0056-NA1': array([[-13820394.24579453,   4995700.97802925],
#         [-13820900.16320004,   4995862.31860808]])
#     }
#     Args:
#         feature (gpd.GeoDataFrame): clipped transects within roi
#     Returns:
#         dict: dict of np.arrays of transect start and end points
#         of form {
#             '<transect_id>': array([[start point],
#                         [end point]]),}
#     """
#     features = []
#     # Use explode to break multilinestrings in linestrings
#     feature_exploded = feature.explode(ignore_index=True)
#     # For each linestring portion of feature convert to lat,lon tuples
#     lat_lng = feature_exploded.apply(
#         lambda row: {str(row.id): np.array(np.array(row.geometry.coords).tolist())},
#         axis=1,
#     )
#     features = list(lat_lng)
#     new_dict = {}
#     for item in list(features):
#         new_dict = {**new_dict, **item}
#     return new_dict