

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import xarray as xr 
from glob import glob
# from tqdm import tqdm
# from scipy import interpolate
# import pytz
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely import get_coordinates
# from shapely.geometry import MultiLineString
from coastseg.common import convert_date_gdf, convert_points_to_linestrings
import shapely
# import scipy

import rasterio
from rasterio.transform import Affine
import rasterio.warp


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
        date_str = os.path.basename(file).split("_")[0]

        gdf["date"] = datetime.strptime(date_str, "%Y-%m-%d")
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


fp_raw = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/TrucVert/truc_vert_insitu/raw/"
transects_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/TrucVert/TrucVertforZooSDS_paper_analysis/"
shoreline_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/TrucVert/"


MSL = 0
MHWS = 1.5

filenames = glob.glob(fp_raw+'*.nc')
filenames = [_ for _ in filenames if '.nc' in _]
filenames = [_ for _ in filenames if 'Grids' not in _]
filenames = [_ for _ in filenames if 'Waves' not in _]
filenames = [_ for _ in filenames if 'Tide' not in _]

# MHWS = 3.7

LATS=[]
LONS=[]
XX = []; YY = []; ZZ = []; D=[]
# read the .nc files and store the date and elevation for each file
for i,fn in enumerate(filenames):

    data = xr.open_dataset(fn,engine='netcdf4', decode_times=False)
    date_str = fn.split('_')[-1].split('.nc')[0]
    D.append(date_str)

    ## the dataset has an error - the coordinates are not, in fact, in EPSG:2154 RGF93 v1 / Lambert-93 as reported
    # so I have to reproject from lat/lon
    elevation = np.array(data.variables['z'][:])
    lon = np.array(data.variables['lon'][:])
    lat = np.array(data.variables['lat'][:])

    s = gpd.GeoSeries([Point(x,y) for x, y in zip(lon,lat)])
    geo_df = gpd.GeoDataFrame(elevation, geometry=s)
    geo_df.crs = {'init': 'epsg:4326'} 
    geo_df = geo_df.to_crs({'init': 'epsg:2154'}) ####2154 = EPSG:2154 RGF93 v1 / Lambert-93

    X = []; Y = []
    for index, row in geo_df.iterrows():
        for pt in list(row['geometry'].coords): 
            x,y = get_coordinates(Point(pt))[0]
            X.append(x)
            Y.append(y)

    indx = np.where((np.hstack(X)>362000) & (np.hstack(X)<365500) & (np.hstack(Y)>6.4095*10**6) & (np.hstack(Y)<6.41735*10**6) )[0]
    X =np.hstack(X)[indx]
    Y = np.hstack(Y)[indx]
    Z = elevation[indx]

    # ind = np.where((Z>contour_level) & (Z<MHWS))[0]
    # Zfilt = Z[ind]    
    # Xfilt = X[ind]    
    # Yfilt = Y[ind]    

    XX.append(X)
    YY.append(Y)
    ZZ.append(Z)
    LATS.append(lat)
    LONS.append(lon)


x = np.arange(np.min(np.hstack(XX)),np.max(np.hstack(XX)),1)
y = np.arange(np.min(np.hstack(YY)),np.max(np.hstack(YY)),1)
xv, yv = np.meshgrid(x,y, indexing='xy')




# SLS_T32_transect_epsg4326.geojson

transect_files = glob.glob(transects_dir + "/name*.geojson")
# print(transect_files)

# read each file into a single geodataframe
transects_gdf = gpd.GeoDataFrame(columns=["geometry", "name"])

for file in transect_files:
    gdf = gpd.read_file(file)
    # assign the filename based on the filename
    id = os.path.basename(file).split("_")[1].split('.geojson')[0]
    gdf["id"] = id
    # need to concatenate the geodataframes
    transects_gdf = gpd.GeoDataFrame(pd.concat([transects_gdf, gdf], ignore_index=True))


transects_gdf.to_crs(2154, inplace=True)
print(transects_gdf.crs)
print(f"transects gdf: {transects_gdf.head(2)}")

# transects_dict = get_transect_points_dict(transects_gdf)
# print(transects_dict)

Txy_s = []
Txy_e = []

for row in transects_gdf.iterrows():
    Txy_s.append(row[1].geometry.coords[0])
    Txy_e.append(row[1].geometry.coords[1])




SS=[]
for xi, yi, zi, date_str,lat,lon in zip(XX,YY,ZZ, D,LATS,LONS):    

    northing = np.array(yi)
    easting = np.array(xi) #data.variables['y'][:]

    grid_z0 = griddata((easting,northing), zi, (xv, yv), method='linear')

    minlon = np.min(lon)
    maxlon = np.max(lon)
    minlat = np.min(lat)
    maxlat = np.max(lat)

    xres = (maxlon - minlon) / grid_z0.shape[1]
    yres = (maxlat - minlat) / grid_z0.shape[0]

    transform = Affine.translation(minlon - xres / 2, minlat - yres / 2) * Affine.scale(xres, yres)

    with rasterio.open(
            f"trucvert_topobathy_{date_str}.tif",
            mode="w",
            driver="GTiff",
            height=grid_z0.shape[0],
            width=grid_z0.shape[1],
            count=1,
            dtype=grid_z0.dtype,
            crs="+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs",
            transform=transform,
    ) as new_dataset:
            new_dataset.write(grid_z0, 1)


    S = []
    for tis,tie in zip(Txy_s,Txy_e):
        # tis=Txy_s[0]
        # tie=Txy_e[0]
        try:
            xl, yl = np.linspace(tis[0]-xv.min(), tie[0]-xv.min(), 100), np.linspace(tis[1]-yv.min(), tie[1]-yv.min(), 100)
            zi = grid_z0[yl.astype(np.int64), xl.astype(np.int64)]
            ind = np.where((zi>0) & (zi<MHWS))[0]
            dz = zi[ind].max() - zi[ind].min()
            dx = xl[ind].max() - xl[ind].min()
            S.append(np.tan(dz/dx))
        except:
            S.append(np.nan)    

    SS.append(S)

    # grid_tmp = grid_z0.copy()
    # grid_tmp[(grid_tmp>0) & (grid_tmp<MHWS)]
    # dx,dy = np.gradient(grid_tmp)
    # mag=np.sqrt(dx**2 + dy**2)
    # S.append(np.nanmedian(mag))

    cs=plt.contour(xv,yv,grid_z0,(-99,MSL,99), colors='k')
    plt.close()

    # t = [len(cs.collections[1].get_paths()[k]) for k in range(len(cs.collections[1].get_paths()))]
    t = [len(cs.get_paths()[k]) for k in range(len(cs.get_paths()))]
    
    try:
        ind = np.argmax(t)

        # p = cs.collections[1].get_paths()[ind]
        p = cs.get_paths()[ind]
        v = p.vertices
        x = v[:,0]
        y = v[:,1]

        df = pd.DataFrame({'x':x,'y':y,'z':np.zeros(len(x))})
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:2154" #crs="EPSG:2154"
        )

        gdf2 = gdf.groupby(['z'])['geometry'].apply(lambda x: LineString(x.tolist()))
        gdf2 = gpd.GeoDataFrame(gdf2,  crs="EPSG:2154")
        gdf2.to_file(f"{date_str}_MSL_line.json", driver="GeoJSON")
    except:
        pass




df = pd.DataFrame(data=np.vstack(SS).T,  index=np.unique(transects_gdf['id']), columns=D).T
df.to_csv('TrucVert_fieldsurvey_slopes.csv')



plt.figure(figsize=(16,8))

plt.subplot(121)
plt.plot(np.linspace(0,108*40,108), df.mean(axis=0), label='mean')
plt.plot(np.linspace(0,108*40,108), df.median(axis=0), label='median')
plt.xlabel('Alongshore distance (m)')
plt.ylabel('Slope (m/m)')
plt.legend()


plt.subplot(122)
plt.plot(df.mean(axis=1), label='mean')
plt.plot(df.median(axis=1), label='median')
# plt.xlabel('Alongshore distance (m)')
plt.ylabel('Slope (m/m)')
plt.legend()

plt.savefig('TrucVert_slope_space_time.png',dpi=200, bbox_inches='tight')
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
plt.savefig('TrucVert_Avslope_MonthOfYear.png',dpi=200, bbox_inches='tight')
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
plt.pcolormesh(np.arange(0,12), np.arange(0,108), np.vstack(MM), vmin=0.02, vmax=0.1)
plt.colorbar(extend='both')
plt.xlabel('Month of Year')
plt.ylabel('Transect ID')
plt.gca().invert_yaxis()
plt.savefig('TrucVert_spacetime_monthly.png',dpi=200, bbox_inches='tight')
plt.close()



################################################################


################################################################
transects_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/TrucVert/TrucVertforZooSDS_paper_analysis"
shoreline_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/TrucVert/TrucVertforZooSDS_paper_analysis/msl"

##############################################################
transect_files = glob.glob(transects_dir + "/name_*.geojson")

# read each file into a single geodataframe
transects_gdf = gpd.GeoDataFrame(columns=["geometry", "id"])

for file in transect_files:
    gdf = gpd.read_file(file)
    # assign the filename based on the filename
    id = os.path.basename(file).split("_")[1]
    gdf["id"] = id
    # need to concatenate the geodataframes
    transects_gdf = gpd.GeoDataFrame(pd.concat([transects_gdf, gdf], ignore_index=True))

print(transects_gdf.crs)

# estimate the crs of the transects
crs = transects_gdf.estimate_utm_crs()
transects_gdf.to_crs(crs, inplace=True)


##############################################################
shoreline_files = glob.glob(shoreline_dir + "/*MSL*.json")
if not shoreline_files:
    raise ValueError(f"No shoreline files found in the directory {shoreline_dir}")

# print(f"shoreline files: {shoreline_files}")
shoreline_gdf = combine_geojson_files_with_dates(shoreline_files)
print(shoreline_gdf.head(2))
print(f"shoreline.crs: {shoreline_gdf.crs}")



# load transects, project to utm, get start x and y coords
transects_gdf = wgs84_to_utm_df(transects_gdf)
crs = transects_gdf.crs

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
id = np.unique(joined_df.transect_id.values)[54]
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
plt.savefig('TrucVert_MSL_timeseries_example.png', dpi=300, bbox_inches='tight')
plt.close()


joined_df.to_csv('TrucVert_MSL_fieldsurvey_shoreline_chainage.csv')

joined_gdf = gpd.GeoDataFrame(
    joined_df, geometry=gpd.points_from_xy(joined_df.x, joined_df.y), crs="EPSG:4326"
)

joined_gdf.to_file("TrucVert_MSL_fieldsurvey_shoreline_chainage.geojson", driver="GeoJSON") 






# # contour levels for each site
# contour_dict = {'NARRABEEN':  {'MSL':  0,      'MHWS': 0.7  }, # survey datum is MSL
#                 'DUCK':       {'MSL':  -0.128, 'MHWS': 0.457}, # survey datum is NAVD88
#                 'TORREYPINES':{'MSL':  0.774 , 'MHWS': 1.566}, # survey datum is NAVD88
#                 'TRUCVERT':   {'MSL':  0     , 'MHWS': 1.5  }, # survey datum is MSL
#                }


# sitename = 'TRUCVERT'
# # select contour level
# contour = 'MSL' # other option is 'MHWS'
# contour_level = contour_dict[sitename][contour]


# ################################################################

# # this scripts assumes all transects are epsg 4326

# shoreline_files = sorted(glob(shoreline_dir + "/*line*.json"))

# shoreline_gdf = combine_geojson_files_with_dates(shoreline_files)
# print(shoreline_gdf.head(2))
# print(f"shoreline.crs: {shoreline_gdf.crs}")
# shorelines_dict = {}




# dates = []
# shorelines = []

# for row in shoreline_gdf.iterrows():
#     date_str = row[1].date.strftime("%Y-%m-%d")
#     print(date_str)
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
# df.to_csv('TrucVert_fieldsurvey_shoreline_chainage.csv')




# def compute_intersection_QC(output, 
#                             transects,
#                             along_dist = 50, 
#                             min_points = 3,
#                             max_std = 50,
#                             max_range = 3000, 
#                             min_chainage = -1000, 
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



# times = ds_disk.time.values

# for i in range(len(times)):

#     zmat = np.array(ds_disk.sel(t=i).z)
#     xmat = np.array(ds_disk.sel(t=0).E)
#     ymat = np.array(ds_disk.sel(t=0).N)
#     # if i==0:
#     xv, yv = np.meshgrid(xmat,ymat, indexing='xy')
#     ts = pd.to_datetime(str(times[i]))
#     d = ts.strftime('%Y_%m_%d')




# for i,fn in enumerate(filenames):

#     data = Dataset(os.path.join(fp_files,fn))
#     date_str = fn.split('_')[0].split('map')[1]
#     date = pytz.utc.localize(datetime.strptime(date_str,'%Y%m%d'))
#     survey_data[date_str] = dict([])
#     survey_data[date_str]['date'] = date
#     elevation = np.array(data.variables['mapz'][:])
#     elevation[elevation < -100] = np.nan
#     # store in dictionary
#     survey_data[date_str]['elevation'] = elevation
#     survey_data[date_str]['x'] = np.array(data.variables['xc'][:])
#     survey_data[date_str]['y'] = np.array(data.variables['yc'][:])
#     survey_data[date_str]['pf'] = np.array(data.variables['alg'][:])
#     survey_data[date_str]['lat'] = np.array(data.variables['latc'][:])
#     survey_data[date_str]['lon'] = np.array(data.variables['lonc'][:])    

#     # plot the last DEM for illustration
#     if i == len(filenames) - 1:
#         fig, ax = plt.subplots(1,1,figsize=(12,8),tight_layout=True)
#         ax.grid(which='major',ls=':',c='0.5')
#         ax.axis('equal')
#         sc = ax.scatter(survey_data[date_str]['lon'],survey_data[date_str]['lat'],c=survey_data[date_str]['elevation'])
#         ax.set(title='Survey on %s'%date_str,xlabel='longitude',ylabel='latitude')
#         plt.colorbar(sc,label='m NAVD88')
#         # plt.savefig(os.path.join(fp_figs,'survey_%s.jpg'%date_str))
#         # plt.close(plt.gcf())

# print('Processed %d surveys'%len(filenames))



# slopes = []
# # select transect to plot for illustration
# transects_to_plot = ['PF592']
# # uncomment to plot all transects
# # transects_to_plot = pf_names
# for i in range(len(pf_names)):
#     topo_profiles[pf_names[i]] = {'dates':[],'chainages':[],'slopes':[]}
#     # plot one of the profiles for illustration
#     if pf_names[i] in transects_to_plot:
#         fig, ax = plt.subplots(1,1,figsize=(12,8),tight_layout=True)
#         ax.grid(which='major', linestyle=':', color='0.5')
#     # loop through each survey and extract the contour
#     for n in list(survey_data.keys()):
#         survey = survey_data[n]
#         date = survey_data[n]['date']
#         elevations = survey['elevation'][:,i]
#         # remove nans
#         idx_nan = np.isnan(elevations)
#         # if less than 5 survey points along the transect, skip it
#         if len(elevations) - sum(idx_nan) <= 5: continue
#         # create transect with its eastings northings coordinates
#         transect = np.zeros([survey['x'].shape[0],2])
#         transect[:,0] = survey['x'][:,i]
#         transect[:,1] = survey['y'][:,i]
#         # flip it so that origin is on land pointing seawards
#         transect = np.flipud(transect)
#         # calculate chainage
#         x = np.sqrt((transect[:,0]-transect[0,0])**2 + (transect[:,1]-transect[0,1])**2)
#         # also flip elevations to match chainages
#         z = np.flipud(elevations) 
#         # remove nans
#         idx_nan = np.isnan(z)
#         x = x[~idx_nan]
#         z = z[~idx_nan]
#         # use interpolation to extract the chainage at the contour level
#         f = interpolate.interp1d(z, x, bounds_error=False)
#         chainage_contour_level = float(f(contour_level))            
#         topo_profiles[pf_names[i]]['chainages'].append(chainage_contour_level)
#         topo_profiles[pf_names[i]]['dates'].append(date)  
#         if pf_names[i] in transects_to_plot:
#             ax.plot(x,z,'-',c='0.6',lw=1)
#             ax.plot(chainage_contour_level,contour_level,'r.')
#         # calculate beach slope (MSL to MHWS)
#         idx = np.where(np.logical_and(z >= contour_dict[sitename]['MSL'],
#                                       z <= contour_dict[sitename]['MHWS']))[0]
#         if len(x[idx])<2:
#             slope = np.nan
#         elif np.max(np.diff(x[idx])) > 5:
#             slope = np.nan
#         else:
#             slope = -np.polyfit(x[idx], z[idx], 1)[0]
#         slopes.append(slope)
#     topo_profiles[pf_names[i]]['slopes'] = slopes
#     print('%s - av. slope %.3f - std %.3f'%(pf_names[i],np.nanmean(slopes),np.nanstd(slopes)))

#     # convert to np.array
#     topo_profiles[pf_names[i]]['chainages'] = np.array(topo_profiles[pf_names[i]]['chainages'])
#     n_surveys[i] = len(topo_profiles[pf_names[i]]['dates'])
#     if len(topo_profiles[pf_names[i]]['dates']) == 0:
#         topo_profiles.pop(pf_names[i])
#         continue
#     if pf_names[i] in transects_to_plot:
#         ax.set(title = 'Transect %s  - %d surveys'%(pf_names[i],len(topo_profiles[pf_names[i]]['dates'])),
#                xlabel='chainage [m]', ylabel='elevation [m]', ylim=[-1,5],
#                xlim=[np.nanmin(topo_profiles[pf_names[i]]['chainages'])-5,
#                      np.nanmax(topo_profiles[pf_names[i]]['chainages'])+5])    
#         # fig.savefig(os.path.join(fp_figs, 'transect_%s.jpg'%pf_names[i]), dpi=100)
#         # plt.close(plt.gcf())


# # save time-series in a pickle file
# fp_save = os.path.join(data_folder, '%s_groundtruth_%s.pkl'%(sitename,contour))
# with open(fp_save, 'wb') as f:
#     pickle.dump(topo_profiles, f)
# print('Time-series for the %.1f m contour along %d transects were saved at %s'%(contour_level,len(pf_names),fp_save))

# # Extracting time-series for 79 transects using the 0.8 m contour...

# # plot survey density
# fig, ax = plt.subplots(1,1,figsize=(12,6), tight_layout=True)
# ax.grid(which='major', linestyle=':', color='0.5')
# ax.bar(x=survey_data[date_str]['pf'][0,:],height=n_surveys,width=1,fc='C0',ec='None',alpha=0.75)
# ax.set(xlabel='transect number',ylabel='number of surveys',title='Survey density');
# fig.savefig(os.path.join(data_folder, '%s_survey_density.jpg'%sitename), dpi=100);

# # plot time-series along specific transect
# selected_transects = ['PF540','PF580']
# fig = plt.figure(figsize=[15,8], tight_layout=True)
# fig.suptitle('Time-series of shoreline change at %s ( %.1fm contour)'%(sitename,contour_level))
# gs = gridspec.GridSpec(len(selected_transects),1)
# gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.12)
# for i,key in enumerate(selected_transects):
#     ax = fig.add_subplot(gs[i,0])
#     ax.grid(linestyle=':', color='0.5')
#     ax.plot(topo_profiles[key]['dates'], topo_profiles[key]['chainages'], '-o', ms=4, mfc='w')
#     ax.set_ylabel('distance [m]', fontsize=12)
#     ax.text(0.1,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
#             va='top', transform=ax.transAxes, fontsize=14)  
# fig.savefig(os.path.join(data_folder, '%s_insitu_timeseries_%s.jpg'%(sitename,contour)), dpi=100);


# # read polygon ROI
# fn_polygon = os.path.join(data_folder, '%s_polygon.geojson'%sitename)
# gdf_polygon = gpd.read_file(fn_polygon)
# print('Loaded polygon in epsg:%d'%gdf_polygon.crs.to_epsg())
# polygon = np.array(gdf_polygon.loc[0,'geometry'].exterior.coords)
# # read reference shoreline
# fn_refsl = os.path.join(data_folder, '%s_reference_shoreline.geojson'%sitename)
# gdf_refsl = gpd.read_file(fn_refsl)
# print('Loaded reference shoreline in epsg:%d'%gdf_refsl.crs.to_epsg())
# refsl = np.array(gdf_refsl.loc[0,'geometry'].coords)
# # read transects
# fn_transects = os.path.join(data_folder, '%s_transects.geojson'%sitename)
# gdf_transects = gpd.read_file(fn_transects)
# print('Loaded transects in epsg:%d'%gdf_transects.crs.to_epsg())
# # put transects into a dictionary with their name
# transects = dict([])
# for i in gdf_transects.index:
#     transects[gdf_transects.loc[i,'name']] = np.array(gdf_transects.loc[i,'geometry'].coords)


# # plot transects and polygon
# fig,ax = plt.subplots(1,1,figsize=(10,8),tight_layout=True)
# ax.grid(which='major',ls=':',c='0.5',lw=1)
# ax.axis('equal')
# ax.set(xlabel='Longitude',ylabel='Latitude',title='ROI and transects for %s'%sitename)
# for i,key in enumerate(list(transects.keys())):
#     if i % 5 == 0 : # plot one every 5 transects
#         ax.plot(transects[key][0,0],transects[key][0,1], 'bo',mfc='None',ms=6)
#         ax.plot(transects[key][:,0],transects[key][:,1],'k-',lw=1)
#         ax.text(transects[key][0,0]+0.007, transects[key][0,1], key,
#                 va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))
# ax.plot(polygon[:,0],polygon[:,1],'r-o',mfc='None',ms=10)
# ax.plot(refsl[:,0],refsl[:,1],'b-')
# print('Plotted polygon, reference shoreline and transects')
# fig.savefig(os.path.join(data_folder, '%s_inputs.jpg'%sitename), dpi=100);

# # load tide time-series
# fn_tides = os.path.join(data_folder,'%s_tides.csv'%sitename)
# tide_data = pd.read_csv(fn_tides, parse_dates=['dates'])
# dates_ts = [_.to_pydatetime() for _ in tide_data['dates']]
# tides_ts = np.array(tide_data['tides'])
# print('Loaded tide time-series')


# # plot tide time-series
# fig, ax = plt.subplots(1,1,figsize=(10,4), tight_layout=True)
# ax.grid(which='major', linestyle=':', color='0.5')
# ax.plot(tide_data['dates'], tide_data['tides'], '-',lw=1)
# ax.set(ylabel='tide level [m]',title='Modelled tides at %s'%sitename)
# print('Plotted tide time-series')
# fig.savefig(os.path.join(data_folder, '%s_tides.jpg'%sitename), dpi=100);

# # load wave time-series
# fn_waves = os.path.join(data_folder, sitename + '_waves_ERA5.csv')
# wave_data = pd.read_csv(fn_waves, parse_dates=['dates'])
# dates_ts = [_.to_pydatetime() for _ in wave_data['dates']]
# wave_params = {'swh':[],'mwd':[],'pp1d':[]}
# for key in wave_params.keys():
#     wave_params[key] = list(wave_data[key])
#     idx_str = np.where([isinstance(_,str) for _ in wave_params[key] ])[0]
#     for i in idx_str:
#         wave_params[key][i] = (float(wave_params[key][i].split('[')[-1].split(']')[0]))
#     wave_params[key] = np.array(wave_params[key]) 
# print('Loaded wave time-series')

# # plot wave time-series
# fig, ax = plt.subplots(3,1,figsize=(10,8), tight_layout=True)
# ax[0].grid(which='major', linestyle=':', color='0.5')
# ax[0].plot(dates_ts, wave_params['swh'], '-',lw=1)
# ax[0].axhline(y=np.nanmean(wave_params['swh']),ls='--',c='r',lw=1,label='mean Hs = %.1f'%np.nanmean(wave_params['swh']))
# ax[0].set(ylabel='Hsig [m]',title='ERA5 wave time-series at %s'%sitename)
# ax[0].legend(loc='upper left')
# ax[1].grid(which='major', linestyle=':', color='0.5')
# ax[1].plot(dates_ts, wave_params['pp1d'], '-',lw=1)
# ax[1].axhline(y=np.nanmean(wave_params['pp1d']),ls='--',c='r',lw=1,label='mean Hs = %.1f'%np.nanmean(wave_params['pp1d']))
# ax[1].set(ylabel='Tp [s]')
# ax[1].legend(loc='upper left')
# ax[2].grid(which='major', linestyle=':', color='0.5')
# ax[2].plot(dates_ts, wave_params['mwd'], '-',lw=1)
# ax[2].set(ylabel='Wdir [deg]')
# print('Plotted wave time-series')
# fig.savefig(os.path.join(data_folder, '%s_waves.jpg'%sitename), dpi=200);




