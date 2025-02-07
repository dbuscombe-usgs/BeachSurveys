
# from datetime import datetime
# import os
# import numpy as np
# import matplotlib.pyplot as plt 

import os
import numpy as np
import pandas as pd
import geopandas as gpd
# import warnings
from datetime import datetime
# import pytz
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import xarray as xr 
from glob import glob
from tqdm import tqdm
from scipy import interpolate
import pytz
import calendar

import rasterio
from rasterio.transform import Affine
import rasterio.warp

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

        gdf["date"] = datetime.strptime(date_str, "%Y%m%d")
        gdfs.append(gdf)
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    return combined_gdf


def toTimestamp(d):
  return calendar.timegm(d.timetuple())


#############################################################################################################################





### data = https://www.nature.com/articles/sdata2018115, https://doi.pangaea.de/10.1594/PANGAEA.885014
file = '/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Narrabeen/raw/Sydney/syd_0.0001_gcs.txt'

head = open(file)
lines = head.readlines()[:6]
ncols = int(lines[0].split('\n')[0].split('ncols')[-1].strip())
nrows = int(lines[1].split('\n')[0].split('nrows')[-1].strip())
xllcorner = float(lines[2].split('\n')[0].split('xllcorner')[-1].strip())
yllcorner = float(lines[3].split('\n')[0].split('yllcorner')[-1].strip())
cellsize = float(lines[4].split('\n')[0].split('cellsize')[-1].strip())


xvec = np.arange(xllcorner,xllcorner+(ncols*cellsize),cellsize)
yvec = np.arange(yllcorner,yllcorner+(nrows*cellsize),cellsize)
xv, yv = np.meshgrid(xvec,yvec, indexing='xy')

dat = np.loadtxt(file, skiprows=6)
dat[dat==-9999.0] = np.nan

dat = np.flipud(dat)

minlon = np.min(xvec)
maxlon = np.max(xvec)
minlat = np.min(yvec)
maxlat = np.max(yvec)

xres = (maxlon - minlon) / dat.shape[1]
yres = (maxlat - minlat) / dat.shape[0]

transform = Affine.translation(minlon - xres / 2, minlat - yres / 2) * Affine.scale(xres, yres)

with rasterio.open(
        f"sydney_topobathy.tif",
        mode="w",
        driver="GTiff",
        height=dat.shape[0],
        width=dat.shape[1],
        count=1,
        dtype=dat.dtype,
        crs="+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs",
        transform=transform,
) as new_dataset:
        new_dataset.write(dat, 1)



# # contour levels for each site
# contour_dict = {'NARRABEEN':  {'MSL':  0,      'MHWS': 0.7  }, # survey datum is MSL
#                 'DUCK':       {'MSL':  -0.128, 'MHWS': 0.457}, # survey datum is NAVD88
#                 'TORREYPINES':{'MSL':  0.774 , 'MHWS': 1.566}, # survey datum is NAVD88
#                 'TRUCVERT':   {'MSL':  0     , 'MHWS': 1.5  }, # survey datum is MSL
#                }


# sitename = 'NARRABEEN'

MSL = 0
MHWS = 0.7

fp_raw = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Narrabeen/raw/"
transects_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Narrabeen/NarrabeenforZooSDS_paper_analysis/"
shoreline_dir = r"/media/marda/FOURTB/SDS/Zoo_SDS_paper/field_survey_data/Narrabeen/"



# read the csv file
fp_dataset = os.path.join(fp_raw,'Narrabeen_Profiles.csv')
df = pd.read_csv(fp_dataset)
pf_names = list(np.unique(df['Profile ID']))

# # select contour level
# contour = 'MSL' # other option is 'MHWS'
# contour_level = contour_dict[sitename][contour]
# print('Extracting time-series for %d transects using the %.1f m contour...'%(len(pf_names), contour_level))


###### BEGIN code originaly from https://github.com/SatelliteShorelines/SDS_Benchmark/blob/main/1_preprocess_datasets.ipynb

chainages_interp = np.arange(-1,100,1)

S=[]
# initialise topo_profiles structure
topo_profiles = dict([])
for i in range(len(pf_names)): # for each profile
     # read dates
    df_pf = df.loc[df['Profile ID'] == pf_names[i]]
    dates_str = df['Date']
    dates_unique = np.unique(dates_str)
    # loop through dates
    topo_profiles[pf_names[i]] = {'dates':[],'chainages':[],'slopes':[]}
    slopes = []
    for date in dates_unique:
        # extract chainage and elevation for that date
        df_date = df_pf.loc[dates_str == date]
        chainages = np.array(df_date['Chainage'])
        elevations = np.array(df_date['Elevation'])
        if len(chainages) < 3: continue
        # sort by chainages
        idx_sorted = np.argsort(chainages)
        chainages = chainages[idx_sorted]
        elevations = elevations[idx_sorted]
        # chainages_interp = np.arange(np.min(chainages)-1,np.max(chainages)+1,1)
        elevations_interp = np.interp(chainages_interp,chainages,elevations)
        # use interpolation to extract the chainage at the contour level
        f = interpolate.interp1d(elevations_interp, chainages_interp, bounds_error=False)
        chainage_contour_level = f(MSL)
        topo_profiles[pf_names[i]]['chainages'].append(chainage_contour_level)
        date_utc = pytz.utc.localize(datetime.strptime(date,'%Y-%m-%d'))
        topo_profiles[pf_names[i]]['dates'].append(date_utc)
        # calculate beach slope from MSL to MHWS
        idx = np.where(np.logical_and(elevations_interp >= MSL,elevations_interp <= MHWS))[0]
        if len(chainages_interp[idx])<2:
            slope = np.nan
        elif np.max(np.diff(chainages_interp[idx])) > 1:
            slope = np.nan
        else:
            slope = -np.polyfit(chainages_interp[idx], elevations_interp[idx], 1)[0]
        slopes.append(slope)
    S.append(slopes)
    topo_profiles[pf_names[i]]['slopes'] = slopes
    print('%s - av. slope %.3f std %.3f'%(pf_names[i],np.nanmean(slopes),np.nanstd(slopes)))

###### END code originaly from https://github.com/SatelliteShorelines/SDS_Benchmark/blob/main/1_preprocess_datasets.ipynb



c=[]
for k in topo_profiles.keys():
    tmp = np.array(topo_profiles[k]['chainages'])
    c.append(tmp)


uniq_dates = np.unique(np.hstack((topo_profiles['PF1']['dates'],topo_profiles['PF2']['dates'],topo_profiles['PF4']['dates'],topo_profiles['PF6']['dates'],topo_profiles['PF8']['dates'])))


utimes = np.array([toTimestamp(d) for d in uniq_dates])
tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF1']['dates']])
pf1 = np.interp(utimes,tmp_times,topo_profiles['PF1']['chainages'])

tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF2']['dates']])
pf2 = np.interp(utimes,tmp_times,topo_profiles['PF2']['chainages'])

tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF4']['dates']])
pf4 = np.interp(utimes,tmp_times,topo_profiles['PF4']['chainages'])

tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF6']['dates']])
pf6 = np.interp(utimes,tmp_times,topo_profiles['PF6']['chainages'])

tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF8']['dates']])
pf8 = np.interp(utimes,tmp_times,topo_profiles['PF8']['chainages'])

v = np.vstack((pf1,pf2,pf4,pf6,pf8))

plt.pcolormesh(np.arange(len(uniq_dates)),np.arange(len(topo_profiles)), np.vstack(v))
plt.show()

df = pd.DataFrame(data=np.vstack(v).T,  index=uniq_dates, columns=topo_profiles.keys())
df.to_csv('Narrabeen_MSL_fieldsurvey_shoreline_chainage.csv')





utimes = np.array([toTimestamp(d) for d in uniq_dates])
tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF1']['dates']])
pf1 = np.interp(utimes,tmp_times,topo_profiles['PF1']['slopes'])

tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF2']['dates']])
pf2 = np.interp(utimes,tmp_times,topo_profiles['PF2']['slopes'])

tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF4']['dates']])
pf4 = np.interp(utimes,tmp_times,topo_profiles['PF4']['slopes'])

tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF6']['dates']])
pf6 = np.interp(utimes,tmp_times,topo_profiles['PF6']['slopes'])

tmp_times = np.array([toTimestamp(d) for d in topo_profiles['PF8']['dates']])
pf8 = np.interp(utimes,tmp_times,topo_profiles['PF8']['slopes'])

v = np.vstack((pf1,pf2,pf4,pf6,pf8))



df = pd.DataFrame(data=np.vstack(v).T,  index=uniq_dates, columns=topo_profiles.keys())
df.to_csv('Narrabeen_fieldsurvey_slopes.csv')



plt.figure(figsize=(16,8))

plt.subplot(121)
plt.plot(np.linspace(0,2950,5), df.mean(axis=0), label='mean')
plt.plot(np.linspace(0,2950,5), df.median(axis=0), label='median')
plt.xlabel('Alongshore distance (m)')
plt.ylabel('Slope (m/m)')
plt.legend()


plt.subplot(122)
plt.plot(df.mean(axis=1), label='mean')
plt.plot(df.median(axis=1), label='median')
# plt.xlabel('Alongshore distance (m)')
plt.ylabel('Slope (m/m)')
plt.legend()

plt.savefig('Narrabeen_slope_space_time.png',dpi=200, bbox_inches='tight')
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
plt.savefig('Narrabeen_Avslope_MonthOfYear.png',dpi=200, bbox_inches='tight')
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
plt.pcolormesh(np.arange(0,12), np.arange(0,5), np.vstack(MM))#, vmin=0.06, vmax=0.1)
plt.colorbar()#extend='both')
plt.xlabel('Month of Year')
plt.ylabel('Transect ID')
plt.gca().invert_yaxis()
plt.savefig('Narrabeen_spacetime_monthly.png',dpi=200, bbox_inches='tight')
plt.close()







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



# filenames = glob(fp_raw+'*.nc')
# filenames = [_ for _ in filenames if '.nc' in _]

# # # read DEMs (already interpolated on a grid)
# # fp_files = os.path.join(fp_raw,'torrey_mapped_sand_elevations','torrey_mapped_sand_elevations') # check that path is correct
# # # fp_figs = os.path.join(data_folder,'figs_dem')
# # # if not os.path.exists(fp_figs): os.makedirs(fp_figs)
# # survey_data = dict([])
# # filenames = os.listdir(fp_files)


# # format the data by profile and not by date
# # fp_figs = os.path.join(data_folder,'figs_topo')
# # if not os.path.exists(fp_figs): os.makedirs(fp_figs)
# # topo_profiles = dict([])
# # pf_names = ['PF%s'%(str(int(survey_data[date_str]['pf'][0,i]))) for i in range(survey_data[date_str]['pf'].shape[1])]
# # n_surveys = np.zeros(len(pf_names))



# # select contour level
# contour = 'MSL' # other option is 'MHWS'
# contour_level = contour_dict[sitename][contour]
# # print('Extracting time-series for %d transects using the %.1f m contour...'%(len(pf_names),contour_level))




# # read the .nc files and store the date and elevation for each file
# for i,fn in enumerate(filenames):

#     data = xr.open_dataset(fn,engine='netcdf4')

#     date_str = fn.split('map')[-1].split('_')[0]
#     # date = pytz.utc.localize(datetime.strptime(date_str,'%Y%m%d'))

#     elevation = np.array(data.variables['mapz'][:])
#     elevation[elevation < -100] = np.nan

#     cs=plt.contour(data.xc.values,data.yc.values,elevation,(-99,contour_level,99), colors='k')
#     plt.close()

#     t = [len(cs.collections[1].get_paths()[k]) for k in range(len(cs.collections[1].get_paths()))]

#     ind = np.argmax(t)

#     p = cs.collections[1].get_paths()[ind]
#     v = p.vertices
#     x = v[:,0]
#     y = v[:,1]

#     df = pd.DataFrame({'x':x,'y':y,'z':np.ones(len(x))})
#     gdf = gpd.GeoDataFrame(
#         df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:32611" #crs="EPSG:26911"
#     )

#     gdf2 = gdf.groupby(['z'])['geometry'].apply(lambda x: LineString(x.tolist()))
#     gdf2 = gpd.GeoDataFrame(gdf2,  crs="EPSG:32611")
#     gdf2.to_file(f"{date_str}_line.json", driver="GeoJSON")



# ################################################################

# # this scripts assumes all transects are epsg 4326

# # SLS_T32_transect_epsg4326.geojson

# transect_files = glob(transects_dir + "/name*.geojson")
# # print(transect_files)

# # read each file into a single geodataframe
# transects_gdf = gpd.GeoDataFrame(columns=["geometry", "id"])

# for file in transect_files:
#     gdf = gpd.read_file(file)
#     # assign the filename based on the filename
#     id = os.path.basename(file).split("_")[1]
#     gdf["id"] = id
#     # need to concatenate the geodataframes
#     transects_gdf = gpd.GeoDataFrame(pd.concat([transects_gdf, gdf], ignore_index=True))


# shoreline_files = sorted(glob(shoreline_dir + "/*line*.json"))

# shoreline_gdf = combine_geojson_files_with_dates(shoreline_files)
# print(shoreline_gdf.head(2))
# print(f"shoreline.crs: {shoreline_gdf.crs}")
# shorelines_dict = {}


# transects_gdf.to_crs(32611, inplace=True)
# print(transects_gdf.crs)
# print(f"transects gdf: {transects_gdf.head(2)}")

# transects_dict = get_transect_points_dict(transects_gdf)
# print(transects_dict)


# dates = []
# shorelines = []

# for row in shoreline_gdf.iterrows():
#     date_str = row[1].date.strftime("%Y%m%d")
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


# df = pd.DataFrame(data=np.vstack(v).T,  index=dates, columns=cross_dist.keys())
# df.to_csv('TorreyPines_fieldsurvey_shoreline_chainage.csv')

















# # times = ds_disk.time.values

# # for i in range(len(times)):

# #     zmat = np.array(ds_disk.sel(t=i).z)
# #     xmat = np.array(ds_disk.sel(t=0).E)
# #     ymat = np.array(ds_disk.sel(t=0).N)
# #     # if i==0:
# #     xv, yv = np.meshgrid(xmat,ymat, indexing='xy')
# #     ts = pd.to_datetime(str(times[i]))
# #     d = ts.strftime('%Y_%m_%d')




# # for i,fn in enumerate(filenames):

# #     data = Dataset(os.path.join(fp_files,fn))
# #     date_str = fn.split('_')[0].split('map')[1]
# #     date = pytz.utc.localize(datetime.strptime(date_str,'%Y%m%d'))
# #     survey_data[date_str] = dict([])
# #     survey_data[date_str]['date'] = date
# #     elevation = np.array(data.variables['mapz'][:])
# #     elevation[elevation < -100] = np.nan
# #     # store in dictionary
# #     survey_data[date_str]['elevation'] = elevation
# #     survey_data[date_str]['x'] = np.array(data.variables['xc'][:])
# #     survey_data[date_str]['y'] = np.array(data.variables['yc'][:])
# #     survey_data[date_str]['pf'] = np.array(data.variables['alg'][:])
# #     survey_data[date_str]['lat'] = np.array(data.variables['latc'][:])
# #     survey_data[date_str]['lon'] = np.array(data.variables['lonc'][:])    

# #     # plot the last DEM for illustration
# #     if i == len(filenames) - 1:
# #         fig, ax = plt.subplots(1,1,figsize=(12,8),tight_layout=True)
# #         ax.grid(which='major',ls=':',c='0.5')
# #         ax.axis('equal')
# #         sc = ax.scatter(survey_data[date_str]['lon'],survey_data[date_str]['lat'],c=survey_data[date_str]['elevation'])
# #         ax.set(title='Survey on %s'%date_str,xlabel='longitude',ylabel='latitude')
# #         plt.colorbar(sc,label='m NAVD88')
# #         # plt.savefig(os.path.join(fp_figs,'survey_%s.jpg'%date_str))
# #         # plt.close(plt.gcf())

# # print('Processed %d surveys'%len(filenames))



# # slopes = []
# # # select transect to plot for illustration
# # transects_to_plot = ['PF592']
# # # uncomment to plot all transects
# # # transects_to_plot = pf_names
# # for i in range(len(pf_names)):
# #     topo_profiles[pf_names[i]] = {'dates':[],'chainages':[],'slopes':[]}
# #     # plot one of the profiles for illustration
# #     if pf_names[i] in transects_to_plot:
# #         fig, ax = plt.subplots(1,1,figsize=(12,8),tight_layout=True)
# #         ax.grid(which='major', linestyle=':', color='0.5')
# #     # loop through each survey and extract the contour
# #     for n in list(survey_data.keys()):
# #         survey = survey_data[n]
# #         date = survey_data[n]['date']
# #         elevations = survey['elevation'][:,i]
# #         # remove nans
# #         idx_nan = np.isnan(elevations)
# #         # if less than 5 survey points along the transect, skip it
# #         if len(elevations) - sum(idx_nan) <= 5: continue
# #         # create transect with its eastings northings coordinates
# #         transect = np.zeros([survey['x'].shape[0],2])
# #         transect[:,0] = survey['x'][:,i]
# #         transect[:,1] = survey['y'][:,i]
# #         # flip it so that origin is on land pointing seawards
# #         transect = np.flipud(transect)
# #         # calculate chainage
# #         x = np.sqrt((transect[:,0]-transect[0,0])**2 + (transect[:,1]-transect[0,1])**2)
# #         # also flip elevations to match chainages
# #         z = np.flipud(elevations) 
# #         # remove nans
# #         idx_nan = np.isnan(z)
# #         x = x[~idx_nan]
# #         z = z[~idx_nan]
# #         # use interpolation to extract the chainage at the contour level
# #         f = interpolate.interp1d(z, x, bounds_error=False)
# #         chainage_contour_level = float(f(contour_level))            
# #         topo_profiles[pf_names[i]]['chainages'].append(chainage_contour_level)
# #         topo_profiles[pf_names[i]]['dates'].append(date)  
# #         if pf_names[i] in transects_to_plot:
# #             ax.plot(x,z,'-',c='0.6',lw=1)
# #             ax.plot(chainage_contour_level,contour_level,'r.')
# #         # calculate beach slope (MSL to MHWS)
# #         idx = np.where(np.logical_and(z >= contour_dict[sitename]['MSL'],
# #                                       z <= contour_dict[sitename]['MHWS']))[0]
# #         if len(x[idx])<2:
# #             slope = np.nan
# #         elif np.max(np.diff(x[idx])) > 5:
# #             slope = np.nan
# #         else:
# #             slope = -np.polyfit(x[idx], z[idx], 1)[0]
# #         slopes.append(slope)
# #     topo_profiles[pf_names[i]]['slopes'] = slopes
# #     print('%s - av. slope %.3f - std %.3f'%(pf_names[i],np.nanmean(slopes),np.nanstd(slopes)))

# #     # convert to np.array
# #     topo_profiles[pf_names[i]]['chainages'] = np.array(topo_profiles[pf_names[i]]['chainages'])
# #     n_surveys[i] = len(topo_profiles[pf_names[i]]['dates'])
# #     if len(topo_profiles[pf_names[i]]['dates']) == 0:
# #         topo_profiles.pop(pf_names[i])
# #         continue
# #     if pf_names[i] in transects_to_plot:
# #         ax.set(title = 'Transect %s  - %d surveys'%(pf_names[i],len(topo_profiles[pf_names[i]]['dates'])),
# #                xlabel='chainage [m]', ylabel='elevation [m]', ylim=[-1,5],
# #                xlim=[np.nanmin(topo_profiles[pf_names[i]]['chainages'])-5,
# #                      np.nanmax(topo_profiles[pf_names[i]]['chainages'])+5])    
# #         # fig.savefig(os.path.join(fp_figs, 'transect_%s.jpg'%pf_names[i]), dpi=100)
# #         # plt.close(plt.gcf())


# # # save time-series in a pickle file
# # fp_save = os.path.join(data_folder, '%s_groundtruth_%s.pkl'%(sitename,contour))
# # with open(fp_save, 'wb') as f:
# #     pickle.dump(topo_profiles, f)
# # print('Time-series for the %.1f m contour along %d transects were saved at %s'%(contour_level,len(pf_names),fp_save))

# # # Extracting time-series for 79 transects using the 0.8 m contour...

# # # plot survey density
# # fig, ax = plt.subplots(1,1,figsize=(12,6), tight_layout=True)
# # ax.grid(which='major', linestyle=':', color='0.5')
# # ax.bar(x=survey_data[date_str]['pf'][0,:],height=n_surveys,width=1,fc='C0',ec='None',alpha=0.75)
# # ax.set(xlabel='transect number',ylabel='number of surveys',title='Survey density');
# # fig.savefig(os.path.join(data_folder, '%s_survey_density.jpg'%sitename), dpi=100);

# # # plot time-series along specific transect
# # selected_transects = ['PF540','PF580']
# # fig = plt.figure(figsize=[15,8], tight_layout=True)
# # fig.suptitle('Time-series of shoreline change at %s ( %.1fm contour)'%(sitename,contour_level))
# # gs = gridspec.GridSpec(len(selected_transects),1)
# # gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.12)
# # for i,key in enumerate(selected_transects):
# #     ax = fig.add_subplot(gs[i,0])
# #     ax.grid(linestyle=':', color='0.5')
# #     ax.plot(topo_profiles[key]['dates'], topo_profiles[key]['chainages'], '-o', ms=4, mfc='w')
# #     ax.set_ylabel('distance [m]', fontsize=12)
# #     ax.text(0.1,0.95, key, bbox=dict(boxstyle="square", ec='k',fc='w'), ha='center',
# #             va='top', transform=ax.transAxes, fontsize=14)  
# # fig.savefig(os.path.join(data_folder, '%s_insitu_timeseries_%s.jpg'%(sitename,contour)), dpi=100);


# # # read polygon ROI
# # fn_polygon = os.path.join(data_folder, '%s_polygon.geojson'%sitename)
# # gdf_polygon = gpd.read_file(fn_polygon)
# # print('Loaded polygon in epsg:%d'%gdf_polygon.crs.to_epsg())
# # polygon = np.array(gdf_polygon.loc[0,'geometry'].exterior.coords)
# # # read reference shoreline
# # fn_refsl = os.path.join(data_folder, '%s_reference_shoreline.geojson'%sitename)
# # gdf_refsl = gpd.read_file(fn_refsl)
# # print('Loaded reference shoreline in epsg:%d'%gdf_refsl.crs.to_epsg())
# # refsl = np.array(gdf_refsl.loc[0,'geometry'].coords)
# # # read transects
# # fn_transects = os.path.join(data_folder, '%s_transects.geojson'%sitename)
# # gdf_transects = gpd.read_file(fn_transects)
# # print('Loaded transects in epsg:%d'%gdf_transects.crs.to_epsg())
# # # put transects into a dictionary with their name
# # transects = dict([])
# # for i in gdf_transects.index:
# #     transects[gdf_transects.loc[i,'name']] = np.array(gdf_transects.loc[i,'geometry'].coords)


# # # plot transects and polygon
# # fig,ax = plt.subplots(1,1,figsize=(10,8),tight_layout=True)
# # ax.grid(which='major',ls=':',c='0.5',lw=1)
# # ax.axis('equal')
# # ax.set(xlabel='Longitude',ylabel='Latitude',title='ROI and transects for %s'%sitename)
# # for i,key in enumerate(list(transects.keys())):
# #     if i % 5 == 0 : # plot one every 5 transects
# #         ax.plot(transects[key][0,0],transects[key][0,1], 'bo',mfc='None',ms=6)
# #         ax.plot(transects[key][:,0],transects[key][:,1],'k-',lw=1)
# #         ax.text(transects[key][0,0]+0.007, transects[key][0,1], key,
# #                 va='center', ha='right', bbox=dict(boxstyle="square", ec='k',fc='w'))
# # ax.plot(polygon[:,0],polygon[:,1],'r-o',mfc='None',ms=10)
# # ax.plot(refsl[:,0],refsl[:,1],'b-')
# # print('Plotted polygon, reference shoreline and transects')
# # fig.savefig(os.path.join(data_folder, '%s_inputs.jpg'%sitename), dpi=100);

# # # load tide time-series
# # fn_tides = os.path.join(data_folder,'%s_tides.csv'%sitename)
# # tide_data = pd.read_csv(fn_tides, parse_dates=['dates'])
# # dates_ts = [_.to_pydatetime() for _ in tide_data['dates']]
# # tides_ts = np.array(tide_data['tides'])
# # print('Loaded tide time-series')


# # # plot tide time-series
# # fig, ax = plt.subplots(1,1,figsize=(10,4), tight_layout=True)
# # ax.grid(which='major', linestyle=':', color='0.5')
# # ax.plot(tide_data['dates'], tide_data['tides'], '-',lw=1)
# # ax.set(ylabel='tide level [m]',title='Modelled tides at %s'%sitename)
# # print('Plotted tide time-series')
# # fig.savefig(os.path.join(data_folder, '%s_tides.jpg'%sitename), dpi=100);

# # # load wave time-series
# # fn_waves = os.path.join(data_folder, sitename + '_waves_ERA5.csv')
# # wave_data = pd.read_csv(fn_waves, parse_dates=['dates'])
# # dates_ts = [_.to_pydatetime() for _ in wave_data['dates']]
# # wave_params = {'swh':[],'mwd':[],'pp1d':[]}
# # for key in wave_params.keys():
# #     wave_params[key] = list(wave_data[key])
# #     idx_str = np.where([isinstance(_,str) for _ in wave_params[key] ])[0]
# #     for i in idx_str:
# #         wave_params[key][i] = (float(wave_params[key][i].split('[')[-1].split(']')[0]))
# #     wave_params[key] = np.array(wave_params[key]) 
# # print('Loaded wave time-series')

# # # plot wave time-series
# # fig, ax = plt.subplots(3,1,figsize=(10,8), tight_layout=True)
# # ax[0].grid(which='major', linestyle=':', color='0.5')
# # ax[0].plot(dates_ts, wave_params['swh'], '-',lw=1)
# # ax[0].axhline(y=np.nanmean(wave_params['swh']),ls='--',c='r',lw=1,label='mean Hs = %.1f'%np.nanmean(wave_params['swh']))
# # ax[0].set(ylabel='Hsig [m]',title='ERA5 wave time-series at %s'%sitename)
# # ax[0].legend(loc='upper left')
# # ax[1].grid(which='major', linestyle=':', color='0.5')
# # ax[1].plot(dates_ts, wave_params['pp1d'], '-',lw=1)
# # ax[1].axhline(y=np.nanmean(wave_params['pp1d']),ls='--',c='r',lw=1,label='mean Hs = %.1f'%np.nanmean(wave_params['pp1d']))
# # ax[1].set(ylabel='Tp [s]')
# # ax[1].legend(loc='upper left')
# # ax[2].grid(which='major', linestyle=':', color='0.5')
# # ax[2].plot(dates_ts, wave_params['mwd'], '-',lw=1)
# # ax[2].set(ylabel='Wdir [deg]')
# # print('Plotted wave time-series')
# # fig.savefig(os.path.join(data_folder, '%s_waves.jpg'%sitename), dpi=200);




