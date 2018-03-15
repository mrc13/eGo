#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corr Analysis
"""

# General Packages
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy
import os

import shapely.wkt
from time import localtime, strftime
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

## Logging
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger(__name__)

fh = logging.FileHandler('corr_anal.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

# General Inputs
result_id = 384
data_set = '2018-03-15'
result_dir = 'corr_results/' + str(result_id) + '/data_proc/' + data_set + '/'

now = strftime("%Y-%m-%d %H:%M", localtime())
analysis_dir = 'corr_results/' + str(result_id) + '/analysis/' + now + '/'
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

#%% Data import
try:
    line_df = pd.DataFrame.from_csv(result_dir + 'line_df.csv', encoding='utf-8')
    mv_line_df = pd.DataFrame.from_csv(result_dir + 'mv_line_df.csv', encoding='utf-8')
except:
    logger.warning('No lines imported')

try:
    bus_df = pd.DataFrame.from_csv(result_dir + 'bus_df.csv', encoding='utf-8')
    mv_bus_df = pd.DataFrame.from_csv(result_dir + 'mv_bus_df.csv', encoding='utf-8')
except:
    logger.warning('No buses imported')

try:
    gens_df = pd.DataFrame.from_csv(result_dir + 'gens_df.csv', encoding='utf-8')
except:
    logger.warning('No gens imported')

try:
    trafo_df = pd.DataFrame.from_csv(result_dir + 'trafo_df.csv', encoding='utf-8')
    mv_trafo_df = pd.DataFrame.from_csv(result_dir + 'mv_trafo_df.csv', encoding='utf-8')
except:
    logger.warning('No trafos imported')

try:
    all_hvmv_subst_df = pd.DataFrame.from_csv(result_dir + 'all_hvmv_subst_df.csv', encoding='utf-8')
    snap_idx = pd.Series.from_csv(result_dir + 'snap_idx', encoding='utf-8')
except:
    logger.warning('No Subst. imported')

#%% Further Processing

crs = {'init': 'epsg:4326'}
line_df = gpd.GeoDataFrame(line_df,
                           crs=crs,
                           geometry=line_df.topo.map(shapely.wkt.loads))
line_df = line_df.drop(['geom', 'topo'], axis=1)

crs = {'init': 'epsg:4326'}
mv_line_df = gpd.GeoDataFrame(mv_line_df,
                              crs=crs,
                              geometry=mv_line_df.geom.map(shapely.wkt.loads))
mv_line_df = mv_line_df.drop(['geom'], axis=1)

crs = {'init': 'epsg:4326'}
bus_df = gpd.GeoDataFrame(bus_df,
                          crs=crs,
                          geometry=bus_df.geom.map(shapely.wkt.loads))
bus_df = bus_df.drop(['geom'], axis=1)

crs = {'init': 'epsg:4326'}
mv_bus_df = gpd.GeoDataFrame(mv_bus_df,
                             crs=crs,
                             geometry=mv_bus_df.geom.map(shapely.wkt.loads))
mv_bus_df = mv_bus_df.drop(['geom'], axis=1)

crs = {'init': 'epsg:4326'}
trafo_df = gpd.GeoDataFrame(trafo_df,
                            crs=crs,
                            geometry=trafo_df.point_geom.map(shapely.wkt.loads))
trafo_df = trafo_df.drop(['point_geom'], axis=1)
trafo_df = trafo_df.drop(['geom'], axis=1)
trafo_df['grid_buffer'] = trafo_df['grid_buffer'].map(shapely.wkt.loads)

crs = {'init': 'epsg:4326'}
mv_trafo_df = gpd.GeoDataFrame(mv_trafo_df,
                               crs=crs,
                               geometry=mv_trafo_df.point.map(shapely.wkt.loads))
mv_trafo_df = mv_trafo_df.drop(['geom', 'point'], axis=1)
mv_trafo_df['grid_buffer'] = mv_trafo_df['grid_buffer'].map(shapely.wkt.loads)

crs = {'init': 'epsg:3035'}
mv_griddistricts_df = gpd.GeoDataFrame(mv_trafo_df['grid_buffer'],
                                        crs=crs,
                                        geometry='grid_buffer')
mv_griddistricts_df = mv_griddistricts_df.to_crs({'init': 'epsg:4326'})
mv_trafo_df['grid_buffer'] = mv_griddistricts_df

del mv_griddistricts_df


#%% Data Cleaning

# ToDo: Clean out faulty MV grids, that e.g. have remote generators...

#%% Basic functions and Dicts

hv_levels = pd.unique(line_df['v_nom']).tolist()
mv_levels = pd.unique(mv_line_df['v_nom']).tolist()
all_levels = mv_levels + hv_levels

level_colors = {'LV': 'grey',
                'MV': 'black',
                'HV': 'blue',
                'EHV220': 'green',
                'EHV380': 'orange',
                'unknown': 'grey'}

def get_lev_from_volt (v_voltage): # in kV
    try:
        v = float(v_voltage)
    except:
        return None
    if v <= 1:
        return 'LV'
    elif (v >= 3) & (v <= 30):
        return 'MV'
    elif (v >= 60) & (v <= 110):
        return 'HV'
    elif v == 220:
        return 'EHV220'
    elif v == 380:
        return 'EHV380'
    else: return 'unknown'
def get_volt_from_lev (v_lev):
    if v_lev == 'HV':
        return 110.
    elif v_lev == 'EHV220':
        return 220.
    elif v_lev == 'EHV380':
        return 380.
    else: return None

#%% Basic grid information
logger.info('Basic grid information')

# MV single
index = mv_line_df.mv_grid.unique()
columns = []
mv_grids_df = pd.DataFrame(index=index, columns=columns)

mv_grids_df['length in km'] = mv_line_df.groupby(['mv_grid'])['length'].sum()
mv_grids_df['Transm. cap. in MVAkm'] = \
    mv_line_df.groupby(['mv_grid'])['s_nom_length_TVAkm'].sum() *1e3

mv_grids_df['Avg. feed-in MV'] = - mv_trafo_df[['subst_id',
                                              'p_mean']].set_index('subst_id')

mv_grids_df['Avg. feed-in HV'] = bus_df.loc[~np.isnan(bus_df['MV_grid_id'])]\
                            [['MV_grid_id','p_mean']].set_index('MV_grid_id')

# ToDo: Herausfinden, mit welcher Leistung die MV Trafos angeschlossen werden.
# ToDo: Hierfür besser direkt über eDisGo Generatoren arbeiten.

mv_gens_df = gens_df.merge(all_hvmv_subst_df,
              how='inner',
              left_on='bus',
              right_on='bus_id')

mv_gens_df = mv_gens_df[mv_gens_df['name'] != 'load shedding']
mv_gens_df = mv_gens_df.dropna() # I think this is also load shedding
## Find out how much load shedding is done...

mv_grids_df['Inst. gen. capacity'] = mv_gens_df.groupby(['subst_id'])['p_nom'].sum()


mv_grids_df.to_csv(analysis_dir + 'mv_grids_df.csv', encoding='utf-8')

# MV total
columns = ['MV']
index =   ['Tot. no. of grids',
           'No. of calc. grids',
           'Perc. of calc. grids',
           'Tot. calc. length in km',
           'Avg. len. per grid in km',
           'Estim. tot. len. in km',
           'Avg. transm. cap. in MVAkm',
           'Estim. tot. trans cap. in MVAkm']
mv_grid_info_df = pd.DataFrame(index=index, columns=columns)

mv_grid_info_df.loc['Tot. no. of grids']['MV'] = len(all_hvmv_subst_df)

mv_grid_info_df.loc['No. of calc. grids']['MV'] = len(mv_line_df.mv_grid.unique())

mv_grid_info_df.loc['Perc. of calc. grids']['MV'] = round(
        mv_grid_info_df.loc['No. of calc. grids']['MV']\
        / mv_grid_info_df.loc['Tot. no. of grids']['MV'] * 100, 2)

mv_grid_info_df.loc['Tot. calc. length in km']['MV'] = round(
        mv_line_df['length'].sum(), 2)

mv_grid_info_df.loc['Avg. len. per grid in km']['MV'] = round(
        mv_grids_df['length in km'].mean(), 2)

mv_grid_info_df.loc['Estim. tot. len. in km']['MV'] = round(
        mv_grid_info_df.loc['Avg. len. per grid in km']['MV'] *\
        mv_grid_info_df.loc['Tot. no. of grids']['MV'], 2)

mv_grid_info_df.loc['Avg. transm. cap. in MVAkm']['MV'] = round(
        mv_grids_df['Transm. cap. in MVAkm'].mean(), 2)

mv_grid_info_df.loc['Estim. tot. trans cap. in MVAkm']['MV'] = round(
        mv_grid_info_df.loc['Avg. transm. cap. in MVAkm']['MV'] *\
        mv_grid_info_df.loc['Tot. no. of grids']['MV'], 2)

mv_grid_info_df.to_csv(analysis_dir + 'mv_grid_info_df.csv', encoding='utf-8')

# HV Total
columns = ['HV', 'EHV220', 'EHV380']
index =   ['Total. len. in km',
           'Something']
grid_info_df = pd.DataFrame(index=index, columns=columns)

for col in columns:
    grid_info_df.loc['Total. len. in km'][col] = round(
            line_df.loc[line_df['v_nom'] == get_volt_from_lev(col)]['length'].sum(), 2)

grid_info_df.to_csv(analysis_dir + 'grid_info_df.csv', encoding='utf-8')

# HV/MV Comparison
columns = ['MV', 'HV', 'EHV220', 'EHV380']
index =   ['Total. len. in km',
           'Something']
hvmv_comparison_df = pd.DataFrame(index=index, columns=columns)

hvmv_comparison_df.loc['Total. len. in km']['MV'] = mv_grid_info_df.loc['Estim. tot. len. in km']['MV']

for col in grid_info_df.columns:
    hvmv_comparison_df.loc['Total. len. in km'][col] = grid_info_df.loc['Total. len. in km'][col]


hvmv_comparison_df.to_csv(analysis_dir + 'hvmv_comparison_df.csv', encoding='utf-8')


#%% Plot and Output Data Processing
logger.info('Plot and Output Data Processing')

### Total grid capacity and max and mean loading per voltage level in TVAkm
s_sum_t = pd.DataFrame(0.0, ## in TVAkm
                                   index=snap_idx,
                                   columns=all_levels)

for index, row in line_gdf.iterrows():
    v_nom = row['v_nom']
    s_series = pd.Series(data=row['s'], index=snap_idx)*row['length']*1e-6
    s_sum_t[v_nom] = s_sum_t[v_nom] + s_series

for index, row in mv_line_gdf.iterrows():
    v_nom = row['v_nom']
    s_series = pd.Series(data=row['s'], index=snap_idx)*row['length']*1e-6
    s_sum_t[v_nom] = s_sum_t[v_nom] + s_series

### Total grid overload per voltage level in TVAkm
s_sum_over_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=all_levels)

for index, row in line_gdf.iterrows():
    s_over_series = pd.Series(data=row['s_over'], index=snap_idx)
    TVAkm_over_series = s_over_series * row['s_nom_length_TVAkm']
    v_nom = row['v_nom']
    s_sum_over_t[v_nom] = s_sum_over_t[v_nom] + TVAkm_over_series

for index, row in mv_line_gdf.iterrows():
    s_over_series = pd.Series(data=row['s_over'], index=snap_idx)
    TVAkm_over_series = s_over_series * row['s_nom_length_TVAkm']
    v_nom = row['v_nom']
    s_sum_over_t[v_nom] = s_sum_over_t[v_nom] + TVAkm_over_series

### Total grid capacity and loading per voltage level
trans_cap_df = line_gdf[['s_nom_length_TVAkm', 'v_nom']].groupby('v_nom').sum()
mv_trans_cap_df = mv_line_gdf[['s_nom_length_TVAkm', 'v_nom']].groupby('v_nom').sum()
total_lines_df = mv_trans_cap_df.append(trans_cap_df)
total_lines_df = total_lines_df.rename(
        columns={'s_nom_length_TVAkm': 'total_cap'}) ## All data is in TVAkm
total_lines_df['max_total_loading'] = s_sum_t.max()
total_lines_df['mean_total_loading'] = s_sum_t.mean()


s_sum_rel_t = pd.DataFrame(0.0, ## in TVAkm
                                   index=snap_idx,
                                   columns=all_levels)
for column in s_sum_rel_t:
    s_sum_rel_t[column] = s_sum_t[column] / total_lines_df['total_cap'].loc[column]

s_sum_over_rel_t = pd.DataFrame(0.0, ## in TVAkm
                                   index=snap_idx,
                                   columns=all_levels)
for column in s_sum_over_rel_t:
    s_sum_over_rel_t[column] = s_sum_over_t[column] / total_lines_df['total_cap'].loc[column]

#%% Plots
logger.info('Plottig')

#### Scatter Plot
plt_name = "MV/HV Comparison"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(10,10)

mv_grids_df.plot.scatter(
        x='Avg. feed-in MV',
        y='Avg. feed-in HV',
        color='blue',
        label='MV/HV Comparison',
        ax=ax1)

ax1.plot([-250, 200],
         [-250, 200],
         ls="--",
         color='red')

file_name = 'feed-in_comparison'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')


### Plot and Corr
#### Line Plot
plt_name = "Voltage Level Total Appearent Power"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,4)

s_sum_t.plot(
        kind='line',
        title=plt_name,
        legend=True,
        linewidth=2,
        ax = ax1)

file_name = 'loading_per_level'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')

#### Scatter Plot
plt_name = "110kV and 220kV Load Correlation"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,6)

s_sum_t.plot.scatter(
        x=110.0,
        y=220.0,
        color='green',
        label='110 - 220kV',
        ax=ax1)


regr = linear_model.LinearRegression()
x = s_sum_t[110.0].values.reshape(len(s_sum_t), 1)
y = s_sum_t[220.0].values.reshape(len(s_sum_t), 1)
regr.fit(x, y)
plt.plot(x, regr.predict(x), color='grey', linewidth=1)

file_name = 'loading_corr_110_220'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')

#### Scatter Plot
plt_name = "220kV and 380kV Load Correlation"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,6)

s_sum_t.plot.scatter(
        x=220.0,
        y=380.0,
        color='orange',
        label='220-380kV',
        ax=ax1)

regr = linear_model.LinearRegression()
x = s_sum_t[220.0].values.reshape(len(s_sum_t), 1)
y = s_sum_t[380.0].values.reshape(len(s_sum_t), 1)
regr.fit(x, y)
plt.plot(x, regr.predict(x), color='grey', linewidth=1)

file_name = 'loading_corr_220_380'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')

#### Histogram
plt_name = "Loading Hist"
fig, ax = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(6,4)

s_sum_rel_t.plot.hist(alpha = 0.5, bins=20, ax=ax)

file_name = 'loading_hist'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')


#################################


#################################
### Plot Overload
#### Line Plot
plt_name = "Voltage Level Total Overload"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,4)

s_sum_over_t.plot(
        kind='line',
        title=plt_name,
        legend=True,
        linewidth=2,
        ax = ax1)

file_name = 'overl_per_level'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')

#### Scatter Plot
plt_name = "220kV and 380kV Overload Correlation"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,6)

s_sum_over_t.plot.scatter(
        x=220.0,
        y=380.0,
        color='orange',
        label='220-380kV',
        ax=ax1)

regr = linear_model.LinearRegression()
x = s_sum_over_t[220.0].values.reshape(len(s_sum_t), 1)
y = s_sum_over_t[380.0].values.reshape(len(s_sum_t), 1)
regr.fit(x, y)
plt.plot(x, regr.predict(x), color='grey', linewidth=1)

file_name = 'overl_corr_220_380'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')

#### Scatter Plot
plt_name = "110kV an 220kV Overload Correlation"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,6)

s_sum_over_t.plot.scatter(
        x=110.0,
        y=220.0,
        color='green',
        label='110-220kV',
        ax=ax1)

regr = linear_model.LinearRegression()
x = s_sum_over_t[110.0].values.reshape(len(s_sum_t), 1)
y = s_sum_over_t[220.0].values.reshape(len(s_sum_t), 1)
regr.fit(x, y)
plt.plot(x, regr.predict(x), color='grey', linewidth=1)

file_name = 'overl_corr_110_220'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')

#### Histogram
plt_name = "Loading Hist"
fig, ax = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(6,4)

s_sum_over_rel_t.plot.hist(alpha = 0.5, bins=20, ax=ax)

file_name = 'overloading_hist'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')

##### Corr Test
mean_squared_error(regr.predict(x), y)
r2_score(regr.predict(x), y)

s_sum_t.corr(method='pearson', min_periods=1)
scipy.stats.pearsonr(regr.predict(x), y)[0][0]

#################################

#################################
### Plot Barplot für Netzkapazitüt und Belastung

plt_name = "Grid Capacity per voltage level in TVAkm"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(8,4)

total_lines_df.plot(
        kind='bar',
        title=plt_name,
        ax = ax1)

file_name = 'grid_cap'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
#################################

#################################
### Plot: Ein- und Ausspeisende MV Grids.
plt_name = "MV Grid Feed-in"
marker_sz_mltp = 20
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(8,12)

plot_df = line_gdf.loc[line_gdf['v_nom'] == 380.]
plot_df.plot(
        color='orange',
        linewidth=5,
        ax = ax1)
plot_df = line_gdf.loc[line_gdf['v_nom'] == 220.]
plot_df.plot(
        color='green',
        linewidth=3,
        ax = ax1)
plot_df = line_gdf.loc[line_gdf['v_nom'] == 110.]
plot_df.plot(
        color='blue',
        linewidth=1,
        ax = ax1)
plot_df = mv_line_gdf
plot_df.plot(
        color='grey',
        linewidth=0.4,
        ax = ax1)
# HTis is acutally not correct!!!!!
plot_df= bus_gdf.loc[(~np.isnan(bus_gdf['MV_grid_id'])) & (bus_gdf['p_mean']>=0)]
plot_df.plot(
        markersize= plot_df['p_mean']*marker_sz_mltp,
        color='darkgreen',
        ax = ax1)
plot_df= bus_gdf.loc[(~np.isnan(bus_gdf['MV_grid_id'])) & (bus_gdf['p_mean']<0)]
plot_df.plot(
        markersize= -plot_df['p_mean']*marker_sz_mltp,
        color='darkred',
        ax = ax1)

file_name = 'grid_feed_in'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
#################################

#################################
### Plot: Transformers
plt_name = "all Transformers"
marker_sz_mltp = 20
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(8,12)

plot_df = line_gdf.loc[line_gdf['v_nom'] == 380.]
plot_df.plot(
        color='orange',
        linewidth=5,
        ax = ax1)
plot_df = line_gdf.loc[line_gdf['v_nom'] == 220.]
plot_df.plot(
        color='green',
        linewidth=3,
        ax = ax1)
plot_df = line_gdf.loc[line_gdf['v_nom'] == 110.]
plot_df.plot(
        color='blue',
        linewidth=1,
        ax = ax1)
plot_df = mv_line_gdf
plot_df.plot(
        color='grey',
        linewidth=0.4,
        ax = ax1)

plot_df = trafo_gdf
plot_df.plot(
        color='red',
        markersize=200,
        ax=ax1)
plot_df = mv_trafo_gdf
plot_df.plot(
        color='violet',
        markersize=200,
        ax=ax1)


## Hier Trafos als Punkte Plotten.

file_name = 'all_trafo'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
#################################