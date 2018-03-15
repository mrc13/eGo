#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corr Processing
"""
## General Packages
from sqlalchemy.orm import sessionmaker
import pandas as pd
import geopandas as gpd
from geoalchemy2.shape import to_shape
from matplotlib import pyplot as plt
import numpy as np
import scipy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
from time import localtime, strftime


## Project Packages
from egoio.tools import db
from egoio.db_tables import model_draft

## Local Packages
from ego.tools.specs import get_scn_name_from_result_id
from ego.tools import corr_io

## Logging
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger(__name__)

fh = logging.FileHandler('corr_proc.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

# Mapping
## model_draft
ormclass_result_meta = model_draft.EgoGridPfHvResultMeta
ormclass_hvmv_subst = model_draft.EgoGridHvmvSubstation
ormclass_result_transformer = model_draft.EgoGridPfHvResultTransformer
ormclass_griddistricts = model_draft.EgoGridMvGriddistrict
## corr
ormclass_result_line = corr_io.corr_hv_lines_results
ormclass_result_bus = corr_io.corr_hv_bus_results
ormclass_result_line_t = corr_io.corr_hv_lines_t_result
ormclass_result_bus_t = corr_io.corr_hv_bus_t_results
mv_lines = corr_io.corr_mv_lines_results
mv_buses = corr_io.corr_mv_bus_results


# General inputs
result_id = 384
hv_ov_fkt = 0.7 # When HV lines are considered overloaded
mv_ov_fkt = 0.5

# Result Folder
now = strftime("%Y-%m-%d %H:%M", localtime())
result_dir = 'corr_results/' + str(result_id) + '/' + now + '/'
plot_dir = result_dir

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

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

# DB Access
conn = db.connection(section='oedb')
Session = sessionmaker(bind=conn)
session = Session()

#%% Metadata
logger.info('Metadata')
scn_name = get_scn_name_from_result_id(session, result_id)

meta_settings = session.query( # Here also the correct brnch_fkt can be found.
            ormclass_result_meta.settings
            ).filter(
            ormclass_result_meta.result_id == result_id
            ).scalar(
                    )

snap_idx = session.query(
            ormclass_result_meta.snapshots
            ).filter(
            ormclass_result_meta.result_id == result_id
            ).scalar(
                    )

#%% Lines
# HV Lines
logger.info('Lines')
query = session.query(
        ormclass_result_line.line_id,
        ormclass_result_line.bus0,
        ormclass_result_line.bus1,
        ormclass_result_bus.v_nom,
        ormclass_result_line.cables,
        ormclass_result_line.frequency,
        ormclass_result_line.r,
        ormclass_result_line.x,
        ormclass_result_line.b,
        ormclass_result_line.g,
        ormclass_result_line.s_nom,
        ormclass_result_line.topo,
        ormclass_result_line.geom,
        ormclass_result_line.length,
        ormclass_result_line_t.p0,
        ormclass_result_line_t.p1
        ).join(
                ormclass_result_bus,
                ormclass_result_bus.bus_id == ormclass_result_line.bus0
                ).join(
                ormclass_result_line_t,
                ormclass_result_line_t.line_id == ormclass_result_line.line_id
                        ).filter(
                ormclass_result_line.result_id == result_id,
                ormclass_result_bus.result_id == result_id,
                ormclass_result_line_t.result_id == result_id)

line_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])
line_df = line_df.set_index('line_id')

line_df['s_nom_length_TVAkm'] = line_df.apply(
        lambda x: (x['length'] * float(x['s_nom']))*1e-6, axis=1)

line_df['s'] = line_df.apply(
        lambda x: [abs(number) for number in x['p0']], axis=1) # This is correct, since in lopf no losses.

line_df['s_rel'] = line_df.apply(
        lambda x: [s/float(x['s_nom']) for s in x['s']], axis=1)

line_df['s_over'] = line_df.apply(
        lambda x: [s_rel > hv_ov_fkt for s_rel in x['s_rel']], axis=1)

line_df['geom'] = line_df.apply(
        lambda x: to_shape(x['geom']), axis=1)

line_df['topo'] = line_df.apply(
        lambda x: to_shape(x['topo']), axis=1)

crs = {'init': 'epsg:4326'}
line_df = gpd.GeoDataFrame(line_df, crs=crs, geometry=line_df.topo)

# MV Lines
query = session.query(
        mv_lines.name,
        mv_lines.mv_grid,
        mv_lines.geom,
        mv_lines.bus0,
        mv_lines.bus1,
        mv_lines.s_nom, # s_nom is in MVA
        mv_lines.s, # s is in KVA, changed futher down
        mv_lines.r,
        mv_lines.x,
        mv_lines.v_nom,
        mv_lines.length
        ).filter(
                mv_lines.result_id == result_id)

mv_line_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

mv_line_df = mv_line_df.set_index('name')

mv_line_df['geom'] = mv_line_df.apply(
        lambda x: to_shape(x['geom']), axis=1)

crs = {'init': 'epsg:4326'}
mv_line_df = gpd.GeoDataFrame(mv_line_df, crs=crs, geometry=mv_line_df.geom)

mv_line_df['s_nom_length_TVAkm'] = mv_line_df.apply(
        lambda x: (x['length'] * float(x['s_nom']))*1e-6, axis=1)

mv_line_df['s'] = mv_line_df.apply( ## now s is in MVA
        lambda x: [s/1000 for s in x['s']], axis=1)

mv_line_df['s_rel'] = mv_line_df.apply(
        lambda x: [s/float(x['s_nom']) for s in x['s']], axis=1)

mv_line_df['s_over'] = mv_line_df.apply(
        lambda x: [s_rel > mv_ov_fkt for s_rel in x['s_rel']], axis=1)

## Further Processing Information
hv_levels = pd.unique(line_df['v_nom']).tolist()
mv_levels = pd.unique(mv_line_df['v_nom']).tolist()
all_levels = mv_levels + hv_levels


#%% Buses
logger.info('Buses')
query = session.query(
        ormclass_result_bus.bus_id,
        ormclass_result_bus.v_nom,
        ormclass_result_bus.geom,
        ormclass_result_bus_t.p,
        ormclass_result_bus_t.v_ang,
        ormclass_hvmv_subst.subst_id.label('MV_grid_id')
        ).join(
                ormclass_result_bus_t,
                ormclass_result_bus_t.bus_id == ormclass_result_bus.bus_id
                ).join(
                        ormclass_hvmv_subst,
                        ormclass_hvmv_subst.otg_id == ormclass_result_bus.bus_id,
                        isouter=True
                        ).filter(
                ormclass_result_bus.result_id == result_id,
                ormclass_result_bus_t.result_id == result_id)

bus_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

bus_df = bus_df.set_index('bus_id')

bus_df['geom'] = bus_df.apply(
        lambda x: to_shape(x['geom']), axis=1)

crs = {'init': 'epsg:4326'}
bus_df = gpd.GeoDataFrame(bus_df, crs=crs, geometry=bus_df.geom)

bus_df['p_mean'] = bus_df.apply( # Mean feed in
        lambda x: pd.Series(data= x['p']).mean(), axis=1)

# MV Buses
query = session.query(
        mv_buses.name,
        mv_buses.v_nom,
        mv_buses.geom,
        mv_buses.p,
        mv_buses.q
        ).filter(
                mv_buses.result_id == result_id)

mv_bus_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

mv_bus_df = mv_bus_df.set_index('name')

mv_bus_df['geom'] = mv_bus_df.apply(
        lambda x: to_shape(x['geom']), axis=1)

mv_bus_df['p_mean'] = mv_bus_df.apply(
        lambda x: pd.Series(data= x['p']).mean(), axis=1)

crs = {'init': 'epsg:4326'}
mv_bus_df = gpd.GeoDataFrame(mv_bus_df, crs=crs, geometry='geom')

## Further Processing Information and Checks

feed_in_check = bus_df.p_mean.sum()
if feed_in_check > 1.:
    logger.warning('p_mean sum is not 0!')

#%% Transformers
logger.info('Transformers')
# HV Transformers
query = session.query(
        ormclass_result_transformer.trafo_id,
        ormclass_result_transformer.bus0,
        ormclass_result_transformer.bus1,
        ormclass_result_transformer.geom
        ).filter(
                ormclass_result_transformer.result_id == result_id)

trafo_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])
trafo_df = trafo_df.set_index('trafo_id')

trafo_df['geom'] = trafo_df.apply(
        lambda x: to_shape(x['geom']), axis=1)

trafo_df['point_geom'] = trafo_df.apply(
        lambda x: x['geom'].representative_point(), axis=1)


crs = {'init': 'epsg:4326'}
trafo_df = gpd.GeoDataFrame(trafo_df, crs=crs, geometry='point_geom')

trafo_df['v_nom0'] = trafo_df.apply(
        lambda x: bus_df.loc[x['bus0']]['v_nom'], axis=1)
trafo_df['v_nom1'] = trafo_df.apply(
        lambda x: bus_df.loc[x['bus1']]['v_nom'], axis=1)

## Define spatial Buffer
trafo_df['grid_buffer'] = trafo_df.apply(
        lambda x: x['point_geom'].buffer(0.1), axis=1) ## Buffergröße noch anpassen

#trafo_df.set_geometry('grid_buffer', inplace=True)
#trafo_df.plot()

# MV Transformers
query = session.query(
        ormclass_hvmv_subst.point,
        ormclass_hvmv_subst.subst_id,
        ormclass_hvmv_subst.otg_id)
mv_trafo_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

mv_trafo_df['point'] = mv_trafo_df.apply(
        lambda x: to_shape(x['point']), axis=1)

crs = {'init': 'epsg:4326'}
mv_trafo_df = gpd.GeoDataFrame(mv_trafo_df, crs=crs, geometry='point')

mv_trafo_df['bus0'] = mv_trafo_df.apply(
        lambda x: 'MVStation_' + str(x['subst_id']), axis=1)
mv_trafo_df['bus1'] = mv_trafo_df.apply(
        lambda x: x['otg_id'], axis=1)

mv_trafo_df = mv_trafo_df.merge(mv_bus_df, # Only the Trafos that can actualy be found in the MV buses....
                                  left_on='bus0',
                                  right_index=True,
                                  how='inner')

## Merge with griddistrict geoms
query = session.query(ormclass_griddistricts.subst_id,
                      ormclass_griddistricts.geom)
mv_griddistricts_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])
mv_griddistricts_df = mv_griddistricts_df.set_index('subst_id')
mv_griddistricts_df['geom'] = mv_griddistricts_df.apply(
        lambda x: to_shape(x['geom']), axis=1)
mv_griddistricts_df = mv_griddistricts_df.rename(
        columns={'geom': 'grid_buffer'})

crs = {'init': 'epsg:3035'}
mv_griddistricts_df = gpd.GeoDataFrame(mv_griddistricts_df,
                                        crs=crs,
                                        geometry='grid_buffer')
mv_griddistricts_df = mv_griddistricts_df.to_crs({'init': 'epsg:4326'})

mv_trafo_df = mv_trafo_df.merge(mv_griddistricts_df,
                                  left_on='subst_id',
                                  right_index=True,
                                  how='left')
del mv_griddistricts_df
mv_trafo_df.head()

#mv_trafo_df.set_geometry('grid_buffer', inplace=True)
#mv_trafo_df.plot()

#%% All hvmv Substations
logger.info('All hvmv Substations')
query = session.query(
        ormclass_result_bus.bus_id,
        ormclass_hvmv_subst.subst_id
        ).join(
                ormclass_hvmv_subst,
                ormclass_hvmv_subst.otg_id == ormclass_result_bus.bus_id
                ).filter(
                        ormclass_result_bus.result_id == result_id)

all_hvmv_subst_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

#%% Export processed results
logger.info('Export processed Results')

line_df.to_csv(result_dir + 'line_df.csv', encoding='utf-8')
mv_line_df.to_csv(result_dir + 'mv_line_df.csv', encoding='utf-8')

bus_df.to_csv(result_dir + 'bus_df.csv', encoding='utf-8')
mv_bus_df.to_csv(result_dir + 'mv_bus_df.csv', encoding='utf-8')

trafo_df.to_csv(result_dir + 'trafo_df.csv', encoding='utf-8')
mv_trafo_df.to_csv(result_dir + 'mv_trafo_df.csv', encoding='utf-8')

all_hvmv_subst_df.to_csv(result_dir + 'all_hvmv_subst_df.csv', encoding='utf-8')


#%% Data Cleaning

# ToDo: Clean out faulty MV grids, that e.g. have remote generators...

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





# MV total
columns = ['MV']
index =   ['Tot. no. of grids',
           'No. of calc. grids',
           'Perc. of calc. grids',
           'Tot. calc. length in km',
           'Av. len. per grid in km',
           'Estim. tot. len. in km']
mv_grid_info_df = pd.DataFrame(index=index, columns=columns)

mv_grid_info_df.loc['Tot. no. of grids']['MV'] = len(all_hvmv_subst_df)

mv_grid_info_df.loc['No. of calc. grids']['MV'] = len(mv_line_df.mv_grid.unique())

mv_grid_info_df.loc['Perc. of calc. grids']['MV'] = round(
        mv_grid_info_df.loc['No. of calc. grids']['MV']\
        / mv_grid_info_df.loc['Tot. no. of grids']['MV'] * 100, 2)

mv_grid_info_df.loc['Tot. calc. length in km']['MV'] = round(
        mv_line_df['length'].sum(), 2)

mv_grid_info_df.loc['Av. len. per grid in km']['MV'] = round(
        mv_grids_df['length'].mean(), 2)

mv_grid_info_df.loc['Estim. tot. len. in km']['MV'] = round(
        mv_grid_info_df.loc['Av. len. per grid in km']['MV'] *\
        mv_grid_info_df.loc['Tot. no. of grids']['MV'], 2)









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
