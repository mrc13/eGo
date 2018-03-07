#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corr Processing
"""
from sqlalchemy.orm import sessionmaker
import pandas as pd
import geopandas as gpd
from geoalchemy2.shape import to_shape
from shapely.ops import transform
from functools import partial
from shapely.geometry import Point, LineString
from matplotlib import pyplot as plt
import numpy as np
import scipy
import pyproj
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from egoio.tools import db
from egoio.db_tables import model_draft
from ego.tools.specs import get_scn_name_from_result_id
from ego.tools import corr_io

# Mapping
ormclass_result_meta = model_draft.EgoGridPfHvResultMeta
ormclass_result_line = model_draft.EgoGridPfHvResultLine
ormclass_result_bus = model_draft.EgoGridPfHvResultBus
ormclass_result_line_t = model_draft.EgoGridPfHvResultLineT
ormclass_result_bus_t = model_draft.EgoGridPfHvResultBusT
ormclass_hvmv_subst = model_draft.EgoGridHvmvSubstation
mv_lines = corr_io.corr_mv_lines_results
mv_buses = corr_io.corr_mv_bus_results
ormclass_result_transformer = model_draft.EgoGridPfHvResultTransformer

ormclass_result_gen = model_draft.__getattribute__('EgoGridPfHvResultGenerator')
ormclass_result_gen_t = model_draft.__getattribute__('EgoGridPfHvResultGeneratorT')
ormclass_result_gen_single = model_draft.__getattribute__('EgoSupplyPfGeneratorSingle')
ormclass_result_load = model_draft.__getattribute__('EgoGridPfHvResultLoad')
ormclass_result_load_t = model_draft.__getattribute__('EgoGridPfHvResultLoadT')
ormclass_result_stor = model_draft.__getattribute__('EgoGridPfHvResultStorage')
ormclass_result_stor_t = model_draft.__getattribute__('EgoGridPfHvResultStorageT')
ormclass_source = model_draft.__getattribute__('EgoGridPfHvSource')
ormclass_aggr_w = model_draft.__getattribute__('EgoSupplyAggrWeather')

# Plotting Base
## Geometry transform function based on pyproj.transform
project = partial(
    pyproj.transform,
    pyproj.Proj(init='EPSG:4326'),
    pyproj.Proj(init='EPSG:32633'))

plot_dir = '/home/student/Dropbox/Masterarbeit/Thesis/graphics/pyplots/'
#level_colors = pd.DataFrame({ 'lev' : [110., 220., 380.],
#                              'color' : ['blue', 'green', 'orange']})

# DB Access
conn = db.connection(section='oedb')
Session = sessionmaker(bind=conn)
session = Session()

# Processing
result_id = 359
brnch_fkt = 1. # In eTraGo used factor for branches
hv_ov_fkt = 0.1 # When HV lines are considered overloaded
mv_ov_fkt = 0.01

scn_name = get_scn_name_from_result_id(session, result_id)

## Metadata
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

## Lines
query = session.query(
        ormclass_result_line.line_id,
        ormclass_result_line.bus0,
        ormclass_result_line.bus1,
        ormclass_result_bus.v_nom,
        ormclass_result_line.cables,
        ormclass_result_line.frequency,
        (ormclass_result_line.s_nom / brnch_fkt).label('s_nom'),
        ormclass_result_line.r,
        ormclass_result_line.x,
        ormclass_result_line.b,
        ormclass_result_line.g,
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
line_gdf = gpd.GeoDataFrame(line_df, crs=crs, geometry=line_df.topo)
line_gdf.v_nom.unique()
hv_levels = pd.unique(line_gdf['v_nom']).tolist()


##MV Lines
query = session.query(
        mv_lines.name,
        mv_lines.mv_grid,
        mv_lines.geom,
        mv_lines.bus0,
        mv_lines.bus1,
        mv_lines.s_nom, # s_nom is in MVA
        mv_lines.s, # s is in KVA, changed futher down
        mv_lines.v_nom
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
mv_line_gdf = gpd.GeoDataFrame(mv_line_df, crs=crs, geometry=mv_line_df.geom)

mv_line_gdf['length'] = mv_line_gdf.apply(
        lambda x: transform(project, x['geom']).length / 1e3, axis=1)

mv_line_gdf['s_nom_length_TVAkm'] = mv_line_gdf.apply(
        lambda x: (x['length'] * float(x['s_nom']))*1e-6, axis=1)

mv_line_gdf['s'] = mv_line_gdf.apply( ## now s is in MVA
        lambda x: [s/1000 for s in x['s']], axis=1)

mv_line_gdf['s_rel'] = mv_line_gdf.apply(
        lambda x: [s/float(x['s_nom']) for s in x['s']], axis=1)

mv_line_gdf['s_over'] = mv_line_gdf.apply(
        lambda x: [s_rel > mv_ov_fkt for s_rel in x['s_rel']], axis=1)

mv_levels = pd.unique(mv_line_gdf['v_nom']).tolist()

all_levels = mv_levels + hv_levels

## Buses
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
bus_gdf = gpd.GeoDataFrame(bus_df, crs=crs, geometry=bus_df.geom)

bus_gdf['p_mean'] = bus_gdf.apply( # Mean feed in
        lambda x: pd.Series(data= x['p']).mean(), axis=1)

feed_in_check = bus_gdf.p_mean.sum() # This must result in 0!

## Transformers
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

crs = {'init': 'epsg:4326'}
trafo_gdf = gpd.GeoDataFrame(trafo_df, crs=crs, geometry=trafo_df.geom)

trafo_gdf['v_nom0'] = trafo_gdf.apply(
        lambda x: bus_gdf.loc[x['bus0']]['v_nom'], axis=1)
trafo_gdf['v_nom1'] = trafo_gdf.apply(
        lambda x: bus_gdf.loc[x['bus1']]['v_nom'], axis=1)

## Hier jetzt noch die Punktgeometrien der Trafos verwenden.
## Dann Hv/MV Verbindungen auch als Trafos eintragen und entsprechende Zeitreihen übernehmen
## Dann geeigneten räumlichen Buffer überlegen.


#################################
### Plot
#### Plot Processing
trans_cap_df = line_gdf[['s_nom_length_TVAkm', 'v_nom']].groupby('v_nom').sum()
mv_trans_cap_df = mv_line_gdf[['s_nom_length_TVAkm', 'v_nom']].groupby('v_nom').sum()
trans_cap_df = mv_trans_cap_df.append(trans_cap_df)

plt_name = "Grid Transmission Capacity"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(8,4)

trans_cap_df['s_nom_length_TVAkm'].plot(
        kind='bar',
        title=plt_name,
        color='grey',
        ax = ax1)

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')
#################################

#################################
### Plot and Corr
#### Processing

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

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')

#### Scatter Plot
plt_name = "220kV and 110kV Load Correlation"
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

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')

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
####Corr
mean_squared_error(regr.predict(x), y)
r2_score(regr.predict(x), y)

s_sum_t.corr(method='pearson', min_periods=1)
scipy.stats.pearsonr(regr.predict(x), y)[0][0]

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')

#################################

#################################
### Plot Overload
#### Plotprcessing
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

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')

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

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')
##### Corr
mean_squared_error(regr.predict(x), y)
r2_score(regr.predict(x), y)

s_sum_t.corr(method='pearson', min_periods=1)
scipy.stats.pearsonr(regr.predict(x), y)[0][0]

#### Scatter Plot
plt_name = "220kV and 110kV Overload Correlation"
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

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')
##### Corr
mean_squared_error(regr.predict(x), y)
r2_score(regr.predict(x), y)

s_sum_t.corr(method='pearson', min_periods=1)
scipy.stats.pearsonr(regr.predict(x), y)[0][0]

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

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')
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

## Hier Trafos als Punkte Plotten.

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')
#################################
