#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:05:57 2018

@author: student
"""

"""
This is the correlation analysis tool for thesis of maltesc

"""
__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität"\
                            "Flensburg, Centre for Sustainable Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "maltesc"

# Import
## General Packages
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy
import os
import shapely.wkt
from time import localtime, strftime
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from dateutil import parser
import logging
from sqlalchemy.orm import sessionmaker
import tilemapbase
import math

### Tilemap
tilemapbase.init(create=True)
t = tilemapbase.tiles.OSM

## Project Packages
from egoio.tools import db

## Local Packages
from ego.tools.corr_func import (color_for_s_over,
                                 add_plot_lines_to_ax,
                                 add_weighted_plot_lines_to_ax,
                                 get_lev_from_volt,
                                 add_figure_to_tex,
                                 add_table_to_tex,
                                 render_mpl_table,
                                 render_corr_table,
                                 to_str,
                                 corr,
                                 aggregate_loading,
                                 corr_colors)
# Time
now = strftime("%Y-%m-%d_%H%M", localtime())

# Logging
log_dir = 'corr_corr_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger(__name__)
specs_logger = logging.getLogger('specs')
network_logger = logging.getLogger('network')
pypsa_logger = logging.getLogger('pypsa.pf')

fh = logging.FileHandler(log_dir + '/corr_corr_' + now + '.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

# General Inputs
result_id = 5
data_set = '2018-05-12'

cont_fct_ehv = 0.7
cont_fct_hv = 0.7
cont_fct_mv = 1
over_voltage_mv = 1.05

r_correct_fct = 2       # Because of a mistake in Freileitung oder Kabel.

# Connection
conn = db.connection(section='oedb')
Session = sessionmaker(bind=conn)
session = Session()

# Create visualization tables
ans = str(input("Are you sure to ureset vis DB (y/n)? "))
if ans == 'y':
    session.execute('''
    SELECT model_draft.corr_vis_result_id(:result_id);
    ''', {'result_id': result_id})
    session.commit()

# Directories
analysis_dir = 'corr_results/' + str(result_id) + '/analysis/' + now + '/'
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

## Germany
ger_dir = analysis_dir + 'ger_analysis/'
if not os.path.exists(ger_dir):
    os.makedirs(ger_dir)

ger_plot_dir = ger_dir + 'plots/'
if not os.path.exists(ger_plot_dir):
    os.makedirs(ger_plot_dir)

ger_corr_dir = ger_dir + 'corr/'
if not os.path.exists(ger_corr_dir):
    os.makedirs(ger_corr_dir)

## HV
hv_dir = analysis_dir + 'hv_analysis/'
if not os.path.exists(hv_dir):
    os.makedirs(hv_dir)

hv_plot_dir = hv_dir + 'plots/'
if not os.path.exists(hv_plot_dir):
    os.makedirs(hv_plot_dir)

hv_corr_dir = hv_dir + 'corr/'
if not os.path.exists(hv_corr_dir):
    os.makedirs(hv_corr_dir)

## Dist
dist_dir = analysis_dir + 'dist_analysis/'
if not os.path.exists(dist_dir):
    os.makedirs(dist_dir)

dist_plot_dir = dist_dir + 'plots/'
if not os.path.exists(dist_plot_dir):
    os.makedirs(dist_plot_dir)

dist_corr_dir = dist_dir + 'corr/'
if not os.path.exists(dist_corr_dir):
    os.makedirs(dist_corr_dir)

readme = open(analysis_dir + 'readme','w')
readme.write(r'''
Test analysis...
''')
readme.close()

# Dicts and List
level_colors = {'LV': 'grey',
                'MV': 'black',
                'HV': 'blue',
                'EHV': 'darkred',
                'unknown': 'grey'}

carrier_colors = {
        "run_of_river" : "royalblue",
        "uranium" : "greenyellow",
        "lignite" : "saddlebrown",
        "coal" : "black",
        "waste" : "peru",
        "other_non_renewable" : "salmon",
        "oil" : "grey",
        "gas" : "orange",
        "geothermal" : "indigo",
        "biomass" : "darkgreen",
        "reservoir" : "blue",
        "wind" : "dodgerblue",
        "solar" : "yellow",
        "load shedding" : "black"}

var_rens = ['wind', 'solar']
all_levels = ['MV', 'HV', 'EHV']

#%%
print('Data Import')

result_dir = 'corr_results/' + str(result_id) + '/data_proc/' + data_set + '/'

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
    gens_df = pd.DataFrame.from_csv(result_dir + 'gens_df.csv',
                                    encoding='utf-8')
except:
    logger.warning('No gens imported')

try:
    load_df = pd.DataFrame.from_csv(result_dir + 'load_df.csv',
                                    encoding='utf-8')
except:
    logger.warning('No load imported')

try:
    trafo_df = pd.DataFrame.from_csv(result_dir + 'trafo_df.csv', encoding='utf-8')
    mv_trafo_df = pd.DataFrame.from_csv(result_dir + 'mv_trafo_df.csv', encoding='utf-8')
except:
    logger.warning('No trafos imported')

try:
    all_hvmv_subst_df = pd.DataFrame.from_csv(result_dir + 'all_hvmv_subst_df.csv', encoding='utf-8')
    snap_idx = pd.Series.from_csv(result_dir + 'snap_idx', encoding='utf-8')
    snap_idx = pd.Series([parser.parse(idx) for idx in snap_idx])
except:
    logger.warning('No Subst. imported')

# Shapefiles to Dataframes

nuts_shp = gpd.read_file("data/nuts/nuts.shp")

hv_grids_shp = gpd.read_file("data/110kV/kv110_merged_touches.shp")

#%%
print('Further Processing')

# HV Lines
crs = {'init': 'epsg:4326'}
line_df = gpd.GeoDataFrame(line_df,
                           crs=crs,
                           geometry=line_df.topo.map(shapely.wkt.loads))
line_df = line_df.drop(['geom', 'topo', 's'], axis=1) # s is redundant due to s_rel

line_df['s_nom_length_GVAkm'] = line_df['s_nom_length_TVAkm']*1e3
line_df['s_rel'] = line_df.apply(
        lambda x: eval(x['s_rel']), axis=1)
line_df['lev'] = line_df.apply(
        lambda x: get_lev_from_volt(x['v_nom']), axis=1)

## Loading
line_df['s_len_abs'] = line_df.apply(    # in GVAkm
        lambda x: [
                s * x['s_nom_length_GVAkm']
                for s in x['s_rel']
                ], axis=1)
## Overload
line_df['s_over'] = line_df.apply(          ## Relative in 1 ## No threshhold
        lambda x: [
                n - cont_fct_hv
                if x['lev'] == 'HV'
                else n - cont_fct_ehv
                for n in x['s_rel']], axis=1)

line_df['s_over_abs'] = line_df.apply(      ## Absoulute in GVAkm ## With threshhold
        lambda x: [n * x['s_nom_length_GVAkm'] \
                   if n>0 else 0 for n in x['s_over']], axis=1)

line_df['s_under_abs'] = line_df.apply(      ## Absoulute in GVAkm
        lambda x:  list(
                pd.Series(
                        data=x['s_len_abs'],
                        index=snap_idx)
                - pd.Series(
                        data=x['s_over_abs'],
                        index=snap_idx)), axis=1)

line_df['s_over_bol'] = line_df.apply(      ## Boolean
        lambda x: [True if n > 0 else False for n in x['s_over']], axis=1)

line_df['anytime_over'] = line_df.apply(      ## True if at any time the line presents an overload
        lambda x: True if sum(x['s_over_bol']) > 0 else False, axis=1)

line_df['s_over_max'] = line_df.apply(      ## Relative in 1
        lambda x:  max(x['s_over']) if max(x['s_over']) > 0 else 0, axis=1)
line_df['s_over_abs_max'] = line_df.apply(      ## Max abslute overload GVAkm
        lambda x:  max(x['s_over_abs']), axis=1)

line_df['s_over_dur'] = line_df.apply(      ## Relative in 1
        lambda x:  sum(x['s_over_bol'])/len(snap_idx), axis=1)

# MV Lines
crs = {'init': 'epsg:4326'}
mv_line_df = gpd.GeoDataFrame(mv_line_df,
                              crs=crs,
                              geometry=mv_line_df.geom.map(shapely.wkt.loads))
mv_line_df = mv_line_df.drop(['geom', 's'], axis=1)

mv_line_df['s_nom_length_GVAkm'] = mv_line_df['s_nom_length_TVAkm']*1e3
mv_line_df['s_rel'] = mv_line_df.apply(
        lambda x: eval(x['s_rel']), axis=1)
mv_line_df['lev'] = mv_line_df.apply(
        lambda x: get_lev_from_volt(x['v_nom']), axis=1)

# Loading
mv_line_df['s_len_abs'] = mv_line_df.apply(    # in GVAkm
        lambda x: [
                s * x['s_nom_length_GVAkm']
                for s in x['s_rel']
                ], axis=1)

# Overload
mv_line_df['s_over'] = mv_line_df.apply(          ## Relative in 1
        lambda x: [n - cont_fct_hv for n in x['s_rel']], axis=1)
mv_line_df['s_over_abs'] = mv_line_df.apply(      ## Absoulute in GVAkm
        lambda x: [n * x['s_nom_length_GVAkm'] \
                   if n>0 else 0 for n in x['s_over']], axis=1)

mv_line_df['s_under_abs'] = mv_line_df.apply(      ## Absoulute in GVAkm
    lambda x:  list(
            pd.Series(
                    data=x['s_len_abs'],
                    index=snap_idx)
            - pd.Series(
                    data=x['s_over_abs'],
                    index=snap_idx)), axis=1)

mv_line_df['s_over_bol'] = mv_line_df.apply(      ## Boolean
        lambda x: [True if n > 0 else False for n in x['s_over']], axis=1)

mv_line_df['anytime_over'] = mv_line_df.apply(      ## True if at any time the line presents an overload
        lambda x: True if sum(x['s_over_bol']) > 0 else False, axis=1)

mv_line_df['s_over_max'] = mv_line_df.apply(      ## Relative in 1
        lambda x:  max(x['s_over']) if max(x['s_over']) > 0 else 0, axis=1)
mv_line_df['s_over_abs_max'] = mv_line_df.apply(      ## Max abslute overload GVAkm
        lambda x:  max(x['s_over_abs']), axis=1)

mv_line_df['s_over_dur'] = mv_line_df.apply(      ## Relative in 1
        lambda x:  sum(x['s_over_bol'])/len(snap_idx), axis=1)

# Buses
crs = {'init': 'epsg:4326'}
bus_df = gpd.GeoDataFrame(bus_df,
                          crs=crs,
                          geometry=bus_df.geom.map(shapely.wkt.loads))
bus_df = bus_df.drop(['geom'], axis=1)

bus_df['lev'] = bus_df.apply(
        lambda x: get_lev_from_volt(x['v_nom']), axis=1)

crs = {'init': 'epsg:4326'}
mv_bus_df = gpd.GeoDataFrame(mv_bus_df,
                             crs=crs,
                             geometry=mv_bus_df.geom.map(shapely.wkt.loads))
mv_bus_df = mv_bus_df.drop(['geom'], axis=1)

mv_bus_df['lev'] = mv_bus_df.apply(
        lambda x: get_lev_from_volt(x['v_nom']), axis=1)

# LV Stations
lv_stations = mv_bus_df.loc[
        mv_bus_df.reset_index()[
                'name'
                ].str.startswith('LVStation').tolist()]

lv_stations['v'] = lv_stations.apply(
        lambda x: eval(x['v']), axis=1)

lv_stations['v_over_bol'] = lv_stations.apply(
        lambda x: [True if n > over_voltage_mv else False for n in x['v']],
        axis=1)

# Transformers
## HV
crs = {'init': 'epsg:4326'}
trafo_df = gpd.GeoDataFrame(trafo_df,
                            crs=crs,
                            geometry=trafo_df.point_geom.map(shapely.wkt.loads))
trafo_df = trafo_df.drop(['point_geom', 's', 'geom', 'p0'], axis=1)

trafo_df['s_rel'] = trafo_df.apply(
        lambda x: eval(x['s_rel']), axis=1)

trafo_df['grid_buffer'] = trafo_df['grid_buffer'].map(shapely.wkt.loads)

## MV
crs = {'init': 'epsg:4326'}
mv_trafo_df = gpd.GeoDataFrame(mv_trafo_df,
                               crs=crs,
                               geometry=mv_trafo_df.point.map(shapely.wkt.loads))
mv_trafo_df = mv_trafo_df.drop(['geom', 'point'], axis=1)
mv_trafo_df['grid_buffer'] = mv_trafo_df['grid_buffer'].map(shapely.wkt.loads)


crs = {'init': 'epsg:3035'}
mv_buffer_df = gpd.GeoDataFrame(mv_trafo_df['grid_buffer'],
                                        crs=crs,
                                        geometry='grid_buffer')
mv_buffer_df = mv_buffer_df.to_crs({'init': 'epsg:4326'})
mv_trafo_df['grid_buffer'] = mv_buffer_df

del mv_buffer_df

mv_trafo_df = mv_trafo_df.rename(
        columns={'v_nom': 'v_nom0'})

mv_trafo_df['v_nom1'] = mv_trafo_df.apply(
        lambda x: bus_df.loc[x['bus1']]['v_nom'], axis=1)

mv_trafo_df['p'] = mv_trafo_df.apply(
        lambda x: eval(x['p']), axis=1)

# Generators
gens_df['p'] = gens_df.apply(
        lambda x: eval(x['p']), axis=1)

# Load
load_df['p'] = load_df.apply(
        lambda x: eval(x['p']), axis=1)


#%%
print('Identify foreign buses, lines and generators')

ger_shp = nuts_shp.loc[nuts_shp['nuts_id'] == 'DE']['geometry'].values[0]

bus_df['frgn'] = bus_df.apply(
        lambda x: not x['geometry'].within(ger_shp), axis=1)

frgn_buses = bus_df.loc[bus_df['frgn'] == True].index
line_df['frgn'] = line_df.apply(
        lambda x: (x['bus0'] in frgn_buses) | (x['bus1'] in frgn_buses),
                axis=1)
gens_df['frgn'] = gens_df.apply(
        lambda x: (x['bus'] in frgn_buses),
                axis=1)

#bus_df = bus_df.drop(bus_df.loc[bus_df['frgn'] == True].index, axis=0)
#line_df = line_df.drop(line_df.loc[line_df['frgn'] == True].index, axis=0)
#gens_df = gens_df.drop(gens_df.loc[gens_df['frgn'] == True].index, axis=0)

#%%
print('s_rel_max and rel_time_over Visualization')

ans = str(input("Are you sure to update table (y/n)? "))
if ans == 'y':
    no_rows = len(line_df)
    i = 1
    for idx, row in line_df.iterrows():

        print(to_str(i/no_rows * 100) + '% ')
        s_rel_max = row['s_over_max']
        rel_time_over = row['s_over_dur']
        line_id = idx
        session.execute('''
        UPDATE model_draft.corr_vis_hv_lines as l
            SET s_rel_max = :s_rel_max, rel_time_over = :rel_time_over
            WHERE result_id = :result_id AND line_id = :line_id;
        ''', {
        'result_id': result_id,
        's_rel_max': s_rel_max,
        'rel_time_over': rel_time_over,
        'line_id': line_id})
        i = i+1
    session.commit()

    no_rows = len(mv_line_df)
    i = 1
    for idx, row in mv_line_df.iterrows():

        print(to_str(i/no_rows * 100) + '% ')
        s_rel_max = row['s_over_max']
        rel_time_over = row['s_over_dur']
        line_id = idx
        session.execute('''
        UPDATE model_draft.corr_vis_mv_lines as l
            SET s_rel_max = :s_rel_max, rel_time_over = :rel_time_over
            WHERE result_id = :result_id AND name = :line_id;
        ''', {
        'result_id': result_id,
        's_rel_max': s_rel_max,
        'rel_time_over': rel_time_over,
        'line_id': line_id})
        i = i+1
    session.commit()
#session.rollback()

#%%
print('HV Analysis')

# EHV overload
s_sum_len_t_ehv, s_sum_len_over_t_ehv, s_sum_len_under_t_ehv = aggregate_loading(
        line_df,
        ['EHV'],
        snap_idx)

# HV grid districts
hv_districts = hv_grids_shp.dissolve(by='operator')

# District Loop
for idx0, row0 in hv_districts.iterrows():
    operator = idx0
    print(operator)
    district_geom = row0['geometry']

    hv_districts.at[operator, 'considered'] = True

    ## Selections of HV components
    ### Selecting HV lines
    hv_dist_lines_df = line_df.loc[
            line_df['lev'] =='HV'
            ]
    hv_dist_lines_df = hv_dist_lines_df.loc[
            line_df['geometry'].intersects(district_geom)
            ]
    if hv_dist_lines_df.empty:
        hv_districts.at[operator, 'considered'] = False
        hv_districts.at[operator, 'drop_reason'] = 'No lines'
        continue
    ### Selecting HV buses
    hv_dist_buses_df = bus_df.loc[
            bus_df['lev'] == 'HV'
            ]
    hv_dist_buses_df = hv_dist_buses_df.loc[
            bus_df['geometry'].within(district_geom)
            ]
    ### Selecting HV generators
    hv_dist_gens_df = gens_df.loc[
            gens_df['bus'].isin(hv_dist_buses_df.index)
            ]

    if hv_dist_gens_df.empty:
        hv_districts.at[operator, 'considered'] = False
        hv_districts.at[operator, 'drop_reason'] = 'No gens'
        continue

    ### Selecting HV loads
    hv_dist_load_df = load_df.loc[
        load_df['bus'].isin(hv_dist_buses_df.index)
        ]

    ## Further HV grid information
    length = hv_dist_lines_df['length'].sum()
    hv_districts.at[operator, 'length'] = length

    capacity = hv_dist_lines_df['s_nom_length_GVAkm'].sum()
    hv_districts.at[operator, 'trans_cap'] = capacity

    if length <= 500:
        hv_districts.at[operator, 'considered'] = False
        hv_districts.at[operator, 'drop_reason'] = 'short HV grid'
        continue


    ## Loading and Overloading
    s_sum_len_t, s_sum_len_over_t, s_sum_len_under_t = aggregate_loading(
            hv_dist_lines_df,
            ['HV'],
            snap_idx)

    ## Overload correlation HV/EHV
    x = s_sum_len_over_t['HV']
    y = s_sum_len_over_t_ehv['EHV']

    r, lcl, ucl = corr(x, y)
    hv_districts.at[operator, 'HV/EHV_r'] = r
    hv_districts.at[operator, 'HV/EHV_lcl'] = lcl
    hv_districts.at[operator, 'HV/EHV_ucl'] = ucl

    if math.isnan(r):
        hv_districts.at[operator, 'considered'] = False
        hv_districts.at[operator, 'drop_reason'] = 'no Correlation'

# Plots
## HV district overview
plt_name = "HV districts"
file_dir = hv_plot_dir

fig, ax = plt.subplots(dpi=400)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

extent = tilemapbase.Extent.from_lonlat(
        5.5,
        15.3,
        47.3,
        55.2)
extent = extent.to_project_3857()
plotter = tilemapbase.Plotter(extent, t, width=800)
plotter.plot(ax, t)

plot_df = hv_districts.loc[hv_districts['considered']].reset_index()
plot_df.crs = {"init": "EPSG:4326"}
plot_df = plot_df.to_crs({"init": "EPSG:3857"})
plot_df['center_geom'] = plot_df['geometry'].apply(
        lambda x: x.representative_point().coords[:])
plot_df['center_geom'] = [
        coords[0] for coords in plot_df['center_geom']]
plot_df.plot(
        column='operator',
        ax=ax,
        alpha=0.6,
        linewidth=0.3, edgecolor='grey')
plot_df.apply(lambda x: ax.text(
        x.center_geom[0],
        x.center_geom[1],
        x.operator,
        ha='center',
        fontsize=3.5),axis=1);

### Save
file_name = 'hv_districts'
fig.savefig(
        file_dir + file_name + '.png',
        bbox_inches='tight',
        dpi=300)
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)

## HV district HV/EHV corr
plt_name = "HV districts"
file_dir = hv_plot_dir

fig, ax = plt.subplots(dpi=200)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

extent = tilemapbase.Extent.from_lonlat(
        5.5,
        15.3,
        47.3,
        55.2)
extent = extent.to_project_3857()
plotter = tilemapbase.Plotter(extent, t, width=800)
plotter.plot(ax, t)

plot_df = hv_districts.loc[hv_districts['considered']].reset_index()
plot_df.crs = {"init": "EPSG:4326"}
plot_df = plot_df.to_crs({"init": "EPSG:3857"})
plot_df['center_geom'] = plot_df['geometry'].apply(
        lambda x: x.representative_point().coords[:])
plot_df['center_geom'] = [
        coords[0] for coords in plot_df['center_geom']]
plot_df['color'] = plot_df.apply(
        lambda x: corr_colors(x['HV/EHV_r']), axis=1)

plot_df.plot(
        ax=ax,
        alpha=0.6,
        color=plot_df['color'],
        linewidth=0.3, edgecolor='grey')

plot_df.apply(lambda x: ax.text(
        x.center_geom[0],
        x.center_geom[1],
        x.operator,
        ha='center',
        fontsize=3.5),axis=1);

patches = []
for i in np.arange(1.0, -1.25, -0.25):
    patches.append(
            mpatches.Patch(
                    color=corr_colors(i),
                    label=str(i) ))

ax.legend(handles=patches, loc=4, prop={'size': 4})

### Save
file_name = 'hv_districts_corrs'
fig.savefig(
        file_dir + file_name + '.png',
        bbox_inches='tight',
        dpi=300)
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)











#######




#
#plot_df = line_df[
#            line_df['lev'] == 'HV']
#plot_df.crs = {"init": "EPSG:4326"}
#plot_df = plot_df.to_crs({"init": "EPSG:3857"})
#
#ax = add_plot_lines_to_ax(plot_df, ax, level_colors, 0.3)


