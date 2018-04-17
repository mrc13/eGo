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
from matplotlib.ticker import FormatStrFormatter

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from dateutil import parser

from ego.tools.corr_func import (color_for_s_over,
                                 add_plot_lines_to_ax,
                                 add_weighted_plot_lines_to_ax,
                                 get_lev_from_volt,
                                 add_figure_to_tex,
                                 add_table_to_tex,
                                 render_mpl_table,
                                 render_corr_table,
                                 to_str)

## Logging
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger(__name__)

#fh = logging.FileHandler('corr_anal.log', mode='w')
#fh.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fh.setFormatter(formatter)
#
#logger.addHandler(fh)

# General Inputs
cont_fct_ehv = 0.85
cont_fct_hv = 0.7
cont_fct_mv = 1
over_voltage_mv = 1.05

r_correct_fct = 2       # Because of a mistake in Freileitung oder Kabel.
result_id = 384
data_set = '2018-03-25'
result_dir = 'corr_results/' + str(result_id) + '/data_proc/' + data_set + '/'

# Directories
now = strftime("%Y-%m-%d_%H%M", localtime())

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
I have calculated 200 hours for whole Germany. 404 Ding0 grids.
I have chosen 1.0 for MV overload and 0.85 for HV overload
Now with Generators
Based on 2018-03-23
''')
readme.close()
#%% Basic functions and Dicts

level_colors = {'LV': 'grey',
                'MV': 'black',
                'HV': 'blue',
                'EHV220': 'green',
                'EHV380': 'orange',
                'EHV': 'darkred',
                'unknown': 'grey'}

all_levels = ['MV', 'HV', 'EHV220', 'EHV380']
all_aggr_levs = ['MV', 'HV', 'EHV']

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

#%% Further Processing
# HV Lines
crs = {'init': 'epsg:4326'}
line_df = gpd.GeoDataFrame(line_df,
                           crs=crs,
                           geometry=line_df.topo.map(shapely.wkt.loads))
line_df = line_df.drop(['geom', 'topo', 's'], axis=1) # s is redundant due to s_rel

#no_line = len(line_df)
#line_df = line_df.drop(line_df.loc[line_df['length'] < 1].index)
#short_lines = no_line - len(line_df)

line_df['s_nom_length_GVAkm'] = line_df['s_nom_length_TVAkm']*1e3
line_df['s_rel'] = line_df.apply(
        lambda x: eval(x['s_rel']), axis=1)
line_df['lev'] = line_df.apply(
        lambda x: get_lev_from_volt(x['v_nom']), axis=1)
line_df['aggr_lev'] = line_df.apply(
        lambda x: get_lev_from_volt(x['v_nom'], v_aggr=True), axis=1)

## Loading
line_df['s_len_abs'] = line_df.apply(    # in GVAkm
        lambda x: [
                s * x['s_nom_length_GVAkm']
                for s in x['s_rel']
                ], axis=1)
## Overload
line_df['s_over'] = line_df.apply(          ## Relative in 1
        lambda x: [
                n - cont_fct_hv
                if x['lev'] == 'HV'
                else n - cont_fct_ehv
                for n in x['s_rel']], axis=1)

line_df['s_over_abs'] = line_df.apply(      ## Absoulute in GVAkm
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
line_df['s_over_dur'] = line_df.apply(      ## Relative in 1
        lambda x:  sum(x['s_over_bol'])/len(snap_idx), axis=1)

# MV Lines
crs = {'init': 'epsg:4326'}
mv_line_df = gpd.GeoDataFrame(mv_line_df,
                              crs=crs,
                              geometry=mv_line_df.geom.map(shapely.wkt.loads))
mv_line_df = mv_line_df.drop(['geom', 's'], axis=1)

#no_line = len(mv_line_df)
#mv_line_df = mv_line_df.drop(mv_line_df.loc[mv_line_df['length'] < 0.1].index)
#short_mv_lines = no_line - len(mv_line_df)

mv_line_df['s_nom_length_GVAkm'] = mv_line_df['s_nom_length_TVAkm']*1e3
mv_line_df['s_rel'] = mv_line_df.apply(
        lambda x: eval(x['s_rel']), axis=1)
mv_line_df['lev'] = mv_line_df.apply(
        lambda x: get_lev_from_volt(x['v_nom']), axis=1)
mv_line_df['aggr_lev'] = mv_line_df.apply(
        lambda x: get_lev_from_volt(x['v_nom'], v_aggr=True), axis=1)
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
bus_df['aggr_lev'] = bus_df.apply(
        lambda x: get_lev_from_volt(x['v_nom'], v_aggr=True), axis=1)

crs = {'init': 'epsg:4326'}
mv_bus_df = gpd.GeoDataFrame(mv_bus_df,
                             crs=crs,
                             geometry=mv_bus_df.geom.map(shapely.wkt.loads))
mv_bus_df = mv_bus_df.drop(['geom'], axis=1)

mv_bus_df['lev'] = mv_bus_df.apply(
        lambda x: get_lev_from_volt(x['v_nom']), axis=1)
mv_bus_df['lev'] = mv_bus_df.apply(
        lambda x: get_lev_from_volt(x['v_nom'], v_aggr=True), axis=1)

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


#%% Delete country links and buses (Germany as basis)
    ### This comes here, cause foreign lines, buses and gens are generally excluded

## Country links
#cntry_links = [24430, 24431, 24432, 24433, 24410, 24411, 24414, 24415, 24416, 24417, 24418, 24419, 24420, 24421, 24422, 24423, 24424, 24425, 24426, 24427, 24428, 24429, 24434, 24435, 24436, 24437, 24438, 24439, 24440, 24441, 24442, 24443, 24444, 24445, 24446, 24447, 24448, 24449, 24450, 24451, 24452, 24453, 24454, 24455, 24456, 24457, 24458, 24459, 2288, 2323, 2402, 24460, 24461, 24462, 3555, 24463, 3848, 3923, 24464, 24465, 4085, 4198, 4453, 4521, 24466, 24467, 4783, 24468, 4868, 24469, 24470, 24471, 24472, 5300, 5384, 5552, 5520, 24473, 24474, 6229, 6230, 6290, 6440, 6480, 6730, 6792, 6815, 6896, 6991, 7120, 7382, 7395, 7437, 7445, 7464, 7466, 7467, 7535, 7700, 7763, 7775, 7821, 7886, 7932, 7991, 7992, 8029, 8059, 8691, 8718, 9729, 10882, 10930, 10992, 11087, 11169, 11282, 11436, 11445, 11561, 11662, 11942, 12007, 12362, 12436, 12686, 12697, 13022, 13025, 13071, 13064, 13148, 13270, 13308, 13310, 13337, 13361, 13415, 13719, 13848, 13850, 13913, 13921, 13972, 14077, 14139, 14152, 14176, 15047, 15195, 15340, 15907, 16093, 16135, 16140, 16349, 16577, 16844, 17150, 17460, 17756, 17821, 17906, 17954, 18646, 18651, 19627, 19767, 19995, 20031, 20082, 20320, 21279, 21412, 22354, 22390, 22457, 22994, 23162, 23441, 23484, 23623, 23596, 23650, 23655, 23706, 23700, 23701, 23746, 23752, 23774, 23911, 24147, 24316, 24254, 24295]
#
#line_df['within_ger'] = pd.Series(
#        {x: x not in cntry_links for x in line_df.index}
#        )

ger_shp = nuts_shp.loc[nuts_shp['nuts_id'] == 'DE']['geometry'].values[0]

#line_df['within_ger'] = line_df.apply(
#        lambda x: x['geometry'].within(ger_shp), axis=1)

#line_df = line_df.drop(line_df.loc[line_df['within_ger'] == False].index, axis=0)

bus_df['frgn'] = bus_df.apply(
        lambda x: not x['geometry'].within(ger_shp), axis=1)

frgn_buses = bus_df.loc[bus_df['frgn'] == True].index # This method is similar to the lines within method. Result should be equal
line_df['frgn_bus'] = line_df.apply(
        lambda x: (x['bus0'] in frgn_buses) | (x['bus1'] in frgn_buses),
                axis=1)
gens_df['frgn'] = gens_df.apply(
        lambda x: (x['bus'] in frgn_buses),
                axis=1)

bus_df = bus_df.drop(bus_df.loc[bus_df['frgn'] == True].index, axis=0)
line_df = line_df.drop(line_df.loc[line_df['frgn_bus'] == True].index, axis=0)
gens_df = gens_df.drop(gens_df.loc[gens_df['frgn'] == True].index, axis=0)

#%% Basic grid information Calcs.
logger.info('Basic grid information')

# MV single
index = mv_line_df.mv_grid.unique()
columns = []
mv_grids_df = pd.DataFrame(index=index, columns=columns)

mv_grids_df['length in km'] = mv_line_df.groupby(['mv_grid'])['length'].sum()
mv_grids_df['Transm. cap. in GVAkm'] = \
    mv_line_df.groupby(['mv_grid'])['s_nom_length_TVAkm'].sum() *1e3

mv_grids_df['Avg. feed-in MV'] = - mv_trafo_df[['subst_id',
                                              'p_mean']].set_index('subst_id')

mv_grids_df['Avg. feed-in HV'] = bus_df.loc[~np.isnan(bus_df['MV_grid_id'])]\
                            [['MV_grid_id','p_mean']].set_index('MV_grid_id')

#TODO: Herausfinden, mit welcher Leistung die MV Trafos angeschlossen werden.
#TODO: Hierfür besser direkt über eDisGo Generatoren arbeiten.

mv_gens_df = gens_df.merge(all_hvmv_subst_df,
              how='inner',
              left_on='bus',
              right_on='bus_id')

mv_gens_df = mv_gens_df[mv_gens_df['name'] != 'load shedding']

mv_grids_df['Inst. gen. capacity'] = mv_gens_df.groupby(['subst_id'])['p_nom'].sum()
mv_grids_df['Inst. gen. capacity'] = mv_grids_df['Inst. gen. capacity'].fillna(0)

mv_grids_df['Inst. wind cap. in GW'] \
    = mv_gens_df.loc[
            mv_gens_df['name'] == 'wind'
            ].groupby(['subst_id'])['p_nom'].sum()
mv_grids_df['Inst. wind cap. in GW'] = mv_grids_df['Inst. wind cap. in GW'].fillna(0)

## Save
file_name = 'mv_grid_single'
file_dir = analysis_dir
df = mv_grids_df

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')

# MV total
columns = ['MV']
index =   ['Tot. no. of grids',
           'No. of calc. grids',
           'Perc. of calc. grids',
           'Tot. calc. length in km',
           'Avg. len. per grid in km',
           'Estim. tot. len. in km',
           'Estim. tot. overl. in km',
           'Tot. calc. cap in GVAkm',
           'Avg. transm. cap. in GVAkm',
           'Estim. tot. trans cap. in GVAkm',
           'X/R ratio']
mv_grid_info_df = pd.DataFrame(index=index, columns=columns)

mv_grid_info_df.loc['Tot. no. of grids']['MV'] = len(all_hvmv_subst_df)

mv_grid_info_df.loc['No. of calc. grids']['MV'] = len(mv_line_df.mv_grid.unique())

mv_grid_info_df.loc['Perc. of calc. grids']['MV'] = (
        mv_grid_info_df.loc['No. of calc. grids']['MV']\
        / mv_grid_info_df.loc['Tot. no. of grids']['MV'] * 100)

mv_grid_info_df.loc['Tot. calc. length in km']['MV'] = round(
        mv_line_df['length'].sum(), 2)

mv_grid_info_df.loc['Avg. len. per grid in km']['MV'] = round(
        mv_grids_df['length in km'].mean(), 2)

mv_grid_info_df.loc['Estim. tot. len. in km']['MV'] = round( # Länge evtl. besser direkt aus Ding0 berechnen
        mv_grid_info_df.loc['Avg. len. per grid in km']['MV'] *\
        mv_grid_info_df.loc['Tot. no. of grids']['MV'], 2)

mv_grid_info_df.loc[
        'Estim. tot. overl. in km'
        ]['MV'] = (
        sum(mv_line_df.loc[mv_line_df['anytime_over'] == True]['length'])/
        (mv_grid_info_df.loc['Perc. of calc. grids']['MV'] / 100))

mv_grid_info_df.loc['Tot. calc. cap in GVAkm']['MV'] = round(
        mv_grids_df['Transm. cap. in GVAkm'].sum(), 2)

mv_grid_info_df.loc['Avg. transm. cap. in GVAkm']['MV'] = round(
        mv_grids_df['Transm. cap. in GVAkm'].mean(), 2)

mv_grid_info_df.loc['Estim. tot. trans cap. in GVAkm']['MV'] = round(
        mv_grid_info_df.loc['Avg. transm. cap. in GVAkm']['MV'] *\
        mv_grid_info_df.loc['Tot. no. of grids']['MV'], 2)

mv_rel_calc = mv_grid_info_df.loc['No. of calc. grids']['MV'] / \
        mv_grid_info_df.loc['Tot. no. of grids']['MV']

x_to_r = []
for idx, row in mv_line_df.iterrows():
    x = row['x']
    r = row['r']
    try:
        x_to_r.append(x/r * (row['length']
                             /mv_grid_info_df.loc['Tot. calc. length in km']['MV']))
    except:
        logger.warning('no r=0')

mv_grid_info_df.loc['X/R ratio']['MV'] = round( np.sum(x_to_r) ,2)
del x_to_r, x, r

## Save
title = 'MV grid overview'
file_name = 'mv_grid_info'
file_dir = analysis_dir
df = mv_grid_info_df

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
render_df = df.applymap(lambda x: to_str(x))
fig, ax = render_mpl_table(render_df,
                           header_columns=0,
                           col_width=3.0,
                           first_width=7.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)

# HV Total
columns = ['HV', 'EHV220', 'EHV380']
index =   ['Total. len. in km',
           'Total. cap. in TVAkm',
           'X/R ratio']
grid_info_df = pd.DataFrame(index=index, columns=columns)

for col in columns:
    grid_info_df.loc['Total. len. in km'][col] = round(
            line_df.loc[
                    line_df['lev'] == col
                    ]['length'].sum(), 2)

for col in columns:
    grid_info_df.loc['Total. cap. in TVAkm'][col] = round(
            line_df.loc[
                    line_df['lev'] == col
                    ]['s_nom_length_TVAkm'].sum(), 2)

for col in columns:
    x_to_r = []
    for idx, row in line_df.loc[line_df['lev'] == col].iterrows():
        x = row['x']
        r = row['r']
        if (col == 'EHV220'):
            r = r/r_correct_fct
        try:
            x_to_r.append(x/r * (row['length']
                                 /grid_info_df.loc['Total. len. in km'][col]))
        except:
            logger.warning('no r=0')

    grid_info_df.loc['X/R ratio'][col] = round( np.sum(x_to_r) ,2)
    # TODO: Check if something is wrong with 220kV!!!
    # Check the standard values for 220kV

## Save
title = 'Grid overview'
file_name = 'grid_info'
file_dir = analysis_dir
df = grid_info_df

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
render_df = df.applymap(lambda x: to_str(x))
fig, ax = render_mpl_table(render_df,
                           header_columns=0,
                           col_width=3.0,
                           first_width=3.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)


# HV/MV Comparison
columns = ['MV', 'HV', 'EHV220', 'EHV380']
index =   ['Total. len. in km',
           'Total. overl. in km',
           'Total. cap. in TVAkm',
           'X/R ratio']
hvmv_comparison_df = pd.DataFrame(index=index, columns=columns)

hvmv_comparison_df.loc['Total. len. in km']['MV'] = mv_grid_info_df.loc['Estim. tot. len. in km']['MV']
for col in grid_info_df.columns:
    hvmv_comparison_df.loc['Total. len. in km'][col] = grid_info_df.loc['Total. len. in km'][col]

hvmv_comparison_df.loc['Total. overl. in km']['MV'] = mv_grid_info_df.loc[
        'Estim. tot. overl. in km']['MV']
for col in grid_info_df.columns:
    hvmv_comparison_df.loc['Total. overl. in km'][col] = sum(line_df.loc[(line_df['anytime_over'] == True) & (line_df['lev'] == col)]['length'])

hvmv_comparison_df.loc['Total. cap. in TVAkm']['MV'] = round(mv_grid_info_df.loc['Estim. tot. trans cap. in GVAkm']['MV']/1e3, 2)
for col in grid_info_df.columns:
    hvmv_comparison_df.loc['Total. cap. in TVAkm'][col] = grid_info_df.loc['Total. cap. in TVAkm'][col]

hvmv_comparison_df.loc['X/R ratio']['MV'] = mv_grid_info_df.loc['X/R ratio']['MV']
for col in grid_info_df.columns:
    hvmv_comparison_df.loc['X/R ratio'][col] = grid_info_df.loc['X/R ratio'][col]

## Save
title = 'Total grid overview'
file_name = 'total_overview'
file_dir = analysis_dir
df = hvmv_comparison_df

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
render_df = df.applymap(lambda x: to_str(x))
fig, ax = render_mpl_table(render_df,
                           header_columns=0,
                           col_width=3.0,
                           first_width=3.5)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)

# Generators
index = gens_df.name.unique()
columns = []
gen_info = pd.DataFrame(index=index, columns=columns)

gen_info['Inst. cap. in GW'] = round(gens_df.groupby(['name'])['p_nom'].sum()/1e3, 2)
gen_info = gen_info.drop(['load shedding'])

## Save
title = 'Generators overview'
file_name = 'gen_overview'
file_dir = analysis_dir
df = gen_info

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
render_df = df.applymap(lambda x: to_str(x))
fig, ax = render_mpl_table(render_df,
                           header_columns=0,
                           col_width=3.0,
                           first_width=5.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)


#%% Electrical Overview

plt_name = "Electrical Overview"
file_dir = analysis_dir
file_name = 's_nom_hist'

fig, ax = plt.subplots(1, 4, sharey=True) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,4)
vals = []
colors = []
levs = []
for idx, lev in enumerate(all_levels):

    levs.append(lev)
    if lev == 'MV':
        df = mv_line_df
        df = df.loc[df['length']>0.3]
    else:
        df = line_df
        df = df.loc[df['length']>3]

    vals.append(df.loc[df['lev'] == lev]['s_nom'])
    colors.append(level_colors[lev])

    weights = np.ones_like(vals[idx])/float(len(vals[idx])) * 100

    counts, bins, patches = ax[idx].hist(
                  vals[idx],
                  color=colors[idx],
                  weights=weights,
                  bins=8,
                  alpha=0.7)

    ax[idx].set_xticks(bins)
    ax[idx].set_xticklabels(bins, rotation=45, rotation_mode="anchor", ha="right")
    ax[idx].xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
#    for label in ax[idx].xaxis.get_ticklabels()[::2]:
#        label.set_visible(False)       # Every second visible

    ax[idx].legend((lev,))


ax[0].set(ylabel='Relative number of lines in %')
ax[0].set(xlabel='Thermal rated power in MVA')

## Save
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)


#%% Corr Germany Calcs
#Todo: Nuts Abfragen hier noch reinbringen

# Line Overload per voltage level
## Total
s_sum_len_over_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=all_levels)

len_over_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=all_levels)


for df in [line_df, mv_line_df]:

    for index, row in df.iterrows():
        lev = get_lev_from_volt(row['v_nom'])

        #cap
        s_over_series = pd.Series(data=row['s_over_abs'], index=snap_idx)

        s_sum_len_over_t[lev] = s_sum_len_over_t[lev] + s_over_series

        #length
        s_over_series = pd.Series(data=row['s_over_bol'], index=snap_idx)
        len_over_series = s_over_series * row['length']

        len_over_t[lev] = len_over_t[lev] + len_over_series

s_sum_len_over_t['MV'] = s_sum_len_over_t['MV'] / mv_rel_calc
len_over_t['MV'] = len_over_t['MV'] / mv_rel_calc # For MV, the calculated values must be estimated for the entire grid

## Relative
s_sum_len_over_t_norm = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=all_levels)
len_over_t_norm = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=all_levels)

for col in s_sum_len_over_t_norm.columns:
    s_sum_len_over_t_norm[col] = s_sum_len_over_t[col] / (hvmv_comparison_df.loc['Total. cap. in TVAkm'][col] * 1e3) * 100
for col in len_over_t_norm.columns:
    len_over_t_norm[col] = len_over_t[col] / hvmv_comparison_df.loc['Total. len. in km'][col] * 100

# Generators
columns = [car for car in carrier_colors.keys()]
gen_dispatch_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=columns)

for idx, row in gens_df.iterrows():
    name = row['name']
    p_series = pd.Series(data=row['p'], index=snap_idx)
    gen_dispatch_t[name] = gen_dispatch_t[name] + p_series

#gen_dispatch_t['total'] = gen_dispatch_t.apply(
#        lambda x: pd.Series([x[n] for n in columns]).sum(), axis=1)

# Load
load_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=['load'])

for idx, row in load_df.iterrows():
    p_series = pd.Series(data=row['p'], index=snap_idx)
    load_t['load'] = load_t['load'] + p_series

# Voltage
voltage_over_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=['MV_volt'])

for idx, row in lv_stations.iterrows():
    volt_over_series = pd.Series(data=row['v_over_bol'], index=snap_idx)
    voltage_over_t ['MV_volt'] = voltage_over_t ['MV_volt'] + volt_over_series

voltage_over_t_norm = voltage_over_t / len(lv_stations) * 100


#%% Corr Germany Corr and Plots

# Corr
considered_carriers = ['solar',
                       'wind',
                       'coal',
                       'lignite',
                       'uranium']
gen_dispatch_t_subset = gen_dispatch_t[considered_carriers]
corr_ger_df = s_sum_len_over_t.merge(gen_dispatch_t_subset,
                       left_index=True,
                       right_index=True
                       ).merge(
                               load_t,
                               left_index=True,
                               right_index=True
                               ).merge(
                                       voltage_over_t,
                                       left_index=True,
                                       right_index=True).corr(method='pearson')

corr_ger_df = corr_ger_df.drop(considered_carriers, axis=0)
corr_ger_df = corr_ger_df.drop('load', axis=0)

## Save
title = 'Correlation Germany'
file_name = 'corr_ger'
file_dir = ger_corr_dir
df = corr_ger_df

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
fig, ax = render_corr_table(df,
                           header_columns=0,
                           col_width=1.5,
                           first_width=1.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)


# Plot
## Overview plot
plt_name = "Overview Germany"
file_dir = ger_plot_dir

fig, ax = plt.subplots(4, sharex=True) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,10)

frm = s_sum_len_over_t.plot(
        kind='area',
        legend=True,
        color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
        linewidth=.5,
        alpha=0.7,
        ax = ax[0])
leg = ax[0].legend(loc='upper right',
          ncol=2, fancybox=True, shadow=True, fontsize=9)
leg.get_frame().set_alpha(0.5)
ax[0].set(ylabel='Overloading in GVAkm')

frm = len_over_t_norm.plot(
        kind='line',
        legend=False,
        color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
        linewidth=2,
        alpha=0.7,
        ax = ax[1])
#leg = ax[1].legend(loc='upper right',
#          ncol=2, fancybox=True, shadow=True, fontsize=9)
#leg.get_frame().set_alpha(0.5)
ax[1].set(ylabel='Overl. length in %')

frm = voltage_over_t_norm.plot(
        kind='line',
        legend=True,
        color='black',
        linewidth=2,
        alpha=0.7,
        ax = ax[2])
leg = ax[2].legend(loc='upper right',
          ncol=1, fancybox=True, shadow=True, fontsize=9)
leg.get_frame().set_alpha(0.5)
ax[2].set(ylabel='Volt. issues in % of buses')

frm = (gen_dispatch_t/1e3).plot(
        kind='area',
        legend=True,
        color=[carrier_colors[name] for name in  gen_dispatch_t.columns],
        linewidth=.5,
        alpha=0.7,
        ax = ax[3])
leg = ax[3].legend(loc='upper right',
          ncol=7, fancybox=True, shadow=True, fontsize=9)
leg.get_frame().set_alpha(0.5)
ax[3].set(ylabel='Generation in GW')

file_name = 'overview_germany'
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)

##% Scatter Plots
plt_name = "Correlation of Overloaded grid Length"
file_dir = ger_plot_dir
for x_lev in len_over_t.columns:
    fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
    fig.set_size_inches(12,6)
    ax1.set_xlim(0, max(len_over_t[x_lev]))
    y_max = 0
    for y_lev in len_over_t.columns:
        y_max_lev = max(len_over_t[y_lev])
        if y_max_lev > y_max:
            y_max = y_max_lev
        ax1.set_ylim(0, y_max)
        x_volt = get_volt_from_lev(x_lev)
        y_volt = get_volt_from_lev(y_lev)
        if x_volt == y_volt:
            continue
        plt_name = x_lev +' and ' + y_lev +' Loading Correlation'

## Dass 380 am meisten ansteigt bei HV und MV lässt sich erklären, da die RE, die das Netz überlasten aus dem VN kommen (angeschlossen sind)
        len_over_t.plot.scatter(
                x=x_lev,
                y=y_lev,
                color=level_colors[y_lev],
                label=y_lev,
                alpha=0.5,
                ax=ax1)

        regr = linear_model.LinearRegression()
        x = len_over_t[x_lev].values.reshape(len(len_over_t), 1)
        y = len_over_t[y_lev].values.reshape(len(len_over_t), 1)
        regr.fit(x, y)
        plt.plot(x, regr.predict(x), color=level_colors[y_lev], linewidth=1)
        plt.ylabel('Overloaded lines in km')
        plt.xlabel(x_lev + ', Overloaded lines in km')

    file_name = 'loading_corr_' + x_lev
    fig.savefig(file_dir + file_name + '.png')
    add_figure_to_tex (plt_name, file_dir, file_name)
    plt.close(fig)

##% Histograms
plt_name = "Retative Overloaded Length Histogram"
file_dir = ger_plot_dir
fig, ax = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(8,4)

vals, colors, levs, weights = [], [], [], []

for col in len_over_t_norm.columns:
   levs.append(col)
   vals.append(len_over_t_norm[col])
   weights.append(
           np.ones_like(len_over_t_norm[col])
           /float(len(len_over_t_norm[col]))
           * 100)
   colors.append(level_colors[col])

ax = plt.hist(x=vals,
              color=colors,
              bins=10,
              alpha=0.7,
              weights=weights)
plt.xlabel("Overloaded Length in % of total length")
plt.legend(levs) # Very interesting result. MV grid overloads and HV occurr very local!

file_name = 'overloaded_length_hist'
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)


# All s_rel over Comparison per level

plt_name = "Relative overload Hist"
file_dir = ger_plot_dir

s_rel_over_per_lev = { key : [] for key in all_levels }
for df in [line_df, mv_line_df]:
    for index, row in df.iterrows():
        lev = get_lev_from_volt(row['v_nom'])
        s_rel_series = row['s_over']
        for s_rel in s_rel_series:
            if s_rel > 0:
                s_rel_over_per_lev[lev].append(s_rel)

fig, ax = plt.subplots(1,1)
fig.set_size_inches(8,4)

vals, colors, levs, weights = [], [], [], []

for key, value in s_rel_over_per_lev.items():
   levs.append(key)
   vals.append(value)
   weights.append(
           np.ones_like(value)
           /float(len(value))
           * 100)
   colors.append(level_colors[key])

bins = [0, .1, .2, .3, .4, .5, .6, .7, 0.8, 0.9, 1., 1.1, 1.2, 1.3]
ax = plt.hist(x=vals,
              color=colors,
              bins=bins,
              weights=weights,
              alpha = 0.7)
plt.xlabel("Relative overload")
plt.legend(levs)

file_name = 'rel_overload_hist'
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)


# Spatial line plots
## Max overload
plt_name = "Grid Germany (maximum overload per line)"
file_dir = ger_plot_dir

fig, ax1 = plt.subplots(1)
fig.set_size_inches(12,14)


plot_df = line_df
ax1 = add_plot_lines_to_ax(
        plot_df.loc[plot_df['lev'] == 'HV'],
        v_ax=ax1,
        v_level_colors=level_colors,
        v_size=0.2)

plot_df = line_df
ax1 = add_plot_lines_to_ax(
        plot_df.loc[
                (plot_df['lev'] == 'EHV220') | (plot_df['lev'] == 'EHV380')
                ],
        v_ax=ax1,
        v_level_colors=level_colors,
        v_size=1)

plot_df = line_df
plot_df['color'] = plot_df.apply(
        lambda x: color_for_s_over(x['s_over_max']), axis=1)

plot_df = plot_df.loc[
                ((plot_df['lev'] == 'EHV220') | (plot_df['lev'] == 'EHV380'))
                   ]

ax1 = add_weighted_plot_lines_to_ax(
        plot_df,
        v_ax=ax1,
        v_size=3)

file_name = 'ger_grid_max_overload'
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)

## Overload hours
plt_name = "Grid Germany (overload duration)"
file_dir = ger_plot_dir

fig, ax1 = plt.subplots(1)
fig.set_size_inches(12,14)


plot_df = line_df
ax1 = add_plot_lines_to_ax(
        plot_df.loc[plot_df['lev'] == 'HV'],
        v_ax=ax1,
        v_level_colors=level_colors,
        v_size=0.2)

plot_df = line_df
ax1 = add_plot_lines_to_ax(
        plot_df.loc[
                (plot_df['lev'] == 'EHV220') | (plot_df['lev'] == 'EHV380')
                ],
        v_ax=ax1,
        v_level_colors=level_colors,
        v_size=1)

plot_df = line_df
plot_df['color'] = plot_df.apply(
        lambda x: color_for_s_over(x['s_over_dur']*2), axis=1)

plot_df = plot_df.loc[
                ((plot_df['lev'] == 'EHV220') | (plot_df['lev'] == 'EHV380'))
                   ] ## not HV!!!!!!

ax1 = add_weighted_plot_lines_to_ax(
        plot_df,
        v_ax=ax1,
        v_size=3)

file_name = 'ger_grid_dur_overload'
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)

## Anytime overloaded
plt_name = "Grid Germany (overloaded anytime)"
file_dir = ger_plot_dir

fig, ax1 = plt.subplots(1)
fig.set_size_inches(12,14)

plot_df = line_df
ax1 = add_plot_lines_to_ax(
        plot_df.loc[plot_df['lev'] == 'HV'],
        v_ax=ax1,
        v_level_colors=level_colors,
        v_size=0.2)

plot_df = line_df
ax1 = add_plot_lines_to_ax(
        plot_df.loc[
                (plot_df['lev'] == 'EHV220') | (plot_df['lev'] == 'EHV380')
                ],
        v_ax=ax1,
        v_level_colors=level_colors,
        v_size=1)

plot_df = line_df.loc[
        (line_df['anytime_over'] == True)
        & (line_df['lev'] != 'HV')]
plot_df.plot(color='red',
             linewidth=3,
             ax=ax1)

file_name = 'ger_grid_any_overload'
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)

#%% HV Districts

# Processing
hv_grids_shp = hv_grids_shp.dissolve(by='operator')

hv_grids_shp['center_geom'] = hv_grids_shp['geometry'].apply(
        lambda x: x.representative_point().coords[:])
hv_grids_shp['center_geom'] = [
        coords[0] for coords in hv_grids_shp['center_geom']]
hv_grids_shp = hv_grids_shp.reset_index()

considered_carriers = ['solar', # For correlation
                           'wind',
                           'coal',
                           'lignite',
                           'uranium']

# Overview Dataframe
index = hv_grids_shp['operator']
hv_dist_df = pd.DataFrame(index=index)
hv_dist_corr_tot_df = pd.DataFrame(index=index)
hv_dist_corr_over_df = pd.DataFrame(index=index)

# District Loop
for idx0, row0 in hv_grids_shp.iterrows():
    operator = row0['operator']
    district_geom = row0['geometry']

    hv_dist_df.at[operator, 'considered'] = True

## Select all relevant lines, buses and their levels

    hv_dist_trafo_df = trafo_df.loc[
            trafo_df['geometry'].within(district_geom)
            ]
    hv_dist_volt = list(set(
        hv_dist_trafo_df['v_nom0'].tolist()
        + hv_dist_trafo_df['v_nom1'].tolist()))
    hv_dist_levs = sorted(list(set([
            get_lev_from_volt(volt, v_aggr=True)
            for volt in hv_dist_volt])), reverse=True)

    hv_dist_lines_df = line_df.loc[
            line_df['aggr_lev'].isin(hv_dist_levs)
            ]
    hv_dist_lines_df = hv_dist_lines_df.loc[
            line_df['geometry'].intersects(district_geom)
            ]
    if hv_dist_lines_df.empty:
        hv_dist_df.at[operator, 'considered'] = False
        hv_dist_df.at[operator, 'drop_reason'] = 'No lines'
        continue

    for lev in hv_dist_levs:
        hv_dist_df.at[operator, lev] = True

    hv_dist_buses_df = bus_df.loc[
            bus_df['aggr_lev'].isin(hv_dist_levs)
            ]
    hv_dist_buses_df = hv_dist_buses_df.loc[
            bus_df['geometry'].within(district_geom)
            ]

    hv_dist_gens_df = gens_df.loc[
            gens_df['bus'].isin(hv_dist_buses_df.index)
            ]
    if hv_dist_gens_df.empty:
        hv_dist_df.at[operator, 'considered'] = False
        hv_dist_df.at[operator, 'drop_reason'] = 'No gens'
        continue

    hv_dist_load_df = load_df.loc[
            load_df['bus'].isin(hv_dist_buses_df.index)
            ]

    if not 'HV' in hv_dist_levs:
        hv_dist_df.at[operator, 'considered'] = False
        hv_dist_df.at[operator, 'drop_reason'] = 'No HV'
        continue
## Further data on this grid
### Generation Capacities

    for car in considered_carriers:
        if car in set(hv_dist_gens_df['name']):
            hv_dist_df.at[
                    operator,
                    'cap_'+ car
                    ] = hv_dist_gens_df.loc[
                            hv_dist_gens_df['name'] == car
                            ]['p_nom'].sum()

## Calculate grid capacity and length per level
    columns = hv_dist_levs
    index =   ['Cap. in GVAkm', 'Len. in km']
    hv_dist_cap_df = pd.DataFrame(index=index, columns=columns)

    cap = hv_dist_lines_df.groupby('aggr_lev')['s_nom_length_GVAkm'].sum()
    for idx, val in cap.iteritems():
        hv_dist_cap_df.loc['Cap. in GVAkm'][idx] = val
        hv_dist_df.at[operator, 'cap_'+idx] = val

    length = hv_dist_lines_df.groupby('aggr_lev')['length'].sum()
    for idx, val in length.iteritems():
        hv_dist_cap_df.loc['Len. in km'][idx] = val
        hv_dist_df.at[operator, 'len_'+idx] = val

    if length['HV'] <= 500:
        hv_dist_df.at[operator, 'considered'] = False
        hv_dist_df.at[operator, 'drop_reason'] = 'short HV grid'
        continue

## Overload Dataframes
### Absolute
    s_sum_len_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=hv_dist_levs)
    s_sum_len_over_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=hv_dist_levs)
    s_sum_len_under_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=hv_dist_levs)

#    len_over_t = pd.DataFrame(0.0,
#                                   index=snap_idx,
#                                   columns=hv_dist_levs)

    for idx1, row1 in hv_dist_lines_df.iterrows():
        lev = row1['aggr_lev']

        #### s_len
        s_len_series = pd.Series(
                data=row1['s_len_abs'],
                index=snap_idx)

        s_sum_len_t[lev] = (s_sum_len_t[lev]
                                 + s_len_series)
        #### s_over
        s_over_series = pd.Series(
                data=row1['s_over_abs'],
                index=snap_idx)

        s_sum_len_over_t[lev] = (s_sum_len_over_t[lev]
                                 + s_over_series)

        #### s_under
        s_under_series = pd.Series(
                data=row1['s_under_abs'],
                index=snap_idx)

        s_sum_len_under_t[lev] = (s_sum_len_under_t[lev]
                                 + s_under_series)
#        #### length
#        len_over_series = pd.Series(
#                data=row1['s_over_bol'],
#                index=snap_idx) * row1['length']

#        len_over_t[lev] = len_over_t[lev] + len_over_series

#    if max(s_sum_len_over_t['HV']) <= 10:
#        hv_dist_df.at[operator, 'considered'] = False
#        hv_dist_df.at[operator, 'drop_reason'] = 'No HV overl.'
#        continue

### Relative
#    s_sum_len_over_t_norm = pd.DataFrame(
#            0.0,
#            index=snap_idx,
#            columns=hv_dist_levs)
#    len_over_t_norm = pd.DataFrame(
#            0.0,
#            index=snap_idx,
#            columns=hv_dist_levs)

#    for col in s_sum_len_over_t_norm.columns:
#        s_sum_len_over_t_norm[col] = (
#                s_sum_len_over_t[col]
#                / hv_dist_cap_df.loc['Cap. in GVAkm'][col]
#                * 100)
#    for col in len_over_t_norm.columns:
#        len_over_t_norm[col] = (
#                len_over_t[col]
#                / hv_dist_cap_df.loc['Len. in km'][col]
#                * 100)

## Generation
    columns = [
        car for car in  carrier_colors.keys()
        if (car in set(hv_dist_gens_df['name']))
        ]

    gen_dispatch_t = pd.DataFrame(
            0.0,
            index=snap_idx,
            columns=columns)

    var_dispatch_t = pd.DataFrame(
            0.0,
            index=snap_idx,
            columns=['var_rens'])

    for idx1, row1 in hv_dist_gens_df.iterrows():
        name = row1['name']
        p_series = pd.Series(data=row1['p'], index=snap_idx)
        gen_dispatch_t[name] = gen_dispatch_t[name] + p_series

        if name in var_rens:
            var_dispatch_t['var_rens'] = (
                    var_dispatch_t['var_rens']
                    + p_series)

## Load
    load_t = pd.DataFrame(
            0.0,
            index=snap_idx,
            columns=['load'])

    for idx, row in hv_dist_load_df.iterrows():
        p_series = pd.Series(data=row['p'], index=snap_idx)
        load_t['load'] = load_t['load'] + p_series

    res_load_t = pd.DataFrame(
            0.0,
            index=snap_idx,
            columns=['res_load'])
    res_load_t['res_load'] = load_t['load'] - var_dispatch_t['var_rens']

    hv_dist_df.at[operator, 'mean_res_l.'] = res_load_t.mean()
# Correlation
    gen_dispatch_t_subset = pd.DataFrame(
            index=snap_idx)

    for carrier in considered_carriers:
        if carrier in set(gen_dispatch_t.columns):
            gen_dispatch_t_subset[carrier] = gen_dispatch_t[carrier]

    s_sum_len_over_t_thresh = s_sum_len_over_t.loc[
            (s_sum_len_over_t != 0).any(axis=1)
            ]
## Overloading and Loading correlation
    if hv_dist_df.loc[operator, 'EHV'] == True:
### Overload  ing
        s_sum_len_over_corr = s_sum_len_over_t_thresh.corr(
                method='pearson')
        hv_dist_corr_over_df.at[
                operator,
                'HV_EHV'] = s_sum_len_over_corr.loc['HV']['EHV']
### Loading
        s_sum_len_corr = s_sum_len_t.corr(
                method='pearson')
        hv_dist_corr_tot_df.at[
                operator,
                'HV_EHV'] = s_sum_len_corr.loc['HV']['EHV']
## Line Overloading and Loading with everything else
### Overloading
    corr_hv_over_df = s_sum_len_over_t.merge(
            gen_dispatch_t_subset,
            left_index=True,
            right_index=True
            ).merge(
                    load_t,
                    left_index=True,
                    right_index=True
                    ).merge(
                        var_dispatch_t,
                        left_index=True,
                        right_index=True
                        ).merge(
                            res_load_t,
                            left_index=True,
                            right_index=True
                                )

    corr_hv_over_thresh_df = corr_hv_over_df.loc[
            (corr_hv_over_df != 0).any(axis=1)]

    hv_over_corr = corr_hv_over_thresh_df.corr(method='pearson')
### Loading
    corr_hv_df = s_sum_len_t.merge(
            gen_dispatch_t_subset,
            left_index=True,
            right_index=True
            ).merge(
                    load_t,
                    left_index=True,
                    right_index=True
                    ).merge(
                        var_dispatch_t,
                        left_index=True,
                        right_index=True
                        ).merge(
                            res_load_t,
                            left_index=True,
                            right_index=True
                                )

    corr_hv_thresh_df = corr_hv_df.loc[
            (corr_hv_df != 0).any(axis=1)]

    hv_corr = corr_hv_thresh_df.corr(method='pearson')

    for col in list(gen_dispatch_t_subset.columns) + ['load', 'var_rens', 'res_load']:
        for lev in hv_dist_levs:
            hv_dist_corr_over_df.at[
                    operator,
                    lev + '_' + col
                    ] = hv_over_corr.loc[lev][col]
            hv_dist_corr_tot_df.at[
                    operator,
                    lev + '_' + col
                    ] = hv_corr.loc[lev][col]

#    corr_hv_df = corr_hv_df.drop(set(gen_dispatch_t_subset.columns), axis=0)
#    corr_hv_df = corr_hv_df.drop('var_rens', axis=0)
#    corr_hv_df = corr_hv_df.drop('load', axis=0)
#    corr_hv_df = corr_hv_df.drop('res_load', axis=0)
#
#    if 'EHV' in hv_dist_levs:
#        hv_dist_df.at[
#                operator,
#                'corr_HV_EHV'
#                ] = corr_hv_df.loc['HV']['EHV']
#
#    hv_dist_df.at[
#            operator,
#            'corr_HV_res_load'
#            ] = corr_hv_df.loc['HV']['res_load']
#
### Save
#    title = 'Correlation HV'
#    file_name = 'corr_hv_' + operator
#    file_dir = hv_corr_dir
#    df = corr_hv_df
#
#    df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
#    fig, ax = render_corr_table(df,
#                               header_columns=0,
#                               col_width=1.5,
#                               first_width=1.0)
#    fig.savefig(file_dir + file_name + '.png')
#    add_table_to_tex(title, file_dir, file_name)
#    plt.close(fig)


# Plots
    ## Line
    plt_name = "HV District overloading"
    file_dir = hv_plot_dir

    fig, ax = plt.subplots(3, sharex=True) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
    fig.set_size_inches(10,6)

    y_max = sum(s_sum_len_t.max())
    for lev in list(s_sum_len_t.max().sort_values(ascending=False).index):

        s_sum_len_under_t_lev = s_sum_len_under_t[lev].rename('normal ' + lev)
        s_sum_len_over_t_lev = s_sum_len_over_t[lev].rename('overloaded ' + lev)


        plot_df = s_sum_len_under_t_lev.to_frame().join(s_sum_len_over_t_lev)
        plot_df.plot(
            kind='area',
            legend=True,
            color=[level_colors[lev], 'orange'],
            linewidth=0.2,
            ax = ax[0])

#    s_sum_len_over_t.plot(
#            kind='line',
#            legend=True,
#            color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
#            linewidth=0.5,
#            alpha=0.2,
#            ax = ax[0])
#    s_sum_len_under_t.plot(
#            kind='line',
#            legend=True,
#            color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
#            linewidth=2,
#            ax = ax[0])
#    s_sum_len_t.plot(
#            kind='line',
#            legend=True,
#            color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
#            linewidth=3,
#            ax = ax[0])
    ax[0].set_ylim([0,y_max])
    leg = ax[0].legend(loc='upper right',
              ncol=1, fancybox=True, shadow=True, fontsize=9)
    leg.get_frame().set_alpha(0.5)
    ax[0].set(ylabel='Loading. in GVAkm')

#    len_over_t_norm.plot(
#        kind='line',
#        legend=False,
#        color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
#        linewidth=2,
#        alpha=0.7,
#        ax = ax[1])
#    ax[1].set(ylabel='Overl. len. in %')

    gen_dispatch_t.plot(
            kind='area',
            legend=True,
            color=[carrier_colors[name] for name in  gen_dispatch_t.columns],
            linewidth=.5,
            alpha=0.7,
            ax = ax[1])

    leg = ax[1].legend(loc='upper right',
              ncol=5, fancybox=True, shadow=True, fontsize=9)
    leg.get_frame().set_alpha(0.5)
    ax[1].set(ylabel='Gen. in MW')

    load_t.plot(
            kind='line',
            legend=True,
            color='grey',
            linewidth=3,
            alpha=0.9,
            ax=ax[2])
    res_load_t.plot(
            kind='line',
            legend=True,
            color='red',
            linewidth=3,
            alpha=0.9,
            ax=ax[2])
    leg = ax[2].legend(loc='upper right',
              ncol=1, fancybox=True, shadow=True, fontsize=9)
    leg.get_frame().set_alpha(0.5)
    ax[2].set(ylabel='Load in MW')

    file_name = 'hv_district_overloading_' + operator
    fig.savefig(file_dir + file_name + '.png')
    add_figure_to_tex(plt_name, file_dir, file_name)
    plt.close(fig)

    ## Spatial
    plt_name = "HV Grid District " + operator
    file_dir = hv_plot_dir
    fig, ax1 = plt.subplots(1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
    fig.set_size_inches(6,6)
    xmin, ymin, xmax, ymax = district_geom.bounds

    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])

    plot_df = hv_dist_lines_df[
            hv_dist_lines_df['aggr_lev'] == 'HV'
            ]
    ax1 = add_plot_lines_to_ax(plot_df, ax1, level_colors, 1)

    plot_df = hv_dist_lines_df[
            hv_dist_lines_df['aggr_lev'] == 'EHV']
    ax1 = add_plot_lines_to_ax(plot_df, ax1, level_colors, 3)

    hv_dist_trafo_df.plot(ax=ax1,
                   alpha = 1,
                   color = 'r',
                   marker='o',
                   markersize=300,
                   facecolors='none'
                   )
    hv_grids_shp[hv_grids_shp['operator'] == operator].plot(
            ax=ax1,
            alpha = 0.3,
            color = 'y')
#    center_geom = hv_grids_shp.loc[
#            hv_grids_shp['operator'] == operator
#            ]['center_geom'].values[0]
#    type(center_geom)
#    ax1.text(
#        center_geom[0],
#        center_geom[1],
#        operator,
#        ha='center')

    file_name = 'hv_district_' + operator
    fig.savefig(file_dir + file_name + '.png')
    add_figure_to_tex(plt_name, file_dir, file_name)
    plt.close(fig)


## HV district overniew
plt_name = "HV districts"
file_dir = hv_plot_dir

fig, ax1 = plt.subplots(1)
fig.set_size_inches(12,14)

cons_grids = hv_dist_df.loc[
        hv_dist_df['considered']==True
        ].index
plot_df = hv_grids_shp#.loc[
#        hv_grids_shp['operator'].isin(cons_grids)
#        ]

plot_df.plot(column='operator', ax=ax1, alpha=0.7)
plot_df.apply(lambda x: ax1.text(
        x.center_geom[0],
        x.center_geom[1],
        x.operator,
        ha='center'),axis=1);

file_name = 'hv_districts'
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)

## HV district overniew
plt_name = "HV grids"
file_dir = hv_plot_dir

fig, ax1 = plt.subplots(1)
fig.set_size_inches(12,14)

cons_grids = hv_dist_df.loc[
        hv_dist_df['considered']==True
        ].index
plot_df = hv_grids_shp.loc[
        hv_grids_shp['operator'].isin(cons_grids)
        ]

plot_df.plot(column='operator', ax=ax1, alpha=0.4)
plot_df.apply(lambda x: ax1.text(
        x.center_geom[0],
        x.center_geom[1],
        x.operator,
        ha='center'),axis=1);

plot_df = line_df
ax1 = add_plot_lines_to_ax(
        plot_df.loc[plot_df['lev'] == 'HV'],
        v_ax=ax1,
        v_level_colors=level_colors,
        v_size=0.2)

ax1 = add_plot_lines_to_ax(
        plot_df.loc[plot_df['aggr_lev'] == 'EHV'],
        v_ax=ax1,
        v_level_colors=level_colors,
        v_size=0.4)

file_name = 'hv_grids'
fig.savefig(file_dir + file_name + '.png')
add_figure_to_tex (plt_name, file_dir, file_name)
plt.close(fig)

## Save
title = 'Correlation HV Overloading'
file_name = 'corr_hv_over'
file_dir = hv_corr_dir
df = hv_dist_corr_over_df.loc[hv_dist_df['considered']]

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
fig, ax = render_corr_table(df,
                           header_columns=0,
                           col_width=1.5,
                           first_width=2.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)
plt.close(fig)

title = 'Correlation HV Loading'
file_name = 'corr_hv_loading'
file_dir = hv_corr_dir
df = hv_dist_corr_tot_df.loc[hv_dist_df['considered']]

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
fig, ax = render_corr_table(df,
                           header_columns=0,
                           col_width=1.5,
                           first_width=2.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)
plt.close(fig)

# Second level correlation
## Loading
considered_columns = [
        'cap_solar',
        'cap_wind',
        'cap_coal',
        'cap_lignite',
        'mean_res_l.',
        'len_HV',
        'len_EHV']
second_corr_hv = hv_dist_df[
        considered_columns
        ][hv_dist_df['considered']].merge(
            hv_dist_corr_tot_df,
            left_index=True,
            right_index=True
            ).corr(method='pearson')

second_corr_hv = second_corr_hv.drop(hv_dist_corr_tot_df.columns, axis=0)
second_corr_hv = second_corr_hv.drop(considered_columns, axis=1)


title = 'Second Level HV Correlation (Loading'
file_name = 'second_corr_hv_loading'
file_dir = hv_corr_dir
df = second_corr_hv

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
fig, ax = render_corr_table(df,
                           header_columns=0,
                           col_width=1.5,
                           first_width=2.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)
plt.close(fig)

## Overloading
second_corr_hv = hv_dist_df[
        considered_columns
        ][hv_dist_df['considered']].merge(
            hv_dist_corr_over_df,
            left_index=True,
            right_index=True
            ).corr(method='pearson')

second_corr_hv = second_corr_hv.drop(hv_dist_corr_tot_df.columns, axis=0)
second_corr_hv = second_corr_hv.drop(considered_columns, axis=1)


title = 'Second Level HV Correlation (Overloading)'
file_name = 'second_corr_hv_overloading'
file_dir = hv_corr_dir
df = second_corr_hv

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
fig, ax = render_corr_table(df,
                           header_columns=0,
                           col_width=1.5,
                           first_width=2.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)
plt.close(fig)
#%% Corr District Calcs

columns = [['mv_grid',
            'lev0',
            'lev1',
            'r',
            'lev0_rel_overl_max',
            'lev1_rel_overl_max',
            'cap0',
            'cap1']]
dist_df = pd.DataFrame(columns=columns)

mv_trafo_df = mv_trafo_df.set_geometry('grid_buffer')
# Loop through HV/MV Trafos
for index, row in mv_trafo_df.iterrows():

    mv_grid_id = row['subst_id']
    grid_buffer = row['grid_buffer']
    bus1 = row['bus1']

    dist_volts = []

    ## HV/MV Trafo voltages
    dist_volts.append(row['v_nom0'])
    dist_volts.append(row['v_nom1'])

    dist_levs = [get_lev_from_volt(volt) for volt in dist_volts]

    # Select all relevant lines
    ## MV
    dist_mv_lines_df = mv_line_df.loc[mv_line_df['mv_grid'] == mv_grid_id]

    ## HV
    dist_hv_lines_df = line_df.loc[
            [x in dist_volts for x in line_df['v_nom']]
            ]

    dist_hv_lines_df = dist_hv_lines_df.loc[
            dist_hv_lines_df['geometry'].intersects(grid_buffer)
            ]

    # Select all relevant LV stations
    dist_lv_stations = lv_stations.loc[
            lv_stations['geometry'].within(grid_buffer)
            ]

    # Calculate grid capacity per level
    columns = dist_levs
    index =   ['s_nom_length_MVAkm']
    dist_cap_df = pd.DataFrame(index=index, columns=columns)

    ## HV
    hv_cap = dist_hv_lines_df.groupby('lev')['s_nom_length_GVAkm'].sum()
    for idx, val in hv_cap.iteritems():
        dist_cap_df.loc['s_nom_length_MVAkm'][idx] = val*1e3
    ## MV
    mv_cap = dist_mv_lines_df.groupby('lev')['s_nom_length_GVAkm'].sum()
    for idx, val in mv_cap.iteritems():
        dist_cap_df.loc['s_nom_length_MVAkm'][idx] = val*1e3

    # Overload Dataframe
    dist_s_sum_len_over_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=dist_levs)

    dist_s_sum_len_over_t_norm = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=dist_levs)

    for df in [dist_mv_lines_df, dist_hv_lines_df]:

        for i, r in df.iterrows():
            lev = r['lev']

            s_over_series = pd.Series(data=r['s_over_abs'],
                                      index=snap_idx)*1e3       # Then in MVAkm

            dist_s_sum_len_over_t[lev] = (  dist_s_sum_len_over_t[lev]
                                            + s_over_series)

    for col in dist_levs:
        dist_s_sum_len_over_t_norm[col] = (
                dist_s_sum_len_over_t[col]
                / dist_cap_df.loc['s_nom_length_MVAkm'][col]
                * 100)
    # Generation
    dist_gen_df = gens_df.loc[gens_df['bus'] == bus1]
    columns = [
            car for car in  carrier_colors.keys()
            if (car in set(dist_gen_df['name']))
            ]

    dist_gen_dispatch_t = pd.DataFrame(0.0,
                                       index=snap_idx,
                                       columns=columns)

    for idx, row in dist_gen_df.iterrows():
        name = row['name']
        p_series = pd.Series(data=row['p'], index=snap_idx)
        dist_gen_dispatch_t[name] = dist_gen_dispatch_t[name] + p_series

    # Load
    dist_load_df = load_df.loc[load_df['bus'] == bus1]
    dist_load_t = pd.DataFrame(0.0,
                                       index=snap_idx,
                                       columns=['load'])

    for idx, row in dist_load_df.iterrows():
        p_series = pd.Series(data=row['p'], index=snap_idx)
        dist_load_t['load'] = dist_load_t['load'] + p_series

    # Voltage

    dist_voltage_over_t = pd.DataFrame(0.0,
                                       index=snap_idx,
                                       columns=['MV_volt'])

    for idx, row in dist_lv_stations.iterrows():
        volt_over_series = pd.Series(data=row['v_over_bol'], index=snap_idx)
        dist_voltage_over_t['MV_volt'] = dist_voltage_over_t['MV_volt'] + volt_over_series

    dist_voltage_over_t_norm = dist_voltage_over_t / len(dist_lv_stations) * 100


    # Correlation
    threshed_df = dist_s_sum_len_over_t.loc[(dist_s_sum_len_over_t != 0).all(axis=1)]

    corr_df = threshed_df.corr()

    lev0 = corr_df.columns[0]
    lev1 = corr_df.columns[1]
    r = corr_df.iloc[0][1]
    lev0_rel_overl_max = dist_s_sum_len_over_t_norm[lev0].max()
    lev1_rel_overl_max = dist_s_sum_len_over_t_norm[lev1].max()         # Better no normalized here...
    cap0 = dist_cap_df.loc['s_nom_length_MVAkm'][lev0]
    cap1 = dist_cap_df.loc['s_nom_length_MVAkm'][lev1]

    dist_df = dist_df.append({'mv_grid': mv_grid_id,
                    'lev0': lev0,
                    'lev1': lev1,
                    'r': r,
                    'lev0_rel_overl_max': lev0_rel_overl_max,
                    'lev1_rel_overl_max': lev1_rel_overl_max,
                    'cap0': cap0,
                    'cap1': cap1},
                ignore_index=True)


    # Cleaning out levels with low overload for plots
    make_plots = True
    for column in dist_s_sum_len_over_t_norm.columns:
        max_over = dist_s_sum_len_over_t_norm[column].max()
        if max_over < 0.2:
            make_plots = False

    # Plots
    ## Line
    if make_plots == True:
        plt_name = "District overloading"
        file_dir = dist_plot_dir

        fig, ax = plt.subplots(3, sharex=True) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
        fig.set_size_inches(10,6)

        dist_s_sum_len_over_t.plot(
                kind='line',
                legend=True,
                color=[level_colors[lev] for lev in  dist_s_sum_len_over_t.columns],
                linewidth=2,
                ax = ax[0])
        leg = ax[0].legend(loc='upper right',
                  ncol=1, fancybox=True, shadow=True, fontsize=9)
        leg.get_frame().set_alpha(0.5)
        ax[0].set(ylabel='Overloading in MVAkm')

        dist_voltage_over_t_norm.plot(
                kind='line',
                legend=True,
                color='black',
                linewidth=1,
                ax = ax[1])
        leg = ax[1].legend(loc='upper right',
                  ncol=1, fancybox=True, shadow=True, fontsize=9)
        leg.get_frame().set_alpha(0.5)
        ax[1].set(ylabel='Volt. issues in % of buses')

        dist_gen_dispatch_t.plot(
                kind='area',
                legend=True,
                color=[carrier_colors[name] for name in  dist_gen_dispatch_t.columns],
                linewidth=.5,
                alpha=0.7,
                ax = ax[2])
        dist_load_t.plot(
                kind='line',
                legend=True,
                color='grey',
                linewidth=3,
                alpha=0.9,
                ax=ax[2])
        leg = ax[2].legend(loc='upper right',
                  ncol=3, fancybox=True, shadow=True, fontsize=9)
        leg.get_frame().set_alpha(0.5)
        ax[2].set(ylabel='Generation in MW')


        file_name = 'district_overloading_mv_grid_' + str(mv_grid_id)
        fig.savefig(file_dir + file_name + '.png')
        add_figure_to_tex(plt_name, file_dir, file_name)
        plt.close(fig)

        ## Spatial
        plt_name = "Grid District"
        file_dir = dist_plot_dir
        fig, ax1 = plt.subplots(1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
        fig.set_size_inches(6,6)
        xmin, ymin, xmax, ymax = grid_buffer.bounds

        ax1.set_xlim([xmin,xmax])
        ax1.set_ylim([ymin,ymax])

        mv_trafo_df = mv_trafo_df.set_geometry('grid_buffer')
        mv_trafo_df[mv_trafo_df['subst_id'] == mv_grid_id].plot(ax=ax1,
                   alpha = 0.3,
                   color = 'y'
                   )
        mv_trafo_df = mv_trafo_df.set_geometry('geometry')
        mv_trafo_df[mv_trafo_df['subst_id'] == mv_grid_id].plot(ax=ax1,
                   alpha = 1,
                   color = 'r',
                   marker='o',
                   markersize=300,
                   facecolors='none'
                   )

        ax1 = add_plot_lines_to_ax(dist_hv_lines_df, ax1, level_colors, 3)
        ax1 = add_plot_lines_to_ax(dist_mv_lines_df, ax1, level_colors, 1)

        file_name = 'district_' + str(mv_grid_id)
        fig.savefig(file_dir + file_name + '.png')
        add_figure_to_tex(plt_name, file_dir, file_name)
        plt.close(fig)

        ## Scatter
        plt_name = "Correlation of District Overload"
        file_dir = dist_plot_dir

        plot_df = dist_s_sum_len_over_t

        plot_df = plot_df.loc[(plot_df != 0).all(axis=1)]    # Watchout - this is an important detail!!!
        if plot_df.empty:
            continue

        fig, ax1 = plt.subplots(1)
        fig.set_size_inches(12,6)

        lev0 = plot_df.columns[0]
        lev1 = plot_df.columns[1]

        ax1.set_xlim(0, max(plot_df[lev0]))
        ax1.set_ylim(0, max(plot_df[lev1]))

        plt_name = (lev0
                    +' and '
                    + lev1
                    +' Loading Correlation_'
                    + str(mv_grid_id))

        plot_df.plot.scatter(
                x=lev0,
                y=lev1,
                color='grey',
                label=lev1,
                alpha=0.6,
                ax=ax1)

        regr = linear_model.LinearRegression()
        x = plot_df[lev0].values.reshape(len(plot_df), 1)
        y = plot_df[lev1].values.reshape(len(plot_df), 1)
        regr.fit(x, y)
        plt.plot(x, regr.predict(x), color='red', linewidth=1)
        plt.xlabel('Overloaded lines in MVAkm, ' + lev0)
        plt.ylabel('Overloaded lines in MVAkm, ' + lev1)

        file_name = 'loading_corr_' + str(mv_grid_id)
        fig.savefig(file_dir + file_name + '.png')
        add_figure_to_tex(plt_name, file_dir, file_name)
        plt.close(fig)


#TODO Plot mit MV grids anhand Überlastung
only_over_dist_df = dist_df.dropna(axis=0, how='any')


#TODO: ganz neue corr tables machen für MV!!!
## Save
title = 'MV grid districts overview'
file_name = 'mv_grid_districts_info'
file_dir = dist_dir
df = only_over_dist_df.set_index('mv_grid').sort_values(by=['r'],ascending=False)

df.to_csv(file_dir + file_name + '.csv', encoding='utf-8')
render_df = df[['lev0', 'lev1', 'r']].applymap(lambda x: to_str(x))
fig, ax = render_mpl_table(render_df,
                           header_columns=0,
                           col_width=1.5,
                           first_width=1.0)
fig.savefig(file_dir + file_name + '.png')
add_table_to_tex(title, file_dir, file_name)







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