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
from dateutil import parser

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
cont_fct_hv = 0.85
cont_fct_mv = 1 # This is ok, simce load will not overload
result_id = 384
data_set = '2018-03-17'
result_dir = 'corr_results/' + str(result_id) + '/data_proc/' + data_set + '/'

# Directories
now = strftime("%Y-%m-%d_%H%M", localtime())

analysis_dir = 'corr_results/' + str(result_id) + '/analysis/' + now + '/'
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

plot_dir = analysis_dir + 'plots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

readme = open(analysis_dir + 'readme','w')
readme.write(r'''
I have calculated 200 hours for whole Germany. 404 Ding0 grids.
I have chosen 1.0 for MV overload and 0.85 for HV overload

''')
readme.close()
#%% Basic functions and Dicts

#hv_levels = pd.unique(line_df['v_nom']).tolist()
#mv_levels = pd.unique(mv_line_df['v_nom']).tolist()
#all_levels = mv_levels + hv_levels

level_colors = {'LV': 'grey',
                'MV': 'black',
                'HV': 'blue',
                'EHV220': 'green',
                'EHV380': 'orange',
                'unknown': 'grey'}

all_levels = ['MV', 'HV', 'EHV220', 'EHV380']

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
    if v_lev == 'MV':
        return 20. # This is not always true
    elif v_lev == 'HV':
        return 110.
    elif v_lev == 'EHV220':
        return 220.
    elif v_lev == 'EHV380':
        return 380.
    else: return None

def get_hour_of_year (v_d):
    return ((v_d.timetuple().tm_yday-1) * 24 + v_d.hour + 1)

def add_figure_to_tex (v_file_name, v_title):
    tex_file = open(plot_dir + v_file_name + '.txt','w')
    tex_file.write(r'''
\begin{figure}[htbp]
	\centering
	\includegraphics[width=\textwidth]{graphics/pyplots/%s/plots/%s.png}
	\caption{%s}
	\label{img:%s}
\end{figure}
    ''' % (now, v_file_name, v_title, v_file_name))
    tex_file.close()

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
    snap_idx = pd.Series([parser.parse(idx) for idx in snap_idx])
except:
    logger.warning('No Subst. imported')

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

## Overload
line_df['s_over'] = line_df.apply(
        lambda x: [n - cont_fct_hv for n in x['s_rel']], axis=1)
line_df['s_over_bol'] = line_df.apply(
        lambda x: [True if n > 0 else False for n in x['s_over']], axis=1)
line_df['s_over_abs'] = line_df.apply(
        lambda x: [n * x['s_nom_length_GVAkm'] \
                   if n else 0 for n in x['s_over_bol']], axis=1)


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
# Overload
mv_line_df['s_over'] = mv_line_df.apply(
        lambda x: [n - cont_fct_hv for n in x['s_rel']], axis=1)
mv_line_df['s_over_bol'] = mv_line_df.apply(
        lambda x: [True if n > 0 else False for n in x['s_over']], axis=1)
mv_line_df['s_over_abs'] = mv_line_df.apply(
        lambda x: [n * x['s_nom_length_GVAkm'] \
                   if n else 0 for n in x['s_over_bol']], axis=1)

# Buses
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



#%% Basic grid information
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
try:
    mv_gens_df = gens_df.merge(all_hvmv_subst_df,
                  how='inner',
                  left_on='bus',
                  right_on='bus_id')

    mv_gens_df = mv_gens_df[mv_gens_df['name'] != 'load shedding']
    mv_gens_df = mv_gens_df.dropna() # I think this is also load shedding
    ## Find out how much load shedding is done...

    mv_grids_df['Inst. gen. capacity'] = mv_gens_df.groupby(['subst_id'])['p_nom'].sum()
    del mv_gens_df
except:
    logger.warning('Inst. gen. capacity could not be imported')

mv_grids_df.to_csv(analysis_dir + 'mv_grids_df.csv', encoding='utf-8')

# MV total
columns = ['MV']
index =   ['Tot. no. of grids',
           'No. of calc. grids',
           'Perc. of calc. grids',
           'Tot. calc. length in km',
           'Avg. len. per grid in km',
           'Estim. tot. len. in km',
           'Tot. calc. cap in GVAkm',
           'Avg. transm. cap. in GVAkm',
           'Estim. tot. trans cap. in GVAkm']
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

mv_grid_info_df.loc['Estim. tot. len. in km']['MV'] = round( # Länge evtl. besser direkt aus Ding0 berechnen
        mv_grid_info_df.loc['Avg. len. per grid in km']['MV'] *\
        mv_grid_info_df.loc['Tot. no. of grids']['MV'], 2)

mv_grid_info_df.loc['Tot. calc. cap in GVAkm']['MV'] = round(
        mv_grids_df['Transm. cap. in GVAkm'].sum(), 2)

mv_grid_info_df.loc['Avg. transm. cap. in GVAkm']['MV'] = round(
        mv_grids_df['Transm. cap. in GVAkm'].mean(), 2)

mv_grid_info_df.loc['Estim. tot. trans cap. in GVAkm']['MV'] = round(
        mv_grid_info_df.loc['Avg. transm. cap. in GVAkm']['MV'] *\
        mv_grid_info_df.loc['Tot. no. of grids']['MV'], 2)

mv_grid_info_df.to_csv(analysis_dir + 'mv_grid_info_df.csv', encoding='utf-8')

mv_rel_calc = mv_grid_info_df.loc['No. of calc. grids']['MV'] / \
        mv_grid_info_df.loc['Tot. no. of grids']['MV']

# HV Total
columns = ['HV', 'EHV220', 'EHV380']
index =   ['Total. len. in km',
           'Total. cap. in TVAkm']
grid_info_df = pd.DataFrame(index=index, columns=columns)

for col in columns:
    grid_info_df.loc['Total. len. in km'][col] = round(
            line_df.loc[line_df['v_nom'] == get_volt_from_lev(col)]['length'].sum(), 2)

for col in columns:
    grid_info_df.loc['Total. cap. in TVAkm'][col] = round(
            line_df.loc[line_df['v_nom'] == get_volt_from_lev(col)]['s_nom_length_TVAkm'].sum(), 2)

grid_info_df.to_csv(analysis_dir + 'grid_info_df.csv', encoding='utf-8')

# HV/MV Comparison
columns = ['MV', 'HV', 'EHV220', 'EHV380']
index =   ['Total. len. in km',
           'Total. cap. in TVAkm']
hvmv_comparison_df = pd.DataFrame(index=index, columns=columns)

hvmv_comparison_df.loc['Total. len. in km']['MV'] = mv_grid_info_df.loc['Estim. tot. len. in km']['MV']
for col in grid_info_df.columns:
    hvmv_comparison_df.loc['Total. len. in km'][col] = grid_info_df.loc['Total. len. in km'][col]

hvmv_comparison_df.loc['Total. cap. in TVAkm']['MV'] = mv_grid_info_df.loc['Estim. tot. trans cap. in GVAkm']['MV']/1e3
for col in grid_info_df.columns:
    hvmv_comparison_df.loc['Total. cap. in TVAkm'][col] = grid_info_df.loc['Total. cap. in TVAkm'][col]

del columns
del index

hvmv_comparison_df.to_csv(analysis_dir + 'hvmv_comparison_df.csv', encoding='utf-8')

#%% Electrical Overview

plt_name = "Electrical Overview"
fig, ax = plt.subplots(1, 4) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,5)
vals = []
colors = []
levs = []
for idx, lev in enumerate(all_levels):

    levs.append(lev)
    if lev == 'MV':
        df = mv_line_df
    else:
        df = line_df
    vals.append(df.loc[df['lev'] == lev]['s_nom'])
    colors.append(level_colors[lev])

    #bins = range(0, 2000, 200)
    pd.Series(vals[idx]).hist(
                  color=colors[idx],
                  ax=ax[idx],
                  bins=10, alpha = 0.7)

    plt.xlabel("S_nom")
    ax[idx].legend((lev,))

file_name = 's_nom_hist'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
add_figure_to_tex(file_name, plt_name)


#%% Corr Germany Calcs

# Total grid overload per voltage level in GVAkm and km  and relative
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


#%% Corr Germany Plots
# Corr
corr_s_sum_len_over_t = s_sum_len_over_t.corr(method='pearson')
corr_s_sum_len_over_t.to_csv(analysis_dir + 'corr_s_sum_len_over_t.csv', encoding='utf-8')

corr_len_over_t = len_over_t.corr(method='pearson')
corr_len_over_t.to_csv(analysis_dir + 'corr_len_over_t.csv', encoding='utf-8')
# Werte müssen noch gerundet werden!!!! Rundung überall beachten!!
## Plot
##% Line Plots
##%% Capacity
plt_name = "Total Line Overloading Germany"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,4)

frm = s_sum_len_over_t.plot(
        kind='line',
        title=plt_name,
        legend=True,
        color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
        linewidth=2,
        ax = ax1)
plt.ylabel('Total overloading in GVAkm')
file_name = 'overl_per_level_in_GVAkm'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
add_figure_to_tex(file_name, plt_name)

plt_name = "Relative total Line Overloading Germany"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,4)

frm = s_sum_len_over_t_norm.plot(
        kind='line',
        title=plt_name,
        legend=True,
        color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
        linewidth=2,
        ax = ax1)
plt.ylabel('Relative total overloading in percent')
file_name = 'rel_overl_per_level_in_perc'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
add_figure_to_tex(file_name, plt_name)

##%% Length
plt_name = "Length of overloaded Lines Germany"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,4)

frm = len_over_t.plot(
        kind='line',
        title=plt_name,
        legend=True,
        color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
        linewidth=2,
        ax = ax1)
plt.ylabel('length of overloaded lines in km')
file_name = 'overl_per_level_in_km'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
add_figure_to_tex(file_name, plt_name)

plt_name = "Length of overloaded Lines Germany (normalized)"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,4)

frm = len_over_t_norm.plot(
        kind='line',
        title=plt_name,
        legend=True,
        color=[level_colors[lev] for lev in  s_sum_len_over_t.columns],
        linewidth=2,
        ax = ax1)
plt.ylabel('Total overloading in % of grid length')
file_name = 'overl_per_level_in_perc'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
add_figure_to_tex(file_name, plt_name)

##% Scatter Plots
plt_name = "Correlation of Overloaded grid Length"
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
    fig.savefig(plot_dir + file_name + '.pdf')
    fig.savefig(plot_dir + file_name + '.png')
    add_figure_to_tex(file_name, plt_name)

##% Histograms
plt_name = "Overloaded Length Histogram"
fig, ax = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(8,4)

vals = []
colors = []
levs = []
for col in len_over_t.columns:
   levs.append(col)
   vals.append(len_over_t[col])
   colors.append(level_colors[col])

ax = plt.hist(x=vals, color=colors, bins=10, normed=True, alpha=0.7)
plt.xlabel("Overloaded Length")
plt.legend(levs)

file_name = 'overloaded_length_hist'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
add_figure_to_tex(file_name, plt_name)

# All s_rel over Comparison per level
s_rel_over_per_lev = { key : [] for key in all_levels }
for df in [line_df, mv_line_df]:
    for index, row in df.iterrows():
        lev = get_lev_from_volt(row['v_nom'])
        s_rel_series = row['s_over']
        for s_rel in s_rel_series:
            if s_rel > 0:
                s_rel_over_per_lev[lev].append(s_rel)


plt_name = "Relative overload Hist"
fig, ax = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(8,4)

vals = []
colors = []
levs = []
for key, value in s_rel_over_per_lev.items():
   levs.append(key)
   vals.append(value)
   colors.append(level_colors[key])

bins = [0, .1, .2, .3, .4, .5, .6, .7, 0.8, 0.9, 1., 1.1, 1.2, 1.3]
ax = plt.hist(x=vals, color=colors, bins=bins, normed=True, alpha = 0.7)
plt.xlabel("Relative overload")
plt.legend(levs)

file_name = 'rel_overload_hist'
fig.savefig(plot_dir + file_name + '.pdf')
fig.savefig(plot_dir + file_name + '.png')
add_figure_to_tex(file_name, plt_name)







#TODO Hier noch Zeitreihen für Load
#TODO Hier noch Zeitreihen für Generatoren (auch Deutschlandweit)

#TODO Gucken welche Leitungen am häufigsten überlastet werden.
#TODO Da auf Deutschlandebene quasi keine Korrelation, die Korrelation mit der Windeinspeisung suchen und gucken ob höher, als z.b. mit Kohle




#%% Corr District Calcs








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