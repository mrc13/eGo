#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corr Processing
"""
from sqlalchemy.orm import sessionmaker
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from egoio.tools import db
from egoio.db_tables import model_draft
from ego.tools.specs import get_scn_name_from_result_id

# Mapping
ormclass_result_meta = model_draft.EgoGridPfHvResultMeta
ormclass_result_line = model_draft.EgoGridPfHvResultLine
ormclass_result_bus = model_draft.EgoGridPfHvResultBus
ormclass_result_line_t = model_draft.EgoGridPfHvResultLineT

ormclass_result_bus_t = model_draft.__getattribute__('EgoGridPfHvResultBusT')
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
plot_dir = '/home/student/Dropbox/Masterarbeit/Thesis/graphics/pyplots/'

level_colors = pd.DataFrame({ 'lev' : [110., 220., 380.],
                              'color' : ['blue', 'green', 'orange']})


# DB Access
conn = db.connection(section='oedb')
Session = sessionmaker(bind=conn)
session = Session()

# Processing
result_id = 359

scn_name = get_scn_name_from_result_id(session, result_id)

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
        ormclass_result_line.s_nom, ### auf cap faktor achten!!!!!!!!!!!1
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

trans_cap_df = line_df[['s_nom_length_TVAkm', 'v_nom']].groupby('v_nom').sum()

#################################
### Plot
plt_name = "Grid Transmission Capacity"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(8,4)

colors = []
for index, row in trans_cap_df.iterrows():
    colors.append(level_colors.loc[level_colors['lev'] == index]['color'].values[0])


trans_cap_df['s_nom_length_TVAkm'].plot(
        kind='bar',
        title=plt_name,
        color='grey',
        ax = ax1)

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')   
#################################

line_df['s'] = line_df.apply(
        lambda x: [abs(number) for number in x['p0']], axis=1) # This is correct, since in lopf no losses.

s_sum_t = pd.DataFrame(0.0,
                                   index=snap_idx,
                                   columns=trans_cap_df.index)

for v_nom in trans_cap_df.index:
    for index, row in line_df.loc[line_df['v_nom'] == v_nom].iterrows():
        s_series = pd.Series(data=row['s'], index=snap_idx) 
        s_sum_t[v_nom] = s_sum_t[v_nom] + s_series      
    
#################################
### Plot
plt_name = "Voltage Level Total Appearent Power"
fig, ax1 = plt.subplots(1,1) # This says what kind of plot I want (this case a plot with a single subplot, thus just a plot)
fig.set_size_inches(12,4)

colors = []
for column in s_sum_t:
    colors.append(level_colors.loc[level_colors['lev'] == column]['color'].values[0])

s_sum_t.plot(
        kind='line',
        title=plt_name,
        legend=True,
        linewidth=2,
        color=colors,
        ax = ax1)

fig.savefig(plot_dir + plt_name + '.pdf')
fig.savefig(plot_dir + plt_name + '.png')
#################################

print(s_sum_t.corr())

## Now, define overload here and plot overloaded TVAkm!!

## Buses 

query = session.query(
        ormclass_result_bus.bus_id,
        ormclass_result_bus.v_nom,
        ormclass_result_bus.geom
        ).filter(
                ormclass_result_bus.result_id == result_id)

bus_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])


