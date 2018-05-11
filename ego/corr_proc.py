#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corr Processing
"""
## General Packages
from sqlalchemy.orm import sessionmaker
import pandas as pd
from geoalchemy2.shape import to_shape
#import numpy as np
import os
from time import localtime, strftime

## Project Packages
from egoio.tools import db
from egoio.db_tables import model_draft, grid

## Local Packages
from ego.tools.specs import (
        get_scn_name_from_result_id,
        get_settings_from_result_id)
from ego.tools import corr_io

# Directories
now = strftime("%Y-%m-%d_%H%M", localtime())

log_dir = 'proc_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

## Logging
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger(__name__)

fh = logging.FileHandler(log_dir + '/corr_proc_' + now + '.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

# Mapping
## model_draft
ormclass_result_meta = model_draft.EgoGridPfHvResultMeta

ormclass_result_transformer = model_draft.EgoGridPfHvResultTransformer
ormclass_result_transformer_t = model_draft.EgoGridPfHvResultTransformerT
ormclass_result_gen = model_draft.EgoGridPfHvResultGenerator
ormclass_result_gen_t = model_draft.EgoGridPfHvResultGeneratorT
ormclass_result_line = model_draft.EgoGridPfHvResultLine
ormclass_result_bus = model_draft.EgoGridPfHvResultBus
ormclass_result_line_t = model_draft.EgoGridPfHvResultLineT
ormclass_result_bus_t = model_draft.EgoGridPfHvResultBusT
ormclass_result_load = model_draft.EgoGridPfHvResultLoad
ormclass_result_load_t = model_draft.EgoGridPfHvResultLoadT
ormclass_result_storage = model_draft.EgoGridPfHvResultStorage
ormclass_result_storage_t = model_draft.EgoGridPfHvResultStorageT

ormclass_source = model_draft.EgoGridPfHvSource

## grid
#ormclass_griddistricts = model_draft.EgoGridMvGriddistrict
ormclass_griddistricts = grid.EgoDpMvGriddistrict
#ormclass_hvmv_subst = model_draft.EgoGridHvmvSubstation
ormclass_hvmv_subst = grid.EgoDpHvmvSubstation

## corr
mv_lines = corr_io.corr_mv_lines_results
mv_buses = corr_io.corr_mv_bus_results

# General inputs
result_id = int(input("Type result ID: "))

# Result Folder
now = strftime("%Y-%m-%d", localtime())
result_dir = 'corr_results/' + str(result_id) + '/data_proc/' + now + '/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# DB Access
conn = db.connection(section='oedb')
Session = sessionmaker(bind=conn)
session = Session()

#%% Metadata
logger.info('Metadata')
scn_name = get_scn_name_from_result_id(session, result_id)

settings = get_settings_from_result_id(session, result_id)
grid_version = settings['gridversion']

if grid_version == 'None':
    raise NotImplementedError("To be implemented")

logger.info('Grid version: ' + str(grid_version))

snap_idx = session.query(
            ormclass_result_meta.snapshots
            ).filter(
            ormclass_result_meta.result_id == result_id
            ).scalar(
                    )

pd.Series(snap_idx).to_csv(result_dir + 'snap_idx', encoding='utf-8')

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

line_df['geom'] = line_df.apply(
        lambda x: to_shape(x['geom']), axis=1)

line_df['topo'] = line_df.apply(
        lambda x: to_shape(x['topo']), axis=1)

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

mv_line_df['s_nom_length_TVAkm'] = mv_line_df.apply(
        lambda x: (x['length'] * float(x['s_nom']))*1e-6, axis=1)

mv_line_df['s'] = mv_line_df.apply( ## now s is in MVA
        lambda x: [s/1000 for s in x['s']], axis=1)

mv_line_df['s_rel'] = mv_line_df.apply(
        lambda x: [s/float(x['s_nom']) for s in x['s']], axis=1)

line_df.to_csv(result_dir + 'line_df.csv', encoding='utf-8')
mv_line_df.to_csv(result_dir + 'mv_line_df.csv', encoding='utf-8')

#%% Load
logger.info('Load')
query = session.query(
        ormclass_result_load.load_id,
        ormclass_result_load.bus,
        ormclass_result_load.e_annual,
        ormclass_result_load_t.p
        ).join(
                ormclass_result_load_t,
                ormclass_result_load_t.load_id == ormclass_result_load.load_id
                ).filter(
                ormclass_result_load.result_id == result_id,
                ormclass_result_load_t.result_id == result_id)

load_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])
load_df = load_df.set_index('load_id')
load_df.to_csv(result_dir + 'load_df.csv', encoding='utf-8')

#%% All hvmv Substations
logger.info('All hvmv Substations')
query = session.query(
        ormclass_result_bus.bus_id,
        ormclass_hvmv_subst.subst_id
        ).join(
                ormclass_hvmv_subst,
                ormclass_hvmv_subst.otg_id == ormclass_result_bus.bus_id
                ).filter(
                        ormclass_result_bus.result_id == result_id,
                        ormclass_hvmv_subst.version == grid_version)

all_hvmv_subst_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

all_hvmv_subst_df.to_csv(result_dir + 'all_hvmv_subst_df.csv', encoding='utf-8')

#%% Buses
logger.info('Buses')

query = session.query(
        ormclass_result_bus.bus_id,
        ormclass_result_bus.v_nom,
        ormclass_result_bus.geom,
        ormclass_result_bus_t.p,
        ormclass_result_bus_t.v_ang
        ).join(
                ormclass_result_bus_t,
                ormclass_result_bus_t.bus_id == ormclass_result_bus.bus_id
                ).filter(
                ormclass_result_bus.result_id == result_id,
                ormclass_result_bus_t.result_id == result_id)

bus_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

bus_df = bus_df.set_index('bus_id')

bus_df = bus_df.merge(
        all_hvmv_subst_df.set_index('bus_id'),
        left_index=True,
        right_index=True,
        how='left')

bus_df['geom'] = bus_df.apply(
        lambda x: to_shape(x['geom']), axis=1)

bus_df['p_mean'] = bus_df.apply( # Mean feed in
        lambda x: pd.Series(data= x['p']).mean(), axis=1)

# MV Buses
query = session.query(
        mv_buses.name,
        mv_buses.v_nom,
        mv_buses.geom,
        mv_buses.p,
        mv_buses.v
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

bus_df.to_csv(result_dir + 'bus_df.csv', encoding='utf-8')
mv_bus_df.to_csv(result_dir + 'mv_bus_df.csv', encoding='utf-8')

#%% Generators
logger.info('Generators')
query = session.query(
        ormclass_result_gen.generator_id, # This ID is an aggregate ID (single generators aggregated)
        ormclass_result_gen.bus,
        ormclass_result_gen.p_nom,
        ormclass_source.name,
        ormclass_result_gen_t.p
        ).join(ormclass_result_gen_t,
               ormclass_result_gen_t.generator_id == ormclass_result_gen.generator_id
                ).join(
                ormclass_source,
                ormclass_source.source_id == ormclass_result_gen.source
                ).filter(
                ormclass_result_gen.result_id == result_id,
                ormclass_result_gen_t.result_id == result_id)

gens_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])
gens_df = gens_df.set_index('generator_id')

gens_df.to_csv(result_dir + 'gens_df.csv', encoding='utf-8')

#%% Storage
logger.info('Storage')
query = session.query(
        ormclass_result_storage.storage_id,
        ormclass_result_storage.bus,
        ormclass_result_storage.p_nom,
        ormclass_source.name,
        ormclass_result_storage_t.p,
        ormclass_result_storage_t.state_of_charge
        ).join(ormclass_result_storage_t,
               ormclass_result_storage_t.storage_id == ormclass_result_storage.storage_id
                ).join(
                ormclass_source,
                ormclass_source.source_id == ormclass_result_storage.source
                ).filter(
                ormclass_result_storage.result_id == result_id,
                ormclass_result_storage_t.result_id == result_id)

storage_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])
storage_df = storage_df.set_index('storage_id')

storage_df.to_csv(result_dir + 'storage_df.csv', encoding='utf-8')

#%% Transformers
logger.info('Transformers')
# HV Transformers
query = session.query(
        ormclass_result_transformer.trafo_id,
        ormclass_result_transformer.bus0,
        ormclass_result_transformer.bus1,
        ormclass_result_transformer.s_nom,
        ormclass_result_transformer.geom,
        ormclass_result_transformer_t.p0
        ).join(
                ormclass_result_transformer_t,
                ormclass_result_transformer_t.trafo_id == ormclass_result_transformer.trafo_id
                ).filter(
                ormclass_result_transformer.result_id == result_id,
                ormclass_result_transformer_t.result_id == result_id)

trafo_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])
trafo_df = trafo_df.set_index('trafo_id')

trafo_df['s'] = trafo_df.apply(
        lambda x: [abs(number) for number in x['p0']], axis=1) # This is correct, since in lopf no losses.

trafo_df['s_rel'] = trafo_df.apply(
        lambda x: [s/float(x['s_nom']) for s in x['s']], axis=1)

trafo_df['geom'] = trafo_df.apply(
        lambda x: to_shape(x['geom']), axis=1)

trafo_df['point_geom'] = trafo_df.apply(
        lambda x: x['geom'].representative_point(), axis=1)

trafo_df['v_nom0'] = trafo_df.apply(
        lambda x: bus_df.loc[x['bus0']]['v_nom'], axis=1)
trafo_df['v_nom1'] = trafo_df.apply(
        lambda x: bus_df.loc[x['bus1']]['v_nom'], axis=1)

## Define spatial Buffer
trafo_df['grid_buffer'] = trafo_df.apply(
        lambda x: x['point_geom'].buffer(0.1), axis=1) ## Buffergröße noch anpassen

# MV Transformers
query = session.query(
        ormclass_hvmv_subst.point,
        ormclass_hvmv_subst.subst_id,
        ormclass_hvmv_subst.otg_id
        ).filter(ormclass_hvmv_subst.version == grid_version)

mv_trafo_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

mv_trafo_df['point'] = mv_trafo_df.apply(
        lambda x: to_shape(x['point']), axis=1)

mv_trafo_df['bus0'] = mv_trafo_df.apply(
        lambda x: 'MVStation_' + str(x['subst_id']), axis=1)
mv_trafo_df['bus1'] = mv_trafo_df.apply(
        lambda x: x['otg_id'], axis=1)

mv_trafo_df = mv_trafo_df.merge(mv_bus_df, # Only the Trafos that can actualy be found in the MV buses....
                                  left_on='bus0',
                                  right_index=True,
                                  how='inner')

## Merge with griddistrict geoms
query = session.query(
        ormclass_griddistricts.subst_id,
        ormclass_griddistricts.geom
        ).filter(ormclass_griddistricts.version == grid_version)
mv_griddistricts_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

mv_griddistricts_df = mv_griddistricts_df.set_index('subst_id')
mv_griddistricts_df['geom'] = mv_griddistricts_df.apply(
        lambda x: to_shape(x['geom']), axis=1)
mv_griddistricts_df = mv_griddistricts_df.rename(
        columns={'geom': 'grid_buffer'})

mv_trafo_df = mv_trafo_df.merge(mv_griddistricts_df,
                                  left_on='subst_id',
                                  right_index=True,
                                  how='left')
del mv_griddistricts_df

trafo_df.to_csv(result_dir + 'trafo_df.csv', encoding='utf-8')
mv_trafo_df.to_csv(result_dir + 'mv_trafo_df.csv', encoding='utf-8')


