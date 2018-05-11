"""
This is the main file for Master thesis of maltesc

"""
__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität"\
                            "Flensburg, Centre for Sustainable Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "maltesc"


## Local Packages
from ego.tools.specs import (
        get_etragospecs_from_db,
        get_scn_name_from_result_id,
        get_settings_from_result_id)
from ego.tools import corr_io

## Project Packages
from edisgo.grid.network import Network, Scenario
from egoio.tools import db
from egoio.db_tables import model_draft, grid

## General Packages
import os
import pandas as pd
from shapely.geometry import LineString
import geoalchemy2.shape as shape
from sqlalchemy.orm import sessionmaker
import os.path
from random import randint, shuffle
import pickle
from time import localtime, strftime

from math import sqrt, pi

import logging

# Directories
now = strftime("%Y-%m-%d_%H%M", localtime())

## Logging

logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger(__name__)
specs_logger = logging.getLogger('specs')
network_logger = logging.getLogger('network')
pypsa_logger = logging.getLogger('pypsa.pf')

fh = logging.FileHandler('corr_edisgo_' + now + '.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)
specs_logger.addHandler(fh)
network_logger.addHandler(fh)
pypsa_logger.addHandler(fh)


# General inputs
result_id = int(input("Type result ID: "))
ding0_files = 'data/ding0_grids'
add_results = True

random_mv_grids = True
reproduce_choice = 'rand_grids'

## Connection
try:
    conn = db.connection(section='oedb')
    Session = sessionmaker(bind=conn)
    session = Session()
except:
    logger.error('Failed connection to Database',  exc_info=True)


settings = get_settings_from_result_id(session, result_id)
grid_version = settings['gridversion']

if grid_version == 'None':
    grid_version = None

if grid_version == None:
    raise NotImplementedError("To be implemented")

print('Grid Version:')
print(grid_version)

schema = model_draft if grid_version is None else grid

## Mapping
mv_lines = corr_io.corr_mv_lines_results
mv_buses = corr_io.corr_mv_bus_results
ormclass_result_bus = model_draft.EgoGridPfHvResultBus
ormclass_hvmv_subst = schema.EgoDpHvmvSubstation

mv_grids = []
for file in os.listdir(ding0_files):
    if file.endswith('.pkl'):
        mv_grids.append(
                int(file.replace('ding0_grids__', '').replace('.pkl', '')))

if random_mv_grids:
    if reproduce_choice:
        print('random grids are reproduced.')
        with open (reproduce_choice, 'rb') as fp:
            mv_grids = pickle.load(fp)

    else:
        shuffle(mv_grids)
        mv_grids = mv_grids[0:random_mv_grids]
        with open('rand_grids', 'wb') as fp:
            pickle.dump(mv_grids, fp)

#chosen_idx = []
#if random_mv_grids:
#    for p in range(0,random_mv_grids):
#        choice = randint(0,len(mv_grids)-1)
#        while choice in chosen_idx:
#            print(str(choice) + ' has been chosen before - sample again...')
#            choice = randint(0,len(mv_grids)-1)
#        chosen_idx.append(choice)
#    chosen_grids = [mv_grids[i] for i in chosen_idx]
#
#    mv_grids = chosen_grids

print('chosen_grids:')
print(mv_grids)

query = session.query(
        ormclass_hvmv_subst.subst_id,
        ormclass_hvmv_subst.otg_id
        ).filter(
        ormclass_hvmv_subst.subst_id.in_(mv_grids),
        ormclass_hvmv_subst.version == grid_version)

bus_id_df = pd.DataFrame(query.all(),
                      columns=[column['name'] for
                               column in
                               query.column_descriptions])

n_buses = len(bus_id_df)

# Start eDisgo
logger.info('eDisGo with result_id: ' + str(result_id))

if add_results == False:
    try:
        logger.info('Deleting old results')
        session.execute('''
        DELETE FROM model_draft.corr_mv_lines_results
            WHERE result_id = :result_id;

        DELETE FROM model_draft.corr_mv_bus_results
            WHERE result_id = :result_id;

        ''', {'result_id': result_id})
        session.commit()
    except:
        logger.error('Clear old results failed',  exc_info=True)
    skip_grids = []
else:
    query = session.query(
        mv_buses.mv_grid
        ).filter(
                        mv_buses.result_id == result_id).distinct()

    skip_grids = [r[0] for r in query]
    print('Skip grids:')
    print(skip_grids)

try:
    scn_name = get_scn_name_from_result_id(session, result_id) # SH Status Quo becomes Status Quo
except:
    logger.error('Failed to get scn_name',  exc_info=True)

for idx, row in bus_id_df.iterrows():
    bus_id = row['otg_id']
    mv_grid_id = row['subst_id']
    if mv_grid_id in skip_grids:
        logger.info('Bus '+ str(bus_id) + ' with MV grid ' + str(mv_grid_id) + 'skipped, because result is already in db')
        continue

    logger.info('Bus '+ str(bus_id) + ' with MV grid ' + str(mv_grid_id))
    progress = int(idx / n_buses * 100)
    logger.info('Progess: '+ str(progress) + ' %')

    try:
        specs = get_etragospecs_from_db(session, bus_id, result_id)
    except:
        logger.error('Specs could not be retrieved',  exc_info=True)
        continue

    ding0_file_path = ding0_files + '/ding0_grids__' + str(mv_grid_id) + '.pkl'
    if not os.path.isfile(ding0_file_path):
        logger.warning('Not MV grid file for MV grid ID: ' + str(mv_grid_id))
        continue
    try:
        scenario = Scenario(etrago_specs=specs,
                    power_flow=(),
                    mv_grid_id=mv_grid_id,
                    scenario_name=scn_name)
#            scenario = Scenario(
#                        power_flow='worst-case',
#                        mv_grid_id=mv_grid_id,
#                        scenario_name= scn_name)
    except:
        logger.error('Scenario could not be initiated for MV_grid ' + str(mv_grid_id),  exc_info=True)
        continue
    try:
        network = Network.import_from_ding0(ding0_file_path,
                                    id=mv_grid_id,
                                    scenario=scenario)
    except:
        logger.error('Network could not be initiated for MV_grid ' + str(mv_grid_id),  exc_info=True)
        continue

    try:
        if scn_name == 'NEP 2035':
            logger.info('Importing Generators')
            network.import_generators()

        network.analyze()#mode='mv')

    except:
        logger.error('No Generators imported or Network could not be analyzed',  exc_info=True)
        continue

#        costs = network.results.grid_expansion_costs
#        print(costs)

    try:
        buses = network.mv_grid.graph.nodes()# network.mv_grid.graph represents a networkx container, and nodes are extracted
        bus = {'name': [], 'geom': []} # Dictionary of lists
        for b in buses:
            bus_name = repr(b) # Knowlededge comes form pypsa_id form edisgo
            bus['name'].append(bus_name)
            bus['geom'].append(b.geom)

        grid_volt = network.mv_grid.voltage_nom # Alle mv-grids haben das selbe voltage-level!
        # Was ich mir hier querie ist nur das MV-grid - also alles eine Spanung.
        # In PyPSA stecken hingegen auch die LV-grids drin - daher passt das nicht alles zusammen.

        bus_df = pd.DataFrame(bus).set_index('name')
        bus_df = bus_df.join(network.pypsa.generators[['type', 'control']]) # Like this (right) join, only mv generators are included

        for idx, row in bus_df.iterrows():
            new_mv_bus = mv_buses()
            new_mv_bus.name = idx
            pypsa_name = "Bus_" + new_mv_bus.name
            new_mv_bus.control = row['control']
            new_mv_bus.type = row['type']
            new_mv_bus.v_nom = grid_volt
            try:
                new_mv_bus.v = network.results.v_res(nodes=None, level='mv')[idx]
            except:
                logger.warning("No voltage series for bus " + str(idx))
                new_mv_bus.v = None
#            try:
#                new_mv_bus.v_ang = network.pypsa.buses_t.v_ang[pypsa_name]
#            except:
#                new_mv_bus.v_ang = None
#                logger.warning("No voltage angle for bus " + str(idx))
            try:
                new_mv_bus.p = network.pypsa.buses_t.p[pypsa_name]
                new_mv_bus.q = network.pypsa.buses_t.q[pypsa_name]
            except:
                new_mv_bus.p = None
                new_mv_bus.q = None
#                logger.warning("No p and q for bus " + str(idx))
            new_mv_bus.mv_grid = mv_grid_id
            new_mv_bus.result_id = result_id

            new_mv_bus.geom = shape.from_shape(row['geom'], srid=4326)
            session.add(new_mv_bus)
#            logger.info('Inserting bus ' + str(idx) + ' to ram')

#       pypsa_df = network.pypsa.buses[['v_nom', 'control']] # Hier stecken auch die LV grids drin! Brauche ich erstmal nicht, denn von den MVs ist das voltage level konstant!

        lines = network.mv_grid.graph.lines()
        line = {'name': [],
                'bus0': [],
                'bus1': [],
                #'type': [],
                'x': [],
                'r': [],
                's_nom': [], # Hier habe ich meine maximale Leistung
                'length': []
                }

        omega = 2 * pi * 50
        for l in lines:
            line['name'].append(repr(l['line']))
            line['bus0'].append(l['adj_nodes'][0])
            line['bus1'].append(l['adj_nodes'][1])
            line['s_nom'].append(
                sqrt(3) * l['line'].type['I_max_th'] * l['line'].type['U_n'] / 1e3)
            line['x'].append(
                    l['line'].type['L'] * omega / 1e3 * l['line'].length)
            line['r'].append(l['line'].type['R'] * l['line'].length)
            line['length'].append(l['line'].length)

        lines_df = pd.DataFrame(line).set_index('name')

        lines_df['v_nom'] = grid_volt
        lines_df['mv_grid'] = mv_grid_id
        lines_df['result_id'] = result_id

        # Hierüber kann man die buses verbinden. Problem: Spannung der Buses ist nur in Pypsa - dort allerdings die Namen anders!
        line_geom = {'geom': []}
        for idx, row in lines_df.iterrows():
            bus0 = repr(row['bus0'])
            bus1 = repr(row['bus1'])
            geom0 = bus_df.loc[bus0]['geom']
            geom1 = bus_df.loc[bus1]['geom']
            line_geom['geom'].append(LineString([geom0, geom1]))
        lines_df['geom'] = line_geom['geom']

        for idx, row in lines_df.iterrows():
            new_mv_lines = mv_lines()
            new_mv_lines.name = idx
            new_mv_lines.bus0 = repr(row['bus0'])
            new_mv_lines.bus1 = repr(row['bus1'])
            new_mv_lines.s_nom = row['s_nom']
            new_mv_lines.s = network.results.s_res(components=None)[idx]
            new_mv_lines.v_nom = row['v_nom']
            new_mv_lines.mv_grid = row['mv_grid']
            new_mv_lines.result_id = row['result_id']
            new_mv_lines.x = row['x']
            new_mv_lines.r = row['r']
            new_mv_lines.length = row['length']

            new_mv_lines.geom = shape.from_shape(row['geom'], srid=4326)

            session.add(new_mv_lines)

#            logger.info('Inserting line ' + str(idx) + ' to ram')

        logger.info('Committing MV grid '+str(mv_grid_id)+ ' to DB')
        session.commit()

    except:
        logger.error('eDisGo results could not be written to DB',  exc_info=True)
        session.rollback()
        continue

logger.info('eDisGo Calculation complete!')
