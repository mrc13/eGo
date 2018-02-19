"""
This is the main file for Master thesis of maltesc

"""
__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität"\
                            "Flensburg, Centre for Sustainable Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "maltesc"

import pandas as pd
#import os


from tools.plots import (make_all_plots,plot_line_loading, plot_stacked_gen,
                                 add_coordinates, curtailment, gen_dist,
                                 storage_distribution, igeoplot)
from tools.utilities import get_scenario_setting, get_time_steps
from tools.io import geolocation_buses, etrago_from_oedb
from tools.results import total_storage_charges

from edisgo.grid.network import Network, Scenario, TimeSeries, Results, ETraGoSpecs

import networkx as nx
import matplotlib.pyplot as plt
        
import geopandas as gpd
from shapely.geometry import Point, LineString
from math import sqrt
import geoalchemy2.shape as shape

from sqlalchemy.ext.automap import automap_base
        
from sqlalchemy.orm import sessionmaker
from egoio.tools import db
from egoio.db_tables import model_draft
#from etrago.tools.io import results_to_oedb
from ego.tools.specs import get_etragospecs_from_db, get_mvgrid_from_bus_id, get_scn_name_from_result_id
import logging

## Logging
LOG_FILENAME = '/home/student/Git/eGo/ego/corr_edisgo.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
logger = logging.getLogger('corr_edisgo')

## Connection and implicit mapping
try:
    conn = db.connection(section='oedb')
    Session = sessionmaker(bind=conn)
    session = Session()
    
    Base = automap_base()        
    Base.prepare(conn, schema='corr_analysis', reflect=True) 

    mv_lines = Base.classes.mv_lines_results
    mv_buses = Base.classes.mv_bus_results
except:
    logger.error('Failed connection to one Database',  exc_info=True)
        
## Explicit Mapping
ormclass_result_bus = model_draft.EgoGridPfHvResultBus

result_ids = [360]
for result_id in result_ids:
    logger.info('eDisGo with result_id: ' + str(result_id))
        
    try:
        scn_name = get_scn_name_from_result_id(session, result_id) # SH Status Quo becomes Status Quo
    except:
        logger.error('Failed to get scn_name',  exc_info=True)
        continue
    
    try:
        query = session.query(
                ormclass_result_bus.bus_id
                ).filter(
                                ormclass_result_bus.result_id == result_id)

        etrago_bus_df = pd.DataFrame(query.all(),
                              columns=[column['name'] for
                                       column in
                                       query.column_descriptions])
        n_buses = len(etrago_bus_df)
        cnt = 1
    except:
        logger.error('Failed retrieve etrago buses',  exc_info=True)
        continue
     
    for idx, row in etrago_bus_df.iterrows(): # 27358 and 25741 are two buses with very different MV grids
        print(str(cnt) + '/' + str(n_buses))
        cnt = cnt + 1
        
        bus_id = row['bus_id'] 
        logger.info('Bus ID: ' + str(bus_id))
        
        try:
            mv_grid_id = get_mvgrid_from_bus_id(session, bus_id)
        except:
            logger.error('mv_grid query failed',  exc_info=True)
            continue
            
        if mv_grid_id == None:
            continue
        
        logger.info('MV grid ID: ' + str(mv_grid_id))
        try: 
            specs = get_etragospecs_from_db(session, bus_id, result_id)
        except:
            logger.error('Specs could not be retrieved',  exc_info=True)
            continue   

        try:
            file_path = '/home/student/Git/eGo/ego/data/grids/SH_model_draft/ding0_grids__' + str(mv_grid_id) + '.pkl'
            scenario = Scenario(etrago_specs=specs,
                        power_flow=(),
                        mv_grid_id=mv_grid_id,
                        scenario_name=scn_name)
    #            scenario = Scenario(
    #                        power_flow='worst-case',
    #                        mv_grid_id=mv_grid_id,
    #                        scenario_name= scn_name)
        
            network = Network.import_from_ding0(file_path,
                                        id=mv_grid_id,
                                        scenario=scenario)
        except:
            logger.error('Scenario or Network could not be initiated',  exc_info=True)
            continue    
        
        try:
            network.analyze()
#           network.reinforce()
        except:
            logger.error('Network could not be analyzed',  exc_info=True)
            continue 
        
#        costs = network.results.grid_expansion_costs
#        print(costs)
        try: 
            buses = network.mv_grid.graph.nodes()# network.mv_grid.graph represents a networkx container, and nodes are extracted
            bus = {'name': [], 'geom': []} # Dictionary for latter DF
            for b in buses:
                bus_name = repr(b) # Knowlededge comes form pypsa_id form edisgo
                bus['name'].append(bus_name)
                bus['geom'].append(b.geom)
                
            grid_volt = network.mv_grid.voltage_nom # Alle mv-grids haben das selbe voltage-level!
            # Was ich mir hier querie ist nur das MV-grid - also alles eine Spanung.
            # In PyPSA stecken hingegen auch die LV-grids drin - daher passt das nicht alles zusammen.
            
            bus_df = pd.DataFrame(bus).set_index('name')
            bus_df = bus_df.join(network.pypsa.generators[['type', 'control']]) # Like this (right) join, only mv generators are included
             
                ## Plotting:
#            crs = {'init': 'epsg:4326'}
#            bus_gdf = gpd.GeoDataFrame(bus_df, crs=crs, geometry=bus_df.geom)
#            bus_gdf.plot()
            
        
            for idx, row in bus_df.iterrows():
                new_mv_bus = mv_buses()
                new_mv_bus.name = idx
                
                new_mv_bus.control = row['control']
                new_mv_bus.type = row['type']
                new_mv_bus.v_nom = grid_volt
                try:
                    new_mv_bus.v = network.results.v_res(nodes=None, level='mv')[idx]
                except:
                    new_mv_bus.v = None
                new_mv_bus.mv_grid = mv_grid_id
                new_mv_bus.result_id = result_id
                
                new_mv_bus.geom = shape.from_shape(row['geom'], srid=4326)
                session.add(new_mv_bus)
                logger.info('Inserting bus ' + str(idx) + ' to db')
                
 
    #       pypsa_df = network.pypsa.buses[['v_nom', 'control']] # Hier stecken auch die LV grids drin! Brauche ich erstmal nicht, denn von den MVs ist das voltage level konstant!
    
            lines = network.mv_grid.graph.lines()
            line = {'name': [],
                    'bus0': [],
                    'bus1': [],
                    #'type': [],
                    #'x': [],
                    #'r': [],
                    's_nom': [], # Hier habe ich meine maximale Leistung
                    #'length': []
                    } 
                  
            for l in lines:
                line['name'].append(repr(l['line']))
                line['bus0'].append(l['adj_nodes'][0])
                line['bus1'].append(l['adj_nodes'][1])
                line['s_nom'].append(
                    sqrt(3) * l['line'].type['I_max_th'] * l['line'].type['U_n'] / 1e3)
                
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
            
#            crs = {'init': 'epsg:4326'}
#            lines_gdf = gpd.GeoDataFrame(lines_df, crs=crs, geometry=lines_df.geom)
#            lines_gdf.plot()
        
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
                
                new_mv_lines.geom = shape.from_shape(row['geom'], srid=4326)
                
                session.add(new_mv_lines)
                
                logger.info('Inserting line ' + str(idx) + ' to db')
                
            session.commit()
            
        except:
            logger.error('eDisGo results could not be written to DB',  exc_info=True)
            session.rollback()
            continue     
    
             
