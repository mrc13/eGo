"""
This is the application file for the tool eGo. The application eGo calculates
the distribution and transmission grids of eTraGo and eDisGo.

.. warning::
    Note, that this Repository is under construction and relies on data provided
    by the OEDB. Currently, only members of the openego project team have access
    to this database.

"""
__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität"\
                            "Flensburg, Centre for Sustainable Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "wolfbunke, maltesc"

import pandas as pd
import os

if not 'READTHEDOCS' in os.environ:
    from etrago.appl import etrago
    from tools.plots import (make_all_plots,plot_line_loading, plot_stacked_gen,
                                     add_coordinates, curtailment, gen_dist,
                                     storage_distribution, igeoplot)
    # For importing geopandas you need to install spatialindex on your system http://github.com/libspatialindex/libspatialindex/wiki/1.-Getting-Started
    from tools.utilities import get_scenario_setting, get_time_steps
    from tools.io import geolocation_buses, etrago_from_oedb
    from tools.results import total_storage_charges
    from sqlalchemy.orm import sessionmaker
    from egoio.tools import db
    from etrago.tools.io import results_to_oedb

# ToDo: Logger should be set up more specific
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # import scenario settings **args of eTraGo
    args = get_scenario_setting(json_file='scenario_setting.json')
    print (args)

    try:
        conn = db.connection(section=args['global']['db'])
        Session = sessionmaker(bind=conn)
        session = Session()
    except OperationalError:
        logger.error('Failed connection to Database',  exc_info=True)

    # start calculations of eTraGo if true
    if args['global']['eTraGo']:
        # start eTraGo calculation
        eTraGo = etrago(args['eTraGo'])

        #ToDo save result to db
        # Does not work wait for eTraGo release 0.5.1
        #results_to_oedb(session, eTraGo, args['eTraGo'], grid='hv')

        # add country code to bus and geometry (shapely)
        # eTraGo.buses = eTraGo.buses.drop(['country_code','geometry'], axis=1)
        #test = geolocation_buses(network = eTraGo, session)

        # other plots based on matplotlib
        make_all_plots(eTraGo)
        # make a line loading plot
        plot_line_loading(eTraGo)

        # plot stacked sum of nominal power for each generator type and timestep
        plot_stacked_gen(eTraGo, resolution="MW")

        # plot to show extendable storages
        storage_distribution(eTraGo)

        # plot storage total charges and discharge
        total_storage_charges(eTraGo, plot=True)

    # get eTraGo results form db
    if args['global']['recover']:
        eTraGo = etrago_from_oedb(session,args)


    # use eTraGo results from ego calculations if true
    # ToDo make function edisgo_direct_specs()



    if args['eDisGo']['direct_specs']:
        # ToDo: add this to utilities.py
        logging.info('Retrieving Specs')
        from ego.tools.specs import get_etragospecs_direct, get_mvgrid_from_bus_id
        from egoio.db_tables import model_draft
        
#        for index, row in eTraGo.buses.iterrows():
#            bus_id = index
#            mv_grid_id = get_mvgrid_from_bus_id(session, bus_id)
#            if mv_grid_id == None:
#                continue
#            print("Bus ID: " + str(bus_id) + ", MV grid ID: " + str(mvgrid_id))
#            
#            
        #bus_id = 25741 ### This seems to be a bus qith decent grid distict
        bus_id = 27358 ### This MV grid should be very diferent
        
        specs = get_etragospecs_direct(session, bus_id, eTraGo, args)  
                
        mv_grid_id = get_mvgrid_from_bus_id(session, bus_id)
        
        from datetime import datetime
        from edisgo.grid.network import Network, Scenario, TimeSeries, Results, ETraGoSpecs
        import networkx as nx
        import matplotlib.pyplot as plt


        file_path = '/home/student/Git/eGo/ego/data/grids/SH_model_draft/ding0_grids__' + str(mv_grid_id) + '.pkl'
        print(file_path)
        #mv_grid = open(file_path)

        scn_name = 'Status Quo' # Not whure if SH is possible
#        scenario = Scenario(etrago_specs=specs,
#                    power_flow=(),
#                    mv_grid_id=mv_grid_id,
#                    scenario_name= scn_name)
        scenario = Scenario(
                    power_flow='worst-case',
                    mv_grid_id=mv_grid_id,
                    scenario_name= scn_name)

        network = Network.import_from_ding0(file_path,
                                    id=mv_grid_id,
                                    scenario=scenario)
        # check SQ MV grid
        network.analyze()
#        network.reinforce()


        #    network.results = Results()
        costs = network.results.grid_expansion_costs
        print(costs)
        
        # Meine Auswertungen:
        
        import geopandas as gpd
        from shapely.geometry import Point
        # The geometry is apparently not in the pypsa container - thus I add it manually
        
        buses = network.mv_grid.graph.nodes()# network.mv_grid.graph represents a networkx container, and nodes are extracted
        bus = {'name': [], 'geom': []} # Dictionary for latter DF
        for b in buses:
            bus_name = repr(b) # Knowlededge comes form pypsa_id form edisgo
            bus['name'].append(bus_name)
            bus['geom'].append(b.geom)
            
        bus_volt = network.mv_grid.voltage_nom # Alle mv-grids haben das selbe voltage-level!
        # Was ich mir hier querie ist nur das MV-grid - also alles eine Spanung.
        # In PyPSA stecken hingegen auch die LV-grids drin - daher passt das nicht alles zusammen.
        
        bus_df = pd.DataFrame(bus).set_index('name')
        
            ## Plotting:
        crs = {'init': 'epsg:4326'}
        bus_gdf = gpd.GeoDataFrame(bus_df, crs=crs, geometry=bus_df.geom)
        bus_gdf.plot()
    
        pypsa_df = network.pypsa.buses[['v_nom', 'control']] # Hier stecken auch die LV grids drin! Brauche ich erstmal nicht, denn von den MVs ist das voltage level konstant!
        
#        bus_df = geoms_df.join(pypsa_df, how='outer')  # There are many geoms missing
        
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
        
        # Hierüber kann man die buses verbinden. Problem: Spannung der Buses ist nur in Pypsa - dort allerdings die Namen anders!
        for idx, row in lines_df.iterrows():
            bus0 = repr(row['bus0'])
            print(bus_df.loc[bus0])
        
        # Query zuende schreiben: Hier bekomme ich meine Ströme und spannungen.
        # Noch genau verstehen: Spannungsregelung durch Blindleistungsbereitstellung. Kann ich davon ausgehen, dass ich keine Spannungsprobleme habe (2. Verteilnetzstudie lesen)
        
        network.results.s_res(components=None)['Line_30360001']
        network.results.v_res(nodes=None, level=None) #level='mv' ausprobieren!!
        
       
        # Alte Auswertugen:

        network.pypsa.buses    
        slack_bus = network.pypsa.buses[network.pypsa.buses['control'] == 'Slack']
        
        eDisGo_results_buses =  network.pypsa.buses[['v_nom', 'control']]
        eDisGo_results_buses.loc['Bus_Generator_1512777']
        
#        
#        
#
#        print(network.mv_grid)
#
#        nx.draw(network.mv_grid.graph)
#        plt.draw()
#        plt.show()
#        
        
    

      
        


    # ToDo make loop for all bus ids
    #      make function which links bus_id (subst_id)
    if args['eDisGo']['specs']:
        

        logging.info('Retrieving Specs')
        # ToDo make it more generic
        # ToDo iteration of grids
        # ToDo move part as function to utilities or specs
        bus_id = 27574 #23971
        result_id = args['global']['result_id']
        
        from ego.tools.specs import get_etragospecs_from_db, get_mvgrid_from_bus_id
        from egoio.db_tables import model_draft
        specs = get_etragospecs_from_db(session, bus_id, result_id)
        
        mv_grid = get_mvgrid_from_bus_id(session, bus_id) # This function can be used to call the correct MV grid

    if args['global']['eDisGo']:
        logging.info('Starting eDisGo')

        # ToDo move part as function to utilities or specs
        from datetime import datetime
        from edisgo.grid.network import Network, Scenario, TimeSeries, Results, ETraGoSpecs
        import networkx as nx
        import matplotlib.pyplot as plt

        # ToDo get ding0 grids over db
        # ToDo implemente iteration
        file_path = '/home/dozeumbuw/ego_dev/src/ding0_grids__1802.pkl'

        #mv_grid = open(file_path)

        mv_grid_id = file_path.split('_')[-1].split('.')[0]
        power_flow = (datetime(2011, 5, 26, 12), datetime(2011, 5, 26, 13)) # Where retrieve from? Database or specs?

        timeindex = pd.date_range(power_flow[0], power_flow[1], freq='H')

        scenario = Scenario(etrago_specs=specs,
                    power_flow=(),
                    mv_grid_id=mv_grid_id,
                    scenario_name= args['global']['scn_name'])

        network = Network.import_from_ding0(file_path,
                                    id=mv_grid_id,
                                    scenario=scenario)
        # check SQ MV grid
        network.analyze()

        network.results.v_res(#nodes=network.mv_grid.graph.nodes(),
                level='mv')
        network.results.s_res()

        # A status quo grid (without new renewable gens) should not need reinforcement
        network.reinforce()


        nx.draw(network.mv_grid.graph)
        plt.draw()
        plt.show()

        #    network.results = Results()
        costs = network.results.grid_expansion_costs
        print(costs)


    # make interactive plot with folium
    #logging.info('Starting interactive plot')
    #igeoplot(network=eTraGo, session=session, args=args)    # ToDo: add eDisGo results

    # calculate power plant dispatch without grid utilization (either in pypsa or in renpassgis)

    # result queries...call functions from utilities

    ## total system costs of transmission grid vs. total system costs of all distribution grids results in overall total
    ## details on total system costs:
    ## existing plants: usage, costs for each technology
    ## newly installed plants (storages, grid measures) with size, location, usage, costs
    ## grid losses: amount and costs

    # possible aggregation of results

    # exports: total system costs, plots, csv export files
