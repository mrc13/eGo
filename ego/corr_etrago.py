"""
This is the main file for Master thesis of maltesc

"""
__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität"\
                            "Flensburg, Centre for Sustainable Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "maltesc"

import pandas as pd

from etrago.appl import etrago
from tools.plots import (make_all_plots,plot_line_loading, plot_stacked_gen,
                                 add_coordinates, curtailment, gen_dist,
                                 storage_distribution, igeoplot)
# For importing geopandas you need to install spatialindex on your system http://github.com/libspatialindex/libspatialindex/wiki/1.-Getting-Started
from tools.utilities import get_scenario_setting, get_time_steps
from tools.io import geolocation_buses, etrago_from_oedb
from tools.results import total_storage_charges
from tools.local_db import local_db_access
from sqlalchemy.orm import sessionmaker
from egoio.tools import db
from etrago.tools.io import results_to_oedb


from egoio.db_tables import model_draft

## Logging
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger('corr_etrago_logger')

fh = logging.FileHandler('/home/student/Git/eGo/ego/corr_etrago.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

## eTraGo args
args = {
  'branch_capacity_factor': 2.0,
  'db': 'oedb',
  'export': False,
  'generator_noise': True,
  'gridversion': None,
  'k_mean_clustering': False,
  'line_grouping': False,
  'load_shedding': False,
  'lpfile': False,
  'method': 'lopf',
  'minimize_loading': False,
  'network_clustering': False,
  'parallelisation': False,
  'pf_post_lopf': False,
  'reproduce_noise': True,
  'results': False,
  'scn_name': 'Status Quo',
  'skip_snapshots': False,
  'solver': 'gurobi',
  'storage_extendable': False}

args['user_name'] = 'maltesc'
## eTraGo iteration parameters
b_factor = [2.0]
snapshots = [(1,2)]
comments = ["2 Std, Status Quo"]
    
try:
    conn = db.connection(section='oedb')
    Session = sessionmaker(bind=conn)
    session = Session()
    
#    session_local = local_db_access()
    
except:
    logger.error('Failed connection to one Database',  exc_info=True)


for b, s, c in zip(b_factor, snapshots, comments):

    args['branch_capacity_factor'] = b
    args['start_snapshot'] = s[0]
    args['end_snapshot'] = s[1]
    args['comment'] = c
    
    logger.info('eTraGo args: ' + str(args))
    try:    
    ## eTraGo Calculation
        eTraGo = etrago(args)
    except:
        logger.error('eTraGo returned Error',  exc_info=True)
    try:
        results_to_oedb(session, eTraGo, args) 
        
    except:
        logger.error('Could not save Results to DB',  exc_info=True)
    try:    
        # make a line loading plot
        plot_line_loading(eTraGo)
        
        # plot stacked sum of nominal power for each generator type and timestep
        plot_stacked_gen(eTraGo, resolution="MW")
        
        # plot to show extendable storages
        storage_distribution(eTraGo)
        
        # plot storage total charges and discharge
        total_storage_charges(eTraGo, plot=True)

    except:
        logger.error('Plots did not work',  exc_info=True)