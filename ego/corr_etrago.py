"""
This is the eTraGo file for Master thesis of maltesc

"""
__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität"\
                            "Flensburg, Centre for Sustainable Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "maltesc"

### Project Packages
from etrago.appl import etrago
from etrago.tools.io import results_to_oedb
from egoio.tools import db
from ego.tools.specs import get_scn_name_from_result_id

### Sub Packages

### General Packages
from sqlalchemy.orm import sessionmaker
import logging

## Logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger(__name__)
pypsa_log = logging.getLogger('pypsa') # Listen to Pypsa

fh = logging.FileHandler('corr_etrago.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)
pypsa_log.addHandler(fh)

## eTraGo args
args = {
  'db': 'oedb',
  'export': False,
  'generator_noise': True,
  'gridversion': "v0.3.0",
  'k_mean_clustering': False,
  'line_grouping': False, # gibt warnung aus. Soll garnicht so gut sei laut Clara
  'load_shedding': True,
  'lpfile': False,
  'method': 'lopf',
  'minimize_loading': False,
  'network_clustering': False,
  'snapshot_clustering':False, ## evtl. snapshot clustering noch. ausprobieren
  'parallelisation': False, # This is OK in my case cause no storage optimization. Macht alles nacheinander
  'pf_post_lopf': False, # Weitere Möglichkeit sind noch solver options
  'reproduce_noise': False,
  'results': 'results/version_test',
  'scn_name': 'NEP 2035',
  'skip_snapshots': False,
  'solver': 'gurobi',
  'storage_extendable': False}

args['user_name'] = 'malte_scharf'
args['branch_capacity_factor'] = None
args['start_snapshot'] = 1
args['end_snapshot'] = 24
args['comment'] = "grid version test"
args['rand_snapshots'] = False

try:
    conn = db.connection(section='oedb')
    Session = sessionmaker(bind=conn)
    session = Session()

except:
    logger.error('Failed connection to one Database',  exc_info=True)

logger.info('Calculating Country Links')
#cntry_links = corr_io.get_cntry_links(session, args['scn_name'])
args['cntry_links'] = [24430, 24431, 24432, 24433, 24410, 24411, 24414, 24415, 24416, 24417, 24418, 24419, 24420, 24421, 24422, 24423, 24424, 24425, 24426, 24427, 24428, 24429, 24434, 24435, 24436, 24437, 24438, 24439, 24440, 24441, 24442, 24443, 24444, 24445, 24446, 24447, 24448, 24449, 24450, 24451, 24452, 24453, 24454, 24455, 24456, 24457, 24458, 24459, 2288, 2323, 2402, 24460, 24461, 24462, 3555, 24463, 3848, 3923, 24464, 24465, 4085, 4198, 4453, 4521, 24466, 24467, 4783, 24468, 4868, 24469, 24470, 24471, 24472, 5300, 5384, 5552, 5520, 24473, 24474, 6229, 6230, 6290, 6440, 6480, 6730, 6792, 6815, 6896, 6991, 7120, 7382, 7395, 7437, 7445, 7464, 7466, 7467, 7535, 7700, 7763, 7775, 7821, 7886, 7932, 7991, 7992, 8029, 8059, 8691, 8718, 9729, 10882, 10930, 10992, 11087, 11169, 11282, 11436, 11445, 11561, 11662, 11942, 12007, 12362, 12436, 12686, 12697, 13022, 13025, 13071, 13064, 13148, 13270, 13308, 13310, 13337, 13361, 13415, 13719, 13848, 13850, 13913, 13921, 13972, 14077, 14139, 14152, 14176, 15047, 15195, 15340, 15907, 16093, 16135, 16140, 16349, 16577, 16844, 17150, 17460, 17756, 17821, 17906, 17954, 18646, 18651, 19627, 19767, 19995, 20031, 20082, 20320, 21279, 21412, 22354, 22390, 22457, 22994, 23162, 23441, 23484, 23623, 23596, 23650, 23655, 23706, 23700, 23701, 23746, 23752, 23774, 23911, 24147, 24316, 24254, 24295]


logger.info('eTraGo args: ' + str(args))

try:
## eTraGo Calculation
    logger.info('Start eTraGo calculations')
    eTraGo = etrago(args)
except:
    logger.error('eTraGo returned Error',  exc_info=True)

ans = str(input("Want to save results (y/n)? "))
if ans == 'y':
    try:
        logger.info('Results to DB')
        results_to_oedb(session, eTraGo, args)

    except:
        logger.error('Could not save Results to DB',  exc_info=True)


ans = str(input("Want to update s_nom in results (y/n)? "))
if ans == 'y':
    result_id = str(input("Please type the result_id: "))
    scn_name = get_scn_name_from_result_id(session, result_id)
    session.execute('''
    UPDATE model_draft.ego_grid_pf_hv_result_line as lr
        SET s_nom = (SELECT s_nom
                         FROM model_draft.ego_grid_pf_hv_line as l
                         WHERE   scn_name = :scn_name AND
                                 l.line_id = lr.line_id)
        WHERE result_id = :result_id;
    ''', {'result_id': result_id, 'scn_name': scn_name})
    session.commit()




