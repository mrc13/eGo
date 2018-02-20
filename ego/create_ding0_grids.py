#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create and Update vis tables in corr_analysis

@author: maltesc
"""

import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from sqlalchemy.orm import sessionmaker

from ding0.core import NetworkDing0
from ding0.tools import config as cfg_ding0, results
from egoio.tools import db
from egoio.db_tables import model_draft
import json
from datetime import datetime

plt.close('all')
cfg_ding0.load_config('config_db_tables.cfg')
cfg_ding0.load_config('config_calc.cfg')
cfg_ding0.load_config('config_files.cfg')
cfg_ding0.load_config('config_misc.cfg')

BASEPATH = os.path.join(os.path.expanduser('~'), '.ding0')


def create_results_dirs(base_path):
    """Create base path dir and subdirectories
    Parameters
    ----------
    base_path : str
        The base path has subdirectories for raw and processed results
    """


    if not os.path.exists(base_path):
        print("Creating directory {} for results data.".format(base_path))
        os.mkdir(base_path)
    if not os.path.exists(os.path.join(base_path, 'results')):
        os.mkdir(os.path.join(base_path, 'results'))
    if not os.path.exists(os.path.join(base_path, 'plots')):
        os.mkdir(os.path.join(base_path, 'plots'))
    if not os.path.exists(os.path.join(base_path, 'info')):
        os.mkdir(os.path.join(base_path, 'info'))
    if not os.path.exists(os.path.join(base_path, 'log')):
        os.mkdir(os.path.join(base_path, 'log'))


def run_multiple_grid_districts(mv_grid_districts, run_id, failsafe=False,
                                base_path=None):
    """
    Perform ding0 run on given grid districts
    Parameters
    ----------
    mv_grid_districs : list
        Integers describing grid districts
    run_id: str
        Identifier for a run of Ding0. For example it is used to create a
        subdirectory of os.path.join(`base_path`, 'results')
    failsafe : bool
        Setting to True enables failsafe mode where corrupt grid districts
        (mostly due to data issues) are reported and skipped. Report is to be
         found in the log dir under :code:`~/.ding0` . Default is False.
    base_path : str
        Base path for ding0 data (input, results and logs).
        Default is `None` which sets it to :code:`~/.ding0` (may deviate on
        windows systems).
        Specify your own but keep in mind that it a required a particular
        structure of subdirectories.
    Returns
    -------
    msg : str
        Traceback of error computing corrupt MV grid district
        .. TODO: this is only true if try-except environment is moved into this
            fundion and traceback return is implemented
    Notes
    -----
    Consider that a large amount of MV grid districts may take hours or up to
    days to compute. A computational run for a single grid district may consume
    around 30 secs.
    """
    start = time.time()

    # define base path
    if base_path is None:
        base_path = BASEPATH

   
    conn = db.connection(section='oedb')
    Session = sessionmaker(bind=conn)
    session = Session()
        
    corrupt_grid_districts = pd.DataFrame(columns=['id', 'message'])
        
    for mvgd in mv_grid_districts:
        # instantiate ding0  network object
        nd = NetworkDing0(name='network', run_id=run_id)
        mvgd=int(mvgd)

        if not os.path.exists(os.path.join(base_path, run_id)):
            os.mkdir(os.path.join(base_path, run_id))

        if not failsafe:
            # run DING0 on selected MV Grid District
            msg = nd.run_ding0(session=session,
                         mv_grid_districts_no=[mvgd])

            # save results
            results.save_nd_to_pickle(nd, os.path.join(base_path, run_id))
        else:
            # try to perform ding0 run on grid district
            try:
                msg = nd.run_ding0(session=session,
                         mv_grid_districts_no=[mvgd])
                
                # if not successful, put grid district to report
                if msg:
                    corrupt_grid_districts = corrupt_grid_districts.append(
                        pd.Series({'id': mvgd,
                                   'message': msg[0]}),
                        ignore_index=True)
                # if successful, save results
                else:
                    results.save_nd_to_pickle(nd, os.path.join(base_path,
                                                               run_id))
            except Exception as e:
                print('Corrupt')
                corrupt_grid_districts = corrupt_grid_districts.append(
                    pd.Series({'id': mvgd,
                               'message': e}),
                    ignore_index=True)

                continue

        # Merge metadata of multiple runs
        if 'metadata' not in locals():
            metadata = nd.metadata

        else:
            if isinstance(mvgd, list):
                metadata['mv_grid_districts'].extend(mvgd)
            else:
                metadata['mv_grid_districts'].append(mvgd)


    # Save metadata to disk
    with open(os.path.join(base_path, run_id, 'Ding0_{}.meta'.format(run_id)),
              'w') as f:
        json.dump(metadata, f)

    # report on unsuccessful runs
    corrupt_grid_districts.to_csv(
        os.path.join(
            base_path,
            run_id,
            'corrupt_mv_grid_districts.txt'),
        index=False,
        float_format='%.0f')

    print('Elapsed time for', str(len(mv_grid_districts)),
          'MV grid districts (seconds): {}'.format(time.time() - start))


    return msg


if __name__ == '__main__':
    base_path = BASEPATH

    # set run_id to current timestamp
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")

    # create directories for local results data
    create_results_dirs(base_path)

    conn = db.connection(section='oedb')
    Session = sessionmaker(bind=conn)
    session = Session()
    
    ormclass_hvmv_subst = model_draft.__getattribute__('EgoGridHvmvSubstation')
    
    query = session.query(ormclass_hvmv_subst.otg_id,
                          ormclass_hvmv_subst.subst_id)
    
    hvmv_trans_df = pd.DataFrame(query.all(),
                          columns=[column['name'] for
                                   column in
                                   query.column_descriptions])  
    
    # define grid district by its id (int)
    mv_grid_districts = list(hvmv_trans_df['subst_id'])

    # run grid districts
    run_multiple_grid_districts(mv_grid_districts,
                                run_id,
                                failsafe=True,
base_path=base_path)