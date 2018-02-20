#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create and Update vis tables in corr_analysis

@author: maltesc
"""

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from egoio.tools import db
from egoio.db_tables import model_draft

## Logging
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger('vis_tables_logger')

fh = logging.FileHandler('/home/student/Git/eGo/ego/vis_tables.log', mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

## Connection and implicit mapping
try:
    conn = db.connection(section='oedb')
    Session = sessionmaker(bind=conn)
    session = Session()
    
except:
    logger.error('Failed connection to Database',  exc_info=True)
 
try:  
    print("Decide on the Result to view")
    result_id = int(input("Type result ID: "))  
    
    session.execute('''
    SELECT model_draft.corr_vis_result_id(:result_id);
    ''', {'result_id': result_id})
    session.commit()
except:
    session.rollback()
    logger.error('Failed to visualize results',  exc_info=True)
    
try:       
    run = True
    while run:
        fct = 1.0 # Multiplies s (resulting) with this factor in order to eliminate branch capacity factor   
        snapshot = int(input("Type snapshot: "))  
        
        session.execute('''
        SELECT model_draft.corr_update_srel(:snapshot, :fct);
        ''', {'snapshot': snapshot, 'fct': fct})
        session.commit()
except:
    session.rollback()
    logger.error('Something happened',  exc_info=True)




