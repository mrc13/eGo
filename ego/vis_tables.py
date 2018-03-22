#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create and Update vis tables in corr_analysis

@author: maltesc
"""

from sqlalchemy.orm import sessionmaker
from egoio.tools import db



## Logging
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

logger = logging.getLogger('vis_tables_logger')

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

        snapshot = int(input("Type snapshot: "))

        session.execute('''
        SELECT model_draft.corr_update_srel(:snapshot);
        ''', {'snapshot': snapshot})
        session.commit()
except:
    session.rollback()
    logger.error('Something happened',  exc_info=True)





