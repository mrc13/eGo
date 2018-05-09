#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 12:15:57 2018

@author: student
"""

# Snom results aus grid schema updaten!!!!!


### Project Packages
from egoio.tools import db
from ego.tools.specs import (
        get_scn_name_from_result_id,
        get_settings_from_result_id)

### Sub Packages

### General Packages
from sqlalchemy.orm import sessionmaker

conn = db.connection(section='oedb')
Session = sessionmaker(bind=conn)
session = Session()

result_id = str(input("Please type the result_id: "))

scn_name = get_scn_name_from_result_id(session, result_id)

settings = get_settings_from_result_id(session, result_id)
grid_version = settings['gridversion']
if grid_version == 'None':
    grid_version = None
if grid_version == None:
    raise NotImplementedError("To be implemented")

print('Scn name:')
print(scn_name)
print('Grid Version:')
print(grid_version)

session.execute('''
UPDATE model_draft.ego_grid_pf_hv_result_line as lr
    SET s_nom = (SELECT s_nom
                     FROM grid.ego_pf_hv_line as l
                     WHERE   scn_name = :scn_name AND
                             version = :grid_version AND
                             l.line_id = lr.line_id)
    WHERE result_id = :result_id;
''', {'result_id': result_id, 'scn_name': scn_name, 'grid_version': grid_version})
session.execute('''
UPDATE model_draft.ego_grid_pf_hv_result_transformer as tr
    SET s_nom = (SELECT s_nom
                     FROM grid.ego_pf_hv_transformer as t
                     WHERE   scn_name = :scn_name AND
                             version = :grid_version AND
                             t.trafo_id = tr.trafo_id)
    WHERE result_id = :result_id;
''', {'result_id': result_id, 'scn_name': scn_name, 'grid_version': grid_version})
session.commit()

## model draft.
#session.execute('''
#UPDATE model_draft.ego_grid_pf_hv_result_line as lr
#    SET s_nom = (SELECT s_nom
#                     FROM model_draft.ego_grid_pf_hv_line as l
#                     WHERE   scn_name = :scn_name AND
#                             l.line_id = lr.line_id)
#    WHERE result_id = :result_id;
#''', {'result_id': result_id, 'scn_name': scn_name})
#session.commit()
