# coding: utf-8
from sqlalchemy import ARRAY, BigInteger, Boolean, CheckConstraint, Column, Date, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, JSON, Numeric, SmallInteger, String, Table, Text, UniqueConstraint, text
from geoalchemy2.types import Geometry, Raster
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.hstore import HSTORE
from sqlalchemy.dialects.postgresql.base import OID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, INTEGER, NUMERIC, TEXT, BIGINT, TIMESTAMP, VARCHAR


Base = declarative_base()
metadata = Base.metadata

class corr_mv_lines_results(Base):
    __tablename__ = 'corr_mv_lines_results'
    __table_args__ = {'schema': 'model_draft'}

    name = Column(Text, primary_key=True)
    bus0 = Column(Text)
    bus1 = Column(Text)
    s_nom = Column(Float(53))
    s = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    v_nom = Column(Float(53))
    mv_grid = Column(BigInteger, primary_key=True)
    result_id = Column(Integer, primary_key=True)
    geom = Column(Geometry('LINESTRING', 4326))
    x = Column(Float(53))
    r = Column(Float(53))
    length = Column(Float(53))

class corr_mv_bus_results(Base):
    __tablename__ = 'corr_mv_bus_results'
    __table_args__ = {'schema': 'model_draft'}

    name = Column(Text, primary_key=True)
    control = Column(Text)
    type = Column(Text)
    v_nom = Column(Float(53))
    v = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    v_ang = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    p = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    q = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    mv_grid = Column(BigInteger, primary_key=True)
    result_id = Column(Integer, primary_key=True)
    geom = Column(Geometry('POINT', 4326))


## General Packages
#import pandas as pd
#import geopandas as gpd
#from geoalchemy2.shape import to_shape
#
### Project Packages
#from egoio.db_tables import model_draft
#from egoio.db_tables import boundaries
#
#
#ormclass_line = model_draft.EgoGridPfHvLine
#ormclass_bus = model_draft.EgoGridPfHvBus
#ormclass_germ = boundaries.BkgVg2501Sta

#def get_cntry_links(session, scn_name):
### Lines
#    query = session.query(
#            ormclass_line.line_id,
#            ormclass_line.topo
#            ).filter(
#                    ormclass_line.scn_name == scn_name)
#
#    line_df = pd.DataFrame(query.all(),
#                          columns=[column['name'] for
#                                   column in
#                                   query.column_descriptions])
#    line_df = line_df.set_index('line_id')
#
#    line_df['topo'] = line_df.apply(
#            lambda x: to_shape(x['topo']), axis=1)
#    crs = {'init': 'epsg:4326'}
#    line_gdf = gpd.GeoDataFrame(line_df, crs=crs, geometry=line_df.topo)
#
#
#    ## Boundaries
#    nuts_gdf = gpd.read_file("data/nuts/nuts.shp")
#    nuts_ger_gdf = nuts_gdf.loc[nuts_gdf['nuts_id'] == 'DE']['geometry'].values[0]
#
#    ## Calculation
#    line_gdf['within_ger'] = line_gdf.apply(
#            lambda x: x['topo'].within(nuts_ger_gdf), axis=1)
#
#    return line_gdf.loc[line_gdf['within_ger']==False].index.tolist()
#
