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

class corr_hv_bus_results(Base):
    __tablename__ = 'corr_hv_bus_results'
    __table_args__ = {'schema': 'model_draft'}

    result_id = Column(BigInteger, primary_key=True, nullable=False)
    bus_id = Column(BigInteger, primary_key=True, nullable=False)
    x = Column(Float(53))
    y = Column(Float(53))
    v_nom = Column(Float(53))
    current_type = Column(Text)
    v_mag_pu_min = Column(Float(53))
    v_mag_pu_max = Column(Float(53))
    geom = Column(Geometry('POINT', 4326))


class corr_hv_bus_t_results(Base):
    __tablename__ = 'corr_hv_bus_t_results'
    __table_args__ = {'schema': 'model_draft'}

    result_id = Column(BigInteger, primary_key=True, nullable=False)
    bus_id = Column(BigInteger, primary_key=True, nullable=False)
    v_mag_pu_set = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    p = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    q = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    v_mag_pu = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    v_ang = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    marginal_price = Column(ARRAY(DOUBLE_PRECISION(precision=53)))

class corr_hv_lines_results(Base):
    __tablename__ = 'corr_hv_lines_results'
    __table_args__ = {'schema': 'model_draft'}

    result_id = Column(BigInteger, primary_key=True, nullable=False)
    line_id = Column(BigInteger, primary_key=True, nullable=False)
    bus0 = Column(BigInteger)
    bus1 = Column(BigInteger)
    x = Column(Numeric)
    r = Column(Numeric)
    g = Column(Numeric)
    b = Column(Numeric)
    s_nom = Column(Numeric)
    s_nom_extendable = Column(Boolean)
    s_nom_min = Column(Float(53))
    s_nom_max = Column(Float(53))
    capital_cost = Column(Float(53))
    length = Column(Float(53))
    cables = Column(Integer)
    frequency = Column(Numeric)
    terrain_factor = Column(Float(53), server_default=text("1"))
    x_pu = Column(Numeric)
    r_pu = Column(Numeric)
    g_pu = Column(Numeric)
    b_pu = Column(Numeric)
    s_nom_opt = Column(Numeric)
    geom = Column(Geometry('MULTILINESTRING', 4326))
    topo = Column(Geometry('LINESTRING', 4326))


class corr_hv_lines_t_result(Base):
    __tablename__ = 'corr_hv_lines_t_results'
    __table_args__ = {'schema': 'model_draft'}

    result_id = Column(BigInteger, primary_key=True, nullable=False)
    line_id = Column(BigInteger, primary_key=True, nullable=False)
    p0 = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    q0 = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    p1 = Column(ARRAY(DOUBLE_PRECISION(precision=53)))
    q1 = Column(ARRAY(DOUBLE_PRECISION(precision=53)))



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
