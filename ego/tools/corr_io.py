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

