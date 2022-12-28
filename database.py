# -*- coding: utf-8 -*-
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
engine = create_engine('sqlite:///SciConcept.sqlite')

Session = sessionmaker(bind=engine)


class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 指定name映射到name字段; name字段为字符串类形，
    name = Column(String(20))
    papers = Column(Integer)
    keywords = Column(Integer)
    real_keywords = Column(Integer)
    create_date = Column(Date)
    status = Column(String(32))


if __name__ == '__main__':
    Base.metadata.create_all(engine, checkfirst=True)
