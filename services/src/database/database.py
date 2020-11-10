import os
import time
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base


ENGINE = create_engine(
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}/{os.environ['POSTGRES_DB']}",
    convert_unicode=False,
)

SESSION = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=ENGINE))

BASE = declarative_base()
BASE.query = SESSION.query_property()


def init():
    from database import models

    while True:
        try:
            BASE.metadata.drop_all(ENGINE)
            BASE.metadata.create_all(ENGINE)
            break
        except OperationalError:
            time.sleep(5)


def wait_connection():
    while True:
        try:
            ENGINE.connect()
            break
        except OperationalError:
            time.sleep(5)
