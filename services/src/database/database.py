import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

ENGINE = create_engine(
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}/{os.environ['POSTGRES_DB']}",
    convert_unicode=False,
)
DB_SESSION = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
)

BASE = declarative_base()
BASE.query = DB_SESSION.query_property()


def init_db():
    import database.models

    BASE.metadata.create_all(ENGINE)
