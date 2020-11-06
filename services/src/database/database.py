import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

ENGINE = create_engine(
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}/{os.environ['POSTGRES_DB']}",
    convert_unicode=False,
)
SESSION = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=ENGINE))

BASE = declarative_base()
BASE.query = SESSION.query_property()


def init_db():
    from database import models

    BASE.metadata.create_all(ENGINE)

    # Initialize all data
    for table in [models.Pricing, models.Synced]:
        try:
            SESSION.query(table).delete()
            SESSION.commit()
        except:
            SESSION.rollback()
