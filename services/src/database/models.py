from sqlalchemy import Column, Integer, FLOAT, String, TIMESTAMP, UniqueConstraint
from database import database as DB


class Pricing(DB.BASE):
    __tablename__ = "pricings"

    id = Column(Integer, primary_key=True)

    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    asset = Column(String, nullable=False)
    open = Column(FLOAT, nullable=False)
    high = Column(FLOAT, nullable=False)
    low = Column(FLOAT, nullable=False)
    close = Column(FLOAT, nullable=False)
    volume = Column(FLOAT, nullable=False)

    __table_args__ = (UniqueConstraint("timestamp", "asset"),)

    def __init__(self, timestamp, asset, open, high, low, close, volume):
        self.timestamp = timestamp
        self.asset = asset
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class Sync(DB.BASE):
    __tablename__ = "syncs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, unique=True)

    def __init__(self, timestamp):
        self.timestamp = timestamp


class Trade(DB.BASE):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, unique=True)

    def __init__(self, timestamp):
        self.timestamp = timestamp
