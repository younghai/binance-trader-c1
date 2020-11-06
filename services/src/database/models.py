from sqlalchemy import Column, Integer, FLOAT, String, DateTime, func
from database.database import BASE


class Pricing(BASE):
    __tablename__ = "pricing"

    id = Column(Integer, primary_key=True)

    timestamp = Column(DateTime(timezone=True), nullable=False)
    asset = Column(String, nullable=False)
    open = Column(FLOAT, nullable=False)
    high = Column(FLOAT, nullable=False)
    low = Column(FLOAT, nullable=False)
    close = Column(FLOAT, nullable=False)
    volume = Column(FLOAT, nullable=False)

    def __init__(self, timestamp, asset, open, high, low, close, volume):
        self.timestamp = timestamp
        self.asset = asset
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class Synced(BASE):
    __tablename__ = "synced"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    def __init__(self, timestamp):
        self.timestamp = timestamp
