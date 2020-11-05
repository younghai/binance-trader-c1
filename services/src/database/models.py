from sqlalchemy import Column, Integer, FLOAT, DateTime, func
from database.database import BASE


class Pricing(BASE):
    _tablename_ = "pricing"

    id = Column(Integer, primary_key=True)
    open = Column(FLOAT, nullable=False)
    high = Column(FLOAT, nullable=False)
    low = Column(FLOAT, nullable=False)
    close = Column(FLOAT, nullable=False)
    volume = Column(FLOAT, nullable=False)

    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_on = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_on = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        onupdate=func.now(),
    )

    def __init__(self, open, high, low, close, volume, timestamp):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timestamp = timestamp
