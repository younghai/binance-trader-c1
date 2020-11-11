from dataclasses import dataclass
from database import database as DB
from database import models
from typing import List, Dict
import pandas as pd


@dataclass
class Usecase:
    sess = DB.SESSION

    def __post_init__(self):
        DB.wait_connection()

    def get_last_sync_on(self):
        timestamp = (
            self.sess.query(models.Sync)
            .order_by(models.Sync.timestamp.desc())
            .first()
            .timestamp
        )
        return pd.Timestamp(timestamp).tz_convert("UTC")

    def get_last_trade_on(self):
        queried = (
            self.sess.query(models.Trade)
            .order_by(models.Trade.timestamp.desc())
            .first()
        )

        if queried is None:
            return None

        return pd.Timestamp(queried.timestamp).tz_convert("UTC")

    def insert_pricings(self, inserts: List[Dict], n_buffer: int = 500):
        tmpl = """
        INSERT INTO
            pricings (
                timestamp,
                asset,
                open,
                high,
                low,
                close,
                volume
            )
        VALUES {};
        """.strip()

        def to_tuple(x, i):
            return (
                f"(:p{i}_1,:p{i}_2,:p{i}_3,:p{i}_4,:p{i}_5,:p{i}_6,:p{i}_7)",
                (
                    {
                        f"p{i}_1": x["timestamp"],
                        f"p{i}_2": x["asset"],
                        f"p{i}_3": x["open"],
                        f"p{i}_4": x["high"],
                        f"p{i}_5": x["low"],
                        f"p{i}_6": x["close"],
                        f"p{i}_7": x["volume"],
                    }
                ),
            )

        for i in range(0, len(inserts), n_buffer):

            items = [
                to_tuple(item, j) for j, item in enumerate(inserts[i : i + n_buffer])
            ]

            query = tmpl.format(",".join([item[0] for item in items]))

            params = {}
            for item in items:
                params.update(item[1])

            self.sess.execute(query, params)

        self.sess.commit()

    def insert_syncs(self, inserts: List[Dict], n_buffer: int = 500):
        tmpl = """
        INSERT INTO
            syncs (
                timestamp
            )
        VALUES {};
        """.strip()

        def to_tuple(x, i):
            return (f"(:p{i}_1)", ({f"p{i}_1": x["timestamp"]}))

        for i in range(0, len(inserts), n_buffer):

            items = [
                to_tuple(item, j) for j, item in enumerate(inserts[i : i + n_buffer])
            ]

            query = tmpl.format(",".join([item[0] for item in items]))

            params = {}
            for item in items:
                params.update(item[1])

            self.sess.execute(query, params)

        self.sess.commit()

    def insert_trade(self, insert: Dict):
        self.sess.add(models.Trade(timestamp=insert["timestamp"]))

        self.sess.commit()

    def update_pricings(self, updates: List[Dict], n_buffer: int = 500):
        tup_str = ""
        min_timestamp = []
        for update in updates:
            tup_str += f"('{update['timestamp'].isoformat()}'::timestamp, '{update['asset']}'),"
            min_timestamp.append(update["timestamp"])
        tup_str = tup_str[:-1]
        min_timestamp = min(min_timestamp)

        db_items = self.sess.execute(
            f"""
            SELECT id,
                    TIMESTAMP,
                    asset,
                    volume
            FROM   pricings
            WHERE  ( TIMESTAMP, asset ) IN (VALUES {tup_str})
                    AND TIMESTAMP >= '{min_timestamp.isoformat()}'::timestamp;
            """
        )

        key = lambda ts, asset: f"{ts.isoformat()}_{asset}"
        db_items_dict = dict()
        for db_item in db_items:
            db_items_dict[key(ts=db_item[1], asset=db_item[2])] = {
                "id": db_item[0],
                "volume": db_item[3],
            }

        ids_to_delete = list()
        to_insert = list()
        for update in updates:
            k = key(ts=update["timestamp"], asset=update["asset"])
            if k in db_items_dict:
                db_item = db_items_dict[k]
                if db_item["volume"] != update["volume"]:
                    # Value changed
                    ids_to_delete.append(db_item["id"])
                else:
                    continue

            to_insert.append(update)

        # Delete if changed
        self.sess.query(models.Pricing).filter(
            models.Pricing.id.in_(ids_to_delete)
        ).delete(synchronize_session=False)

        # Insert
        if len(to_insert):
            self.insert_pricings(inserts=to_insert, n_buffer=n_buffer)
        else:
            self.sess.commit()

    def update_syncs(self, updates: List[Dict], n_buffer: int = 500):
        tup_str = ""
        min_timestamp = []
        for update in updates:
            tup_str += f"('{update['timestamp'].isoformat()}'::timestamp),"
            min_timestamp.append(update["timestamp"])
        tup_str = tup_str[:-1]
        min_timestamp = min(min_timestamp)

        db_items = self.sess.execute(
            f"""
            SELECT TIMESTAMP
            FROM   syncs
            WHERE  ( TIMESTAMP ) IN (VALUES {tup_str})
                    AND TIMESTAMP >= '{min_timestamp.isoformat()}'::timestamp;
            """
        )

        key = lambda ts: f"{ts.isoformat()}"
        db_timestamps = set([key(ts=db_item[0]) for db_item in db_items])

        to_insert = list()
        for update in updates:
            k = key(ts=update["timestamp"])
            if k in db_timestamps:
                continue

            to_insert.append(update)

        # Insert
        if len(to_insert):
            self.insert_syncs(inserts=to_insert, n_buffer=n_buffer)

    def delete_old_records(self, table: str, limit: int):
        assert table in ("pricings", "syncs", "trades")
        if table == "pricings":
            table_class = models.Pricing
        elif table == "syncs":
            table_class = models.Sync
        elif table == "trades":
            table_class = models.Trade
        else:
            raise NotImplementedError

        table_counts = self.sess.execute(
            f"""
            SELECT count(*)
            FROM   {table};
            """
        ).first()[0]

        if table_counts > limit:
            # Delete overflowed records
            id_subqueries = (
                self.sess.query(table_class.id)
                .order_by(table_class.timestamp.asc())
                .limit(table_counts - limit)
            )
            self.sess.query(table_class).filter(
                table_class.id.in_(id_subqueries)
            ).delete(synchronize_session=False)
            self.sess.commit()
