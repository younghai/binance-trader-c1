from dataclasses import dataclass
from database import database as DB
from database import models
from typing import List, Dict


@dataclass
class Usecase:
    sess = DB.SESSION

    def insert_pricings(self, inserts: List[Dict], n_buffer: int = 1000):
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
