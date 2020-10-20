from dataclasses import dataclass
from typing import Dict, List
from joblib import Parallel, delayed
from IPython.display import display
from tqdm import tqdm
import backtester
from .utils import grid
import os


@dataclass
class ReviewerV1:
    grid_params: Dict[str, List]
    backtester_type: str = "BacktesterV1"
    n_jobs: int = 16

    def __post_init__(self):
        self._build()

    def _build(self):
        self.backtesters = [
            getattr(backtester, self.backtester_type)(**params)
            for params in tqdm(list(grid(self.grid_params)))
        ]

        self.backtesters = Parallel(n_jobs=self.n_jobs, verbose=1)(
            [delayed(backtester.run)(display=False) for backtester in self.backtesters]
        )

    def display_metrics(self):
        display(
            pd.concat([backtester.build_metrics() for backtester in self.backtesters])
        )

    def display_report(self, index):
        self.backtesters[index].display_report(
            self.backtesters[index].generate_report()
        )

    def store(self, store_path):
        joblib.dump(self, store_path)
        print("[+] Stored!")
