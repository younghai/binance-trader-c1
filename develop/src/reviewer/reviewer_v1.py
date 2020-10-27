from dataclasses import dataclass
from typing import Dict, List, Union
from joblib import Parallel, delayed
from IPython.display import display, display_markdown
from tqdm import tqdm
import backtester
import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from .utils import grid
import json
from reviewer import paramset


@dataclass
class ReviewerV1:
    reviewer_prefix: str
    grid_params: Union[str, Dict[str, List]]
    backtester_type: str = "BacktesterV1"
    n_jobs: int = 16

    def __post_init__(self):
        if isinstance(self.grid_params, str):
            self.grid_params = getattr(paramset, self.grid_params)

    def run(self):
        self.backtesters = [
            getattr(backtester, self.backtester_type)(
                report_prefix=f"{self.reviewer_prefix}_{idx}", **params
            )
            for idx, params in enumerate(tqdm(list(grid(self.grid_params))))
        ]

        Parallel(n_jobs=self.n_jobs, verbose=1)(
            [delayed(backtester.run)(display=False) for backtester in self.backtesters]
        )

    def load_artifact(self, artifact_type, index):
        assert artifact_type in ("metrics", "report", "params")

        file_path = os.path.join(
            self.grid_params["exp_dir"],
            f"reports/{artifact_type}_{self.reviewer_prefix}_{index}_{self.grid_params['base_currency']}.parquet.zstd",
        )

        if artifact_type in ("metrics", "report"):
            artifact = pd.read_parquet(file_path)
        else:
            artifact = json.load(open(file_path.replace(".parquet.zstd", ".json"), "r"))

        return artifact

    def load_artifacts(self, artifact_type):
        assert artifact_type in ("metrics", "report")

        file_paths = glob(
            os.path.join(
                self.grid_params["exp_dir"],
                f"reports/{artifact_type}_{self.reviewer_prefix}_*_{self.grid_params['base_currency']}.parquet.zstd",
            )
        )
        file_paths = sorted(
            file_paths,
            key=lambda x: int(
                x.split(f"{self.reviewer_prefix}_")[-1].split(
                    f'_{self.grid_params["base_currency"]}'
                )[0]
            ),
        )

        artifacts = [pd.read_parquet(file_path) for file_path in file_paths]

        return artifacts

    def display_metrics(self):
        display(
            pd.concat(self.load_artifacts(artifact_type="metrics")).reset_index(
                drop=True
            )
        )

    def display_report(self, index):
        report = self.load_artifact(artifact_type="report", index=index)

        display_markdown(f"#### Report: {index}", raw=True)
        _, ax = plt.subplots(4, 1, figsize=(12, 12))

        for idx, column in enumerate(["capital", "cache", "return", "trade_return"]):
            if column == "trade_return":
                report[column].dropna().apply(lambda x: sum(x)).plot(ax=ax[idx])

            else:
                report[column].plot(ax=ax[idx])

            ax[idx].set_title(f"historical {column}")

        plt.tight_layout()
        plt.show()

    def display_params(self, index):
        display(pd.Series(self.load_artifact(artifact_type="params", index=index)))
