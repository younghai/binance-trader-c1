from dataclasses import dataclass
from typing import Dict, List
from joblib import Parallel, delayed
from IPython.display import display
from tqdm import tqdm
import backtester
from .utils import grid
import os
import pandas as pd
import joblib
from glob import glob


@dataclass
class ReviewerV1:
    reviewer_prefix: str
    grid_params: Dict[str, List]
    backtester_type: str = "BacktesterV1"
    n_jobs: int = 16

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

    def load_artifacts(self, artifact_type):
        assert artifact_type in ("metrics", "report")

        file_paths = glob(
            os.path.join(
                self.grid_params["exp_dir"],
                f"reports/{artifact_type}_{self.reviewer_prefix}_*_{self.grid_params['base_currency']}.csv",
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

        artifacts = [
            pd.read_csv(file_path, header=0, index_col=0) for file_path in file_paths
        ]

        return artifacts

    def display_metrics(self):
        display(pd.concat(self.load_artifacts(artifact_type="metrics"), axis=1),)

    def display_reports(self):
        for report_id, report in enumerate(self.load_artifacts(artifact_type="report")):
            display_markdown(f"#### Report: {report_id}", raw=True)
            _, ax = plt.subplots(4, 1, figsize=(12, 12))

            for idx, column in enumerate(
                ["capital", "cache", "return", "trade_return"]
            ):
                if column == "trade_return":
                    report[column].dropna().apply(lambda x: sum(x)).plot(ax=ax[idx])

                else:
                    report[column].plot(ax=ax[idx])

                ax[idx].set_title(f"historical {column}")

            plt.tight_layout()
            plt.show()
