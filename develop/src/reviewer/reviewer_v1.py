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
from common_utils_dev import to_abs_path
from tabulate import tabulate


@dataclass
class ReviewerV1:
    dataset_dir: str = to_abs_path(__file__, "../../storage/dataset/v001/")
    exp_dir: str = to_abs_path(__file__, "../../storage/experiments/v001/")
    reviewer_prefix: str = "v001"
    grid_params: Union[str, Dict[str, List]] = "V1_SET1"
    backtester_type: str = "BacktesterV1"
    exec_start: int = 0
    exec_end: int = None
    n_jobs: int = 16

    def __post_init__(self):
        if isinstance(self.grid_params, str):
            self.grid_params = getattr(paramset, self.grid_params)

        self.grid_params["dataset_dir"] = self.dataset_dir
        self.grid_params["exp_dir"] = self.exp_dir

        self._build_backtesters()

    def _exists_artifact(self, index):
        exists = []
        for artifact_type in ["metrics", "report", "params"]:
            file_path = os.path.join(
                self.grid_params["exp_dir"],
                f"reports/{artifact_type}_{self.reviewer_prefix}_{index}_{self.grid_params['base_currency']}.parquet.zstd",
            )

            if artifact_type in ("params"):
                exists.append(
                    os.path.exists(file_path.replace(".parquet.zstd", ".json"))
                )
                continue

            exists.append(os.path.exists(file_path))

        exists = all(exists)

        if exists is True:
            print(f"[!] Found backtests already done: {index}")

        return exists

    def _build_backtesters(self):
        def _is_valid_params(param):
            if "entry_qay_threshold" in param:
                if (param["entry_qay_threshold"] == 9) and (
                    param["entry_qay_prob_threshold"] >= 0.4
                ):
                    return False

            if "entry_qby_threshold" in param:
                if (param["entry_qby_threshold"] == 9) and (
                    param["entry_qby_prob_threshold"] >= 0.4
                ):
                    return False

            return True

        grid_params = list(grid(self.grid_params))

        # Filter grid_params
        grid_params = [
            grid_param
            for grid_param in grid_params
            if _is_valid_params(param=grid_param) is True
        ]

        # Build backtesters
        self.backtesters = [
            getattr(backtester, self.backtester_type)(
                report_prefix=f"{self.reviewer_prefix}_{idx}", **params
            )
            for idx, params in enumerate(grid_params)
        ][self.exec_start : self.exec_end]
        self.backtesters = [
            backtester
            for backtester in self.backtesters
            if self._exists_artifact(index=backtester.report_prefix.split("_")[-1])
            is not True
        ]

    def _load_artifact(self, artifact_type, index):
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

    def _load_artifacts(self, artifact_type, with_index=False):
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
        index = pd.Index(
            [
                int(
                    file_path.split(f"{artifact_type}_{self.reviewer_prefix}_")[
                        -1
                    ].split(f"_{self.grid_params['base_currency']}.parquet.zstd")[0]
                )
                for file_path in file_paths
            ]
        )

        if with_index is True:
            return artifacts, index

        return artifacts

    def _build_metrics(self):
        artifacts, index = self._load_artifacts(
            artifact_type="metrics", with_index=True
        )
        metrics = pd.concat(artifacts)
        metrics.index = index

        return metrics

    def display_params(self, index, in_shell=False):
        params = (
            pd.Series(self._load_artifact(artifact_type="params", index=index))
            .rename("params")
            .to_frame()
        )

        if in_shell is True:
            print(tabulate(params, headers="keys", tablefmt="psql"))
        else:
            display(params)

    def display_report(self, index, in_shell=False):
        report = self._load_artifact(artifact_type="report", index=index)

        display_markdown(f"#### Report: {index}", raw=True)
        _, ax = plt.subplots(4, 1, figsize=(12, 12))

        for idx, column in enumerate(["capital", "cache", "return", "trade_return"]):
            if column == "trade_return":
                report[column].dropna().apply(lambda x: sum(x)).plot(ax=ax[idx])

            else:
                report[column].plot(ax=ax[idx])

            ax[idx].set_title(f"historical {column}")

        plt.tight_layout()

        if in_shell is True:
            plt.show(block=True)
        else:
            plt.show()

    def display_metrics(self, in_shell=False):
        metrics = self._build_metrics()

        if in_shell is True:
            print(tabulate(metrics, headers="keys", tablefmt="psql"))
        else:
            display(metrics)

    def display(self, in_shell=False):
        self.display_metrics(in_shell=in_shell)

        metrics = self._build_metrics()
        best_index = metrics["total_return"].sort_values(ascending=False).index[0]

        display_markdown(f"### [+] Best index: {best_index}", raw=True)

        display(metrics.loc[best_index])
        self.display_params(index=best_index, in_shell=in_shell)
        self.display_report(index=best_index, in_shell=in_shell)

    def run(self, in_shell=False):
        print(f"[+] Found backtests to start: {len(self.backtesters)}")

        Parallel(n_jobs=self.n_jobs, verbose=1)(
            [delayed(backtester.run)(display=False) for backtester in self.backtesters]
        )

        self.display(in_shell=in_shell)


if __name__ == "__main__":
    import fire

    fire.Fire(ReviewerV1)
