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
import fancytable as ft


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

    def _load_data_dict(self):
        data_dict = {}
        data_dict["labels"] = pd.read_parquet(
            os.path.join(self.exp_dir, "generated_output/labels.parquet.zstd")
        )
        data_dict["predictions"] = pd.read_parquet(
            os.path.join(self.exp_dir, "generated_output/predictions.parquet.zstd")
        )

        return data_dict

    def _display_timeseries(self, data_dict):
        columns = data_dict["predictions"].columns
        _, ax = plt.subplots(len(columns), 1, figsize=(24, 2.5 * len(columns)))

        for idx, column in enumerate(columns):
            data_dict["labels"][column].rename("label").plot(ax=ax[idx], alpha=0.5)
            data_dict["predictions"][column].rename("prediction").plot(ax=ax[idx])
            ax[idx].legend()
            ax[idx].set_title(column)

        plt.tight_layout()
        plt.show()

    def _build_levels(self, data):
        levels = {}
        for column in data.columns:
            levels[column] = pd.qcut(data[column], 10, labels=False, retbins=False)

        return pd.concat(levels, axis=1)

    def _build_total_performance(self, data_dict):
        total_performance = (data_dict["labels"] * data_dict["predictions"] >= 0).mean()
        total_performance["mean"] = total_performance.mean()

        return total_performance

    def _build_performance_on_levels(self, data_dict, levels):
        performance = data_dict["labels"] * data_dict["predictions"] >= 0

        performance_on_levels = []
        for column in performance.columns:
            performance_on_levels.append(
                performance[column].groupby(levels[column]).mean()
            )

        performance_on_levels = pd.concat(performance_on_levels, axis=1)
        performance_on_levels["mean"] = performance_on_levels.mean(axis=1)

        return performance_on_levels

    def display_performance(self):
        data_dict = self._load_data_dict()

        display_markdown("#### Timeseries", raw=True)
        self._display_timeseries(data_dict=data_dict)

        display_markdown("#### Total Performance", raw=True)
        total_performance = self._build_total_performance(data_dict=data_dict)
        display(ft.display(total_performance.rename("bin_acc").to_frame().T, axis=1))

        # Build levels
        label_levels = self._build_levels(data=data_dict["labels"])
        prediction_levels = self._build_levels(data=data_dict["predictions"])
        abs_prediction_levels = self._build_levels(data=data_dict["predictions"].abs())

        display_markdown("#### Performance on label levels", raw=True)
        display(
            ft.display(
                self._build_performance_on_levels(
                    data_dict=data_dict, levels=label_levels
                )
            )
        )

        display_markdown("#### Performance on prediction levels", raw=True)
        display(
            ft.display(
                self._build_performance_on_levels(
                    data_dict=data_dict, levels=prediction_levels
                )
            )
        )

        display_markdown("#### Performance on abs(prediction) levels", raw=True)
        display(
            ft.display(
                self._build_performance_on_levels(
                    data_dict=data_dict, levels=abs_prediction_levels
                )
            )
        )

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
            if param["adjust_prediction"] is True:
                if isinstance(param["exit_threshold"], (int, float)):
                    return False

            if param["exit_threshold"] != "auto":
                if param["achieve_ratio"] != 1:
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
        display_markdown(f"#### Params: {index}", raw=True)

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
        self.display_performance()

        print(f"[+] Found backtests to start: {len(self.backtesters)}")

        Parallel(n_jobs=self.n_jobs, verbose=1)(
            [delayed(backtester.run)(display=False) for backtester in self.backtesters]
        )

        self.display(in_shell=in_shell)


if __name__ == "__main__":
    import fire

    fire.Fire(ReviewerV1)
