import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from IPython.display import display


def agg_error_metrics(
    data,
    y_true_col="orbit_mean_density",
    y_pred_col="pred_orbit_mean_density",
    metrics_to_include=["rmse", "mae", "r2", "mape", "bias"],
    return_actual=True,
    return_pred=True,
):
    y_true, y_pred = data[y_true_col], data[y_pred_col]

    metrics = {}
    metrics["n_records"] = len(data)
    metrics["n_states"] = len(data.index.unique())

    if "rmse" in metrics_to_include:
        metrics["rmse"] = mean_squared_error(y_true, y_pred) ** (0.5)

    if "wrmse" in metrics_to_include:
        weights = data["weight"] if "weight" in data.columns else 1
        mse = np.average((y_true - y_pred) ** 2, weights=weights)
        metrics["wrmse"] = np.sqrt(mse)

    if "mae" in metrics_to_include:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)

    if "r2" in metrics_to_include:
        metrics["r2"] = r2_score(y_true, y_pred)

    if "mape" in metrics_to_include:
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)

    if "smape" in metrics_to_include:
        metrics["smape"] = _smape(y_true, y_pred)

    if "bias" in metrics_to_include:
        metrics["bias"] = np.mean(y_true - y_pred)

    if return_actual:
        metrics["avg_gt"] = np.mean(y_true)

    if return_pred:
        metrics["avg_pred"] = np.mean(y_pred)

    return pd.Series(metrics)


def _smape(y_true, y_pred):
    smape = (
        1
        / len(y_true)
        * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    )

    return smape


def eval_agg(pred_df, grouper=["year"], is_include_mean_std=False, **kwargs):
    eval_agg_df = pred_df.groupby(grouper).apply(
        agg_error_metrics, include_groups=False, **kwargs
    )
    if "file_id" in grouper:
        eval_agg_df = eval_agg_df.sort_values("orbit_mean_density", ascending=False)
    if (is_include_mean_std) & (len(eval_agg_df) > 1) & (grouper != ["all"]):
        eval_agg_df = pd.concat(
            [
                eval_agg_df,
                eval_agg_df.mean()
                .to_frame()
                .T.assign(a="mean")
                .set_index("a")
                .rename_axis(None, axis=0),
                eval_agg_df.std()
                .to_frame()
                .T.assign(a="std")
                .set_index("a")
                .rename_axis(None, axis=0),
            ],
            axis=0,
        )
    eval_agg_df = eval_agg_df.assign(
        n_records=lambda x: x["n_records"].astype(int),
        n_states=lambda x: x["n_states"].astype(int),
    )

    return eval_agg_df


def eval_all(
    pred_df,
    grouper_list=[["fold"], ["file_prefix"]],
    is_include_mean_std=True,
    **kwargs,
):
    for grouper in grouper_list:
        print(grouper)
        display(
            eval_agg(
                pred_df,
                grouper=grouper,
                is_include_mean_std=is_include_mean_std,
                **kwargs,
            )
        )
        print("\n")


def eval_agg_with_msis(
    df, grouper, metrics_to_include=["rmse", "r2"], is_include_mean_std=False, **kwargs
):
    df = df.copy()
    metrics_pred = dict(
        zip(metrics_to_include, [x + "_pred" for x in metrics_to_include])
    )
    metrics_msis = dict(
        zip(metrics_to_include, [x + "_msis" for x in metrics_to_include])
    )
    metrics_comb = [
        f"{metric}_{pred}" for metric in metrics_to_include for pred in ["pred", "msis"]
    ]
    smr_pred = (
        eval_agg(
            df,
            grouper=grouper,
            is_include_mean_std=False,
            metrics_to_include=metrics_to_include,
            **kwargs,
        )
        .assign(n_records=lambda x: x["n_records"].astype(int))
        .rename(columns=metrics_pred)
    )
    smr_msis = (
        eval_agg(
            df,
            grouper=grouper,
            is_include_mean_std=is_include_mean_std,
            y_pred_col="msis_density",
            metrics_to_include=metrics_to_include,
            **kwargs,
        )
        .assign(n_records=lambda x: x["n_records"].astype(int))
        .rename(columns=metrics_msis)
        .rename(columns={"avg_pred": "avg_msis"})
    )
    smr_comb = pd.concat(
        [smr_pred, smr_msis[metrics_msis.values()], smr_msis["avg_msis"]], axis=1
    )[["n_records", "n_states"] + metrics_comb + ["avg_gt", "avg_pred", "avg_msis"]]

    return smr_comb


def eval_skill(
    df, grouper, metrics_to_include=["rmse", "r2"], is_include_mean_std=False, **kwargs
):
    smr_comb = eval_agg_with_msis(
        df, grouper=grouper, metrics_to_include=metrics_to_include, **kwargs
    )
    smr_skll = (
        eval_agg_with_msis(df, grouper=grouper + ["horizon", "weight"])
        .reset_index()
        .groupby(grouper)
        .apply(
            _weighted_avg, ["rmse_pred", "rmse_msis"], "weight", include_groups=False
        )
        .assign(
            skill=lambda x: 1 - (x["rmse_pred"] / x["rmse_msis"]),
        )
    )
    smr_comb = pd.concat([smr_comb, smr_skll], axis=1)
    if (is_include_mean_std) & (len(smr_comb) > 1) & (grouper != ["all"]):
        smr_comb = pd.concat(
            [
                smr_comb,
                smr_comb.mean()
                .to_frame()
                .T.assign(a="mean")
                .set_index("a")
                .rename_axis(None, axis=0),
                smr_comb.std()
                .to_frame()
                .T.assign(a="std")
                .set_index("a")
                .rename_axis(None, axis=0),
            ],
            axis=0,
        )
        smr_comb = smr_comb.assign(
            n_records=lambda x: x["n_records"].astype(int),
            n_states=lambda x: x["n_states"].astype(int),
        )
    smr_comb.columns = [
        "n_records",
        "n_states",
        "rmse_pred",
        "rmse_msis",
        "r2_pred",
        "r2_msis",
        "avg_gt",
        "avg_pred",
        "avg_msis",
        "wrmse_pred",
        "wrmse_msis",
        "skill",
    ]

    return smr_comb


def eval_skill_all(
    pred_df,
    grouper_list=[["fold"], ["file_prefix"]],
    is_include_mean_std=True,
    return_table=False,
    **kwargs,
):
    tables = []
    for grouper in grouper_list:
        print(grouper)
        _res = eval_skill(
            pred_df, grouper=grouper, is_include_mean_std=is_include_mean_std, **kwargs
        )
        display(_res)
        print("\n")
        tables.append(_res)
    if return_table:
        return tables


def _weighted_avg(group, value_cols, weight_col):
    weights = group[weight_col]
    return pd.Series(
        {col: (group[col] * weights).sum() / weights.sum() for col in value_cols}
    )
