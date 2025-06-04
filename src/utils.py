import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from loguru import logger


def seed_everything(seed=2024):
    random.seed(seed)
    np.random.seed(seed)


def create_folds(df, cv_split, seed, mode="base", scols=["BS"], is_print=False):
    df["fold"] = -1

    if mode == "base":
        kf = KFold(n_splits=cv_split, shuffle=True, random_state=seed)
        kf_split = kf.split(df)

    elif mode == "stratified":
        kf = StratifiedKFold(n_splits=cv_split, shuffle=True, random_state=seed)
        kf_split = kf.split(df, y=df[scols])

    for fold, (train_idx, val_idx) in enumerate(kf_split):
        if is_print:
            print(len(train_idx), len(val_idx))
        df.loc[val_idx, "fold"] = fold

    return df


def get_weight(epsilon=1e-5, delta_times=None):
    if not delta_times:
        delta_times = np.array(range(1, 433)) * 10
    total_duration = max(delta_times[-1] - delta_times[0], 1e-12)
    decay_rate = -np.log(epsilon) / total_duration
    weights = np.exp(-decay_rate * (delta_times - delta_times[0]))

    return weights


def get_weight_df(df, epsilon=1e-5, delta_times=None):
    if "weight" in df.columns:
        return df

    df = df.copy()
    weights = get_weight(epsilon=epsilon, delta_times=delta_times)
    weights_df = pd.DataFrame(weights, columns=["weight"])
    weights_df["horizon"] = np.array(range(1, 433))

    if "horizon" not in df.columns:
        df = get_horizon(df)

    df = pd.merge(df.reset_index(), weights_df, on="horizon", how="left")
    df = df.set_index("file_id")

    return df


def get_horizon(df):
    df = df.copy()
    df = df.assign(horizon=lambda x: (x.groupby("file_id").cumcount() + 1))

    return df


def get_gfold(df):
    df = df.copy()
    df["gfold"] = np.where(
        df["file_prefix"].isin(["grace1", "grace2"]), "grace", df["file_prefix"]
    )

    return df


def get_base_df(df):
    df = df.copy()
    df_base = df.reset_index()[["file_id", "timestamp"]].assign(
        end_timestamp=lambda x: x["timestamp"] + pd.Timedelta(days=3),
    )
    df_base["ts"] = df_base.apply(
        lambda row: pd.date_range(
            row["timestamp"], row["end_timestamp"], freq="10min", inclusive="left"
        ),
        axis=1,
    )
    df_base = (
        df_base.explode("ts")
        .drop(columns=["timestamp", "end_timestamp"])
        .reset_index(drop=True)
    )
    df_base = get_horizon(df_base)
    return df_base


class LoguruStream:
    def write(self, message):
        if message.strip():
            logger.info(message.strip())

    def flush(self):
        pass
