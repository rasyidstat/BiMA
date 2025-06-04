import numpy as np
import pandas as pd


def parse_time_features(
    df, col="timestamp", level=["hour", "doy", "month"], use_cycle=False, prefix=""
):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    if "hour" in level:
        df[f"{prefix}hour"] = df[col].dt.hour
        if use_cycle:
            df[f"{prefix}hour_sin"] = np.sin(2 * np.pi * df[f"{prefix}hour"] / 24)
            df[f"{prefix}hour_cos"] = np.cos(2 * np.pi * df[f"{prefix}hour"] / 24)
    if "minute" in level:
        df[f"{prefix}minute"] = df[col].dt.minute
    if "second" in level:
        df[f"{prefix}second"] = df[col].dt.second
    if "doy" in level:
        df[f"{prefix}doy"] = df[col].dt.dayofyear
        if use_cycle:
            df[f"{prefix}doy_sin"] = np.sin(2 * np.pi * df[f"{prefix}doy"] / 365.25)
            df[f"{prefix}doy_cos"] = np.cos(2 * np.pi * df[f"{prefix}doy"] / 365.25)
    if "month" in level:
        df[f"{prefix}month"] = df[col].dt.month
    if "year" in level:
        df[f"{prefix}year"] = df[col].dt.year

    return df


def get_solar_time(df, col="timestamp", longitude_col="longitude_deg"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    solar_offset = pd.to_timedelta(df[longitude_col] / 360 * 24, unit="h")
    df["solar_time"] = df[col] + solar_offset

    return df


def transform_density(df, col="orbit_mean_density", retransform=False):
    df = df.copy()
    if retransform:
        df[col] = df[col] / 10e12
    else:
        df[col] = df[col] * 10e12

    return df


def get_omni_rolling_features(df, cfg):
    df = df.copy()
    df_omni = pd.DataFrame()

    roll_window_long = cfg["omni"]["roll_long"]
    omni_features = cfg["omni"]["features"]
    agg_funcs = cfg["omni"]["agg_funcs"]

    for window in roll_window_long:
        _df = (
            df.groupby("file_id")
            .tail(window)
            .groupby("file_id")[omni_features]
            .agg(agg_funcs)
        )
        _df.columns = [f"__r{window}_".join(col).strip() for col in _df.columns.values]
        df_omni = pd.concat([df_omni, _df], axis=1)

    return df_omni


def get_ground_truth(df_gt: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    agg_method = cfg["target_transform"]["agg"]

    if agg_method == "first":
        df_gt_agg = (
            df_gt.groupby("file_id")["orbit_mean_density"]
            .first()
            .reset_index()
            .set_index("file_id")
        )
    elif agg_method == "raw":
        df_gt_agg = df_gt.copy()
        df_gt_agg = df_gt_agg.dropna(subset=["orbit_mean_density"])[
            ["horizon", "orbit_mean_density"]
        ]
    else:
        df_gt_agg = (
            df_gt.groupby("file_id")["orbit_mean_density"].agg(agg_method).to_frame()
        )

    return df_gt_agg


def get_satellite_year_meta(df):
    """
    Mark last 1-year of each satellite as test data
    For GRACE1 where sample size is low, use last 7-year as test data
    """
    df = df.copy()
    smr_sat_year = df.groupby(["file_prefix", "year"]).agg(
        n_records=("date", "size"), n_days=("date", "nunique")
    )
    smr_sat_year = smr_sat_year.reset_index().assign(
        rnk=lambda x: x.groupby(["file_prefix"])["year"].rank(ascending=False),
        is_test=lambda x: np.where(
            (x["rnk"] == 1) | ((x["rnk"] <= 7) & (x["file_prefix"] == "grace1")),
            True,
            False,
        ),
    )

    return smr_sat_year
