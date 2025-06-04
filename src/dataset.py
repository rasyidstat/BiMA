import os
import pandas as pd
from src.config import META_DATA_DIR, GT_DATA_DIR, OMNI_DATA_DIR, omni_mask_dict


def read_initial_states(dirname=META_DATA_DIR, n_samples=None, mask_cols=None):
    file_glob = dirname.glob("*.csv")
    if n_samples:
        file_glob = list(file_glob)[:n_samples]

    df = pd.concat([pd.read_csv(x) for x in file_glob])
    df.columns = _clean_column(df.columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year
    df["file_id"] = df["file_id"].astype(str).str.zfill(5)
    df = df.sort_values("file_id").set_index("file_id")

    if mask_cols:
        df[mask_cols] = df[mask_cols].mask(df[mask_cols] >= 9.990000e29)

    return df


def read_initial_states_file(fname, mask_cols=None):
    df = pd.read_csv(fname)
    df.columns = _clean_column(df.columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["year"] = df["timestamp"].dt.year
    df["file_id"] = df["file_id"].astype(str).str.zfill(5)
    df = df.sort_values("file_id").set_index("file_id")

    if mask_cols:
        df[mask_cols] = df[mask_cols].mask(df[mask_cols] >= 9.990000e29)

    return df


def read_gt(dirname=GT_DATA_DIR, n_samples=None, mask_cols=None):
    file_glob = dirname.glob("*.csv")
    if n_samples:
        file_glob = list(file_glob)[:n_samples]
    df = pd.concat(
        [
            pd.read_csv(x).assign(
                file_id=os.path.basename(x.as_posix().replace("gr-of1", "grof1")).split(
                    "-"
                )[1],
                file_prefix=os.path.basename(
                    x.as_posix().replace("gr-of1", "grof1")
                ).split("-")[0],
            )
            for x in file_glob
        ]
    )
    df.columns = ["timestamp", "orbit_mean_density", "file_id", "file_prefix"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df = df.sort_values(["file_id", "timestamp"]).set_index("file_id")

    if mask_cols:
        df[mask_cols] = df[mask_cols].mask(df[mask_cols] >= 9.990000e32)

    return df


def read_omni(
    dirname=OMNI_DATA_DIR, n_samples=None, mask_dict=omni_mask_dict, file_ids=None
):
    file_glob = dirname.glob("*.csv")
    if n_samples:
        file_glob = list(file_glob)[:n_samples]
    if file_ids:
        prefixes = [f"omni2-{id_}" for id_ in file_ids]
        file_glob = [
            p
            for p in list(file_glob)
            if any(p.stem.startswith(prefix) for prefix in prefixes)
        ]
    df = pd.concat(
        [
            pd.read_csv(x).assign(
                file_id=x.stem.split("-")[1],
            )
            for x in file_glob
        ]
    )
    df.columns = _clean_column(df.columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df = df.sort_values(["file_id", "timestamp"]).set_index("file_id")

    if mask_dict:
        for val, mask_cols in mask_dict.items():
            df[mask_cols] = df[mask_cols].mask(df[mask_cols] >= val)

    return df


def _clean_column(x):
    x = (
        x.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return x


if __name__ == "__main__":
    df_init = read_initial_states(
        mask_cols=["latitude_deg", "longitude_deg", "altitude_km"]
    )
    df_init.to_feather("data/interim/initial_states.feather")
    df_gt = read_gt(mask_cols=["orbit_mean_density"])
    df_gt.to_feather("data/interim/sat_density.feather")
    df_omni = read_omni()
    df_omni.to_feather("data/interim/omni.feather")
