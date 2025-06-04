import sys
import numpy as np
import pandas as pd
import importlib
import json
from pathlib import Path
from src.utils import get_base_df, LoguruStream
from src.dataset import read_initial_states_file, read_omni
from src.features import parse_time_features, get_solar_time, get_omni_rolling_features
from src.models import get_msis_density, predict_lgb
from src.qa import log_data_summary

from loguru import logger

import pymsis
import os, shutil

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 10000)

_DATA_FNAME: str = "SW-All.csv"
_MY_F107_AP_PATH: Path = Path(os.getcwd()) / _DATA_FNAME
_F107_AP_PATH: Path = Path(pymsis.__file__).parent / _DATA_FNAME
if not _F107_AP_PATH.exists():
    shutil.copyfile(_MY_F107_AP_PATH, _F107_AP_PATH)
    print(f"Copied {_MY_F107_AP_PATH} to {_F107_AP_PATH}.")

DEBUG = True if os.path.exists(".gitignore") else False
logger.info(f"Debug mode: {DEBUG}")
if DEBUG:
    logger.add(Path("tests") / "log.txt", level="INFO")
    sys.stdout = sys.stderr = LoguruStream()

logger.info("Load base config")
CFG_DIR = "lag3det_final"
cfg = importlib.import_module(f"configs.{CFG_DIR}.raw").cfg
exp_list = ["raw_ratio", "raw_ratio_log"]

logger.info("Get directory")
SUB_DIR = Path("/app/ingested_program")
TEST_DATA_DIR = Path("/app/data/dataset/test")
INPUT_DATA_FILE = Path("/app/input_data/initial_states.csv")
TEST_PREDS_FP = Path("/app/output/prediction.json")
if DEBUG:
    SUB_DIR = Path("")
    TEST_DATA_DIR = Path("data/raw")
    INPUT_DATA_FILE = Path("data/raw/meta/06672_to_08118-initial_states.csv")
    TEST_PREDS_FP = Path("tests/prediction.json")
OMNI_DATA_DIR = TEST_DATA_DIR / "omni2"
MODEL_DIR = SUB_DIR / "models" / CFG_DIR

logger.info("Read raw data")
df_init = read_initial_states_file(
    INPUT_DATA_FILE, mask_cols=["latitude_deg", "longitude_deg", "altitude_km"]
)
df_init = df_init.drop(columns=["unnamed:_0"], errors="ignore")
if DEBUG:
    df_init = df_init.head(10)
df_omni = read_omni(dirname=OMNI_DATA_DIR, file_ids=list(df_init.index))
df_omni = df_omni.drop(columns=["unnamed:_0"], errors="ignore")

logger.info(f"df_init shape: {df_init.shape}, df_init IDs: {df_init.index.nunique()}")
log_data_summary(df_init, "df_init", "Initial states")
logger.info(f"df_omni shape: {df_omni.shape}, df_omni IDs: {df_omni.index.nunique()}")
log_data_summary(df_omni[cfg["omni"]["features"]], "df_omni", "Omni data")

use_cycle = cfg.get("use_cycle", False)
if cfg["ts_features"]:
    logger.info("Parse time features")
    ts_features = cfg["ts_features"]
    df_init = parse_time_features(df_init, use_cycle=use_cycle, level=ts_features)
df_init["altitude_km"] = np.where(
    df_init["altitude_km"] > 100_000,
    df_init["altitude_km"] / 1000,
    df_init["altitude_km"],
)
df_init = df_init.ffill()


if cfg["use_solar_time"]:
    logger.info("Use solar time")
    df_init = get_solar_time(df_init)
    df_init = parse_time_features(
        df_init,
        col="solar_time",
        prefix="solar_",
        use_cycle=use_cycle,
        level=ts_features,
    )
    cfg["remove_features"].append("solar_time")


logger.info("Parse OMNI features")
df_omni_features = get_omni_rolling_features(df_omni, cfg)
logger.info(f"df_omni_features shape: {df_omni_features.shape}")

df_train = pd.merge(df_init.reset_index(), df_omni_features.reset_index())
logger.info(
    f"Check missing values:\n{df_train.isna().sum().sort_values(ascending=False)}",
)
df_train = get_msis_density(df_train, omni_cols=[])
logger.info(df_train.describe().T)
if cfg["target_transform"]["agg"] == "raw":
    HORIZON = 432
else:
    HORIZON = 1

logger.info("Predict model")
df_pred = []
for exp_name in exp_list:
    cfg = importlib.import_module(f"configs.{CFG_DIR}.{exp_name}").cfg
    cfg["exp_name"] = exp_name
    for h in range(HORIZON):
        df_pred.append(
            predict_lgb(
                df_train.assign(horizon=h + 1), cfg, model_output_dir=MODEL_DIR
            ).assign(exp_name=exp_name)
        )
df_pred = pd.concat(df_pred)
logger.info(df_pred.groupby("exp_name").describe())

df_pred_ens = df_pred.groupby(["file_id", "horizon"])[
    ["msis_density", "pred_orbit_mean_density"]
].mean()
df_pred_ens["pred"] = df_pred_ens["pred_orbit_mean_density"] / 10e12
df_base = get_base_df(df_init)
df_pred_ens = pd.merge(df_base, df_pred_ens.reset_index())
assert len(df_pred_ens) == len(df_init) * HORIZON

logger.info("Write prediction in json")
predictions = (
    df_pred_ens.groupby("file_id")
    .apply(
        lambda g: {
            "Timestamp": g["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "Orbit Mean Density (kg/m^3)": g["pred"].tolist(),
        },
        include_groups=False,
    )
    .to_dict()
)
with open(TEST_PREDS_FP, "w") as outfile:
    json.dump(predictions, outfile)
logger.info("Finished")
