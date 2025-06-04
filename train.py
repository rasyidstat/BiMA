import sys
import pandas as pd
import json
from pathlib import Path
from src.config import ITR_DATA_DIR
from src.utils import create_folds, get_weight_df, get_gfold, LoguruStream
from src.features import (
    parse_time_features,
    get_solar_time,
    transform_density,
    get_omni_rolling_features,
    get_ground_truth,
    get_satellite_year_meta,
)
from src.models import get_msis_density, train_lgb
from src.eval import eval_skill_all, eval_skill

from loguru import logger

import warnings
import argparse
import importlib

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for Atmospheric Density Forecasting",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("-c", "--config")
    parser.add_argument("-s", "--seed", type=int, default=2024)
    parser.add_argument("-dm", "--dirname_main", type=str, default="final")
    parser.add_argument("-sf", "--dirname_suffix", type=str, default="")
    parser.add_argument("-vf", "--validate_future", action="store_true", default=False)
    parser.add_argument("-sp", "--save_prediction", action="store_true", default=False)
    parser.add_argument(
        "-tm", "--train_mode", nargs="+", default=["validate", "full_train"]
    )
    return parser.parse_args()


args = parse_args()
exp_name = args.config
cfg = importlib.import_module(f"configs.{args.dirname_main}.{exp_name}").cfg
cfg.update(vars(args))
cfg["exp_name"] = f"{exp_name}{args.dirname_suffix}"

OUTPUT_DIR = Path("runs") / cfg["dirname_main"] / cfg["exp_name"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

json.dump(cfg, open(f"{OUTPUT_DIR}/config.json", "w"), indent=4)

logger.add(OUTPUT_DIR / "log.txt", level="INFO")
sys.stdout = sys.stderr = LoguruStream()

logger.info("Read raw data")
df_init = pd.read_feather(ITR_DATA_DIR / "initial_states.feather")
df_gt = pd.read_feather(ITR_DATA_DIR / "sat_density.feather")
df_omni = pd.read_feather(ITR_DATA_DIR / "omni.feather")

logger.info("Prepare data")
df_init = df_init.merge(
    df_gt.reset_index()[["file_id", "file_prefix"]]
    .drop_duplicates()
    .set_index("file_id"),
    right_index=True,
    left_index=True,
)
df_gt = df_gt.sort_values(["file_id", "timestamp"])
df_gt = transform_density(df_gt)
df_gt = get_weight_df(df_gt)


use_cycle = cfg.get("use_cycle", False)
if cfg["ts_features"]:
    logger.info("Parse time features")
    ts_features = cfg["ts_features"]
    df_init = parse_time_features(df_init, use_cycle=use_cycle, level=ts_features)
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


logger.info("Get ground truth")
df_gt_agg = get_ground_truth(df_gt, cfg)
logger.info(f"df_gt_agg shape: {df_gt_agg.shape}")
df_train = pd.merge(df_init.reset_index(), df_omni_features.reset_index())

df_train = get_msis_density(df_train, omni_cols=[])
df_train = create_folds(df_train, cv_split=5, seed=cfg["seed"])
df_train = pd.merge(df_train, df_gt_agg.reset_index())
logger.info(f"df_train shape: {df_train.shape}")

if cfg["validate_future"]:
    df_meta = get_satellite_year_meta(df_init)
    df_train = pd.merge(df_train, df_meta[["file_prefix", "year", "is_test"]])
    df_train = get_gfold(df_train)
    cfg["fold_col"] = "gfold"

if "gfold" in df_train.columns:
    cfg["remove_features"].append("gfold")

if "validate" in cfg["train_mode"]:
    logger.info("Train CV model")
    df_pred, mdls = train_lgb(
        df_train, cfg, model_output_dir=MODEL_DIR, future_val=cfg["validate_future"]
    )

    df_pred_base = pd.merge(
        df_gt.reset_index()[
            ["file_id", "orbit_mean_density", "horizon", "weight"]
        ].dropna(subset=["orbit_mean_density"]),
        df_pred.drop(columns=["orbit_mean_density"]),
    ).set_index("file_id")
    logger.info(f"df_pred_base: {df_pred_base.shape}")
    if cfg["save_prediction"]:
        fname = "pred_vf" if cfg["validate_future"] else "pred"
        df_pred_base.reset_index().to_feather(OUTPUT_DIR / f"{fname}.feather")

    logger.info("Evaluate model")
    if cfg["validate_future"]:
        df_pred_base = df_pred_base.assign(
            is_future=lambda x: x["gfold"] != x["gfold_iter"]
        )
        smr_eval = eval_skill(df_pred_base, ["is_future", "file_prefix"])
        logger.info(smr_eval)
        logger.info("Evaluate all")
        df_eval_all = eval_skill_all(df_pred_base)
        logger.info("Evaluate group only")
        df_eval = eval_skill_all(df_pred_base.query("is_future==False"))
    else:
        df_eval = eval_skill_all(df_pred_base)

if "full_train" in cfg["train_mode"]:
    logger.info("Train model with all data")
    MODEL_DIR_FT = Path("models") / cfg["dirname_main"]
    MODEL_DIR_FT.mkdir(parents=True, exist_ok=True)
    _, mdl = train_lgb(df_train, cfg, model_output_dir=MODEL_DIR_FT, full_train=True)
