import numpy as np
import pandas as pd
import lightgbm as lgb
from pymsis import msis
from loguru import logger
from src.utils import seed_everything


def train_lgb(df, cfg, model_output_dir=None, full_train=False, future_val=False):
    df = df.copy()
    cfg = cfg.copy()
    seed_everything(cfg["seed"])

    pred_col = "pred_" + cfg["target_col"]
    features = _get_features(df, cfg["remove_features"])
    idx_cols = [col for col in cfg["idx_cols"] if col in list(df)]

    logger.info(features)

    df = _apply_target_transform(df, cfg)
    df_pred, models = [], []

    if full_train:
        logger.info("Training on full data")
        train_data = _prepare_dataset(df, features, cfg["target_col"])
        cfg["lgb_train_params"]["n_round"] = cfg["lgb_train_params"]["ft_round"]
        model = _train_single_model(train_data, None, cfg, valid_names=["train"])
        pred_df = df[idx_cols].copy()
        pred_df[pred_col] = model.predict(df[features])
        df_pred.append(pred_df)
        models.append(model)
        if model_output_dir:
            logger.info("Save model")
            model.save_model(model_output_dir / f"{cfg['exp_name']}.txt")
    else:
        logger.info("Training with cross-validation")
        for fold in df[cfg["fold_col"]].unique():
            logger.info(f"Fold or group: {fold}")
            is_valid = df[cfg["fold_col"]] == fold
            if future_val:
                is_valid = is_valid | (df["is_test"])
            train_df, valid_df = df[~is_valid], df[is_valid]
            train_data = _prepare_dataset(train_df, features, cfg["target_col"])
            valid_data = _prepare_dataset(valid_df, features, cfg["target_col"])
            logger.info(
                f"Train data: {train_df[features].shape}, valid data: {valid_df[features].shape}"
            )
            logger.info(
                f"Train states: {train_df['file_id'].nunique()}, valid states: {valid_df['file_id'].nunique()}"
            )

            model = _train_single_model(
                train_data, valid_data, cfg, valid_names=["train", "valid"]
            )
            if future_val:
                pred_df = valid_df[idx_cols + ["gfold"]].copy()
                pred_df["gfold_iter"] = fold
            else:
                pred_df = valid_df[idx_cols].copy()
            pred_df[pred_col] = model.predict(valid_df[features])
            df_pred.append(pred_df)
            models.append(model)
            if model_output_dir:
                logger.info(f"Save model for fold {fold}")
                model.save_model(model_output_dir / f"{cfg['exp_name']}_f{fold}.txt")

    df_pred = pd.concat(df_pred).reset_index(drop=True)
    df_pred = _inverse_target_transform(df_pred, cfg, pred_col)
    return df_pred, models


def predict_lgb(df, cfg, model_output_dir):
    df = df.copy()
    cfg = cfg.copy()

    features = _get_features(df, cfg["remove_features"])
    model_file = model_output_dir / f"{cfg['exp_name']}.txt"
    model = lgb.Booster(model_file=model_file)

    pred_col = "pred_orbit_mean_density"
    df[pred_col] = model.predict(df[features])
    df_pred = df[["file_id", "msis_density", "horizon", pred_col]].copy()
    df_pred = _inverse_target_transform(df_pred, cfg, pred_col)

    return df_pred


def _get_features(df, remove_features):
    df = df.copy()
    remove_features_ = [col for col in remove_features if col in list(df)]
    all_features = [col for col in list(df) if col not in (remove_features_)]

    return all_features


def _apply_target_transform(df, cfg):
    mode = cfg["target_transform"]["mode"]
    target = cfg["target_col"]

    if mode == "diff":
        logger.info("Use target diff")
        df[target] = df[target] - df["msis_density"]
    elif mode == "ratio":
        logger.info("Use target ratio")
        df[target] = df[target] / df["msis_density"]

    if cfg["target_transform"]["log"]:
        logger.info("Use log")
        df[target] = np.log1p(df[target])
    return df


def _inverse_target_transform(df, cfg, pred_col):
    if cfg["target_transform"]["log"]:
        df[pred_col] = np.expm1(df[pred_col])

    mode = cfg["target_transform"]["mode"]
    if mode == "diff":
        df[pred_col] += df["msis_density"]
    elif mode == "ratio":
        df[pred_col] *= df["msis_density"]
    return df


def _prepare_dataset(df, features, target):
    return lgb.Dataset(df[features], label=df[target])


def _train_single_model(train_data, valid_data, cfg, valid_names):
    callbacks = []

    if not cfg.get("train_full", False) and cfg["lgb_train_params"].get("es_round"):
        callbacks.append(lgb.early_stopping(cfg["lgb_train_params"]["es_round"]))
    callbacks.append(lgb.log_evaluation(cfg["lgb_train_params"]["verbose"]))

    valid_sets = [train_data] if valid_data is None else [train_data, valid_data]
    return lgb.train(
        cfg["lgb_params"],
        train_data,
        num_boost_round=cfg["lgb_train_params"]["n_round"],
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def get_msis_density(
    df, omni_cols=["f107s", "f107as", "aps"], col_name="msis_density", transform=True
):
    df = df.copy()
    kwargs = {}
    _c = 10e12 if transform else 1

    if "f107s" in omni_cols:
        kwargs["f107s"] = df["f10.7_index"].to_numpy()
    if "f107as" in omni_cols:
        kwargs["f107as"] = df["f107as"].to_numpy()
    if "ap_index_nt" in omni_cols:
        kwargs["aps"] = df["ap_index_nt"].to_numpy()

    result = msis.run(
        dates=df["timestamp"].to_numpy(),
        lons=df["longitude_deg"].to_numpy(),
        lats=df["latitude_deg"].to_numpy(),
        alts=df["altitude_km"].to_numpy(),
        **kwargs,
    )
    df[col_name] = result[:, 0] * _c

    return df
