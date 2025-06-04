cfg = dict(
    seed=2025,
    fold_col="fold",
    cat_features=[],
    idx_cols=[
        "fold",
        "file_id",
        "orbit_mean_density",
        "msis_density",
        "file_prefix",
        "year",
        "horizon",
    ],
    remove_features=[
        "fold",
        "file_id",
        "orbit_mean_density",
        "file_prefix",
        "date",
        "timestamp",
        "year",
        "is_test",
        "hour",
        "doy",
        "solar_hour",
        "solar_doy",
    ],
    target_col="orbit_mean_density",
    target_transform=dict(
        mode="raw",  # ["raw", "diff", "ratio"],
        agg="raw",  # ["mean", "median", "first", "raw"]
        log=False,
    ),
    use_solar_time=True,
    use_cycle=True,
    ts_features=["hour", "doy"],
    omni=dict(
        features=[
            "lyman_alpha",
            "f10.7_index",
            "r_sunspot_no",
            "scalar_b_nt",
            "vector_b_magnitude_nt",
            "magnetosonic_mach_number",
            "dst_index_nt",
            "alpha_prot_ratio",
            "quasy_invariant",
            "au_index_nt",
            "alfen_mach_number",
            "ae_index_nt",
            "al_index_nt",
            "kp_index",
            "sigma_ratio",
            "sigma_theta_v_degrees",
            "ap_index_nt",
            "rms_field_vector_nt",
            "pc_index",
            "rms_bz_gse_nt",
        ],
        agg_funcs=["mean", "max"],
        roll_long=[6, 24, 72],
    ),
)

lgb_params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "learning_rate": 0.075,
    "num_leaves": 2**4 - 1,
    "min_data_in_leaf": 2**5 - 1,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "max_depth": -1,
    "random_state": cfg["seed"],
}

lgb_train_params = {
    "n_round": 2400,  # number of training rounds
    "ft_round": 2400,  # number of training rounds (for full training)
    # "es_round": 100,  # early stopping rounds
    "verbose": 200,
}

cfg = {**cfg, "lgb_params": lgb_params, "lgb_train_params": lgb_train_params}
