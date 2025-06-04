# Path
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
EXT_DATA_DIR = Path("data/external")
ITR_DATA_DIR = Path("data/interim")
PROC_DATA_DIR = Path("data/processed")

META_DATA_DIR = RAW_DATA_DIR / "meta"
OMNI_DATA_DIR = RAW_DATA_DIR / "omni2"
GT_DATA_DIR = RAW_DATA_DIR / "sat_density"

# Base config
cfg = dict(
    seed=2024,
    remove_features=["fold", "file_id", "orbit_mean_density"],
    target_col="orbit_mean_density",
)


omni_mask_dict = {
    9.999: ["alpha_prot_ratio", "sigma_ratio"],
    9.9999: ["quasy_invariant"],
    99.0: ["id_for_sw_plasma_spacecraft", "id_for_imf_spacecraft"],
    99.9: ["magnetosonic_mach_number"],
    99.99: ["flow_pressure"],
    999.0: ["num_points_imf_averages", "num_points_plasma_averages"],
    999.9: [
        "sigma_n_n_cm3",
        "scalar_b_nt",
        "pc_index",
        "vector_b_magnitude_nt",
        "lat_angle_of_b_gse",
        "long_angle_of_b_gse",
        "f10.7_index",
        "alfen_mach_number",
        "bz_nt_gsm",
        "sigma_theta_v_degrees",
        "sigma_phi_v_degrees",
        "bx_nt_gse_gsm",
        "by_nt_gsm",
        "bz_nt_gse",
        "sw_plasma_flow_lat_angle",
        "sw_plasma_flow_long_angle",
        "by_nt_gse",
        "sw_proton_density_n_cm3",
        "rms_magnitude_nt",
        "rms_field_vector_nt",
        "rms_bx_gse_nt",
        "rms_by_gse_nt",
        "rms_bz_gse_nt",
    ],
    999.99: ["plasma_beta", "e_electric_field"],
    9999.0: ["ae_index_nt", "sigma_v_km_s", "sw_plasma_speed_km_s"],
    99999.0: ["au_index_nt", "al_index_nt"],
    99999.99: [
        "proton_flux_>60_mev",
        "proton_flux_>30_mev",
        "proton_flux_>10_mev",
        "proton_flux_>4_mev",
        "proton_flux_>2_mev",
    ],
    999999.99: ["proton_flux_>1_mev"],
    9999999.0: ["sigma_t_k", "sw_plasma_temperature_k"],
}
