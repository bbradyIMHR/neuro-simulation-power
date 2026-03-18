#!/usr/bin/env python3
"""
Monte Carlo power simulation for voxel-wise fMRI regression
===========================================================

PURPOSE
-------
This script estimates statistical power for a whole-brain voxel-wise regression,
such as correlating subject-level activation maps with a clinical score across participants.

At each voxel v, the second-level model is:

    Y_i(v) = intercept(v) + beta(v) * X_i + error_i(v)

where:
    - Y_i(v) is the activation value for subject i at voxel v
    - X_i is the clinical score for subject i
    - beta(v) is the true brain-behavior slope at voxel v


Pilot subject activation maps are loaded from a single MATLAB v7.3 .mat file
containing a masked voxel matrix, rather than from individual .nii/.nii.gz files.

Expected .mat contents
----------------------
For the attached file, the relevant datasets are:
    - data_matrix : shape (n_mask_voxels, n_subjects) or (n_subjects, n_mask_voxels)
    - mask_idx    : shape (X, Y, Z), binary mask used to create data_matrix
    - subjects    : subject IDs

The script validates that:
    1) the NIfTI brain mask matches the .mat mask exactly
    2) the number of masked voxels in the matrix matches the mask voxel count
    3) the number of subject IDs matches the matrix subject dimension
    4) the optional ROI mask matches the brain-mask space and overlaps the brain mask

SIMULATION LOGIC
----------------
For each Monte Carlo iteration, the script:
    1) bootstraps pilot subject maps to get realistic subject-level base maps
    2) simulates clinical scores X_i
    3) injects a true slope map beta(v) into the subject maps
    4) runs a voxel-wise regression across subjects
    5) computes a t-statistic at every voxel
    6) thresholds the t-map using a whole-brain FWE threshold based on max-|t| under the null
    7) applies a minimum cluster size k
    8) records whether the true signal region was detected

IMPORTANT
---------
This script estimates whole-brain FWE control using an empirical null distribution
of the maximum absolute t-statistic across the brain ("max-|t|").
That is valid for FWE control and is practical for power simulations.

It is NOT literally SPM's RFT/FWE threshold.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label, generate_binary_structure
from tqdm import tqdm


# =============================================================================
# USER CONFIGURATION
# =============================================================================

CONFIG = {
    # -------------------------------------------------------------------------
    # INPUT DATA
    # -------------------------------------------------------------------------

    # str
    # MATLAB v7.3 file containing all pilot subject maps in one masked matrix
    "pilot_mat_path": "/Users/brendanbrady/Documents/Funding_Opportunities/2026_Advancing 2S:LGBTQI/power_calculations/H2/n39_additionalsubjects/Minoritystress_n39_subject_voxel_matrix.mat",

    # str
    # Dataset name inside the .mat file containing the masked voxel matrix
    "pilot_mat_data_key": "data_matrix",

    # str
    # Dataset name inside the .mat file containing subject IDs
    "pilot_mat_subjects_key": "subjects",

    # Optional[str]
    # Dataset name inside the .mat file containing the binary mask used
    # to generate the matrix. Strongly recommended for validation.
    "pilot_mat_mask_key": "mask_idx",

    # str
    # Orientation of the matrix in the .mat file:
    #   "voxels_by_subjects" -> shape (n_voxels, n_subjects)
    #   "subjects_by_voxels" -> shape (n_subjects, n_voxels)
    "pilot_mat_orientation": "voxels_by_subjects",

    # str
    # Path to a binary whole-brain mask NIfTI.
    # This MUST match the .mat mask exactly.
    "brain_mask_path": "/Users/brendanbrady/Documents/Funding_Opportunities/2026_Advancing 2S:LGBTQI/power_calculations/H2/n39_additionalsubjects/Minoritystress_n39_wholesample_intersection_mask.nii",

    # tuple/list of length 3 or None
    # The .mat mask needs axis reordering before comparison/use.
    "pilot_mat_mask_transpose_axes": [2, 1, 0],

    # Optional[str]
    # Optional ROI/signal mask NIfTI.
    # Required if signal_mode == "roi_mask".
    "signal_roi_mask_path": None,

    # Optional[str]
    # Optional CSV containing pilot clinical scores.
    # Must have columns:
    #   subject_id, clinical_score
    "pilot_scores_csv": "/Users/brendanbrady/Documents/Funding_Opportunities/2026_Advancing 2S:LGBTQI/power_calculations/H2/clinical_scores.csv",

    # -------------------------------------------------------------------------
    # SIMULATION DESIGN
    # -------------------------------------------------------------------------

    # list[int]
    # Candidate sample sizes to evaluate.
    "candidate_sample_sizes": [160],

    # int
    # Number of Monte Carlo datasets used to estimate POWER for each N.
    "n_mc_power": 2000,

    # int
    # Number of Monte Carlo NULL datasets used to estimate the whole-brain FWE threshold.
    "n_mc_null": 500,

    # float
    # Family-wise error rate.
    "alpha_fwe": 0.05,

    # int
    # Minimum cluster extent.
    "cluster_extent_k": 10,

    # bool
    # If True, use a two-sided threshold based on |t|.
    # If False, use one-sided threshold on positive t only.
    "two_sided_test": True,

    # -------------------------------------------------------------------------
    # CLINICAL SCORE SIMULATION
    # -------------------------------------------------------------------------

    # str
    # "normal"          -> simulate X ~ Normal(mean, sd)
    # "bootstrap_pilot" -> sample scores with replacement from pilot clinical scores
    "clinical_score_mode": "bootstrap_pilot",

    # float
    # Mean of clinical score distribution when clinical_score_mode == "normal"
    "clinical_score_mean": 0.0,

    # float
    # SD of clinical score distribution when clinical_score_mode == "normal"
    "clinical_score_sd": 1.0,

    # bool
    # Whether to z-score the simulated clinical scores within each simulated dataset.
    "zscore_simulated_scores": True,

    # -------------------------------------------------------------------------
    # BASE MAP CONSTRUCTION
    # -------------------------------------------------------------------------

    # bool
    # If True and pilot scores are available:
    #   regress the pilot clinical scores out of the pilot maps first
    #   so that bootstrapped base maps are closer to "noise/background"
    "residualize_pilot_maps": True,

    # -------------------------------------------------------------------------
    # TRUE SIGNAL SPECIFICATION
    # -------------------------------------------------------------------------

    # str
    # "sphere"   -> create a spherical signal region
    # "roi_mask" -> use the provided ROI mask
    "signal_mode": "sphere",

    # list[int] of length 3
    # Center voxel for spherical signal region, if signal_mode == "sphere"
    "signal_center_voxel": [32, 40, 28],

    # int or float
    # Radius of spherical signal region in voxel units
    "signal_radius_vox": 5,

    # float
    # Optional Gaussian smoothing sigma (in voxel units) applied to the beta map.
    "signal_smoothing_sigma_vox": 1.0,

    # float
    # Fixed slope amplitude to inject if not estimating from pilot ROI.
    "fixed_beta_amplitude": 0.55,
    # CORRESPONDING r = beta/voxelwiseSD = beta/2 = 0.21
    #beta = r * SD_Y/SD_X
    # SD_X = SD of predictor (i.e., clinical score) - these are standardized, thus SD_X = 1
    # SD_Y = SD of activation of the same voxel across subjs = 2 (mean) or 1.8 (median)

    # bool
    # If True:
    #   estimate the average pilot slope inside the ROI and use that instead
    #   of fixed_beta_amplitude
    #
    # Requires:
    #   - pilot_scores_csv
    #   - signal_roi_mask_path
    "estimate_beta_from_pilot_roi": False,

    # float
    # Multiplier applied to the pilot-derived beta estimate.
    "pilot_beta_shrink_factor": 1,

    # -------------------------------------------------------------------------
    # OPTIONAL NUISANCE COVARIATES
    # -------------------------------------------------------------------------

    # list[dict]
    # Example:
    # "covariates": [
    #     {"name": "age", "mode": "normal", "mean": 40, "sd": 12},
    # ]
    "covariates":[
    {"name": "cov1", "mode": "normal", "mean": 0, "sd": 1},
    {"name": "cov2", "mode": "normal", "mean": 0, "sd": 1},
    {"name": "cov3", "mode": "normal", "mean": 0, "sd": 1},
    {"name": "cov4", "mode": "normal", "mean": 0, "sd": 1},
    {"name": "cov5", "mode": "normal", "mean": 0, "sd": 1},
    ],

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    "output_dir": "./fmri_voxelwise_regression_power_output",
    "random_seed": 12345,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LoadedData:
    """
    Container for loaded and mask-vectorized data.

    Attributes
    ----------
    pilot_maps_2d : np.ndarray, shape (n_pilot_subjects, n_mask_voxels), dtype float32
        Subject-level pilot maps already aligned to the brain mask.

    affine : np.ndarray, shape (4, 4), dtype float64
        NIfTI affine matrix for saving output images.

    header : nib.Nifti1Header
        NIfTI header copied from the brain mask image.

    brain_mask_3d : np.ndarray, shape (X, Y, Z), dtype bool
        Whole-brain binary analysis mask.

    mask_indices : np.ndarray, shape (n_mask_voxels,), dtype int64
        Flat indices of mask voxels in the full 3D volume.

    map_paths : list[pathlib.Path]
        Placeholder subject identifiers stored as Path objects for compatibility.

    pilot_scores : Optional[np.ndarray], shape (n_pilot_subjects,), dtype float64
        Pilot clinical scores aligned to the map order, or None.

    signal_roi_3d : Optional[np.ndarray], shape (X, Y, Z), dtype bool
        Optional ROI mask for defining true signal or estimating pilot slope.

    subject_ids : list[str]
        Subject IDs loaded from the .mat file.
    """
    pilot_maps_2d: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header
    brain_mask_3d: np.ndarray
    mask_indices: np.ndarray
    map_paths: List[Path]
    pilot_scores: Optional[np.ndarray]
    signal_roi_3d: Optional[np.ndarray]
    subject_ids: List[str]


# =============================================================================
# FILE / IO HELPERS
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """
    Load a NIfTI image.
    """
    return nib.load(str(path))


def save_nifti_from_masked_vector(
    masked_vec: np.ndarray,
    mask_3d: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    out_path: str | Path,
) -> None:
    """
    Save a 1D masked vector back to a full 3D NIfTI volume.
    """
    vol = np.zeros(mask_3d.shape, dtype=np.float32)
    vol[mask_3d] = masked_vec.astype(np.float32)
    img = nib.Nifti1Image(vol, affine=affine, header=header)
    nib.save(img, str(out_path))


# =============================================================================
# MATLAB .MAT HELPERS
# =============================================================================

def validate_config(config: Dict) -> None:
    """
    Validate high-level config consistency before running.
    """
    required_keys = [
        "pilot_mat_path",
        "pilot_mat_data_key",
        "pilot_mat_subjects_key",
        "pilot_mat_orientation",
        "brain_mask_path",
        "signal_mode",
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required CONFIG keys: {missing}")

    if config["pilot_mat_orientation"] not in {"voxels_by_subjects", "subjects_by_voxels"}:
        raise ValueError(
            "pilot_mat_orientation must be either 'voxels_by_subjects' or 'subjects_by_voxels'."
        )

    if config["signal_mode"] not in {"sphere", "roi_mask"}:
        raise ValueError("signal_mode must be either 'sphere' or 'roi_mask'.")

    if config["signal_mode"] == "roi_mask" and config["signal_roi_mask_path"] is None:
        raise ValueError("signal_mode='roi_mask' requires signal_roi_mask_path.")

    if config["estimate_beta_from_pilot_roi"]:
        if config["pilot_scores_csv"] is None:
            raise ValueError("estimate_beta_from_pilot_roi=True requires pilot_scores_csv.")
        if config["signal_roi_mask_path"] is None:
            raise ValueError("estimate_beta_from_pilot_roi=True requires signal_roi_mask_path.")

    if config["clinical_score_mode"] == "bootstrap_pilot" and config["pilot_scores_csv"] is None:
        raise ValueError("clinical_score_mode='bootstrap_pilot' requires pilot_scores_csv.")

    if not Path(config["pilot_mat_path"]).exists():
        raise FileNotFoundError(f"pilot_mat_path does not exist: {config['pilot_mat_path']}")

    if not Path(config["brain_mask_path"]).exists():
        raise FileNotFoundError(f"brain_mask_path does not exist: {config['brain_mask_path']}")

    if config["signal_roi_mask_path"] is not None and not Path(config["signal_roi_mask_path"]).exists():
        raise FileNotFoundError(
            f"signal_roi_mask_path does not exist: {config['signal_roi_mask_path']}"
        )

    if config["pilot_scores_csv"] is not None and not Path(config["pilot_scores_csv"]).exists():
        raise FileNotFoundError(f"pilot_scores_csv does not exist: {config['pilot_scores_csv']}")

    if not isinstance(config["candidate_sample_sizes"], list) or len(config["candidate_sample_sizes"]) == 0:
        raise ValueError("candidate_sample_sizes must be a non-empty list of integers.")

    if any(int(n) <= 1 for n in config["candidate_sample_sizes"]):
        raise ValueError("All candidate sample sizes must be > 1.")

    if config["n_mc_power"] <= 0 or config["n_mc_null"] <= 0:
        raise ValueError("n_mc_power and n_mc_null must both be > 0.")

    if not (0 < config["alpha_fwe"] < 1):
        raise ValueError("alpha_fwe must be between 0 and 1.")

    if int(config["cluster_extent_k"]) < 1:
        raise ValueError("cluster_extent_k must be >= 1.")


def read_matlab_v73_subject_ids(mat_file: h5py.File, dataset_name: str) -> List[str]:
    """
    Read subject IDs from a MATLAB v7.3 HDF5 dataset containing cell-array strings.

    This supports the common MATLAB v7.3 representation where a cell array of strings
    is stored as object references to uint16 character arrays.

    Parameters
    ----------
    mat_file : h5py.File
    dataset_name : str

    Returns
    -------
    list[str]
    """
    if dataset_name not in mat_file:
        raise KeyError(f"Dataset '{dataset_name}' not found in .mat file.")

    ds = mat_file[dataset_name]

    if ds.dtype.kind != "O":
        raise ValueError(
            f"Expected '{dataset_name}' to be an object-reference dataset for MATLAB strings, "
            f"but got dtype {ds.dtype}"
        )

    refs = np.array(ds)

    subject_ids = []
    for idx in np.ndindex(refs.shape):
        ref = refs[idx]
        if ref == 0:
            continue

        char_array = np.array(mat_file[ref])

        if np.issubdtype(char_array.dtype, np.integer):
            sid = "".join(chr(int(c)) for c in char_array.flatten())
        else:
            sid = "".join(str(c) for c in char_array.flatten())

        sid = sid.strip()
        if sid != "":
            subject_ids.append(sid)

    if len(subject_ids) == 0:
        raise ValueError(f"No subject IDs could be decoded from dataset '{dataset_name}'.")

    return subject_ids


def load_mat_mask(
    mat_path: str | Path,
    mask_key: str,
    transpose_axes: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Load a binary mask from a MATLAB v7.3 .mat file.

    Parameters
    ----------
    mat_path : str or Path
    mask_key : str
    transpose_axes : Optional[tuple[int, int, int]]
        Optional axis permutation applied after loading.
        For your file, use (2, 1, 0).

    Returns
    -------
    np.ndarray, shape (X, Y, Z), dtype bool
    """
    with h5py.File(mat_path, "r") as f:
        if mask_key not in f:
            raise KeyError(f"Dataset '{mask_key}' not found in .mat file.")
        mask = np.array(f[mask_key])

    if mask.ndim != 3:
        raise ValueError(f"MATLAB mask dataset '{mask_key}' must be 3D, but got shape {mask.shape}")

    mask = mask.astype(bool)

    if transpose_axes is not None:
        mask = np.transpose(mask, axes=tuple(transpose_axes))

    return mask


def load_pilot_maps_from_mat(
    mat_path: str | Path,
    data_key: str,
    subjects_key: str,
    orientation: str = "voxels_by_subjects",
) -> Tuple[np.ndarray, List[str]]:
    """
    Load pilot subject maps from a MATLAB v7.3 .mat file.

    Parameters
    ----------
    mat_path : str or Path
        Path to .mat file.

    data_key : str
        Name of dataset containing the masked voxel matrix.

    subjects_key : str
        Name of dataset containing subject IDs.

    orientation : str
        Either:
            "voxels_by_subjects"  -> matrix shape (n_voxels, n_subjects)
            "subjects_by_voxels"  -> matrix shape (n_subjects, n_voxels)

    Returns
    -------
    pilot_maps_2d : np.ndarray, shape (n_subjects, n_voxels), dtype float32
    subject_ids : list[str]
    """
    with h5py.File(mat_path, "r") as f:
        if data_key not in f:
            raise KeyError(f"Dataset '{data_key}' not found in .mat file.")

        data_matrix = np.array(f[data_key], dtype=np.float32)
        subject_ids = read_matlab_v73_subject_ids(f, subjects_key)

    if data_matrix.ndim != 2:
        raise ValueError(f"'{data_key}' must be a 2D matrix, but got shape {data_matrix.shape}")

    if orientation == "voxels_by_subjects":
        pilot_maps_2d = data_matrix.T
    elif orientation == "subjects_by_voxels":
        pilot_maps_2d = data_matrix
    else:
        raise ValueError(f"Unsupported pilot_mat_orientation: {orientation}")

    if pilot_maps_2d.shape[0] != len(subject_ids):
        raise ValueError(
            "Number of subjects in matrix does not match number of subject IDs.\n"
            f"pilot_maps_2d.shape[0] = {pilot_maps_2d.shape[0]}\n"
            f"len(subject_ids)       = {len(subject_ids)}"
        )

    if pilot_maps_2d.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 pilot subjects, but got {pilot_maps_2d.shape[0]}"
        )

    if not np.all(np.isfinite(pilot_maps_2d)):
        raise ValueError("pilot_maps_2d contains NaN or infinite values.")

    return pilot_maps_2d.astype(np.float32), subject_ids


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(config: Dict) -> LoadedData:
    """
    Load pilot maps from a MATLAB .mat file, brain mask from NIfTI,
    optional clinical scores, and optional ROI mask.

    Returns
    -------
    LoadedData
    """
    # -------------------------------------------------------------------------
    # Load brain mask NIfTI
    # -------------------------------------------------------------------------
    brain_mask_img = load_nifti(config["brain_mask_path"])
    brain_mask_data = brain_mask_img.get_fdata()

    if brain_mask_data.ndim != 3:
        raise ValueError(
            f"brain_mask_path must point to a 3D mask image, but got shape {brain_mask_data.shape}"
        )

    brain_mask_3d = brain_mask_data > 0

    if brain_mask_3d.sum() == 0:
        raise ValueError("Supplied NIfTI brain mask is empty.")

    affine = brain_mask_img.affine
    header = brain_mask_img.header.copy()
    mask_indices = np.flatnonzero(brain_mask_3d.ravel())
    n_mask_vox = int(brain_mask_3d.sum())

    # -------------------------------------------------------------------------
    # Validate .mat mask against NIfTI mask
    # -------------------------------------------------------------------------
    if config.get("pilot_mat_mask_key") is not None:
        mat_mask_3d = load_mat_mask(
            mat_path=config["pilot_mat_path"],
            mask_key=config["pilot_mat_mask_key"],
            transpose_axes=tuple(config["pilot_mat_mask_transpose_axes"])
            if config.get("pilot_mat_mask_transpose_axes") is not None
            else None,
        )

        if mat_mask_3d.shape != brain_mask_3d.shape:
            raise ValueError(
                "Shape mismatch between .mat mask and NIfTI brain mask.\n"
                f".mat mask shape  = {mat_mask_3d.shape}\n"
                f"NIfTI mask shape = {brain_mask_3d.shape}"
            )

        if not np.array_equal(mat_mask_3d, brain_mask_3d):
            n_diff = int(np.sum(mat_mask_3d != brain_mask_3d))
            raise ValueError(
                "The mask inside the .mat file does not exactly match the supplied NIfTI brain mask.\n"
                f"Number of differing voxels: {n_diff}\n"
                "This would invalidate voxel ordering between the matrix and mask."
            )

    # -------------------------------------------------------------------------
    # Load pilot subject maps from .mat
    # -------------------------------------------------------------------------
    pilot_maps_2d, subject_ids = load_pilot_maps_from_mat(
        mat_path=config["pilot_mat_path"],
        data_key=config["pilot_mat_data_key"],
        subjects_key=config["pilot_mat_subjects_key"],
        orientation=config["pilot_mat_orientation"],
    )

    # Validate matrix voxel count against brain mask
    if pilot_maps_2d.shape[1] != n_mask_vox:
        raise ValueError(
            "Voxel count mismatch between .mat data_matrix and brain mask.\n"
            f"pilot_maps_2d.shape[1] = {pilot_maps_2d.shape[1]}\n"
            f"brain_mask voxels      = {n_mask_vox}"
        )

    # Placeholder list for compatibility with existing code structure
    map_paths = [Path(sid) for sid in subject_ids]

    # -------------------------------------------------------------------------
    # Load pilot clinical scores if provided
    # -------------------------------------------------------------------------
    pilot_scores = None

    if config["pilot_scores_csv"] is not None:
        scores_df = pd.read_csv(config["pilot_scores_csv"], dtype={"subject_id": str})
        scores_df["subject_id"] = scores_df["subject_id"].str.zfill(2)


        required_cols = {"subject_id", "clinical_score"}
        if not required_cols.issubset(scores_df.columns):
            raise ValueError(
                f"pilot_scores_csv must contain columns {required_cols}, "
                f"but got {list(scores_df.columns)}"
            )

        if scores_df["subject_id"].duplicated().any():
            dup_ids = scores_df.loc[scores_df["subject_id"].duplicated(), "subject_id"].tolist()
            raise ValueError(
                f"pilot_scores_csv contains duplicated subject_id values. Examples: {dup_ids[:5]}"
            )

        score_map = dict(
            zip(
                scores_df["subject_id"].astype(str),
                scores_df["clinical_score"].astype(float),
            )
        )

        missing = [sid for sid in subject_ids if sid not in score_map]
        if missing:
            raise ValueError(
                f"{len(missing)} subject IDs from the .mat file were not found in the score CSV. "
                f"Examples: {missing[:5]}"
            )

        pilot_scores = np.array([score_map[sid] for sid in subject_ids], dtype=float)

        if not np.all(np.isfinite(pilot_scores)):
            raise ValueError("pilot_scores contains NaN or infinite values.")

        if len(pilot_scores) != pilot_maps_2d.shape[0]:
            raise ValueError(
                "Number of aligned pilot scores does not match number of subjects.\n"
                f"len(pilot_scores)     = {len(pilot_scores)}\n"
                f"pilot_maps_2d.shape[0]= {pilot_maps_2d.shape[0]}"
            )

    # -------------------------------------------------------------------------
    # Load optional signal ROI mask
    # -------------------------------------------------------------------------
    signal_roi_3d = None

    if config["signal_roi_mask_path"] is not None:
        roi_img = load_nifti(config["signal_roi_mask_path"])
        roi_data = roi_img.get_fdata()

        if roi_data.ndim != 3:
            raise ValueError(
                f"signal_roi_mask_path must point to a 3D mask image, but got shape {roi_data.shape}"
            )

        roi = roi_data > 0

        if roi.shape != brain_mask_3d.shape:
            raise ValueError(
                "Signal ROI mask shape does not match brain mask.\n"
                f"ROI shape        = {roi.shape}\n"
                f"Brain mask shape = {brain_mask_3d.shape}"
            )

        signal_roi_3d = roi & brain_mask_3d

        if signal_roi_3d.sum() == 0:
            raise ValueError("Signal ROI mask is empty after intersecting with the brain mask.")

    return LoadedData(
        pilot_maps_2d=pilot_maps_2d,
        affine=affine,
        header=header,
        brain_mask_3d=brain_mask_3d,
        mask_indices=mask_indices,
        map_paths=map_paths,
        pilot_scores=pilot_scores,
        signal_roi_3d=signal_roi_3d,
        subject_ids=subject_ids,
    )


# =============================================================================
# BASIC MATH / ARRAY HELPERS
# =============================================================================

def zscore_1d(x: np.ndarray) -> np.ndarray:
    """
    Z-score a 1D vector using sample SD (ddof=1).
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(f"zscore_1d expects a 1D array, but got shape {x.shape}")

    if not np.all(np.isfinite(x)):
        raise ValueError("Cannot z-score vector containing NaN or infinite values.")

    mean_x = float(np.mean(x))
    sd_x = float(np.std(x, ddof=1))

    if not np.isfinite(mean_x) or not np.isfinite(sd_x):
        raise ValueError("Mean or SD became non-finite during z-scoring.")

    if sd_x <= 0:
        raise ValueError("Cannot z-score a constant vector.")

    z = (x - mean_x) / sd_x

    if not np.all(np.isfinite(z)):
        raise ValueError("zscore_1d produced NaN or infinite values.")

    return z


def make_spherical_mask(
    shape: Tuple[int, int, int],
    center: Tuple[int, int, int],
    radius: float
) -> np.ndarray:
    """
    Create a spherical boolean mask in voxel coordinates.
    """
    if len(shape) != 3:
        raise ValueError(f"Expected 3D shape, got {shape}")

    if len(center) != 3:
        raise ValueError(f"Expected 3D center, got {center}")

    zz, yy, xx = np.indices(shape)
    cx, cy, cz = center

    if not (0 <= cx < shape[0] and 0 <= cy < shape[1] and 0 <= cz < shape[2]):
        raise ValueError(
            f"signal_center_voxel {center} is outside image bounds {shape}"
        )

    dist2 = (zz - cx) ** 2 + (yy - cy) ** 2 + (xx - cz) ** 2
    return dist2 <= radius ** 2


def masked_vec_to_3d(vec: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    """
    Convert a masked 1D vector back to a full 3D volume.
    """
    if vec.ndim != 1:
        raise ValueError(f"masked_vec_to_3d expects a 1D vector, got shape {vec.shape}")

    if vec.size != int(mask_3d.sum()):
        raise ValueError(
            "Vector length does not match number of True voxels in mask.\n"
            f"vec.size         = {vec.size}\n"
            f"mask_3d.sum()    = {int(mask_3d.sum())}"
        )

    out = np.zeros(mask_3d.shape, dtype=vec.dtype)
    out[mask_3d] = vec
    return out


# =============================================================================
# SIGNAL / BETA MAP CONSTRUCTION
# =============================================================================



def estimate_peak_voxel_beta_from_mni(
    pilot_maps_2d: np.ndarray,
    pilot_scores: np.ndarray,
    brain_mask_3d: np.ndarray,
    affine: np.ndarray,
    mni_coord: tuple[float, float, float],
) -> dict:
    """
    Estimate the pilot regression slope at a reported MNI peak voxel.

    Returns beta for:
        voxel_value ~ intercept + zscored_clinical_score

    So the returned slope is directly interpretable as:
        change in voxel intensity per +1 SD increase in clinical score
    """
    Y = np.asarray(pilot_maps_2d, dtype=np.float64)
    x = np.asarray(pilot_scores, dtype=np.float64)

    if Y.ndim != 2:
        raise ValueError(f"pilot_maps_2d must be 2D, got shape {Y.shape}")
    if x.ndim != 1 or len(x) != Y.shape[0]:
        raise ValueError(
            f"pilot_scores must have length {Y.shape[0]}, got shape {x.shape}"
        )

    # z-score predictor so beta matches your simulation meaning
    xz = (x - x.mean()) / x.std(ddof=1)

    # Convert MNI -> voxel coordinates
    voxel_float = nib.affines.apply_affine(np.linalg.inv(affine), np.array(mni_coord))
    voxel_ijk = np.round(voxel_float).astype(int)

    if np.any(voxel_ijk < 0) or np.any(voxel_ijk >= np.array(brain_mask_3d.shape)):
        raise ValueError(
            f"MNI coordinate {mni_coord} maps outside image bounds: voxel {tuple(voxel_ijk)}"
        )

    if not brain_mask_3d[tuple(voxel_ijk)]:
        raise ValueError(
            f"MNI coordinate {mni_coord} maps to voxel {tuple(voxel_ijk)}, "
            "which is outside the analysis mask."
        )

    # Find this voxel's column in the masked matrix
    flat_voxel_idx = np.ravel_multi_index(tuple(voxel_ijk), brain_mask_3d.shape)
    mask_flat_indices = np.flatnonzero(brain_mask_3d.ravel())
    col_idx = np.where(mask_flat_indices == flat_voxel_idx)[0]

    if len(col_idx) != 1:
        raise ValueError("Could not uniquely map voxel to masked matrix column.")

    col_idx = int(col_idx[0])

    # Subject-level voxel values at the peak
    y = Y[:, col_idx]

    # OLS: y = intercept + beta * xz
    X = np.column_stack([np.ones(len(xz)), xz])
    betas = np.linalg.inv(X.T @ X) @ X.T @ y
    fitted = X @ betas
    resid = y - fitted

    df = len(xz) - 2
    rss = np.sum(resid ** 2)
    sigma2 = rss / df
    se_beta = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    t_beta = betas[1] / se_beta
    r = np.corrcoef(xz, y)[0, 1]

    return {
        "mni_coord": tuple(mni_coord),
        "voxel_ijk": tuple(voxel_ijk.tolist()),
        "matrix_column_index": col_idx,
        "beta_zscoreX": float(betas[1]),
        "intercept": float(betas[0]),
        "se_beta": float(se_beta),
        "t_stat": float(t_beta),
        "correlation_r": float(r),
        "voxel_sd": float(np.std(y, ddof=1)),
        "voxel_values": y,
    }


def estimate_mean_pilot_slope_in_roi(
    pilot_maps_2d: np.ndarray,
    pilot_scores: np.ndarray,
    roi_3d: np.ndarray,
    brain_mask_3d: np.ndarray,
) -> float:
    """
    Estimate the mean pilot regression slope inside an ROI.

    Model in pilot:
        Y(v) = intercept + beta(v) * zscore(X) + error

    This function returns the average slope across ROI voxels.

    Parameters
    ----------
    pilot_maps_2d : np.ndarray, shape (n_pilot_subjects, n_mask_voxels)
    pilot_scores : np.ndarray, shape (n_pilot_subjects,)
    roi_3d : np.ndarray, shape (X, Y, Z), dtype bool
    brain_mask_3d : np.ndarray, shape (X, Y, Z), dtype bool

    Returns
    -------
    float
        Mean slope across ROI voxels.
    """
    Y = np.asarray(pilot_maps_2d, dtype=np.float64)
    x = zscore_1d(pilot_scores).astype(np.float64)

    if Y.ndim != 2:
        raise ValueError(f"pilot_maps_2d must be 2D, but got shape {Y.shape}")
    if x.ndim != 1:
        raise ValueError(f"pilot_scores must be 1D, but got shape {x.shape}")
    if Y.shape[0] != len(x):
        raise ValueError(
            "Number of subjects in pilot_maps_2d does not match length of pilot_scores.\n"
            f"pilot_maps_2d.shape[0] = {Y.shape[0]}\n"
            f"len(pilot_scores)      = {len(x)}"
        )
    if roi_3d.shape != brain_mask_3d.shape:
        raise ValueError(
            f"roi_3d shape {roi_3d.shape} does not match brain_mask_3d shape {brain_mask_3d.shape}"
        )

    if not np.all(np.isfinite(Y)):
        raise ValueError("pilot_maps_2d contains NaN or infinite values.")
    if not np.all(np.isfinite(x)):
        raise ValueError("pilot_scores contain NaN or infinite values.")

    roi_masked = roi_3d[brain_mask_3d]

    if roi_masked.sum() == 0:
        raise ValueError("ROI is empty within the analysis mask.")
    if Y.shape[1] != roi_masked.size:
        raise ValueError(
            "Voxel dimension of pilot_maps_2d does not match masked brain size.\n"
            f"pilot_maps_2d.shape[1] = {Y.shape[1]}\n"
            f"masked brain voxels    = {roi_masked.size}"
        )

    y = Y[:, roi_masked]

    X = np.column_stack([np.ones(len(x), dtype=np.float64), x])

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Could not invert X'X while estimating pilot slope in ROI."
        ) from e

    betas = XtX_inv @ X.T @ y
    slopes = betas[1]

    if not np.all(np.isfinite(slopes)):
        raise ValueError("Estimated pilot ROI slopes contain NaN or infinite values.")

    return float(np.mean(slopes))


def build_signal_region_and_beta_map(
    data: LoadedData,
    config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the true signal region and the true slope map beta(v).
    """
    if config["signal_mode"] == "sphere":
        sphere = make_spherical_mask(
            shape=data.brain_mask_3d.shape,
            center=tuple(config["signal_center_voxel"]),
            radius=config["signal_radius_vox"],
        )
        signal_region_3d = sphere & data.brain_mask_3d

    elif config["signal_mode"] == "roi_mask":
        if data.signal_roi_3d is None:
            raise ValueError("signal_mode='roi_mask' requires signal_roi_mask_path.")
        signal_region_3d = data.signal_roi_3d.copy()

    else:
        raise ValueError(f"Unsupported signal_mode: {config['signal_mode']}")

    if signal_region_3d.sum() == 0:
        raise ValueError("Signal region is empty after intersecting with the brain mask.")

    beta_amp = config["fixed_beta_amplitude"]

    if config["estimate_beta_from_pilot_roi"]:
        if data.pilot_scores is None:
            raise ValueError("estimate_beta_from_pilot_roi=True requires pilot_scores_csv.")
        if data.signal_roi_3d is None:
            raise ValueError("estimate_beta_from_pilot_roi=True requires signal_roi_mask_path.")

        beta_amp = estimate_mean_pilot_slope_in_roi(
            pilot_maps_2d=data.pilot_maps_2d,
            pilot_scores=data.pilot_scores,
            roi_3d=data.signal_roi_3d,
            brain_mask_3d=data.brain_mask_3d,
        )

        beta_amp *= config["pilot_beta_shrink_factor"]

    beta_3d = np.zeros(data.brain_mask_3d.shape, dtype=np.float32)
    beta_3d[signal_region_3d] = float(beta_amp)

    sigma = float(config["signal_smoothing_sigma_vox"])
    if sigma > 0:
        beta_3d = gaussian_filter(beta_3d, sigma=sigma)
        beta_3d *= data.brain_mask_3d

    beta_masked_1d = beta_3d[data.brain_mask_3d].astype(np.float32)

    if not np.all(np.isfinite(beta_masked_1d)):
        raise ValueError("beta_map contains NaN or infinite values after construction.")

    return signal_region_3d, beta_masked_1d


# =============================================================================
# PILOT MAP RESIDUALIZATION
# =============================================================================

def residualize_pilot_maps_against_scores(
    pilot_maps_2d: np.ndarray,
    pilot_scores: np.ndarray,
) -> np.ndarray:
    """
    Regress the pilot clinical score out of each voxel across pilot subjects.

    Pilot model:
        Y(v) = intercept + b(v) * X + e(v)

    This function returns:
        intercept + residuals

    so that the pilot brain-behavior association is approximately removed, but
    the voxelwise mean structure is preserved.

    Parameters
    ----------
    pilot_maps_2d : np.ndarray, shape (n_pilot_subjects, n_mask_voxels)
    pilot_scores : np.ndarray, shape (n_pilot_subjects,)

    Returns
    -------
    np.ndarray, shape (n_pilot_subjects, n_mask_voxels), dtype float32
    """
    Y = np.asarray(pilot_maps_2d, dtype=np.float64)
    x = zscore_1d(pilot_scores).astype(np.float64)

    if Y.ndim != 2:
        raise ValueError(f"pilot_maps_2d must be 2D, but got shape {Y.shape}")
    if x.ndim != 1:
        raise ValueError(f"pilot_scores must be 1D, but got shape {x.shape}")
    if Y.shape[0] != len(x):
        raise ValueError(
            "Number of subjects in pilot_maps_2d does not match length of pilot_scores.\n"
            f"pilot_maps_2d.shape[0] = {Y.shape[0]}\n"
            f"len(pilot_scores)      = {len(x)}"
        )

    if not np.all(np.isfinite(Y)):
        raise ValueError("pilot_maps_2d contains NaN or infinite values before residualization.")
    if not np.all(np.isfinite(x)):
        raise ValueError("pilot_scores contain NaN or infinite values before residualization.")

    X = np.column_stack([np.ones(len(x), dtype=np.float64), x])

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Could not invert X'X during pilot-map residualization."
        ) from e

    betas = XtX_inv @ X.T @ Y
    fitted = X @ betas
    residuals = Y - fitted

    intercept_only = np.outer(np.ones(len(x), dtype=np.float64), betas[0])
    out = intercept_only + residuals

    if not np.all(np.isfinite(out)):
        raise ValueError("Residualized pilot maps contain NaN or infinite values.")

    return out.astype(np.float32)


# =============================================================================
# SIMULATION HELPERS
# =============================================================================

def simulate_covariates(
    n: int,
    cov_config: List[Dict],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate nuisance covariates.
    """
    cols = []

    for c in cov_config:
        mode = c["mode"]

        if mode == "normal":
            vals = rng.normal(c["mean"], c["sd"], size=n)
        else:
            raise ValueError(f"Unsupported covariate mode: {mode}")

        cols.append(vals)

    if not cols:
        return np.empty((n, 0), dtype=np.float64)

    out = np.column_stack(cols)

    if not np.all(np.isfinite(out)):
        raise ValueError("Simulated covariates contain NaN or infinite values.")

    return out


def simulate_clinical_scores(
    n: int,
    config: Dict,
    rng: np.random.Generator,
    pilot_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate the primary clinical predictor X.
    """
    mode = config["clinical_score_mode"]

    if mode == "normal":
        x = rng.normal(
            config["clinical_score_mean"],
            config["clinical_score_sd"],
            size=n
        )

    elif mode == "bootstrap_pilot":
        if pilot_scores is None:
            raise ValueError("clinical_score_mode='bootstrap_pilot' requires pilot_scores_csv.")
        idx = rng.integers(0, len(pilot_scores), size=n)
        x = pilot_scores[idx].copy()

    else:
        raise ValueError(f"Unsupported clinical_score_mode: {mode}")

    if config["zscore_simulated_scores"]:
        x = zscore_1d(x)

    if not np.all(np.isfinite(x)):
        raise ValueError("Simulated clinical scores contain NaN or infinite values.")

    return x.astype(float)


def bootstrap_base_maps(
    pilot_maps_2d: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Bootstrap subject-level base maps from the pilot maps.
    """
    idx = rng.integers(0, pilot_maps_2d.shape[0], size=n)
    out = pilot_maps_2d[idx].copy()

    if not np.all(np.isfinite(out)):
        raise ValueError("Bootstrapped base maps contain NaN or infinite values.")

    return out


# =============================================================================
# VOXEL-WISE REGRESSION
# =============================================================================

def fit_voxelwise_ols_tmap(
    Y: np.ndarray,
    x: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit voxel-wise OLS regression across subjects and compute the t-statistic
    for the PRIMARY predictor x at every voxel.

    MODEL
    -----
    At each voxel v:

        Y(v) = intercept + b1(v) * x + nuisance_covariates + error

    Parameters
    ----------
    Y : np.ndarray, shape (n_subjects, n_mask_voxels)
        Subject maps for one simulated dataset.

    x : np.ndarray, shape (n_subjects,)
        Primary predictor (clinical score).

    covariates : Optional[np.ndarray], shape (n_subjects, n_covariates)
        Optional nuisance covariates.

    Returns
    -------
    slope_map : np.ndarray, shape (n_mask_voxels,), dtype float64
    se_map : np.ndarray, shape (n_mask_voxels,), dtype float64
    t_map : np.ndarray, shape (n_mask_voxels,), dtype float64
    """
    Y = np.asarray(Y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (subjects x voxels), but got shape {Y.shape}")

    n = Y.shape[0]

    if x.ndim != 1 or len(x) != n:
        raise ValueError(
            f"x must be a 1D vector of length {n}, but got shape {x.shape}"
        )

    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains NaN or infinite values.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values.")

    if covariates is None:
        X = np.column_stack([np.ones(n, dtype=np.float64), x])
        predictor_idx = 1
    else:
        covariates = np.asarray(covariates, dtype=np.float64)

        if covariates.ndim != 2 or covariates.shape[0] != n:
            raise ValueError(
                f"covariates must have shape (n_subjects, n_covariates), got {covariates.shape}"
            )
        if not np.all(np.isfinite(covariates)):
            raise ValueError("covariates contain NaN or infinite values.")

        X = np.column_stack([np.ones(n, dtype=np.float64), x, covariates])
        predictor_idx = 1

    p = X.shape[1]

    if n <= p:
        raise ValueError(f"Need n > number of model parameters. Got n={n}, p={p}")

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"Design matrix is singular or ill-conditioned for n={n}. "
            "This can happen if x or covariates are constant/collinear."
        ) from e

    betas = XtX_inv @ X.T @ Y
    fitted = X @ betas
    resid = Y - fitted

    df = n - p
    rss = np.sum(resid ** 2, axis=0)
    sigma2 = rss / df

    slope = betas[predictor_idx]
    se = np.sqrt(sigma2 * XtX_inv[predictor_idx, predictor_idx])

    with np.errstate(divide="ignore", invalid="ignore"):
        tmap = slope / se

    tmap[~np.isfinite(tmap)] = 0.0

    return slope, se, tmap


# =============================================================================
# THRESHOLDING / CLUSTER DETECTION
# =============================================================================

def get_max_t_threshold(
    null_max_t: np.ndarray,
    alpha_fwe: float = 0.05,
) -> float:
    """
    Convert a null distribution of maximum statistics into a whole-brain FWE threshold.
    """
    if null_max_t.ndim != 1 or len(null_max_t) == 0:
        raise ValueError("null_max_t must be a non-empty 1D array.")

    if not np.all(np.isfinite(null_max_t)):
        raise ValueError("null_max_t contains NaN or infinite values.")

    q = 1.0 - alpha_fwe
    return float(np.quantile(null_max_t, q))


def cluster_detect(
    supra_vec: np.ndarray,
    brain_mask_3d: np.ndarray,
    signal_region_3d: np.ndarray,
    k: int,
) -> Dict[str, bool | int]:
    """
    Determine whether any suprathreshold cluster of size >= k overlaps the true signal region.
    """
    supra_3d = masked_vec_to_3d(supra_vec.astype(bool), brain_mask_3d)
    conn = generate_binary_structure(rank=3, connectivity=3)
    labeled, n_clust = label(supra_3d, structure=conn)

    detected = False
    n_pass = 0
    largest_cluster = 0

    for cl_id in range(1, n_clust + 1):
        cl_mask = labeled == cl_id
        size = int(cl_mask.sum())

        if size < k:
            continue

        n_pass += 1
        largest_cluster = max(largest_cluster, size)

        if np.any(cl_mask & signal_region_3d):
            detected = True

    return {
        "detected": bool(detected),
        "n_clusters_passing_k": int(n_pass),
        "largest_cluster": int(largest_cluster),
    }


# =============================================================================
# NULL THRESHOLD ESTIMATION
# =============================================================================

def compute_null_threshold_for_n(
    n: int,
    pilot_maps_base: np.ndarray,
    config: Dict,
    rng: np.random.Generator,
    pilot_scores: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate the whole-brain FWE threshold for sample size n.
    """
    null_max_t = np.zeros(config["n_mc_null"], dtype=float)

    for it in tqdm(range(config["n_mc_null"]), desc=f"Null threshold N={n}", leave=False):
        base = bootstrap_base_maps(pilot_maps_base, n=n, rng=rng)

        x = simulate_clinical_scores(
            n=n,
            config=config,
            rng=rng,
            pilot_scores=pilot_scores
        )

        covs = simulate_covariates(n, config["covariates"], rng) if config["covariates"] else None
        _, _, tmap = fit_voxelwise_ols_tmap(base, x=x, covariates=covs)

        if config["two_sided_test"]:
            null_max_t[it] = np.max(np.abs(tmap))
        else:
            null_max_t[it] = np.max(tmap)

    return get_max_t_threshold(null_max_t, alpha_fwe=config["alpha_fwe"])


# =============================================================================
# POWER ESTIMATION FOR ONE SAMPLE SIZE
# =============================================================================

def run_power_for_n(
    n: int,
    pilot_maps_base: np.ndarray,
    beta_map_1d: np.ndarray,
    signal_region_3d: np.ndarray,
    data: LoadedData,
    config: Dict,
    rng: np.random.Generator,
    t_thr: float,
) -> pd.DataFrame:
    """
    Estimate power for a single sample size n.
    """
    rows = []
    signal_region_masked = signal_region_3d[data.brain_mask_3d]

    for it in tqdm(range(config["n_mc_power"]), desc=f"Power N={n}", leave=False):
        base = bootstrap_base_maps(pilot_maps_base, n=n, rng=rng)

        x = simulate_clinical_scores(
            n=n,
            config=config,
            rng=rng,
            pilot_scores=data.pilot_scores,
        )

        covs = simulate_covariates(n, config["covariates"], rng) if config["covariates"] else None

        Y = base + np.outer(x, beta_map_1d)

        slope, se, tmap = fit_voxelwise_ols_tmap(Y, x=x, covariates=covs)

        if config["two_sided_test"]:
            supra = np.abs(tmap) >= t_thr
        else:
            supra = tmap >= t_thr

        det = cluster_detect(
            supra_vec=supra,
            brain_mask_3d=data.brain_mask_3d,
            signal_region_3d=signal_region_3d,
            k=config["cluster_extent_k"],
        )

        rows.append({
            "sample_size": n,
            "iteration": it + 1,
            "t_threshold_fwe": t_thr,
            "detected": int(det["detected"]),
            "n_clusters_passing_k": det["n_clusters_passing_k"],
            "largest_cluster": det["largest_cluster"],
            "peak_abs_t": float(np.max(np.abs(tmap))),
            "mean_slope_in_signal": float(np.mean(slope[signal_region_masked])),
            "mean_se_in_signal": float(np.mean(se[signal_region_masked])),
        })

    return pd.DataFrame(rows)


# =============================================================================
# ESTIMATE BETA FROM PILOT RESULT
# =============================================================================

def estimate_representative_voxel_sd(
    pilot_maps_2d: np.ndarray,
    summary: str = "median",
) -> dict:
    """
    Estimate a representative cross-subject voxelwise SD from the pilot maps.

    PURPOSE
    -------
    The exact reported peak voxel could not be robustly localized in the pilot
    matrix space, so this function estimates a representative voxel SD across
    the brain mask. That SD is then used to approximately convert a reported
    peak t-statistic into the beta scale used by the simulation.

    PARAMETERS
    ----------
    pilot_maps_2d : np.ndarray, shape (n_subjects, n_voxels)
        Subject-by-voxel pilot matrix.

    summary : str, default="median"
        Summary statistic used as the representative SD.
        Options:
            - "median"        : robust default
            - "mean"
            - "percentile_75" : somewhat more optimistic

    RETURNS
    -------
    dict
        Dictionary containing:
            - selected_summary
            - selected_sd
            - voxel_sd_min
            - voxel_sd_p25
            - voxel_sd_median
            - voxel_sd_mean
            - voxel_sd_p75
            - voxel_sd_max
    """
    Y = np.asarray(pilot_maps_2d, dtype=np.float64)

    if Y.ndim != 2:
        raise ValueError(f"pilot_maps_2d must be 2D, got shape {Y.shape}")

    if not np.all(np.isfinite(Y)):
        raise ValueError("pilot_maps_2d contains NaN or infinite values.")

    # SD across subjects at each voxel
    voxel_sds = np.std(Y, axis=0, ddof=1)

    if not np.all(np.isfinite(voxel_sds)):
        raise ValueError("voxelwise SDs contain NaN or infinite values.")

    if summary == "median":
        selected_sd = float(np.median(voxel_sds))
    elif summary == "mean":
        selected_sd = float(np.mean(voxel_sds))
    elif summary == "percentile_75":
        selected_sd = float(np.percentile(voxel_sds, 75))
    else:
        raise ValueError(
            f"Unsupported summary='{summary}'. Use 'median', 'mean', or 'percentile_75'."
        )

    return {
        "selected_summary": summary,
        "selected_sd": selected_sd,
        "voxel_sd_min": float(np.min(voxel_sds)),
        "voxel_sd_p25": float(np.percentile(voxel_sds, 25)),
        "voxel_sd_median": float(np.median(voxel_sds)),
        "voxel_sd_mean": float(np.mean(voxel_sds)),
        "voxel_sd_p75": float(np.percentile(voxel_sds, 75)),
        "voxel_sd_max": float(np.max(voxel_sds)),
    }


def estimate_beta_from_reported_peak_t(
    pilot_maps_2d: np.ndarray,
    t_value: float,
    n_subjects: int,
    sign: str = "negative",
    sd_summary: str = "median",
    shrink_factors: tuple[float, ...] = (0.25, 1/3, 0.5),
) -> dict:
    """
    Estimate simulation beta values from a reported pilot peak-voxel t-statistic.

    PURPOSE
    -------
    This function provides a practical, traceable way to convert a reported
    pilot peak-voxel result into the beta scale used by the Monte Carlo simulation.

    LOGIC
    -----
    Step 1:
        Estimate a representative voxelwise SD across subjects from the pilot maps.

    Step 2:
        Convert the reported t-statistic into a correlation magnitude using:

            r = t / sqrt(t^2 + df)

        where:
            df = n_subjects - 2

        Then assign the reported sign ("negative" or "positive").

    Step 3:
        Convert the correlation into a slope beta under the assumption that the
        predictor will be z-scored in the simulation:

            beta = r * SD_Y

        where:
            SD_Y is the representative voxel SD from Step 1.

    Step 4:
        Apply shrink factors to the implied peak beta, since peak voxels are
        usually inflated and should not typically be used directly.

    IMPORTANT
    ---------
    This is an approximation intended for simulation calibration, not an exact
    reconstruction of the original pilot model.

    PARAMETERS
    ----------
    pilot_maps_2d : np.ndarray, shape (n_subjects, n_voxels)
        Subject-by-voxel pilot matrix.

    t_value : float
        Reported peak-voxel t-statistic from the pilot analysis.
        Example: 7.60

    n_subjects : int
        Number of subjects in the pilot analysis.
        Example: 20

    sign : str, default="negative"
        Direction of the reported association.
        Must be "negative" or "positive".

    sd_summary : str, default="median"
        Which representative voxel SD to use.
        Options:
            - "median"
            - "mean"
            - "percentile_75"

    shrink_factors : tuple[float, ...], default=(0.25, 1/3, 0.5)
        Multipliers applied to the implied peak beta.

    RETURNS
    -------
    dict
        Dictionary containing:
            - reported_t
            - n_subjects
            - degrees_of_freedom
            - sign
            - representative_sd_summary
            - representative_voxel_sd
            - representative_sd_distribution
            - implied_peak_correlation_r
            - implied_peak_beta_signed
            - implied_peak_beta_magnitude
            - shrunken_beta_recommendations

    INTERPRETATION
    --------------
    If the simulation z-scores the clinical score, then beta means:

        "change in voxel intensity per +1 SD increase in the clinical score"

    So the shrunken beta magnitudes returned here can be used directly as
    candidate values for CONFIG["fixed_beta_amplitude"].
    """
    import math

    if not np.isfinite(t_value):
        raise ValueError(f"t_value must be finite, got {t_value}")

    if int(n_subjects) < 3:
        raise ValueError(f"n_subjects must be >= 3, got {n_subjects}")

    if sign not in {"negative", "positive"}:
        raise ValueError(f"sign must be 'negative' or 'positive', got {sign}")

    if len(shrink_factors) == 0:
        raise ValueError("shrink_factors must contain at least one value")

    for sf in shrink_factors:
        if not np.isfinite(sf) or sf <= 0:
            raise ValueError(f"All shrink_factors must be positive finite numbers, got {sf}")

    # Step 1: representative voxel SD from pilot maps
    sd_info = estimate_representative_voxel_sd(
        pilot_maps_2d=pilot_maps_2d,
        summary=sd_summary,
    )
    representative_sd = float(sd_info["selected_sd"])

    # Step 2: t -> r for a simple regression with one predictor
    df = int(n_subjects) - 2
    r_mag = abs(float(t_value)) / math.sqrt(float(t_value) ** 2 + df)

    if sign == "negative":
        r = -r_mag
    else:
        r = r_mag

    # Step 3: r -> beta, assuming predictor will be z-scored in the simulation
    beta_peak_signed = r * representative_sd
    beta_peak_magnitude = abs(beta_peak_signed)

    # Step 4: shrink the peak beta to generate more conservative simulation values
    shrunken = {}
    for sf in shrink_factors:
        shrunken[float(sf)] = {
            "beta_signed": float(beta_peak_signed * sf),
            "beta_magnitude": float(beta_peak_magnitude * sf),
        }

    return {
        "reported_t": float(t_value),
        "n_subjects": int(n_subjects),
        "degrees_of_freedom": int(df),
        "sign": sign,
        "representative_sd_summary": sd_summary,
        "representative_voxel_sd": representative_sd,
        "representative_sd_distribution": sd_info,
        "implied_peak_correlation_r": float(r),
        "implied_peak_beta_signed": float(beta_peak_signed),
        "implied_peak_beta_magnitude": float(beta_peak_magnitude),
        "shrunken_beta_recommendations": shrunken,
        "notes": (
            "This estimate uses a representative whole-brain voxelwise SD from the pilot maps "
            "because the exact reported peak voxel could not be robustly localized in the "
            "pilot matrix space. The implied peak beta should generally be treated as an upper "
            "anchor and shrunk before use in simulations."
        ),
    }


def print_beta_estimation_summary(beta_info: dict) -> None:
    """
    Pretty-print the output of estimate_beta_from_reported_peak_t().
    """
    print("\nApproximate beta calibration from reported pilot peak")
    print("----------------------------------------------------")
    print(f"Reported peak t-statistic:         {beta_info['reported_t']:.4f}")
    print(f"Pilot N:                           {beta_info['n_subjects']}")
    print(f"Degrees of freedom:                {beta_info['degrees_of_freedom']}")
    print(f"Reported sign:                     {beta_info['sign']}")
    print(f"Representative SD summary used:    {beta_info['representative_sd_summary']}")
    print(f"Representative voxel SD:           {beta_info['representative_voxel_sd']:.4f}")
    print(f"Implied peak correlation r:        {beta_info['implied_peak_correlation_r']:.4f}")
    print(f"Implied peak beta (signed):        {beta_info['implied_peak_beta_signed']:.4f}")
    print(f"Implied peak beta (magnitude):     {beta_info['implied_peak_beta_magnitude']:.4f}")

    sd_dist = beta_info["representative_sd_distribution"]
    print("\nRepresentative voxel SD distribution across brain mask")
    print("------------------------------------------------------")
    print(f"Min:                               {sd_dist['voxel_sd_min']:.4f}")
    print(f"25th percentile:                   {sd_dist['voxel_sd_p25']:.4f}")
    print(f"Median:                            {sd_dist['voxel_sd_median']:.4f}")
    print(f"Mean:                              {sd_dist['voxel_sd_mean']:.4f}")
    print(f"75th percentile:                   {sd_dist['voxel_sd_p75']:.4f}")
    print(f"Max:                               {sd_dist['voxel_sd_max']:.4f}")

    print("\nShrunken beta recommendations")
    print("-----------------------------")
    for sf, vals in beta_info["shrunken_beta_recommendations"].items():
        print(
            f"shrink={sf:.4f}  "
            f"beta_signed={vals['beta_signed']:.4f}  "
            f"beta_magnitude={vals['beta_magnitude']:.4f}"
        )


# =============================================================================
# MAIN DRIVER
# =============================================================================

def main(config: Dict) -> None:
    """
    Main execution function.
    """
    validate_config(config)

    rng = np.random.default_rng(config["random_seed"])
    out_dir = ensure_dir(config["output_dir"])

    print("Loading data...")
    data = load_data(config)

    print(f"Pilot maps loaded: {data.pilot_maps_2d.shape[0]}")
    print(f"Masked voxels: {data.pilot_maps_2d.shape[1]}")
    print(f"Subject IDs loaded: {len(data.subject_ids)}")
    print(f"First 5 subject IDs: {data.subject_ids[:5]}")



    # -------------------------------------------------------------------------
    # Approximate beta calibration from reported pilot peak result
    #
    # Reported pilot result:
    #   negative correlation between EDS and activation
    #   peak voxel t = 7.60
    #   N = 20
    #
    # -------------------------------------------------------------------------
    beta_info = estimate_beta_from_reported_peak_t(
        pilot_maps_2d=data.pilot_maps_2d,
        t_value=7.60,
        n_subjects=20,
        sign="negative",
        sd_summary="median",                 # recommended default
        shrink_factors=(0.25, 0.5, 0.75, 1),
    )

    print_beta_estimation_summary(beta_info)
    # ------------------------------------------------------------------


    # printing diagnostics
    print("pilot_maps_2d dtype:", data.pilot_maps_2d.dtype)
    print("pilot_maps_2d shape:", data.pilot_maps_2d.shape)
    print("pilot_maps finite:", np.isfinite(data.pilot_maps_2d).all())
    print("pilot_maps min:", np.nanmin(data.pilot_maps_2d))
    print("pilot_maps max:", np.nanmax(data.pilot_maps_2d))
    print("pilot_maps mean:", np.nanmean(data.pilot_maps_2d))
    print("pilot_maps std:", np.nanstd(data.pilot_maps_2d))
    print("n NaN:", np.isnan(data.pilot_maps_2d).sum())
    print("n Inf:", np.isinf(data.pilot_maps_2d).sum())


    # check pilot scores
    print("pilot_scores:", data.pilot_scores)
    print("pilot_scores finite:", np.isfinite(data.pilot_scores).all())
    print("pilot_scores std:", np.std(data.pilot_scores, ddof=1))

    pilot_maps_base = data.pilot_maps_2d.copy()
    
    if config["residualize_pilot_maps"]:
        if data.pilot_scores is None:
            warnings.warn(
                "residualize_pilot_maps=True but no pilot_scores_csv supplied. "
                "Proceeding without residualization."
            )
        else:
            print("Residualizing pilot maps against pilot clinical scores...")
            pilot_maps_base = residualize_pilot_maps_against_scores(
                pilot_maps_2d=pilot_maps_base,
                pilot_scores=data.pilot_scores,
            )

    # more diagnositics
    print("residualized finite:", np.isfinite(pilot_maps_base).all())
    print("residualized min:", np.nanmin(pilot_maps_base))
    print("residualized max:", np.nanmax(pilot_maps_base))
    print("residualized mean:", np.nanmean(pilot_maps_base))
    print("residualized std:", np.nanstd(pilot_maps_base))

    print("Building signal region and beta map...")
    signal_region_3d, beta_map_1d = build_signal_region_and_beta_map(data, config)

    save_nifti_from_masked_vector(
        masked_vec=beta_map_1d,
        mask_3d=data.brain_mask_3d,
        affine=data.affine,
        header=data.header,
        out_path=out_dir / "beta_map_injected.nii.gz",
    )

    nib.save(
        nib.Nifti1Image(signal_region_3d.astype(np.uint8), affine=data.affine, header=data.header),
        str(out_dir / "signal_region_mask.nii.gz"),
    )

    print(f"Signal voxels: {int(signal_region_3d.sum())}")
    print(f"Injected beta max: {float(beta_map_1d.max()):.6f}")
    
    with open(out_dir / "config_used.json", "w") as f:
        json.dump(config, f, indent=2)

    summary_rows = []
    all_iter_rows = []

    for n in config["candidate_sample_sizes"]:
        print(f"\n=== Candidate N = {n} ===")

        t_thr = compute_null_threshold_for_n(
            n=n,
            pilot_maps_base=pilot_maps_base,
            config=config,
            rng=rng,
            pilot_scores=data.pilot_scores,
        )

        print(f"Estimated whole-brain FWE threshold (max-|t|) for N={n}: {t_thr:.4f}")

        df_n = run_power_for_n(
            n=n,
            pilot_maps_base=pilot_maps_base,
            beta_map_1d=beta_map_1d,
            signal_region_3d=signal_region_3d,
            data=data,
            config=config,
            rng=rng,
            t_thr=t_thr,
        )

        power = float(df_n["detected"].mean())

        summary_rows.append({
            "sample_size": n,
            "power": power,
            "t_threshold_fwe": t_thr,
            "cluster_extent_k": config["cluster_extent_k"],
            "n_mc_power": config["n_mc_power"],
            "n_mc_null": config["n_mc_null"],
        })

        all_iter_rows.append(df_n)

        print(f"Estimated power for N={n}: {power:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    iter_df = pd.concat(all_iter_rows, ignore_index=True)

    summary_df.to_csv(out_dir / "power_summary.csv", index=False)
    iter_df.to_csv(out_dir / "power_iterations.csv", index=False)

    print("\nDone.")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main(CONFIG)