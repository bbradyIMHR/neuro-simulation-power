from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    User-editable configuration for pilot-based ROI effect size estimation.

    Required inputs
    ---------------
    subject_voxel_mat_file : Path
        Path to a MATLAB v7.3 .mat file containing:
            - data_matrix : subject-by-voxel or voxel-by-subject matrix
            - mask_idx    : 3D binary analysis mask
            - subjects    : optional subject IDs stored as MATLAB strings/cell array

    roi_mask_file : Path
        Path to a NIfTI (.nii or .nii.gz) ROI mask to be resampled to the
        reference image grid.

    reference_grid_file : Path
        Path to a NIfTI (.nii or .nii.gz) image defining the target spatial grid.
        This should match the grid used by the 3D mask stored in the .mat file.

    group_assignments : dict
        Dictionary mapping subject IDs to group labels. Expected labels in this
        script are "eg" and "cg".
    """

    subject_voxel_mat_file: Path
    roi_mask_file: Path
    reference_grid_file: Path
    group_assignments: Dict[str, str]

    roi_percentile: float = 95.0
    roi_binarize_threshold: float = 0.0
    restrict_roi_to_analysis_mask: bool = True
    group_difference_percentile: float = 95.0


# =============================================================================
# MATLAB string loading helpers
# =============================================================================

def decode_matlab_hdf5_string(h5_file: h5py.File, ref) -> str:
    """
    Decode a MATLAB v7.3 string or cell-string stored as an HDF5 object reference.
    """
    arr = np.array(h5_file[ref][:]).squeeze()

    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(x)) for x in arr.ravel())

    return "".join(str(x) for x in arr.ravel())


def load_subject_ids_from_mat(h5_file: h5py.File) -> List[str]:
    """
    Load subject IDs from the optional 'subjects' variable in a MATLAB v7.3 file.
    """
    if "subjects" not in h5_file:
        return []

    subject_dataset = np.array(h5_file["subjects"][:])
    subject_ids: List[str] = []

    # MATLAB cell arrays are read in column-major order
    for ref in subject_dataset.ravel(order="F"):
        subject_ids.append(decode_matlab_hdf5_string(h5_file, ref))

    return subject_ids


# =============================================================================
# Data loading
# =============================================================================

def load_subject_voxel_matrix_and_mask(
    mat_file: str | Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a MATLAB-exported subject-by-voxel matrix and its corresponding 3D mask.

    Expected .mat contents
    ----------------------
    data_matrix : 2D numeric array
        Subject-by-voxel matrix, or voxel-by-subject matrix.
    mask_idx : 3D binary array
        Analysis mask used to define the included voxels.
    subjects : optional
        Subject IDs.

    Returns
    -------
    data_matrix : np.ndarray
        Shape (n_subjects, n_voxels)
    mask_idx : np.ndarray
        Shape (X, Y, Z), dtype=bool
    subject_ids : list of str
        Subject IDs corresponding to matrix rows, if available
    """
    mat_file = Path(mat_file)

    with h5py.File(mat_file, "r") as h5_file:
        if "data_matrix" not in h5_file:
            raise KeyError("Expected variable 'data_matrix' not found in .mat file.")
        if "mask_idx" not in h5_file:
            raise KeyError("Expected variable 'mask_idx' not found in .mat file.")

        data_matrix = np.array(h5_file["data_matrix"][:])
        mask_idx = np.array(h5_file["mask_idx"][:]).astype(bool)
        subject_ids = load_subject_ids_from_mat(h5_file)

    if data_matrix.ndim != 2:
        raise ValueError(f"'data_matrix' must be 2D. Got shape {data_matrix.shape}.")
    if mask_idx.ndim != 3:
        raise ValueError(f"'mask_idx' must be 3D. Got shape {mask_idx.shape}.")

    n_mask_voxels = int(mask_idx.sum())

    # Handle possible MATLAB / h5py orientation mismatch
    if data_matrix.shape[1] == n_mask_voxels:
        pass
    elif data_matrix.shape[0] == n_mask_voxels:
        data_matrix = data_matrix.T
    else:
        raise ValueError(
            f"Matrix shape {data_matrix.shape} is incompatible with "
            f"mask voxel count {n_mask_voxels}."
        )

    n_subjects, n_voxels = data_matrix.shape

    if n_voxels != n_mask_voxels:
        raise ValueError(
            f"After orientation handling, data_matrix has {n_voxels} voxel columns "
            f"but mask contains {n_mask_voxels} voxels."
        )

    if subject_ids and len(subject_ids) != n_subjects:
        raise ValueError(
            f"Loaded {len(subject_ids)} subject IDs, but matrix has {n_subjects} rows."
        )

    return data_matrix, mask_idx, subject_ids


# =============================================================================
# ROI utilities
# =============================================================================

def load_and_resample_roi_to_reference_grid(
    roi_mask_file: str | Path,
    reference_grid_file: str | Path,
    binarize_threshold: float = 0.0,
) -> np.ndarray:
    """
    Load an ROI mask and resample it onto the reference image grid.

    Parameters
    ----------
    roi_mask_file : str or Path
        Path to an ROI NIfTI file.

    reference_grid_file : str or Path
        Path to a NIfTI file defining the target voxel grid.

    binarize_threshold : float
        Voxels greater than this threshold are included in the ROI mask.

    Returns
    -------
    np.ndarray
        Boolean ROI mask on the reference grid.
    """
    roi_img = nib.load(str(roi_mask_file))
    ref_img = nib.load(str(reference_grid_file))

    roi_resampled = resample_to_img(
        source_img=roi_img,
        target_img=ref_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )

    return (roi_resampled.get_fdata() > binarize_threshold).astype(bool)


def get_roi_column_indices_from_mask(
    analysis_mask: np.ndarray,
    roi_mask: np.ndarray,
    restrict_roi_to_analysis_mask: bool = True,
) -> np.ndarray:
    """
    Map ROI voxels to matrix column indices using MATLAB-compatible voxel ordering.

    Important
    ---------
    The subject-by-voxel matrix is assumed to have been created using MATLAB-style
    logical indexing, which follows column-major (Fortran) ordering.
    """
    analysis_mask = np.asarray(analysis_mask, dtype=bool)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    if analysis_mask.shape != roi_mask.shape:
        raise ValueError(
            f"analysis_mask shape {analysis_mask.shape} does not match "
            f"roi_mask shape {roi_mask.shape}."
        )

    roi_use = roi_mask & analysis_mask if restrict_roi_to_analysis_mask else roi_mask.copy()

    analysis_linear = np.flatnonzero(analysis_mask.ravel(order="F"))
    roi_linear = np.flatnonzero(roi_use.ravel(order="F"))

    linear_to_column = {linear_idx: col_idx for col_idx, linear_idx in enumerate(analysis_linear)}

    missing_voxels = [idx for idx in roi_linear if idx not in linear_to_column]
    if missing_voxels:
        raise ValueError(
            f"{len(missing_voxels)} ROI voxels are not present in the analysis mask. "
            "Check mask alignment or set restrict_roi_to_analysis_mask=True."
        )

    return np.array([linear_to_column[idx] for idx in roi_linear], dtype=int)


# =============================================================================
# Core analysis
# =============================================================================

def compute_subjectwise_brain_summary(
    mat_file: str | Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute subject-wise whole-brain mean and SD across all masked voxels.
    """
    data_matrix, _, subject_ids = load_subject_voxel_matrix_and_mask(mat_file)
    brain_mean = np.mean(data_matrix, axis=1)
    brain_sd = np.std(data_matrix, axis=1, ddof=1)
    return brain_mean, brain_sd, subject_ids


def compute_subjectwise_roi_percentile(
    mat_file: str | Path,
    roi_mask_file: str | Path,
    reference_grid_file: str | Path,
    percentile: float = 95.0,
    binarize_threshold: float = 0.0,
    restrict_roi_to_analysis_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Compute a subject-wise percentile of values within an ROI directly from the
    subject-by-voxel matrix.

    Returns
    -------
    roi_percentile : np.ndarray
        Shape (n_subjects,)
    roi_values : np.ndarray
        Shape (n_subjects, n_roi_voxels)
    subject_ids : list of str
        Subject IDs corresponding to matrix rows
    roi_cols : np.ndarray
        Column indices corresponding to ROI voxels
    """
    data_matrix, analysis_mask, subject_ids = load_subject_voxel_matrix_and_mask(mat_file)

    roi_mask = load_and_resample_roi_to_reference_grid(
        roi_mask_file=roi_mask_file,
        reference_grid_file=reference_grid_file,
        binarize_threshold=binarize_threshold,
    )

    if roi_mask.shape != analysis_mask.shape:
        raise ValueError(
            f"Resampled ROI shape {roi_mask.shape} does not match "
            f"analysis mask shape {analysis_mask.shape}."
        )

    roi_cols = get_roi_column_indices_from_mask(
        analysis_mask=analysis_mask,
        roi_mask=roi_mask,
        restrict_roi_to_analysis_mask=restrict_roi_to_analysis_mask,
    )

    roi_values = data_matrix[:, roi_cols]
    roi_percentile = np.percentile(roi_values, percentile, axis=1)

    return roi_percentile, roi_values, subject_ids, roi_cols


def compute_pooled_sd(group1_values: np.ndarray, group2_values: np.ndarray) -> np.ndarray:
    """
    Compute pooled SD at each voxel across two groups.
    """
    n1 = group1_values.shape[0]
    n2 = group2_values.shape[0]

    sd1 = np.std(group1_values, axis=0, ddof=1)
    sd2 = np.std(group2_values, axis=0, ddof=1)

    return np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))


def main(config: AnalysisConfig) -> None:
    """
    Run pilot-based ROI effect size estimation and print a concise summary.
    """
    brain_mean, brain_sd, _ = compute_subjectwise_brain_summary(
        config.subject_voxel_mat_file
    )

    roi_percentile, roi_values, subject_ids, roi_cols = compute_subjectwise_roi_percentile(
        mat_file=config.subject_voxel_mat_file,
        roi_mask_file=config.roi_mask_file,
        reference_grid_file=config.reference_grid_file,
        percentile=config.roi_percentile,
        binarize_threshold=config.roi_binarize_threshold,
        restrict_roi_to_analysis_mask=config.restrict_roi_to_analysis_mask,
    )

    missing_subjects = [sid for sid in subject_ids if sid not in config.group_assignments]
    if missing_subjects:
        raise KeyError(
            "Missing group assignments for subject IDs: " + ", ".join(missing_subjects)
        )

    groups = np.array([config.group_assignments[sid] for sid in subject_ids])

    eg_values = roi_values[groups == "eg"]
    cg_values = roi_values[groups == "cg"]

    if eg_values.size == 0 or cg_values.size == 0:
        raise ValueError("One or both groups contain zero subjects.")

    eg_mean_per_voxel = np.mean(eg_values, axis=0)
    cg_mean_per_voxel = np.mean(cg_values, axis=0)
    voxelwise_difference = eg_mean_per_voxel - cg_mean_per_voxel

    beta_for_simulation = np.percentile(
        voxelwise_difference,
        config.group_difference_percentile,
    )

    pooled_sd_voxelwise = compute_pooled_sd(eg_values, cg_values)
    median_pooled_sd = np.median(pooled_sd_voxelwise)

    print("\n" + "=" * 72)
    print("Pilot-informed ROI effect size summary")
    print("=" * 72)
    print(f"Number of subjects:            {len(subject_ids)}")
    print(f"Number of ROI voxels:          {len(roi_cols)}")
    print(f"Experimental group subjects:   {eg_values.shape[0]}")
    print(f"Control group subjects:        {cg_values.shape[0]}")
    print(f"ROI percentile used:           {config.roi_percentile:.1f}")
    print(f"Difference percentile used:    {config.group_difference_percentile:.1f}")

    print("\nWhole-brain summary")
    print("-" * 72)
    print(f"Mean whole-brain beta:         {np.mean(brain_mean):.6f}")
    print(f"Mean whole-brain SD:           {np.mean(brain_sd):.6f}")

    print("\nSimulation summary")
    print("-" * 72)
    print(f"Beta input for simulation:     {beta_for_simulation:.6f}")
    print(f"Median pooled voxel SD:        {median_pooled_sd:.6f}")

    if median_pooled_sd > 0:
        print(f"Approx. standardized effect:   {beta_for_simulation / median_pooled_sd:.6f}")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Example configuration
    #
    # Replace these filenames with your own local input files.
    #
    # 1) subject_voxel_mat_file:
    #    MATLAB v7.3 .mat file containing:
    #      - data_matrix
    #      - mask_idx
    #      - subjects (optional)
    #
    # 2) roi_mask_file:
    #    NIfTI ROI mask file (.nii or .nii.gz)
    #
    # 3) reference_grid_file:
    #    NIfTI image defining the target voxel grid used by mask_idx
    # -------------------------------------------------------------------------
    config = AnalysisConfig(
        subject_voxel_mat_file=Path("subject_voxel_matrix.mat"),
        roi_mask_file=Path("roi_mask.nii"),
        reference_grid_file=Path("reference_grid.nii"),
        group_assignments={
            "S01": "eg",
            "S02": "cg",
            # Add remaining subject-to-group mappings here
        },
        roi_percentile=95.0,
        roi_binarize_threshold=0.0,
        restrict_roi_to_analysis_mask=True,
        group_difference_percentile=95.0,
    )

    main(config)
