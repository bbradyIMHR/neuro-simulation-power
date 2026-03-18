"""
Pilot-informed simulation of subject-level fMRI contrast maps
for a TWO-GROUP design (EG vs CG), with whole-brain voxelwise
FDR correction and a post-threshold cluster extent filter.

STUDY LOGIC BEING SIMULATED
---------------------------
Each subject contributes ONE subject-level contrast map, e.g.:

    contrast_map = ConditionA - ConditionB

We simulate those subject-level contrast maps directly for each group:

    map_i = beta_group_i * signal_template + spatial_noise_i

Then, at the second level, we test:

    mean(EG contrast maps) - mean(CG contrast maps)

using:
    1. voxelwise EG vs CG two-sample t-test across all voxels in the brain mask
    2. Benjamini-Hochberg FDR correction across all voxels in the brain mask
    3. cluster extent threshold k >= 10
    4. detection criterion = any surviving cluster overlaps the injected-signal ROI

MAIN OBJECTS / SHAPES
---------------------
pilot.maps
    np.ndarray, shape (n_pilot, X, Y, Z)

noise_params.mask
    np.ndarray, shape (X, Y, Z), dtype=bool

signal_template
    np.ndarray, shape (X, Y, Z)
    Continuous-valued spatial signal template used during simulation.

signal_roi_mask
    np.ndarray, shape (X, Y, Z), dtype=bool
    Binary region used to define whether the planted signal was detected.

eg_maps
    np.ndarray, shape (n_eg, X, Y, Z)

cg_maps
    np.ndarray, shape (n_cg, X, Y, Z)

DEPENDENCIES
------------
pip install numpy scipy nibabel
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, label, generate_binary_structure
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# =============================================================================
# 1. DATA CONTAINERS
# =============================================================================

@dataclass
class PilotData:
    """
    Stores pilot subject-level contrast maps and image metadata.

    Attributes
    ----------
    maps : np.ndarray
        Shape: (n_pilot, X, Y, Z)
        Subject-level pilot contrast maps.

    affine : np.ndarray
        Shape: (4, 4)
        Affine matrix from the first pilot NIfTI.

    header : nib.Nifti1Header
        Header from the first pilot NIfTI.

    voxel_sizes_mm : tuple[float, float, float]
        Voxel sizes in mm, e.g. (2.0, 2.0, 2.0).

    file_paths : list[Path]
        Input NIfTI file paths.
    """
    maps: np.ndarray                   # shape: (n_pilot, X, Y, Z)
    affine: Optional[np.ndarray] = None
    header: Optional[nib.Nifti1Header] = None
    voxel_sizes_mm: Optional[Tuple[float, float, float]] = None
    file_paths: Optional[List[Path]] = None
    subject_ids: Optional[List[str]] = None
    mask: Optional[np.ndarray] = None


@dataclass
class NoiseParameters:
    """
    Pooled pilot-informed noise parameters.

    Attributes
    ----------
    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Analysis mask.

    mean_map : np.ndarray
        Shape: (X, Y, Z)
        Mean pilot contrast map.

    global_sd : float
        Mean within-mask spatial SD across pilot maps.

    voxelwise_sd : np.ndarray
        Shape: (X, Y, Z)
        Voxelwise SD across pilot maps.

    fwhm_vox : float
        Effective smoothness estimate in voxel units.

    fwhm_mm : float
        Effective smoothness estimate in mm.
    """
    mask: np.ndarray
    mean_map: np.ndarray
    global_sd: float
    voxelwise_sd: np.ndarray
    fwhm_vox: float
    fwhm_mm: float


@dataclass
class WholeBrainPowerResult:
    """
    Output of the whole-brain Monte Carlo power simulation.

    Attributes
    ----------
    power : float
        Proportion of Monte Carlo iterations in which the true signal region
        was detected after peak-level FWE thresholding and cluster thresholding.

    detected_signal : np.ndarray
        Shape: (n_iterations,), dtype=bool
        Whether each iteration successfully detected the signal region.

    p_thresholds : np.ndarray
        Shape: (n_iterations,)
        The critical whole-brain peak-level FWE |t| threshold used in each iteration's design.

    n_fwe_voxels : np.ndarray
        Shape: (n_iterations,)
        Number of voxels surviving the peak-level FWE threshold before the cluster filter.

    n_clustered_voxels : np.ndarray
        Shape: (n_iterations,)
        Number of voxels surviving both the peak-level FWE threshold and the cluster threshold.

    n_surviving_clusters : np.ndarray
        Shape: (n_iterations,)
        Number of clusters surviving the cluster threshold.
    """
    power: float
    detected_signal: np.ndarray
    t_thresholds: np.ndarray
    n_fwe_voxels: np.ndarray
    n_clustered_voxels: np.ndarray
    n_surviving_clusters: np.ndarray
    n_iterations: np.ndarray


# =============================================================================
# 2. LOAD PILOT FILES
# =============================================================================


import h5py

def _decode_hdf5_string_column(f: h5py.File, ref) -> str:
    arr = f[ref][:]
    return ''.join(chr(x) for x in arr.flatten())

def load_pilot_mat(mat_path: Path, reference_nifti: Path) -> PilotData:
    """
    Load pilot maps from a MATLAB .mat voxel matrix and attach
    spatial metadata from a reference NIfTI.
    """

    mat_path = Path(mat_path)
    reference_nifti = Path(reference_nifti)

    # Load reference image metadata
    ref_img = nib.load(str(reference_nifti))
    affine = ref_img.affine.copy()
    header = ref_img.header.copy()
    voxel_sizes_mm = tuple(float(v) for v in header.get_zooms()[:3])
    shape = ref_img.shape

    with h5py.File(mat_path, "r") as f:
        data_matrix = f["data_matrix"][:]      # (voxels, subjects)
        mask = f["mask_idx"][:].astype(bool)

    if mask.shape != shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference NIfTI shape {shape}"
        )

    n_voxels, n_subjects = data_matrix.shape

    if mask.sum() != n_voxels:
        raise ValueError(
            f"Mask contains {mask.sum()} voxels but data_matrix has {n_voxels} rows."
        )

    maps = np.zeros((n_subjects, *shape), dtype=np.float32)

    for i in range(n_subjects):
        vol = np.zeros(shape, dtype=np.float32)
        vol[mask] = data_matrix[:, i]
        maps[i] = vol

    return PilotData(
        maps=maps,
        affine=affine,
        header=header,
        voxel_sizes_mm=voxel_sizes_mm,
        file_paths=[mat_path]
    )

def load_pilot_niftis(file_paths: Sequence[Path]) -> PilotData:
    """
    Load subject-level pilot contrast maps from 3D NIfTI files.

    Parameters
    ----------
    file_paths : sequence of Path
        Paths to 3D NIfTI files. All files must have the same shape and affine.

    Returns
    -------
    PilotData
        Container with:
            maps.shape = (n_pilot, X, Y, Z)
    """
    file_paths = [Path(p) for p in file_paths]
    if len(file_paths) == 0:
        raise ValueError("No pilot files were provided.")

    imgs = [nib.load(str(p)) for p in file_paths]
    shapes = [img.shape for img in imgs]

    if any(len(shape) != 3 for shape in shapes):
        raise ValueError(f"All pilot maps must be 3D NIfTI files. Got shapes: {shapes}")
    if len(set(shapes)) != 1:
        raise ValueError(f"All pilot maps must have the same shape. Got shapes: {shapes}")

    first_img = imgs[0]
    affine = first_img.affine.copy()
    header = first_img.header.copy()
    voxel_sizes_mm = tuple(float(v) for v in header.get_zooms()[:3])

    maps = np.zeros((len(imgs), *first_img.shape), dtype=np.float32)

    for i, img in enumerate(imgs):
        if not np.allclose(img.affine, affine):
            raise ValueError("Not all pilot images share the same affine.")
        maps[i] = img.get_fdata(dtype=np.float32)

    return PilotData(
        maps=maps,
        affine=affine,
        header=header,
        voxel_sizes_mm=voxel_sizes_mm,
        file_paths=file_paths
    )


# =============================================================================
# 3. MASK ESTIMATION
# =============================================================================

def estimate_mask_from_pilot_maps(
    pilot_maps: np.ndarray,
    nonzero_fraction_threshold: float = 0.95,
    finite_only: bool = True
) -> np.ndarray:
    """
    Estimate a conservative analysis mask from pilot maps.

    A voxel is included if it is nonzero in at least the specified fraction
    of subjects, and optionally finite in all subjects.

    Parameters
    ----------
    pilot_maps : np.ndarray
        Shape: (n_pilot, X, Y, Z)

    nonzero_fraction_threshold : float
        Minimum fraction of subjects for which a voxel must be nonzero.

    finite_only : bool
        If True, require all values to be finite.

    Returns
    -------
    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool
    """
    if pilot_maps.ndim != 4:
        raise ValueError(f"Expected pilot_maps shape (n_pilot, X, Y, Z), got {pilot_maps.shape}")

    nonzero_fraction = np.mean(pilot_maps != 0, axis=0)
    mask = nonzero_fraction >= nonzero_fraction_threshold

    if finite_only:
        mask &= np.all(np.isfinite(pilot_maps), axis=0)

    return mask


# =============================================================================
# 4. POOLED NOISE PARAMETER ESTIMATION
# =============================================================================

def estimate_global_sd(pilot_maps: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimate the mean within-mask spatial SD across pilot subjects.

    Parameters
    ----------
    pilot_maps : np.ndarray
        Shape: (n_pilot, X, Y, Z)

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    Returns
    -------
    global_sd : float
    """
    subject_sds = [float(pilot_maps[i][mask].std(ddof=1)) for i in range(pilot_maps.shape[0])]
    return float(np.mean(subject_sds))


def estimate_voxelwise_sd(
    pilot_maps: np.ndarray,
    mask: np.ndarray,
    min_sd: float = 1e-6
) -> np.ndarray:
    """
    Estimate voxelwise SD across pilot subjects.

    Parameters
    ----------
    pilot_maps : np.ndarray
        Shape: (n_pilot, X, Y, Z)

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    min_sd : float
        Lower bound for numerical stability inside the mask.

    Returns
    -------
    voxelwise_sd : np.ndarray
        Shape: (X, Y, Z), dtype=float32
    """
    voxelwise_sd = np.std(pilot_maps, axis=0, ddof=1).astype(np.float32)
    voxelwise_sd[mask] = np.maximum(voxelwise_sd[mask], min_sd)
    voxelwise_sd[~mask] = 0.0
    return voxelwise_sd


def estimate_effective_fwhm_vox(pilot_maps: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimate effective isotropic smoothness in voxel units.

    Method
    ------
    Uses lag-1 correlation along x, y, and z, then backs out the equivalent
    Gaussian sigma and converts to FWHM.
    Based on: Stefan J. Kiebel, Jean-Baptiste Poline, Karl J. Friston, Andrew P. Holmes, Keith J. Worsley,
    Robust Smoothness Estimation in Statistical Parametric Maps Using Standardized Residuals from the General Linear Model,
    NeuroImage, Volume 10, Issue 6, 1999, Pages 756-766, ISSN 1053-8119, https://doi.org/10.1006/nimg.1999.0508.

    Parameters
    ----------
    pilot_maps : np.ndarray
        Shape: (n_pilot, X, Y, Z)

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    Returns
    -------
    fwhm_vox : float
    """
    correlations = []

    for axis in range(3):
        slicer_a = [slice(None)] * 4
        slicer_b = [slice(None)] * 4
        slicer_a[axis + 1] = slice(0, -1)
        slicer_b[axis + 1] = slice(1, None)

        a = pilot_maps[tuple(slicer_a)]
        b = pilot_maps[tuple(slicer_b)]

        mask_a = [slice(None)] * 3
        mask_b = [slice(None)] * 3
        mask_a[axis] = slice(0, -1)
        mask_b[axis] = slice(1, None)

        neighbor_mask = mask[tuple(mask_a)] & mask[tuple(mask_b)]

        a_vals = a[:, neighbor_mask].ravel()
        b_vals = b[:, neighbor_mask].ravel()

        if a_vals.size < 10:
            raise ValueError("Too few neighboring masked voxels to estimate smoothness.")

        r = np.corrcoef(a_vals, b_vals)[0, 1]
        correlations.append(float(r))

    r_mean = float(np.mean(correlations))
    r_mean = np.clip(r_mean, 1e-6, 0.999999)

    sigma_vox = np.sqrt(-1.0 / (4.0 * np.log(r_mean)))
    fwhm_vox = 2.355 * sigma_vox

    return float(fwhm_vox)


def estimate_noise_parameters(
    pilot_maps: np.ndarray,
    voxel_sizes_mm: Tuple[float, float, float],
    mask: Optional[np.ndarray] = None,
    nonzero_fraction_threshold: float = 0.95
) -> NoiseParameters:
    """
    Estimate pooled pilot-informed noise parameters.

    Parameters
    ----------
    pilot_maps : np.ndarray
        Shape: (n_pilot, X, Y, Z)

    voxel_sizes_mm : tuple[float, float, float]
        Voxel sizes in mm.

    mask : np.ndarray or None
        Optional predefined analysis mask. If None, estimate from pilot maps.

    nonzero_fraction_threshold : float
        Used only if mask is None.

    Returns
    -------
    NoiseParameters
    """
    if mask is None:
        mask = estimate_mask_from_pilot_maps(
            pilot_maps=pilot_maps,
            nonzero_fraction_threshold=nonzero_fraction_threshold
        )

    mean_map = np.mean(pilot_maps, axis=0).astype(np.float32)
    mean_map[~mask] = 0.0

    global_sd = estimate_global_sd(pilot_maps, mask)
    voxelwise_sd = estimate_voxelwise_sd(pilot_maps, mask)
    fwhm_vox = estimate_effective_fwhm_vox(pilot_maps, mask)

    mean_voxel_size_mm = float(np.mean(voxel_sizes_mm))
    fwhm_mm = fwhm_vox * mean_voxel_size_mm

    return NoiseParameters(
        mask=mask,
        mean_map=mean_map,
        global_sd=global_sd,
        voxelwise_sd=voxelwise_sd,
        fwhm_vox=fwhm_vox,
        fwhm_mm=fwhm_mm
    )


# =============================================================================
# 5. SIGNAL TEMPLATE CREATION
# =============================================================================

def create_spherical_signal_template(
    shape: Tuple[int, int, int],
    center_xyz: Tuple[int, int, int],
    radius_vox: float,
    smooth_sigma_vox: float = 0.0,
    mask: Optional[np.ndarray] = None,
    peak_value: float = 1.0
) -> np.ndarray:
    """
    Create a spherical signal template.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Image shape, e.g. (64, 64, 64)

    center_xyz : tuple[int, int, int]
        Sphere center in voxel coordinates.

    radius_vox : float
        Radius in voxel units.

    smooth_sigma_vox : float
        Optional Gaussian smoothing sigma in voxel units.

    mask : np.ndarray or None
        Optional mask to zero out values outside the analysis mask.

    peak_value : float
        Template is normalized so that its maximum equals this value.

    Returns
    -------
    signal : np.ndarray
        Shape: (X, Y, Z), dtype=float32
    """
    x = np.arange(shape[0], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    z = np.arange(shape[2], dtype=np.float32)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    dist2 = (
        (xx - center_xyz[0]) ** 2 +
        (yy - center_xyz[1]) ** 2 +
        (zz - center_xyz[2]) ** 2
    )

    signal = (dist2 <= radius_vox ** 2).astype(np.float32)

    if smooth_sigma_vox > 0:
        signal = gaussian_filter(signal, sigma=smooth_sigma_vox)

    if signal.max() > 0:
        signal = signal / signal.max() * peak_value

    if mask is not None:
        signal[~mask] = 0.0

    return signal.astype(np.float32)


# =============================================================================
# 6. SPATIAL NOISE GENERATION
# =============================================================================

def generate_spatially_correlated_noise(
    n_subjects: int,
    mask: np.ndarray,
    fwhm_vox: float,
    global_sd: Optional[float] = None,
    voxelwise_sd: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    dtype=np.float32
) -> np.ndarray:
    """
    Generate subject-level spatially correlated noise maps.

    Exactly one of global_sd or voxelwise_sd must be provided.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    fwhm_vox : float
        Effective smoothness in voxel units.

    global_sd : float or None
        Scalar SD for homoscedastic scaling.

    voxelwise_sd : np.ndarray or None
        Shape: (X, Y, Z)
        Voxelwise SD map for heteroscedastic scaling.

    rng : np.random.Generator or None
        Random number generator.

    dtype
        Output dtype.

    Returns
    -------
    noise_maps : np.ndarray
        Shape: (n_subjects, X, Y, Z)
    """
    if (global_sd is None) == (voxelwise_sd is None):
        raise ValueError("Provide exactly one of global_sd or voxelwise_sd.")

    if rng is None:
        rng = np.random.default_rng()

    sigma_vox = fwhm_vox / 2.355
    shape = mask.shape

    noise_maps = np.zeros((n_subjects, *shape), dtype=dtype)

    for i in range(n_subjects):
        z = rng.standard_normal(size=shape)
        z = gaussian_filter(z, sigma=sigma_vox)

        masked_sd = z[mask].std(ddof=1)
        if masked_sd <= 0:
            raise RuntimeError("Generated noise had near-zero within-mask SD.")

        z = z / masked_sd

        if voxelwise_sd is not None:
            z = z * voxelwise_sd
        else:
            z = z * float(global_sd)

        z[~mask] = 0.0
        noise_maps[i] = z.astype(dtype)

    return noise_maps


# =============================================================================
# 7. SUBJECT-LEVEL CONTRAST MAP SIMULATION FOR EACH GROUP
# =============================================================================

def generate_group_contrast_maps(
    n_subjects: int,
    signal_template: np.ndarray,
    group_beta: float,
    mask: np.ndarray,
    fwhm_vox: float,
    global_sd: Optional[float] = None,
    voxelwise_sd: Optional[np.ndarray] = None,
    subject_beta_sd: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    dtype=np.float32
) -> np.ndarray:
    """
    Generate subject-level contrast maps for one group.

    MODEL
    -----
    For subject i:

        map_i = beta_i * signal_template + noise_i

    where:
        beta_i ~ Normal(group_beta, subject_beta_sd)

    Parameters
    ----------
    n_subjects : int
        Number of subjects in the group.

    signal_template : np.ndarray
        Shape: (X, Y, Z)

    group_beta : float
        Mean group-level signal amplitude.

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    fwhm_vox : float
        Effective smoothness in voxel units.

    global_sd : float or None
        Used for global SD scaling.

    voxelwise_sd : np.ndarray or None
        Shape: (X, Y, Z)
        Used for voxelwise SD scaling.

    subject_beta_sd : float
        Between-subject SD around the group mean beta.

    rng : np.random.Generator or None
        Random number generator.

    dtype
        Output dtype.

    Returns
    -------
    maps : np.ndarray
        Shape: (n_subjects, X, Y, Z)
    """
    if rng is None:
        rng = np.random.default_rng()

    noise_maps = generate_spatially_correlated_noise(
        n_subjects=n_subjects,
        mask=mask,
        fwhm_vox=fwhm_vox,
        global_sd=global_sd,
        voxelwise_sd=voxelwise_sd,
        rng=rng,
        dtype=dtype
    )

    maps = np.zeros_like(noise_maps, dtype=dtype)

    for i in range(n_subjects):
        if subject_beta_sd > 0:
            beta_i = float(rng.normal(loc=group_beta, scale=subject_beta_sd))
        else:
            beta_i = float(group_beta)

        maps[i] = noise_maps[i] + beta_i * signal_template
        maps[i][~mask] = 0.0

    return maps


def generate_two_group_contrast_maps(
    n_eg: int,
    n_cg: int,
    signal_template: np.ndarray,
    beta_eg: float,
    beta_cg: float,
    mask: np.ndarray,
    fwhm_vox: float,
    global_sd: Optional[float] = None,
    voxelwise_sd: Optional[np.ndarray] = None,
    subject_beta_sd: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    dtype=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate subject-level contrast maps for both EG and CG.

    Returns
    -------
    eg_maps : np.ndarray
        Shape: (n_eg, X, Y, Z)

    cg_maps : np.ndarray
        Shape: (n_cg, X, Y, Z)
    """
    if rng is None:
        rng = np.random.default_rng()

    eg_maps = generate_group_contrast_maps(
        n_subjects=n_eg,
        signal_template=signal_template,
        group_beta=beta_eg,
        mask=mask,
        fwhm_vox=fwhm_vox,
        global_sd=global_sd,
        voxelwise_sd=voxelwise_sd,
        subject_beta_sd=subject_beta_sd,
        rng=rng,
        dtype=dtype
    )

    cg_maps = generate_group_contrast_maps(
        n_subjects=n_cg,
        signal_template=signal_template,
        group_beta=beta_cg,
        mask=mask,
        fwhm_vox=fwhm_vox,
        global_sd=global_sd,
        voxelwise_sd=voxelwise_sd,
        subject_beta_sd=subject_beta_sd,
        rng=rng,
        dtype=dtype
    )

    return eg_maps, cg_maps


# =============================================================================
# 8. WHOLE-BRAIN SECOND-LEVEL TESTING: VOXELWISE T/P, FDR, CLUSTERS
# =============================================================================

def two_group_voxelwise_t_and_p_maps(
    eg_maps: np.ndarray,
    cg_maps: np.ndarray,
    mask: np.ndarray,
    equal_var: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute voxelwise EG vs CG t-statistics and two-sided p-values.

    Parameters
    ----------
    eg_maps : np.ndarray
        Shape: (n_eg, X, Y, Z)

    cg_maps : np.ndarray
        Shape: (n_cg, X, Y, Z)

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    equal_var : bool
        If True, use pooled-variance two-sample t-test.
        If False, use Welch's t-test.

    Returns
    -------
    t_map : np.ndarray
        Shape: (X, Y, Z), dtype=float32

    p_map : np.ndarray
        Shape: (X, Y, Z), dtype=float32
    """
    t_map = np.zeros(mask.shape, dtype=np.float32)
    p_map = np.ones(mask.shape, dtype=np.float32)

    eg_flat = eg_maps[:, mask]   # shape: (n_eg, n_vox)
    cg_flat = cg_maps[:, mask]   # shape: (n_cg, n_vox)

    t_vals, p_vals = ttest_ind(
        eg_flat,
        cg_flat,
        axis=0,
        equal_var=equal_var,
        nan_policy="raise"
    )

    t_map[mask] = t_vals.astype(np.float32)
    p_map[mask] = p_vals.astype(np.float32)

    return t_map, p_map


def benjamini_hochberg_fdr_mask(
    p_map: np.ndarray,
    mask: np.ndarray,
    q: float = 0.05
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Apply Benjamini-Hochberg FDR correction across all voxels in the brain mask.

    Parameters
    ----------
    p_map : np.ndarray
        Shape: (X, Y, Z)
        Voxelwise p-values.

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Brain/analysis mask. FDR is applied across all voxels in this mask.

    q : float
        Desired FDR level, e.g. 0.05.

    Returns
    -------
    significant_map : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Voxels surviving BH-FDR.

    p_threshold : float or None
        The largest raw p-value that survives BH.
        Returns None if no voxels survive.
    """
    pvals = p_map[mask].astype(np.float64)
    m = pvals.size

    if m == 0:
        raise ValueError("Mask contains no voxels.")

    order = np.argsort(pvals)
    p_sorted = pvals[order]

    ranks = np.arange(1, m + 1, dtype=np.float64)
    bh_line = (ranks / m) * q

    passed = p_sorted <= bh_line

    significant_map = np.zeros(mask.shape, dtype=bool)

    if not np.any(passed):
        return significant_map, None

    max_idx = np.where(passed)[0].max()
    p_threshold = float(p_sorted[max_idx])

    significant_map[mask] = pvals <= p_threshold
    return significant_map, p_threshold


def apply_cluster_extent_threshold(
    significant_map: np.ndarray,
    min_cluster_size: int = 10,
    connectivity: int = 26
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Apply a minimum cluster extent threshold to a boolean significance map.

    Parameters
    ----------
    significant_map : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Boolean map of voxels that survived voxelwise correction.

    min_cluster_size : int
        Minimum cluster size threshold k.

    connectivity : int
        One of {6, 18, 26}.

    Returns
    -------
    surviving_map : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Voxels that survive the cluster extent threshold.

    labeled_clusters : np.ndarray
        Shape: (X, Y, Z), dtype=int
        Connected-component labels of the input significance map.

    surviving_cluster_sizes : list[int]
        Sizes of clusters that survived.
    """
    if connectivity == 6:
        structure = generate_binary_structure(rank=3, connectivity=1)
    elif connectivity == 18:
        structure = generate_binary_structure(rank=3, connectivity=2)
    elif connectivity == 26:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
    else:
        raise ValueError("connectivity must be one of {6, 18, 26}")

    labeled_clusters, n_clusters = label(significant_map, structure=structure)

    surviving_map = np.zeros_like(significant_map, dtype=bool)
    surviving_cluster_sizes: List[int] = []

    for cluster_id in range(1, n_clusters + 1):
        cluster_voxels = (labeled_clusters == cluster_id)
        cluster_size = int(cluster_voxels.sum())

        if cluster_size >= min_cluster_size:
            surviving_map[cluster_voxels] = True
            surviving_cluster_sizes.append(cluster_size)

    return surviving_map, labeled_clusters, surviving_cluster_sizes


def two_group_whole_brain_fdr_test(
    eg_maps: np.ndarray,
    cg_maps: np.ndarray,
    mask: np.ndarray,
    signal_roi_mask: np.ndarray,
    q: float = 0.05,
    min_cluster_size: int = 10,
    equal_var: bool = True,
    connectivity: int = 26
) -> Dict[str, Any]:
    """
    Whole-brain EG vs CG second-level test with:
        1. voxelwise two-sample t-test
        2. voxelwise BH-FDR across all voxels in the brain mask
        3. cluster extent threshold k
        4. detection defined by overlap with the true signal ROI

    Parameters
    ----------
    eg_maps : np.ndarray
        Shape: (n_eg, X, Y, Z)

    cg_maps : np.ndarray
        Shape: (n_cg, X, Y, Z)

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Brain/analysis mask across which FDR is applied.

    signal_roi_mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Binary ROI where the true signal was injected.

    q : float
        FDR threshold, e.g. 0.05.

    min_cluster_size : int
        Cluster extent threshold k.

    equal_var : bool
        If True use pooled-variance t-test.

    connectivity : int
        Cluster connectivity: 6, 18, or 26.

    Returns
    -------
    result : dict
        Keys:
            "t_map"               : np.ndarray, shape (X, Y, Z)
            "p_map"               : np.ndarray, shape (X, Y, Z)
            "fdr_significant_map" : np.ndarray, shape (X, Y, Z), bool
            "clustered_map"       : np.ndarray, shape (X, Y, Z), bool
            "p_threshold"         : float or None
            "cluster_sizes"       : list[int]
            "detected_signal"     : bool
    """
    t_map, p_map = two_group_voxelwise_t_and_p_maps(
        eg_maps=eg_maps,
        cg_maps=cg_maps,
        mask=mask,
        equal_var=equal_var
    )

    fdr_significant_map, p_threshold = benjamini_hochberg_fdr_mask(
        p_map=p_map,
        mask=mask,
        q=q
    )

    clustered_map, _, cluster_sizes = apply_cluster_extent_threshold(
        significant_map=fdr_significant_map,
        min_cluster_size=min_cluster_size,
        connectivity=connectivity
    )

    detected_signal = bool(np.any(clustered_map & signal_roi_mask))

    return {
        "t_map": t_map,
        "p_map": p_map,
        "fdr_significant_map": fdr_significant_map,
        "clustered_map": clustered_map,
        "p_threshold": p_threshold,
        "cluster_sizes": cluster_sizes,
        "detected_signal": detected_signal
    }

def estimate_peak_fwe_t_threshold(
    n_null_iterations: int,
    n_eg: int,
    n_cg: int,
    noise_params: NoiseParameters,
    use_voxelwise_sd: bool = True,
    equal_var: bool = True,
    seed: int = 123
) -> tuple[float, np.ndarray]:
    """
    Estimate a whole-brain peak-level FWE critical |t| threshold empirically
    from null simulations for a fixed design.

    This approximates peak-level FWE by simulating datasets with NO true group
    difference, computing the voxelwise EG vs CG t-map for each null dataset,
    recording the maximum absolute t-value anywhere in the brain mask, and then
    taking the 95th percentile of that null max-|t| distribution.

    IMPORTANT
    ---------
    This threshold should be recomputed whenever the design changes, e.g.:
        - n_eg
        - n_cg
        - mask/search volume
        - smoothness
        - variance model

    Parameters
    ----------
    n_null_iterations : int
        Number of null datasets used to estimate the threshold.

    n_eg : int
        Number of EG subjects in the design.

    n_cg : int
        Number of CG subjects in the design.

    noise_params : NoiseParameters
        Pooled pilot-informed noise parameters.

    use_voxelwise_sd : bool
        If True use voxelwise SD, otherwise use global SD.

    equal_var : bool
        If True use pooled-variance t-test.

    seed : int
        RNG seed.

    Returns
    -------
    t_fwe_threshold : float
        Empirical whole-brain peak-level FWE critical |t| threshold for alpha=0.05.

    max_abs_t_null : np.ndarray
        Shape: (n_null_iterations,)
        Null distribution of max absolute t-values.
    """
    rng = np.random.default_rng(seed)
    max_abs_t_null = np.zeros(n_null_iterations, dtype=np.float64)

    zero_template = np.zeros(noise_params.mask.shape, dtype=np.float32)

    for it in range(n_null_iterations):
        if it % 25 == 0:
            print(n_eg, it)
        eg_maps, cg_maps = generate_two_group_contrast_maps(
            n_eg=n_eg,
            n_cg=n_cg,
            signal_template=zero_template,
            beta_eg=0.0,
            beta_cg=0.0,
            mask=noise_params.mask,
            fwhm_vox=noise_params.fwhm_vox,
            global_sd=None if use_voxelwise_sd else noise_params.global_sd,
            voxelwise_sd=noise_params.voxelwise_sd if use_voxelwise_sd else None,
            subject_beta_sd=0.0,
            rng=rng
        )

        t_map, _ = two_group_voxelwise_t_and_p_maps(
            eg_maps=eg_maps,
            cg_maps=cg_maps,
            mask=noise_params.mask,
            equal_var=equal_var
        )

        max_abs_t_null[it] = float(np.max(np.abs(t_map[noise_params.mask])))

    t_fwe_threshold = float(np.quantile(max_abs_t_null, 0.95))
    return t_fwe_threshold, max_abs_t_null

def two_group_whole_brain_fwe_test(
    eg_maps: np.ndarray,
    cg_maps: np.ndarray,
    mask: np.ndarray,
    signal_roi_mask: np.ndarray,
    t_fwe_threshold: float,
    min_cluster_size: int = 10,
    equal_var: bool = True,
    connectivity: int = 26
) -> Dict[str, Any]:
    """
    Whole-brain EG vs CG second-level test using a PRECOMPUTED peak-level
    FWE-corrected |t| threshold.

    PIPELINE
    --------
    1. voxelwise two-sample t-test across the brain mask
    2. threshold voxels at abs(t) >= t_fwe_threshold
    3. apply cluster extent threshold k
    4. detection = any surviving cluster overlaps the true signal ROI

    Parameters
    ----------
    eg_maps : np.ndarray
        Shape: (n_eg, X, Y, Z)

    cg_maps : np.ndarray
        Shape: (n_cg, X, Y, Z)

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    signal_roi_mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Binary ROI where the true signal was injected.

    t_fwe_threshold : float
        Precomputed empirical whole-brain peak-level FWE |t| threshold.

    min_cluster_size : int
        Cluster extent threshold k.

    equal_var : bool
        If True use pooled-variance t-test.

    connectivity : int
        Cluster connectivity: 6, 18, or 26.

    Returns
    -------
    result : dict
        Keys:
            "t_map"            : np.ndarray, shape (X, Y, Z)
            "significant_map"  : np.ndarray, shape (X, Y, Z), bool
            "clustered_map"    : np.ndarray, shape (X, Y, Z), bool
            "t_threshold"      : float
            "cluster_sizes"    : list[int]
            "detected_signal"  : bool
    """
    t_map, _ = two_group_voxelwise_t_and_p_maps(
        eg_maps=eg_maps,
        cg_maps=cg_maps,
        mask=mask,
        equal_var=equal_var
    )

    significant_map = (np.abs(t_map) >= t_fwe_threshold) & mask

    clustered_map, _, cluster_sizes = apply_cluster_extent_threshold(
        significant_map=significant_map,
        min_cluster_size=min_cluster_size,
        connectivity=connectivity
    )

    detected_signal = bool(np.any(clustered_map & signal_roi_mask))

    return {
        "t_map": t_map,
        "significant_map": significant_map,
        "clustered_map": clustered_map,
        "t_threshold": float(t_fwe_threshold),
        "cluster_sizes": cluster_sizes,
        "detected_signal": detected_signal
    }
# =============================================================================
# 9. MONTE CARLO POWER FOR WHOLE-BRAIN EG VS CG ANALYSIS
# =============================================================================

def monte_carlo_power_two_group_whole_brain_fdr(
    n_iterations: int,
    n_eg: int,
    n_cg: int,
    signal_template: np.ndarray,
    signal_roi_mask: np.ndarray,
    noise_params: NoiseParameters,
    beta_eg: float,
    beta_cg: float,
    use_voxelwise_sd: bool = True,
    q: float = 0.05,
    min_cluster_size: int = 10,
    subject_beta_sd: float = 0.0,
    equal_var: bool = True,
    connectivity: int = 26,
    seed: int = 123
) -> WholeBrainPowerResult:
    """
    Monte Carlo power for the planned whole-brain EG vs CG analysis:
        voxelwise t-test -> BH-FDR across all brain voxels -> cluster extent k

    Success criterion:
        at least one surviving cluster overlaps the true signal ROI.

    Parameters
    ----------
    n_iterations : int
        Number of Monte Carlo iterations.

    n_eg : int
        Number of EG subjects per iteration.

    n_cg : int
        Number of CG subjects per iteration.

    signal_template : np.ndarray
        Shape: (X, Y, Z)
        Continuous template used to generate the signal.

    signal_roi_mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Binary ROI where the signal is injected.

    noise_params : NoiseParameters
        Pooled pilot-informed noise parameters.

    beta_eg : float
        Mean signal amplitude in EG.

    beta_cg : float
        Mean signal amplitude in CG.

    use_voxelwise_sd : bool
        If True use voxelwise SD, otherwise use global SD.

    q : float
        FDR level.

    min_cluster_size : int
        Cluster extent threshold k.

    subject_beta_sd : float
        Between-subject variability in signal amplitude.

    equal_var : bool
        If True use pooled-variance t-test.

    connectivity : int
        Cluster connectivity: 6, 18, or 26.

    seed : int
        RNG seed.

    Returns
    -------
    WholeBrainPowerResult
    """
    rng = np.random.default_rng(seed)

    detected_signal = np.zeros(n_iterations, dtype=bool)
    p_thresholds = np.full(n_iterations, np.nan, dtype=np.float64)
    n_fdr_voxels = np.zeros(n_iterations, dtype=np.int32)
    n_clustered_voxels = np.zeros(n_iterations, dtype=np.int32)
    n_surviving_clusters = np.zeros(n_iterations, dtype=np.int32)

    for it in range(n_iterations):
        eg_maps, cg_maps = generate_two_group_contrast_maps(
            n_eg=n_eg,
            n_cg=n_cg,
            signal_template=signal_template,
            beta_eg=beta_eg,
            beta_cg=beta_cg,
            mask=noise_params.mask,
            fwhm_vox=noise_params.fwhm_vox,
            global_sd=None if use_voxelwise_sd else noise_params.global_sd,
            voxelwise_sd=noise_params.voxelwise_sd if use_voxelwise_sd else None,
            subject_beta_sd=subject_beta_sd,
            rng=rng
        )

        test_result = two_group_whole_brain_fdr_test(
            eg_maps=eg_maps,
            cg_maps=cg_maps,
            mask=noise_params.mask,
            signal_roi_mask=signal_roi_mask,
            q=q,
            min_cluster_size=min_cluster_size,
            equal_var=equal_var,
            connectivity=connectivity
        )

        detected_signal[it] = test_result["detected_signal"]

        if test_result["p_threshold"] is not None:
            p_thresholds[it] = float(test_result["p_threshold"])

        n_fdr_voxels[it] = int(test_result["fdr_significant_map"].sum())
        n_clustered_voxels[it] = int(test_result["clustered_map"].sum())
        n_surviving_clusters[it] = len(test_result["cluster_sizes"])

    return WholeBrainPowerResult(
        power=float(np.mean(detected_signal)),
        detected_signal=detected_signal,
        p_thresholds=p_thresholds,
        n_fdr_voxels=n_fdr_voxels,
        n_clustered_voxels=n_clustered_voxels,
        n_surviving_clusters=n_surviving_clusters,
        n_iterations=n_iterations,
    )


def monte_carlo_power_two_group_whole_brain_fwe(
    n_iterations: int,
    n_eg: int,
    n_cg: int,
    signal_template: np.ndarray,
    signal_roi_mask: np.ndarray,
    noise_params: NoiseParameters,
    beta_eg: float,
    beta_cg: float,
    t_fwe_threshold: float,
    use_voxelwise_sd: bool = True,
    min_cluster_size: int = 10,
    subject_beta_sd: float = 0.0,
    equal_var: bool = True,
    connectivity: int = 26,
    seed: int = 123
) -> WholeBrainPowerResult:
    """
    Monte Carlo power for the planned whole-brain EG vs CG analysis:
        voxelwise t-test -> peak-level FWE threshold -> cluster extent k

    Success criterion:
        at least one surviving cluster overlaps the true signal ROI.

    Parameters
    ----------
    n_iterations : int
        Number of Monte Carlo iterations.

    n_eg : int
        Number of EG subjects per iteration.

    n_cg : int
        Number of CG subjects per iteration.

    signal_template : np.ndarray
        Shape: (X, Y, Z)
        Continuous template used to generate the signal.

    signal_roi_mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool
        Binary ROI where the signal is injected.

    noise_params : NoiseParameters
        Pooled pilot-informed noise parameters.

    beta_eg : float
        Mean signal amplitude in EG.

    beta_cg : float
        Mean signal amplitude in CG.

    t_fwe_threshold : float
        Precomputed empirical peak-level FWE |t| threshold for this design.

    use_voxelwise_sd : bool
        If True use voxelwise SD, otherwise use global SD.

    min_cluster_size : int
        Cluster extent threshold k.

    subject_beta_sd : float
        Between-subject variability in signal amplitude.

    equal_var : bool
        If True use pooled-variance t-test.

    connectivity : int
        Cluster connectivity: 6, 18, or 26.

    seed : int
        RNG seed.

    Returns
    -------
    WholeBrainPowerResult
    """
    rng = np.random.default_rng(seed)

    detected_signal = np.zeros(n_iterations, dtype=bool)
    t_thresholds = np.full(n_iterations, float(t_fwe_threshold), dtype=np.float64)
    n_fwe_voxels = np.zeros(n_iterations, dtype=np.int32)
    n_clustered_voxels = np.zeros(n_iterations, dtype=np.int32)
    n_surviving_clusters = np.zeros(n_iterations, dtype=np.int32)

    for it in range(n_iterations):
        if it % 25 == 0:
            print(n_eg, it)
        eg_maps, cg_maps = generate_two_group_contrast_maps(
            n_eg=n_eg,
            n_cg=n_cg,
            signal_template=signal_template,
            beta_eg=beta_eg,
            beta_cg=beta_cg,
            mask=noise_params.mask,
            fwhm_vox=noise_params.fwhm_vox,
            global_sd=None if use_voxelwise_sd else noise_params.global_sd,
            voxelwise_sd=noise_params.voxelwise_sd if use_voxelwise_sd else None,
            subject_beta_sd=subject_beta_sd,
            rng=rng
        )

        test_result = two_group_whole_brain_fwe_test(
            eg_maps=eg_maps,
            cg_maps=cg_maps,
            mask=noise_params.mask,
            signal_roi_mask=signal_roi_mask,
            t_fwe_threshold=t_fwe_threshold,
            min_cluster_size=min_cluster_size,
            equal_var=equal_var,
            connectivity=connectivity
        )

        detected_signal[it] = test_result["detected_signal"]
        n_fwe_voxels[it] = int(test_result["significant_map"].sum())
        n_clustered_voxels[it] = int(test_result["clustered_map"].sum())
        n_surviving_clusters[it] = len(test_result["cluster_sizes"])

    return WholeBrainPowerResult(
        power=float(np.mean(detected_signal)),
        detected_signal=detected_signal,
        t_thresholds=t_thresholds,
        n_fwe_voxels=n_fwe_voxels,
        n_clustered_voxels=n_clustered_voxels,
        n_surviving_clusters=n_surviving_clusters,
        n_iterations=n_iterations,
    )



# =============================================================================
# 10. SAVE NIFTI MAPS
# =============================================================================

def save_nifti_maps(
    maps: np.ndarray,
    output_dir: Path,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    prefix: str
) -> List[Path]:
    """
    Save a batch of maps to NIfTI files.

    Parameters
    ----------
    maps : np.ndarray
        Shape: (n_subjects, X, Y, Z)

    output_dir : Path
        Output directory.

    affine : np.ndarray
        Shape: (4, 4)

    header : nib.Nifti1Header
        Header template.

    prefix : str
        Filename prefix.

    Returns
    -------
    saved_paths : list[Path]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for i in range(maps.shape[0]):
        out_path = output_dir / f"{prefix}_{i + 1:04d}.nii.gz"
        img = nib.Nifti1Image(maps[i].astype(np.float32), affine=affine, header=header)
        nib.save(img, str(out_path))
        saved_paths.append(out_path)

    return saved_paths


# =============================================================================
# 11. DIAGNOSTICS
# =============================================================================

def print_mask_diagnostics(mask: np.ndarray, voxel_sizes_mm: Tuple[float, float, float]) -> None:
    """
    Print mask diagnostics.

    Parameters
    ----------
    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    voxel_sizes_mm : tuple[float, float, float]
        Voxel sizes in mm.
    """
    total_voxels = int(np.prod(mask.shape))
    mask_voxels = int(mask.sum())

    voxel_volume_mm3 = float(np.prod(voxel_sizes_mm))
    mask_volume_ml = mask_voxels * voxel_volume_mm3 / 1000.0

    print("Mask diagnostics")
    print("----------------")
    print(f"Mask voxels: {mask_voxels}")
    print(f"Mask proportion: {mask_voxels / total_voxels:.3f}")
    print(f"Approx mask volume (mL): {mask_volume_ml:.1f}")


def summarize_subject_spatial_sds(maps: np.ndarray, mask: np.ndarray, label: str) -> None:
    """
    Summarize within-mask spatial SD across subjects.

    Parameters
    ----------
    maps : np.ndarray
        Shape: (n_subjects, X, Y, Z)

    mask : np.ndarray
        Shape: (X, Y, Z), dtype=bool

    label : str
        Label for printing.
    """
    sds = np.array([maps[i][mask].std(ddof=1) for i in range(maps.shape[0])], dtype=np.float64)

    print(f"{label} spatial SD summary")
    print("--------------------------")
    print(f"Mean SD:   {sds.mean():.6f}")
    print(f"Median SD: {np.median(sds):.6f}")
    print(f"Min SD:    {sds.min():.6f}")
    print(f"Max SD:    {sds.max():.6f}")


# =============================================================================
# 12. MAIN EXAMPLE
# =============================================================================

def main():
    """
    Example usage of the full whole-brain EG vs CG simulation pipeline.

    IMPORTANT
    ---------
    This example is configured to use the mock NIfTI files.
    Update the path below once you have real pilot subject-level contrast maps.
    """
    # -------------------------------------------------------------------------
    # A. Point this to your pilot subject-level contrast maps
    # -------------------------------------------------------------------------
    # pilot_files = sorted(Path("/Users/brendanbrady/Documents/Funding_Opportunities/2026_Advancing 2S:LGBTQI/power_calculations/H1/mock_pilot_niftis").glob("sub-*_con_mock.nii.gz"))

    # if len(pilot_files) == 0:
    #     raise FileNotFoundError(
    #         "No pilot files found. Update the path in main()."
    #     )

    # -------------------------------------------------------------------------
    # B. Load pilot maps
    # -------------------------------------------------------------------------
    #pilot = load_pilot_niftis(pilot_files)
    
    pilot = load_pilot_mat("BetaWeights_forBrendanSimulation/Minoritystress_n20_subject_voxel_matrix.mat", 
                           "BetaWeights_forBrendanSimulation/Minoritystress_n20_wholesample_intersection_mask.nii")
    
    print("Loaded pilot maps")
    print("-----------------")
    print(f"n_pilot: {pilot.maps.shape[0]}")
    print(f"map shape: {pilot.maps.shape[1:]}")
    print(f"voxel sizes (mm): {pilot.voxel_sizes_mm}")
    print()

    # sanity check
    # import matplotlib.pyplot as plt
    # plt.imshow(pilot.maps[0][:,:,40], origin="lower", cmap="gray")
    # plt.title("Subject 1 slice")
    # plt.show()

    # -------------------------------------------------------------------------
    # C. Estimate pooled noise parameters
    # -------------------------------------------------------------------------
    noise_params = estimate_noise_parameters(
        pilot_maps=pilot.maps,
        voxel_sizes_mm=pilot.voxel_sizes_mm,
        mask=None,
        nonzero_fraction_threshold=0.95
    )

    print("Estimated pooled noise parameters")
    print("--------------------------------")
    print(f"global SD: {noise_params.global_sd:.6f}")
    print(f"effective FWHM (vox): {noise_params.fwhm_vox:.6f}")
    print(f"effective FWHM (mm):  {noise_params.fwhm_mm:.6f}")
    print()

    print_mask_diagnostics(noise_params.mask, pilot.voxel_sizes_mm)
    print()
    
    # -------------------------------------------------------------------------
    # D. Define the true signal template and binary signal ROI
    # -------------------------------------------------------------------------
    shape = pilot.maps.shape[1:]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    signal_template = create_spherical_signal_template(
        shape=shape,
        center_xyz=center,
        radius_vox=5.0,
        smooth_sigma_vox=1.5,
        mask=noise_params.mask,
        peak_value=1.0
    )

    # Binary ground-truth signal region used for defining detection
    signal_roi_mask = signal_template > 0.2
    # plt.imshow(signal_template[:,:, shape[2] // 2])
    # plt.show()
    
    # -------------------------------------------------------------------------
    # E. Generate one synthetic dataset for EG and CG
    # -------------------------------------------------------------------------
    

    x = 2.15  # pilot-derived raw beta difference (difference in group means of subject-level ROI 95th percentiles)
    shrinkage = 1
    beta_diff = shrinkage * x
    print('beta diff =', beta_diff)
    print('')


    # quick calibration to ensure injected signal in ROI actually gives the correct effect size


    rng = np.random.default_rng(41)

    eg_maps, cg_maps = generate_two_group_contrast_maps(
        n_eg=80,
        n_cg=80,
        signal_template=signal_template,
        beta_eg=beta_diff, # note that these are the peak voxel effects in the ROI center
        beta_cg=0, # note that these are the peak voxel effects in the ROI center
        mask=noise_params.mask,
        fwhm_vox=noise_params.fwhm_vox,
        global_sd=None,
        voxelwise_sd=noise_params.voxelwise_sd,
        subject_beta_sd=0.03,
        rng=rng
    )

    print("Generated synthetic group maps")
    print("------------------------------")
    print(f"EG maps shape: {eg_maps.shape}")
    print(f"CG maps shape: {cg_maps.shape}")
    print()

    summarize_subject_spatial_sds(eg_maps, noise_params.mask, "EG")
    print()
    summarize_subject_spatial_sds(cg_maps, noise_params.mask, "CG")
    print()
    
  
    #average the maps for each group and plot cross section
    eg_map_mean = np.mean(eg_maps, 0)
    cg_map_mean = np.mean(cg_maps, 0)
    plt.imshow(eg_map_mean[:,:,shape[2] // 2], origin='lower', vmin=-7, vmax=7)
    plt.show()
    plt.imshow(cg_map_mean[:,:,shape[2] // 2], origin='lower', vmin=-7, vmax=7)
    plt.show()

    print(eg_map_mean.shape)
    print(signal_roi_mask.shape)
    print('eg ROI mean beta:', np.mean(eg_map_mean[signal_roi_mask==1]))
    print('cg ROI mean beta:', np.mean(cg_map_mean[signal_roi_mask==1]))
    print('eg ROI max beta:', eg_map_mean[signal_roi_mask==1].max())
    print('cg ROI max beta:', cg_map_mean[signal_roi_mask==1].max())
    diff = eg_map_mean[signal_roi_mask==1] - cg_map_mean[signal_roi_mask==1]
    print('median voxel-wise effect size numerator :',np.median(diff))
    
    
    # -------------------------------------------------------------------------
    # F. Estimate the empirical peak-level FWE threshold for this design and
    #    run one corrected EG vs CG test
    # -------------------------------------------------------------------------
    # t_fwe_threshold, null_max_abs_t = estimate_peak_fwe_t_threshold(
    #     n_null_iterations=1,
    #     n_eg=40,
    #     n_cg=40,
    #     noise_params=noise_params,
    #     use_voxelwise_sd=True,
    #     equal_var=True,
    #     seed=999
    # )

    # single_test = two_group_whole_brain_fwe_test(
    #     eg_maps=eg_maps,
    #     cg_maps=cg_maps,
    #     mask=noise_params.mask,
    #     signal_roi_mask=signal_roi_mask,
    #     t_fwe_threshold=t_fwe_threshold,
    #     min_cluster_size=10,
    #     equal_var=True,
    #     connectivity=26
    # )

    # print("Single whole-brain peak-FWE-corrected test")
    # print("-------------------------------------------")
    # print(f"Peak-FWE critical |t| threshold: {single_test['t_threshold']:.4f}")
    # print(f"Detected true signal: {single_test['detected_signal']}")
    # print(f"Number of peak-FWE-significant voxels: {int(single_test['significant_map'].sum())}")
    # print(f"Number of clustered surviving voxels: {int(single_test['clustered_map'].sum())}")
    # print(f"Surviving cluster sizes: {single_test['cluster_sizes']}")
    # print()
    
    
    # -------------------------------------------------------------------------
    # G. Whole-brain Monte Carlo power
    # -------------------------------------------------------------------------
    
    

    for n_per_group in [30,50,70,90,110,130]:

        t_fwe_threshold_n, _ = estimate_peak_fwe_t_threshold(
            n_null_iterations=300,
            n_eg=n_per_group,
            n_cg=n_per_group,
            noise_params=noise_params,
            use_voxelwise_sd=True,
            equal_var=True,
            seed=1000 + n_per_group
        )

        wb = monte_carlo_power_two_group_whole_brain_fwe(
            n_iterations=2000, 
            n_eg=n_per_group,
            n_cg=n_per_group,
            signal_template=signal_template,
            signal_roi_mask=signal_roi_mask,
            noise_params=noise_params,
            beta_eg=beta_diff, # note that these are the peak voxel effects in the ROI center
            beta_cg=0, # note that these are the peak voxel effects in the ROI center
            t_fwe_threshold=t_fwe_threshold_n,
            use_voxelwise_sd=True,
            min_cluster_size=10,
            subject_beta_sd=0.03,
            equal_var=True,
            connectivity=26,
            seed=123
        )

        print("Whole-brain Monte Carlo power (peak-FWE + cluster extent)")
        print("----------------------------------------------------------")
        print(f"n per group: {n_per_group}")
        print(f"Number of datasets simulated: {wb.n_iterations:.3f}")
        print(f"Estimated power: {wb.power:.3f}")
        print(f"Peak-FWE critical |t| threshold: {t_fwe_threshold_n:.4f}")
        print(f"Mean number of peak-FWE-significant voxels: {wb.n_fwe_voxels.mean():.2f}")
        print(f"Mean number of clustered surviving voxels: {wb.n_clustered_voxels.mean():.2f}")
        print(f"Mean number of surviving clusters: {wb.n_surviving_clusters.mean():.2f}")
        print()

    # -------------------------------------------------------------------------
    # H. Optional: save one simulated EG/CG dataset to disk
    # -------------------------------------------------------------------------
    out_dir = Path("./synthetic_two_group_maps")

    save_nifti_maps(
        maps=eg_maps,
        output_dir=out_dir / "EG",
        affine=pilot.affine,
        header=pilot.header,
        prefix="EG_sub"
    )

    save_nifti_maps(
        maps=cg_maps,
        output_dir=out_dir / "CG",
        affine=pilot.affine,
        header=pilot.header,
        prefix="CG_sub"
    )

    print(f"Saved synthetic maps under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()