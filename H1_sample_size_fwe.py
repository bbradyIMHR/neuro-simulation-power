"""
Pilot-informed simulation of subject-level neuroimaging contrast maps
for a two-group design (EG vs CG), using whole-brain voxelwise testing
with peak-level FWE correction and a post-threshold cluster extent filter.

Overview
--------
Each subject contributes one subject-level contrast map, for example:

    contrast_map = ConditionA - ConditionB

Simulated subject maps are generated as:

    map_i = beta_i * signal_template + spatial_noise_i

Group differences are then tested at the second level using:

    1. voxelwise EG vs CG two-sample t-test across the analysis mask
    2. peak-level FWE correction using an empirically estimated |t| threshold
    3. cluster extent threshold
    4. detection criterion = any surviving cluster overlaps the injected signal ROI

Required input files
--------------------
1. A MATLAB v7.3 .mat file containing:
    - data_matrix : voxel-by-subject matrix
    - mask_idx    : 3D binary analysis mask

2. A reference NIfTI file (.nii or .nii.gz) defining:
    - the target image grid
    - affine/header metadata used for reconstructed subject maps

Dependencies
------------
pip install numpy scipy nibabel h5py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, label, generate_binary_structure
from scipy.stats import ttest_ind


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """
    User-editable configuration for whole-brain two-group Monte Carlo power analysis.

    Required inputs
    ---------------
    pilot_mat_file : Path
        MATLAB v7.3 .mat file containing:
            - data_matrix : voxel-by-subject matrix
            - mask_idx    : 3D binary mask

    reference_nifti_file : Path
        NIfTI file defining the target image grid and image metadata.

    Optional outputs
    ----------------
    save_example_maps : bool
        Whether to save one simulated EG/CG dataset as NIfTI files.

    output_dir : Path
        Output directory used if save_example_maps=True.
    """

    # Required input files
    pilot_mat_file: Path
    reference_nifti_file: Path

    # Signal model
    signal_radius_vox: float = 5.0
    signal_smoothing_sigma_vox: float = 1.5
    signal_peak_value: float = 1.0
    signal_roi_threshold: float = 0.2

    # Group effect model
    beta_eg: float = 2.15
    beta_cg: float = 0.0
    subject_beta_sd: float = 0.03

    # Pilot-derived noise estimation
    nonzero_fraction_threshold: float = 0.95
    use_voxelwise_sd: bool = True

    # Inference settings
    min_cluster_size: int = 10
    connectivity: int = 26
    equal_var: bool = True

    # Null threshold estimation
    n_null_iterations: int = 500

    # Monte Carlo power simulation
    n_power_iterations: int = 2000
    sample_sizes_per_group: Tuple[int, ...] = (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140)

    # Random seeds
    null_seed_base: int = 1000
    simulation_seed: int = 123

    # Optional outputs
    save_example_maps: bool = False
    output_dir: Path = Path("synthetic_two_group_maps")


# =============================================================================
# 2. DATA CONTAINERS
# =============================================================================

@dataclass
class PilotData:
    """Stores pilot subject-level contrast maps and image metadata."""
    maps: np.ndarray
    affine: Optional[np.ndarray] = None
    header: Optional[nib.Nifti1Header] = None
    voxel_sizes_mm: Optional[Tuple[float, float, float]] = None
    file_paths: Optional[List[Path]] = None
    subject_ids: Optional[List[str]] = None
    mask: Optional[np.ndarray] = None


@dataclass
class NoiseParameters:
    """Stores pooled pilot-informed noise parameters."""
    mask: np.ndarray
    mean_map: np.ndarray
    global_sd: float
    voxelwise_sd: np.ndarray
    fwhm_vox: float
    fwhm_mm: float


@dataclass
class WholeBrainPowerResult:
    """Stores whole-brain Monte Carlo power results."""
    power: float
    detected_signal: np.ndarray
    t_thresholds: np.ndarray
    n_fwe_voxels: np.ndarray
    n_clustered_voxels: np.ndarray
    n_surviving_clusters: np.ndarray
    n_iterations: int


# =============================================================================
# 3. PILOT DATA LOADING
# =============================================================================

def load_pilot_mat(mat_file: Path, reference_nifti_file: Path) -> PilotData:
    """
    Load pilot maps from a MATLAB voxel matrix and attach spatial metadata
    from a reference NIfTI file.

    Expected .mat contents
    ----------------------
    data_matrix : 2D array
        Shape (n_voxels, n_subjects)

    mask_idx : 3D binary array
        Analysis mask corresponding to the voxel rows of data_matrix
    """
    mat_file = Path(mat_file)
    reference_nifti_file = Path(reference_nifti_file)

    ref_img = nib.load(str(reference_nifti_file))
    affine = ref_img.affine.copy()
    header = ref_img.header.copy()
    voxel_sizes_mm = tuple(float(v) for v in header.get_zooms()[:3])
    shape = ref_img.shape

    with h5py.File(mat_file, "r") as f:
        if "data_matrix" not in f:
            raise KeyError("Expected variable 'data_matrix' not found in .mat file.")
        if "mask_idx" not in f:
            raise KeyError("Expected variable 'mask_idx' not found in .mat file.")

        data_matrix = f["data_matrix"][:]
        mask = f["mask_idx"][:].astype(bool)

    if mask.shape != shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference NIfTI shape {shape}."
        )

    n_voxels, n_subjects = data_matrix.shape

    if int(mask.sum()) != n_voxels:
        raise ValueError(
            f"Mask contains {int(mask.sum())} voxels but data_matrix has {n_voxels} rows."
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
        file_paths=[mat_file],
        mask=mask,
    )


# =============================================================================
# 4. MASK AND NOISE ESTIMATION
# =============================================================================

def estimate_mask_from_pilot_maps(
    pilot_maps: np.ndarray,
    nonzero_fraction_threshold: float = 0.95,
    finite_only: bool = True,
) -> np.ndarray:
    """Estimate a conservative analysis mask from pilot maps."""
    if pilot_maps.ndim != 4:
        raise ValueError(f"Expected pilot_maps shape (n_subjects, X, Y, Z), got {pilot_maps.shape}")

    nonzero_fraction = np.mean(pilot_maps != 0, axis=0)
    mask = nonzero_fraction >= nonzero_fraction_threshold

    if finite_only:
        mask &= np.all(np.isfinite(pilot_maps), axis=0)

    return mask


def estimate_global_sd(pilot_maps: np.ndarray, mask: np.ndarray) -> float:
    """Estimate the mean within-mask spatial SD across pilot subjects."""
    subject_sds = [float(pilot_maps[i][mask].std(ddof=1)) for i in range(pilot_maps.shape[0])]
    return float(np.mean(subject_sds))


def estimate_voxelwise_sd(
    pilot_maps: np.ndarray,
    mask: np.ndarray,
    min_sd: float = 1e-6,
) -> np.ndarray:
    """Estimate voxelwise SD across pilot subjects."""
    voxelwise_sd = np.std(pilot_maps, axis=0, ddof=1).astype(np.float32)
    voxelwise_sd[mask] = np.maximum(voxelwise_sd[mask], min_sd)
    voxelwise_sd[~mask] = 0.0
    return voxelwise_sd


def estimate_effective_fwhm_vox(pilot_maps: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimate effective isotropic smoothness in voxel units using lag-1 spatial correlation.
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
    return float(2.355 * sigma_vox)


def estimate_noise_parameters(
    pilot_maps: np.ndarray,
    voxel_sizes_mm: Tuple[float, float, float],
    mask: Optional[np.ndarray] = None,
    nonzero_fraction_threshold: float = 0.95,
) -> NoiseParameters:
    """Estimate pooled pilot-informed noise parameters."""
    if mask is None:
        mask = estimate_mask_from_pilot_maps(
            pilot_maps=pilot_maps,
            nonzero_fraction_threshold=nonzero_fraction_threshold,
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
        fwhm_mm=fwhm_mm,
    )


# =============================================================================
# 5. SIGNAL TEMPLATE
# =============================================================================

def create_spherical_signal_template(
    shape: Tuple[int, int, int],
    center_xyz: Tuple[int, int, int],
    radius_vox: float,
    smooth_sigma_vox: float = 0.0,
    mask: Optional[np.ndarray] = None,
    peak_value: float = 1.0,
) -> np.ndarray:
    """Create a spherical signal template."""
    x = np.arange(shape[0], dtype=np.float32)
    y = np.arange(shape[1], dtype=np.float32)
    z = np.arange(shape[2], dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    dist2 = (
        (xx - center_xyz[0]) ** 2
        + (yy - center_xyz[1]) ** 2
        + (zz - center_xyz[2]) ** 2
    )

    signal = (dist2 <= radius_vox**2).astype(np.float32)

    if smooth_sigma_vox > 0:
        signal = gaussian_filter(signal, sigma=smooth_sigma_vox)

    if signal.max() > 0:
        signal = signal / signal.max() * peak_value

    if mask is not None:
        signal[~mask] = 0.0

    return signal.astype(np.float32)


# =============================================================================
# 6. SPATIAL NOISE AND SUBJECT MAP GENERATION
# =============================================================================

def generate_spatially_correlated_noise(
    n_subjects: int,
    mask: np.ndarray,
    fwhm_vox: float,
    global_sd: Optional[float] = None,
    voxelwise_sd: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Generate subject-level spatially correlated noise maps.

    Exactly one of global_sd or voxelwise_sd must be provided.
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
    dtype=np.float32,
) -> np.ndarray:
    """Generate subject-level contrast maps for one group."""
    if rng is None:
        rng = np.random.default_rng()

    noise_maps = generate_spatially_correlated_noise(
        n_subjects=n_subjects,
        mask=mask,
        fwhm_vox=fwhm_vox,
        global_sd=global_sd,
        voxelwise_sd=voxelwise_sd,
        rng=rng,
        dtype=dtype,
    )

    maps = np.zeros_like(noise_maps, dtype=dtype)

    for i in range(n_subjects):
        beta_i = float(rng.normal(group_beta, subject_beta_sd)) if subject_beta_sd > 0 else float(group_beta)
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
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate subject-level contrast maps for both EG and CG."""
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
        dtype=dtype,
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
        dtype=dtype,
    )

    return eg_maps, cg_maps


# =============================================================================
# 7. SECOND-LEVEL TESTING
# =============================================================================

def two_group_voxelwise_t_map(
    eg_maps: np.ndarray,
    cg_maps: np.ndarray,
    mask: np.ndarray,
    equal_var: bool = True,
) -> np.ndarray:
    """Compute voxelwise EG vs CG t-statistics."""
    t_map = np.zeros(mask.shape, dtype=np.float32)

    eg_flat = eg_maps[:, mask]
    cg_flat = cg_maps[:, mask]

    t_vals, _ = ttest_ind(
        eg_flat,
        cg_flat,
        axis=0,
        equal_var=equal_var,
        nan_policy="raise",
    )

    t_map[mask] = t_vals.astype(np.float32)
    return t_map


def apply_cluster_extent_threshold(
    significant_map: np.ndarray,
    min_cluster_size: int = 10,
    connectivity: int = 26,
) -> Tuple[np.ndarray, List[int]]:
    """Apply a minimum cluster extent threshold to a boolean map."""
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
        cluster_voxels = labeled_clusters == cluster_id
        cluster_size = int(cluster_voxels.sum())

        if cluster_size >= min_cluster_size:
            surviving_map[cluster_voxels] = True
            surviving_cluster_sizes.append(cluster_size)

    return surviving_map, surviving_cluster_sizes


def estimate_peak_fwe_t_threshold(
    n_null_iterations: int,
    n_eg: int,
    n_cg: int,
    noise_params: NoiseParameters,
    use_voxelwise_sd: bool = True,
    equal_var: bool = True,
    seed: int = 123,
    progress_every: int = 25,
) -> Tuple[float, np.ndarray]:
    """
    Estimate a whole-brain peak-level FWE |t| threshold empirically from null simulations.
    """
    rng = np.random.default_rng(seed)
    max_abs_t_null = np.zeros(n_null_iterations, dtype=np.float64)
    zero_template = np.zeros(noise_params.mask.shape, dtype=np.float32)

    for it in range(n_null_iterations):
        if progress_every and (it + 1) % progress_every == 0:
            print(f"  Null threshold estimation: {it + 1}/{n_null_iterations}")

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
            rng=rng,
        )

        t_map = two_group_voxelwise_t_map(
            eg_maps=eg_maps,
            cg_maps=cg_maps,
            mask=noise_params.mask,
            equal_var=equal_var,
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
    connectivity: int = 26,
) -> Dict[str, Any]:
    """
    Whole-brain EG vs CG test using a precomputed peak-level FWE |t| threshold.
    """
    t_map = two_group_voxelwise_t_map(
        eg_maps=eg_maps,
        cg_maps=cg_maps,
        mask=mask,
        equal_var=equal_var,
    )

    significant_map = (np.abs(t_map) >= t_fwe_threshold) & mask

    clustered_map, cluster_sizes = apply_cluster_extent_threshold(
        significant_map=significant_map,
        min_cluster_size=min_cluster_size,
        connectivity=connectivity,
    )

    detected_signal = bool(np.any(clustered_map & signal_roi_mask))

    return {
        "t_map": t_map,
        "significant_map": significant_map,
        "clustered_map": clustered_map,
        "t_threshold": float(t_fwe_threshold),
        "cluster_sizes": cluster_sizes,
        "detected_signal": detected_signal,
    }


# =============================================================================
# 8. MONTE CARLO POWER
# =============================================================================

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
    seed: int = 123,
    progress_every: int = 100,
) -> WholeBrainPowerResult:
    """
    Monte Carlo power for voxelwise t-test -> peak-level FWE threshold -> cluster extent.
    """
    rng = np.random.default_rng(seed)

    detected_signal = np.zeros(n_iterations, dtype=bool)
    t_thresholds = np.full(n_iterations, float(t_fwe_threshold), dtype=np.float64)
    n_fwe_voxels = np.zeros(n_iterations, dtype=np.int32)
    n_clustered_voxels = np.zeros(n_iterations, dtype=np.int32)
    n_surviving_clusters = np.zeros(n_iterations, dtype=np.int32)

    for it in range(n_iterations):
        if progress_every and (it + 1) % progress_every == 0:
            print(f"  Power simulation: {it + 1}/{n_iterations}")

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
            rng=rng,
        )

        test_result = two_group_whole_brain_fwe_test(
            eg_maps=eg_maps,
            cg_maps=cg_maps,
            mask=noise_params.mask,
            signal_roi_mask=signal_roi_mask,
            t_fwe_threshold=t_fwe_threshold,
            min_cluster_size=min_cluster_size,
            equal_var=equal_var,
            connectivity=connectivity,
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
# 9. OPTIONAL OUTPUT
# =============================================================================

def save_nifti_maps(
    maps: np.ndarray,
    output_dir: Path,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    prefix: str,
) -> List[Path]:
    """Save a batch of maps to NIfTI files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for i in range(maps.shape[0]):
        out_path = output_dir / f"{prefix}_{i + 1:04d}.nii.gz"
        img = nib.Nifti1Image(maps[i].astype(np.float32), affine=affine, header=header)
        nib.save(img, str(out_path))
        saved_paths.append(out_path)

    return saved_paths


# =============================================================================
# 10. PRINT HELPERS
# =============================================================================

def print_pilot_summary(pilot: PilotData) -> None:
    """Print a concise pilot data summary."""
    print("\n" + "=" * 72)
    print("Pilot data summary")
    print("=" * 72)
    print(f"Number of pilot maps:         {pilot.maps.shape[0]}")
    print(f"Image shape:                  {pilot.maps.shape[1:]}")
    print(f"Voxel sizes (mm):             {pilot.voxel_sizes_mm}")


def print_noise_summary(noise_params: NoiseParameters) -> None:
    """Print a concise noise summary."""
    print("\nNoise parameter summary")
    print("-" * 72)
    print(f"Global SD:                    {noise_params.global_sd:.6f}")
    print(f"Effective FWHM (vox):         {noise_params.fwhm_vox:.6f}")
    print(f"Effective FWHM (mm):          {noise_params.fwhm_mm:.6f}")
    print(f"Analysis mask voxels:         {int(noise_params.mask.sum())}")


def print_power_summary(
    n_per_group: int,
    t_fwe_threshold: float,
    result: WholeBrainPowerResult,
) -> None:
    """Print a concise power summary."""
    print("\n" + "-" * 72)
    print(f"Sample size per group:        {n_per_group}")
    print(f"Monte Carlo iterations:       {result.n_iterations}")
    print(f"Estimated power:              {result.power:.3f}")
    print(f"Peak-FWE |t| threshold:       {t_fwe_threshold:.4f}")
    print(f"Mean FWE-significant voxels:  {result.n_fwe_voxels.mean():.2f}")
    print(f"Mean clustered voxels:        {result.n_clustered_voxels.mean():.2f}")
    print(f"Mean surviving clusters:      {result.n_surviving_clusters.mean():.2f}")


# =============================================================================
# 11. MAIN
# =============================================================================

def main(config: SimulationConfig) -> None:
    """
    Run the full whole-brain two-group Monte Carlo power analysis pipeline.
    """
    # -------------------------------------------------------------------------
    # Load pilot data
    # -------------------------------------------------------------------------
    pilot = load_pilot_mat(
        mat_file=config.pilot_mat_file,
        reference_nifti_file=config.reference_nifti_file,
    )
    print_pilot_summary(pilot)

    # -------------------------------------------------------------------------
    # Estimate pooled noise parameters
    # -------------------------------------------------------------------------
    noise_params = estimate_noise_parameters(
        pilot_maps=pilot.maps,
        voxel_sizes_mm=pilot.voxel_sizes_mm,
        mask=None,
        nonzero_fraction_threshold=config.nonzero_fraction_threshold,
    )
    print_noise_summary(noise_params)

    # -------------------------------------------------------------------------
    # Define signal template and binary signal ROI
    # -------------------------------------------------------------------------
    shape = pilot.maps.shape[1:]
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    signal_template = create_spherical_signal_template(
        shape=shape,
        center_xyz=center,
        radius_vox=config.signal_radius_vox,
        smooth_sigma_vox=config.signal_smoothing_sigma_vox,
        mask=noise_params.mask,
        peak_value=config.signal_peak_value,
    )

    signal_roi_mask = signal_template > config.signal_roi_threshold

    # -------------------------------------------------------------------------
    # Generate one example synthetic dataset
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(config.simulation_seed)

    eg_maps, cg_maps = generate_two_group_contrast_maps(
        n_eg=max(config.sample_sizes_per_group),
        n_cg=max(config.sample_sizes_per_group),
        signal_template=signal_template,
        beta_eg=config.beta_eg,
        beta_cg=config.beta_cg,
        mask=noise_params.mask,
        fwhm_vox=noise_params.fwhm_vox,
        global_sd=None if config.use_voxelwise_sd else noise_params.global_sd,
        voxelwise_sd=noise_params.voxelwise_sd if config.use_voxelwise_sd else None,
        subject_beta_sd=config.subject_beta_sd,
        rng=rng,
    )

    print("\nSimulation setup")
    print("-" * 72)
    print(f"Injected EG beta:             {config.beta_eg:.6f}")
    print(f"Injected CG beta:             {config.beta_cg:.6f}")
    print(f"Between-subject beta SD:      {config.subject_beta_sd:.6f}")
    print(f"Signal ROI voxels:            {int(signal_roi_mask.sum())}")

    # -------------------------------------------------------------------------
    # Power across candidate sample sizes
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Whole-brain Monte Carlo power analysis")
    print("=" * 72)

    for n_per_group in config.sample_sizes_per_group:
        print(f"\nEstimating threshold and power for n = {n_per_group} per group")

        t_fwe_threshold_n, _ = estimate_peak_fwe_t_threshold(
            n_null_iterations=config.n_null_iterations,
            n_eg=n_per_group,
            n_cg=n_per_group,
            noise_params=noise_params,
            use_voxelwise_sd=config.use_voxelwise_sd,
            equal_var=config.equal_var,
            seed=config.null_seed_base + n_per_group,
        )

        result = monte_carlo_power_two_group_whole_brain_fwe(
            n_iterations=config.n_power_iterations,
            n_eg=n_per_group,
            n_cg=n_per_group,
            signal_template=signal_template,
            signal_roi_mask=signal_roi_mask,
            noise_params=noise_params,
            beta_eg=config.beta_eg,
            beta_cg=config.beta_cg,
            t_fwe_threshold=t_fwe_threshold_n,
            use_voxelwise_sd=config.use_voxelwise_sd,
            min_cluster_size=config.min_cluster_size,
            subject_beta_sd=config.subject_beta_sd,
            equal_var=config.equal_var,
            connectivity=config.connectivity,
            seed=config.simulation_seed,
        )

        print_power_summary(
            n_per_group=n_per_group,
            t_fwe_threshold=t_fwe_threshold_n,
            result=result,
        )

    # -------------------------------------------------------------------------
    # Optional: save one example synthetic dataset
    # -------------------------------------------------------------------------
    if config.save_example_maps:
        save_nifti_maps(
            maps=eg_maps,
            output_dir=config.output_dir / "EG",
            affine=pilot.affine,
            header=pilot.header,
            prefix="EG_sub",
        )

        save_nifti_maps(
            maps=cg_maps,
            output_dir=config.output_dir / "CG",
            affine=pilot.affine,
            header=pilot.header,
            prefix="CG_sub",
        )

        print(f"\nSaved example synthetic maps to: {config.output_dir.resolve()}")

    print("\nDone.\n")


if __name__ == "__main__":
    # Replace these example filenames with your own local input files.
    #
    # Required inputs:
    #   1. pilot_mat_file:
    #        MATLAB v7.3 .mat file containing:
    #          - data_matrix
    #          - mask_idx
    #
    #   2. reference_nifti_file:
    #        NIfTI file defining the spatial grid and image metadata
    #
    config = SimulationConfig(
        pilot_mat_file=Path("subject_voxel_matrix.mat"),
        reference_nifti_file=Path("reference_grid.nii"),
        beta_eg=2.15,
        beta_cg=0.0,
        subject_beta_sd=0.03,
        sample_sizes_per_group=(20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140),
        n_null_iterations=500,
        n_power_iterations=2000,
        save_example_maps=False,
    )

    main(config)
