from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import h5py
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


def _decode_matlab_hdf5_string(f: h5py.File, ref) -> str:
    """
    Decode a MATLAB v7.3 string/cell-string stored as an HDF5 object reference.

    Parameters
    ----------
    f : h5py.File
        Open HDF5-backed .mat file.
    ref : h5py.Reference
        Reference to a MATLAB string object.

    Returns
    -------
    str
        Decoded string.
    """
    arr = f[ref][:]
    arr = np.array(arr).squeeze()

    # MATLAB chars are commonly stored as uint16 code points
    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(x)) for x in arr.ravel())

    # Fallback
    return "".join(str(x) for x in arr.ravel())


def _load_subject_ids_from_mat(f: h5py.File) -> List[str]:
    """
    Load subject IDs from the 'subjects' variable in a MATLAB v7.3 .mat file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5-backed .mat file.

    Returns
    -------
    list of str
        Subject IDs in matrix row order.
    """
    if "subjects" not in f:
        return []

    subj_ds = f["subjects"][:]
    subj_ds = np.array(subj_ds)

    subject_ids: List[str] = []

    # Common MATLAB cell-array case: object refs in a 2D array
    for ref in subj_ds.ravel(order="F"):
        subject_ids.append(_decode_matlab_hdf5_string(f, ref))

    return subject_ids


def load_subject_voxel_matrix_and_mask(
    mat_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load MATLAB-exported subject × voxel matrix and mask.
    """
    mat_path = Path(mat_path)

    with h5py.File(mat_path, "r") as f:
        if "data_matrix" not in f:
            raise KeyError("Expected variable 'data_matrix' not found in .mat file.")
        if "mask_idx" not in f:
            raise KeyError("Expected variable 'mask_idx' not found in .mat file.")

        data_matrix = f["data_matrix"][:]
        mask_idx = f["mask_idx"][:].astype(bool)
        subject_ids = _load_subject_ids_from_mat(f)

    data_matrix = np.asarray(data_matrix)

    if data_matrix.ndim != 2:
        raise ValueError(f"'data_matrix' must be 2D. Got shape {data_matrix.shape}")
    if mask_idx.ndim != 3:
        raise ValueError(f"'mask_idx' must be 3D. Got shape {mask_idx.shape}")

    n_mask_voxels = int(mask_idx.sum())

    # MATLAB v7.3 / h5py often yields transposed orientation relative to MATLAB usage.
    if data_matrix.shape[1] == n_mask_voxels:
        # already subjects × voxels
        pass
    elif data_matrix.shape[0] == n_mask_voxels:
        # loaded as voxels × subjects, transpose to subjects × voxels
        data_matrix = data_matrix.T
    else:
        raise ValueError(
            f"data_matrix shape {data_matrix.shape} is incompatible with "
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
            f"Loaded {len(subject_ids)} subject IDs, but data_matrix has {n_subjects} rows."
        )

    return data_matrix, mask_idx, subject_ids


def load_and_resample_roi_to_mask_grid(
    roi_nifti_path: str | Path,
    reference_nifti_path: str | Path,
    binarize_threshold: float = 0.0,
) -> np.ndarray:
    """
    Load an ROI NIfTI and resample it onto the reference image grid.

    Parameters
    ----------
    roi_nifti_path : str or Path
        Path to ROI image (e.g., Neurosynth insula mask).

    reference_nifti_path : str or Path
        Path to reference NIfTI defining the target grid
        (e.g., whole-sample intersection mask NIfTI).

    binarize_threshold : float, default=0.0
        Threshold used after resampling to convert ROI to boolean.
        Voxels > threshold are considered in the ROI.

    Returns
    -------
    roi_mask_resampled : np.ndarray
        Shape matches reference image, dtype=bool.
    """
    roi_img = nib.load(str(roi_nifti_path))
    ref_img = nib.load(str(reference_nifti_path))

    roi_resampled = resample_to_img(
        source_img=roi_img,
        target_img=ref_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )

    roi_mask_resampled = roi_resampled.get_fdata() > binarize_threshold
    return roi_mask_resampled.astype(bool)


def get_roi_column_indices_from_matlab_mask(
    mask_idx: np.ndarray,
    roi_mask: np.ndarray,
    require_inside_mask: bool = True,
) -> np.ndarray:
    """
    Map ROI voxels onto column indices of a MATLAB-exported subject × voxel matrix.

    Critical detail:
    MATLAB logical indexing uses column-major (Fortran) order. This function matches
    that ordering exactly.

    Parameters
    ----------
    mask_idx : np.ndarray
        Shape (X, Y, Z), dtype=bool
        3D analysis mask used to extract the MATLAB subject × voxel matrix.

    roi_mask : np.ndarray
        Shape (X, Y, Z), dtype=bool
        ROI mask on the same grid as mask_idx.

    require_inside_mask : bool, default=True
        If True, ROI voxels are restricted to mask_idx before mapping.
        This is usually what you want.

    Returns
    -------
    roi_cols : np.ndarray
        1D integer array of column indices into data_matrix.
    """
    mask_idx = np.asarray(mask_idx, dtype=bool)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    if mask_idx.shape != roi_mask.shape:
        raise ValueError(
            f"mask_idx shape {mask_idx.shape} does not match roi_mask shape {roi_mask.shape}"
        )

    if require_inside_mask:
        roi_use = roi_mask & mask_idx
    else:
        roi_use = roi_mask.copy()

    # Linear indices using MATLAB-compatible ordering
    mask_linear = np.flatnonzero(mask_idx.ravel(order="F"))
    roi_linear = np.flatnonzero(roi_use.ravel(order="F"))

    # Map from full-volume linear index -> column position in data_matrix
    lookup = {lin: i for i, lin in enumerate(mask_linear)}

    missing = [lin for lin in roi_linear if lin not in lookup]
    if missing:
        raise ValueError(
            f"{len(missing)} ROI voxels are not present in mask_idx. "
            "Set require_inside_mask=True or verify masks are aligned."
        )

    roi_cols = np.array([lookup[lin] for lin in roi_linear], dtype=int)
    return roi_cols


def compute_subjectwise_roi_percentile_from_mat(
    mat_path: str | Path,
    roi_nifti_path: str | Path,
    reference_nifti_path: str | Path,
    percentile: float = 95.0,
    binarize_threshold: float = 0.0,
    require_inside_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Compute a subject-wise percentile of voxel values inside an ROI directly
    from a MATLAB-exported subject × voxel matrix.

    This avoids reconstructing 3D volumes and preserves the original MATLAB
    voxel ordering exactly.

    Parameters
    ----------
    mat_path : str or Path
        Path to MATLAB v7.3 .mat file containing:
        - data_matrix (subjects × voxels)
        - mask_idx (3D mask)
        - subjects (optional)

    roi_nifti_path : str or Path
        Path to ROI NIfTI image.

    reference_nifti_path : str or Path
        Path to reference NIfTI defining the grid onto which the ROI will be resampled.
        This should match the grid of mask_idx, typically the whole-sample intersection mask.

    percentile : float, default=95.0
        Percentile to compute within ROI for each subject.

    binarize_threshold : float, default=0.0
        Threshold applied after resampling ROI to reference grid.

    require_inside_mask : bool, default=True
        Restrict ROI to voxels inside mask_idx before mapping to matrix columns.

    Returns
    -------
    roi_percentile : np.ndarray
        Shape (n_subjects,)
        Percentile value for each subject.

    roi_values : np.ndarray
        Shape (n_subjects, n_roi_voxels)
        Matrix of subject values inside the ROI.

    subject_ids : list of str
        Subject IDs corresponding to rows of roi_values / roi_percentile.

    roi_cols : np.ndarray
        Column indices in data_matrix corresponding to ROI voxels.
    """
    data_matrix, mask_idx, subject_ids = load_subject_voxel_matrix_and_mask(mat_path)

    roi_mask = load_and_resample_roi_to_mask_grid(
        roi_nifti_path=roi_nifti_path,
        reference_nifti_path=reference_nifti_path,
        binarize_threshold=binarize_threshold,
    )

    if roi_mask.shape != mask_idx.shape:
        raise ValueError(
            f"Resampled ROI shape {roi_mask.shape} does not match mask_idx shape {mask_idx.shape}"
        )

    roi_cols = get_roi_column_indices_from_matlab_mask(
        mask_idx=mask_idx,
        roi_mask=roi_mask,
        require_inside_mask=require_inside_mask,
    )

    roi_values = data_matrix[:, roi_cols]  # subjects × ROI voxels
    roi_percentile = np.percentile(roi_values, percentile, axis=1)

    return roi_percentile, roi_values, subject_ids, roi_cols

import numpy as np
import matplotlib.pyplot as plt

def reconstruct_from_selected_columns(subject_vec, mask_idx, selected_cols):
    """
    Reconstruct a 3D volume containing only values from selected columns
    of a MATLAB-exported subject x voxel matrix.
    """
    mask_idx = mask_idx.astype(bool)
    vol = np.zeros(mask_idx.shape, dtype=np.float32)

    flat_vol = vol.ravel(order="F")
    flat_mask_inds = np.flatnonzero(mask_idx.ravel(order="F"))

    selected_linear_inds = flat_mask_inds[selected_cols]
    flat_vol[selected_linear_inds] = subject_vec[selected_cols]

    return flat_vol.reshape(mask_idx.shape, order="F")


def compute_subjectwise_brain_summary_from_mat(
    mat_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute subject-wise mean and SD across all masked brain voxels.
    """
    data_matrix, mask_idx, subject_ids = load_subject_voxel_matrix_and_mask(mat_path)
    brain_mean = np.mean(data_matrix, axis=1)
    brain_sd = np.std(data_matrix, axis=1, ddof=1)
    return brain_mean, brain_sd, subject_ids

if __name__ == "__main__":

    brain_mean, brain_sd, subject_ids = compute_subjectwise_brain_summary_from_mat("BetaWeights_forBrendanSimulation/Minoritystress_n20_subject_voxel_matrix.mat")
    print('global SD:', np.mean(brain_sd))
    print('mean beta:', np.mean(brain_mean))

    
    mat_path = "BetaWeights_forBrendanSimulation/Minoritystress_n20_subject_voxel_matrix.mat"
    roi_nifti_path = "BetaWeights_forBrendanSimulation/Neurosynth_Insula+PosteriorInsula_z>10_SPM.nii"
    reference_nifti_path = "BetaWeights_forBrendanSimulation/Minoritystress_n20_wholesample_intersection_mask.nii"

    p95, roi_values, subject_ids, roi_cols = compute_subjectwise_roi_percentile_from_mat(
        mat_path=mat_path,
        roi_nifti_path=roi_nifti_path,
        reference_nifti_path=reference_nifti_path,
        percentile=95,
        binarize_threshold=0.0,
        require_inside_mask=True,
    )

    group_assignments = {'04': 'eg',
     '09': 'cg',
     '14': 'cg',
     '16': 'cg',
     '19': 'eg',
     '20': 'eg',
     '22': 'eg',
     '29': 'eg',
     '30': 'eg',
     '31': 'cg',
     '32': 'eg',
     '36': 'cg',
     '37': 'eg',
     '51': 'cg',
     '59': 'cg',
     '62': 'cg',
     '64': 'eg',
     '65': 'eg',
     '71': 'cg',
     '76': 'cg'}

    groups = np.array([group_assignments[sid] for sid in subject_ids])

    roi_group1 = roi_values[groups == "eg"]
    roi_group2 = roi_values[groups == "cg"]

    g1_mean_per_vxl = np.mean(roi_group1, axis=0)
    g2_mean_per_vxl = np.mean(roi_group2, axis=0)

    diff = g1_mean_per_vxl - g2_mean_per_vxl
    
    #####
    x = np.percentile(diff, 99)
    print('beta for input into sim = ', x)
    #####

    #denominator of effect size this corresponds to (numerator comes from simulation)
    eg_roi = roi_group1
    cg_roi = roi_group2
    sd1 = np.std(eg_roi, axis=0, ddof=1)   # SD across EG subjects at each voxel
    sd2 = np.std(cg_roi, axis=0, ddof=1)   # SD across CG subjects at each voxel

    n1 = eg_roi.shape[0]
    n2 = cg_roi.shape[0]

    pooled_sd_vox = np.sqrt(
        ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
    )

    denom = np.median(pooled_sd_vox)
    print('median voxel-wise effect size denominator :',denom)








    sanity_check = False
    if sanity_check:
        ####### SANITY CHECK

        # ------------------------------------------------
        # Load matrix + mask for visualization
        # ------------------------------------------------

        data_matrix, mask_idx, _ = load_subject_voxel_matrix_and_mask(mat_path)

        # ------------------------------------------------
        # Reconstruct ROI voxels for one subject
        # ------------------------------------------------

        subj_idx = 0

        roi_only_vol = reconstruct_from_selected_columns(
            subject_vec=data_matrix[subj_idx, :],
            mask_idx=mask_idx,
            selected_cols=roi_cols
        )

        # ------------------------------------------------
        # Plot slices
        # ------------------------------------------------

        import matplotlib.pyplot as plt

        roi_mask = load_and_resample_roi_to_mask_grid(
            roi_nifti_path,
            reference_nifti_path,
            binarize_threshold=0.0,
        )

        for z in [30, 35, 40, 45, 50]:
            plt.figure(figsize=(6,6))
            plt.imshow(roi_only_vol[:,:,z].T, origin="lower", cmap="gray")
            plt.contour(roi_mask[:,:,z].T, levels=[0.5], colors="red")
            plt.title(f"ROI-only reconstructed values, z={z}")
            plt.axis("off")
            plt.show()