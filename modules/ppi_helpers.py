import glob, os
import nibabel as nib
import numpy as np
import scipy
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from sklearn.decomposition import PCA

# Define functions
def load_roi_images(subject_id, roi_directory):
    """
    Load and binarize Region of Interest (ROI) images for a specific subject.

    Parameters:
    - subject_id (str): Identifier for the subject (e.g., '01').
    - roi_directory (str): Path to the directory containing ROI files.

    Returns:
    - dict: A dictionary where keys are ROI names and values are binary masks as NumPy arrays.
    """
    rois = {}
    roi_paths = {
        "fov": glob.glob(os.path.join(roi_directory, f"sub-{subject_id}", "*label-FOV+20_roi.nii"))[0],
        "per": glob.glob(os.path.join(roi_directory, f"sub-{subject_id}", "*label-PER_roi.nii"))[0],
        "opp": glob.glob(os.path.join(roi_directory, f"sub-{subject_id}", "*label-OPP_roi.nii"))[0],
        "ffa": glob.glob(os.path.join(roi_directory, f"sub-{subject_id}", "*label-FFA_roi.nii"))[0],
        "loc": glob.glob(os.path.join(roi_directory, f"sub-{subject_id}", "*label-LOC_roi.nii"))[0],
        # "a1": glob.glob(os.path.join(roi_directory, f"sub-{subject_id}", '*label-A1_roi.nii'))[0]
        }

    for roi_name, path in roi_paths.items():
        data = nib.load(path).get_fdata()
        binary_mask = np.where(data != 0, 1, 0).flatten()
        rois[roi_name] = binary_mask

    return rois

def load_functional_images(subject_id, run_number, func_directory):
    """
    Load functional MRI images for a specific subject and run.

    Parameters:
    - subject_id (str): Subject identifier (e.g., '01').
    - run_number (int): Run number.
    - func_directory (str): Path to the functional images directory.

    Returns:
    - np.array: Functional image data as a NumPy array.
    """
    func_path = os.path.join(
        func_directory,
        f"sub-{subject_id}_task-exp_run-{run_number}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    if os.path.exists(func_path):
        return nib.load(func_path).get_fdata()
    else:
        print(f"WARNING: Functional image not found for subject {subject_id}, run {run_number}")
        return None

def load_and_process_events(subject, run, dir_events):
    # Load the .mat file
    path = os.path.join(
        dir_events,
        f"sub-{subject}_task-exp_run-run-{run}_desc-bike+car+female+male_SPMmulticondition.mat",
    )
    events_data = scipy.io.loadmat(path)
    ev_names, ev_onsets, ev_durations = (
        events_data["names"],
        events_data["onsets"],
        events_data["durations"],
    )

    events = []
    for i in range(ev_names.shape[1]):
        names = [ev_names[0, i][0]] * len(ev_onsets[0, i][0])
        onsets = ev_onsets[0, i][0]
        durations = ev_durations[0, i][0]
        # events.append(pd.DataFrame({"trial_type": names, "onset": onsets, "duration": durations}))

    events.append(
        pd.DataFrame(
            {"trial_type": names, "onset": onsets, "duration": durations}
        ).reset_index(drop=True)  # Ensure proper DataFrame construction
    )

    events = pd.concat(events).sort_values("onset").reset_index(drop=True)

    # Fallback: Assign `y_p` if no task-related trials are found
    if "y_p" not in events["trial_type"].unique():
        events["trial_type"] = "y_p"

    return events

def make_design_matrix(events, motion_confounds, frame_times, hrf_model="glover"):
    """
    Create a design matrix for first-level GLM analysis.

    Parameters:
    - events (pd.DataFrame): Event data.
    - motion_confounds (np.array): Motion regressor data.
    - frame_times (np.array): Frame times (TR intervals).
    - hrf_model (str): Hemodynamic response function model. Default is "glover".

    Returns:
    - pd.DataFrame: Design matrix.
    """
    motion_labels = ["tx", "ty", "tz", "rx", "ry", "rz"]
    return make_first_level_design_matrix(
        frame_times,
        events,
        drift_model="polynomial",
        drift_order=3,
        add_regs=motion_confounds,
        add_reg_names=motion_labels,
        hrf_model=hrf_model,
    )

def apply_pca_and_extract_signal(data, mask):
    """
    Apply PCA to extract signals from masked functional data.

    Parameters:
    - data (np.array): Functional data.
    - mask (np.array): Binary mask.

    Returns:
    - np.array: Signal extracted via PCA.
    - float: Explained variance ratio.
    """
    masked_data = data[:, mask == 1]
    pca = PCA(n_components=1, whiten=True)
    pca.fit(masked_data)
    principal_component = pca.transform(masked_data).flatten()

    return principal_component, pca.explained_variance_ratio_[0]
