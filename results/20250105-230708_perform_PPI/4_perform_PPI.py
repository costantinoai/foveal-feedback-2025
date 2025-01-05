import glob
import inspect
import shutil
import numpy as np
import pandas as pd
import os
import concurrent.futures
from modules.utils import create_run_id
from modules.ppi_helpers import (
    load_roi_images,
    load_functional_images,
    load_and_process_events,
    make_design_matrix,
    apply_pca_and_extract_signal
    )

def save_script_to_file(output_directory):
    """
    Save the calling script to a specified output directory.

    This function obtains the filename of the script that directly calls this function
    (i.e., the "caller frame") and copies that script to a target directory, providing
    reproducibility by capturing the exact code used in the analysis.

    Parameters
    ----------
    output_directory : str
        Path to the directory where the script file will be copied.

    Returns
    -------
    None
    """
    caller_frame = inspect.stack()[1]  # Stack frame of the caller
    script_file = caller_frame.filename
    script_file_out = os.path.join(output_directory, os.path.basename(script_file))
    shutil.copy(script_file, script_file_out)

def process_run(subject_id, run_number, root_directory, roi_directory, func_directory, roi_masks):
    """
    Process a single subject's run data.

    Parameters:
    - subject_id (str): Subject identifier.
    - run_number (int): Run number.
    - root_directory (str): Root directory for BIDS data.
    - roi_directory (str): Directory containing ROI masks.
    - func_directory (str): Directory containing functional data.
    - roi_masks (dict): Binary masks for each ROI.

    Returns:
    - pd.DataFrame: Processed design matrix and extracted signals.
    """
    print(f"Processing subject {subject_id}, run {run_number}...")


    # Load functional images
    func_data = load_functional_images(subject_id, run_number, func_directory)
    if func_data is None:
        return None

    n_scans = func_data.shape[-1]
    func_data_reshaped = func_data.reshape(-1, n_scans).T

    # Load events
    event_directory = os.path.join(root_directory, f"sub-{subject_id}", "func")
    events_df = load_and_process_events(subject_id, run_number, event_directory)

    # Load motion regressors
    motion_path = os.path.join(func_directory, f"sub-{subject_id}_task-exp_run-{run_number}_desc-6HMP_regressors.txt")
    motion_confounds = np.loadtxt(motion_path)

    # Validate motion regressor shape
    if motion_confounds.shape[0] != n_scans:
        raise ValueError("Mismatch in number of scans and motion regressors.")

    # Create design matrix
    frame_times = np.arange(n_scans) * 2.0  # Assuming TR = 2.0 seconds
    design_matrix = make_design_matrix(events_df, motion_confounds, frame_times)

    # Apply PCA to extract signals for each ROI
    for roi_name, mask in roi_masks.items():
        signal, variance_explained = apply_pca_and_extract_signal(func_data_reshaped, mask)
        design_matrix[f"y_{roi_name}"] = signal
        design_matrix[f"variance_{roi_name}"] = variance_explained

    # Compute PPI terms
    task_contrast = 2 * design_matrix["y_p"] - 1
    for roi_name in roi_masks:
        design_matrix[f"y_ppi_{roi_name}"] = task_contrast * design_matrix[f"y_{roi_name}"]

    design_matrix["sub"] = subject_id
    design_matrix["run"] = run_number
    design_matrix["TR"] = design_matrix.index // 2

    print(f"Completed processing for subject {subject_id}, run {run_number}")
    return design_matrix

root_dir = "/data/projects/fov/data/BIDS"

config = {
    "root_dir": root_dir,
    "roi_dir": os.path.join(root_dir, "derivatives", "rois"),
    "func_dir": os.path.join(root_dir, "derivatives", "fMRIprep"),
    "output_dir": f"results/{create_run_id()}_perform_PPI"
    }
os.makedirs(config["output_dir"], exist_ok=True)
save_script_to_file(config["output_dir"])

subjects = [str(i).zfill(2) for i in range(2, 26)]
results = []

for subject in subjects:
    roi_masks = load_roi_images(subject, config["roi_dir"])
    func_sub_dir = os.path.join(config["func_dir"], f"sub-{subject}", "func")

    run_files = glob.glob(os.path.join(func_sub_dir, "*_task-exp_run-*_bold.nii.gz"))
    n_runs = len(run_files)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_run, subject, run_number, config["root_dir"], config["roi_dir"], func_sub_dir, roi_masks
            )
            for run_number in range(1, n_runs + 1)
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()  # Get the result
            if result is not None and not result.empty:  # Check for valid DataFrame
                results.append(result)

final_df = pd.concat(results).groupby(["sub", "run", "TR"]).mean().reset_index()
output_path = os.path.join(config["output_dir"], "ppi_results.csv")
final_df.to_csv(output_path, index=False)
print(f"Saved results to {output_path}")
