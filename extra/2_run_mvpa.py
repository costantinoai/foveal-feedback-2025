# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:39:23 2021

@author: costantino_ai
"""

import os, inspect, shutil
from datetime import datetime
from modules import FOV_ROIS_ORDER, ROIS_ORDER, SUBCAT_ORDER
from modules.utils import set_rnd_seed
from modules.fmri_helpers import (
    get_beta_dataframes,
    get_mask_files,
    initialize_reports_and_results,
    prepare_dataset_for_classification,
    demean_data_per_run,
    filter_and_sort_betas,
    save_results_to_file,
    perform_classification
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

# Define the root directory for derivatives
derivatives_bids = "/data/projects/fov/data/BIDS/derivatives"

# Define parameters
params = {
    "out_dir": "/home/eik-tb/Desktop/mvpa-new/mvpa_c01",  # Output directory for results
    "derivatives_dir": derivatives_bids,  # Base derivatives directory
    "beta_root": os.path.join(derivatives_bids, "SPMglm-fmriprep"),  # Root for beta images
    "rois_root": os.path.join(derivatives_bids, "rois"),  # Root for ROI masks
    "seed": 42,  # Random seed for reproducibility
    "results": {},  # Dictionary to store classification results
    "rois": FOV_ROIS_ORDER + ROIS_ORDER,  # Combine FOV and general ROI orders
}

# Create the output directory if it doesn't exist
os.makedirs(params["out_dir"], exist_ok=True)
save_script_to_file(params["out_dir"])

# Initialize the results structure for storing classification data
params["results"]["classification"] = initialize_reports_and_results(params)

# Convert SUBCAT_ORDER to lowercase for consistent processing
conditions = tuple([condition.lower() for condition in SUBCAT_ORDER])

# Define the subdirectory for task-specific beta images
task_subdir = "desc-bike+car+female+male"

# Set the random seed for reproducibility
set_rnd_seed(params["seed"])

# Print the start timestamp
print(f"START: {datetime.now()}")

# Iterate over subjects (from sub-02 to sub-25)
for sub_id in range(2, 26):
    sub = f"sub-{str(sub_id).zfill(2)}"  # Format subject ID as 'sub-XX'

    print(f"STEP: {sub} - Starting classification for conditions: {conditions}")

    # Load ROI mask files for the current subject
    print("STEP: Loading masks...", end="\r")
    mask_files = get_mask_files(params, sub)
    print("done!")

    # Generate the beta mapping DataFrame
    print("STEP: Generating beta mapping DataFrame...", end="\r")
    beta_loc = os.path.join(params["beta_root"], task_subdir, sub)  # Subject-specific beta directory
    betas_df = get_beta_dataframes(beta_loc)  # Create initial beta DataFrame
    betas_df = filter_and_sort_betas(betas_df, conditions)  # Filter and sort by conditions
    print("done!")

    # Demean beta values by removing the average pattern for each run
    print("STEP: Demeaning beta data per run...", end="\r")
    betas_df = demean_data_per_run(betas_df)
    print("done!")

    # Iterate over each ROI to prepare and classify data
    for roi_name in mask_files.keys():
        print("STEP: Preparing dataset for classification (zeroing NaNs, shuffling)...", end="\r")

        # Prepare the dataset for classification
        X, y, runs_idx = prepare_dataset_for_classification(betas_df, mask_files, roi_name)
        print("done!")

        # Perform classification with leave-one-run-out cross-validation
        testing_accuracy, y_true_agg, y_pred_agg, y_true_train_agg, y_pred_train_agg = perform_classification(
            X, y, runs_idx, C=0.01
        )

        # Update the results dictionary for the current ROI and conditions
        classification_results = params["results"]["classification"][conditions][roi_name]
        classification_results["acc"].append(testing_accuracy)
        classification_results["y_true_aggregated"].append(y_true_agg)
        classification_results["y_pred_aggregated"].append(y_pred_agg)
        classification_results["y_true_aggregated_train"].append(y_true_train_agg)
        classification_results["y_pred_aggregated_train"].append(y_pred_train_agg)

# Final step: Prepare figures and save results
print("STEP: Preparing and plotting figures...", end="\r")

# Save results to a pickle file
save_results_to_file(params)
