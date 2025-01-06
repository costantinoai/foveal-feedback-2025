#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:35:11 2024

@author: costantino_ai
"""

import inspect
import shutil
import random
import os
from copy import deepcopy
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils import (
    standardize_activations_np,
    apply_dimensionality_reduction_np,
    sort_activations,
    classify,
    prepare_dataset,
    load_model_and_preprocess,
    process_images,
    add_gaussian_noise_np_smooth,
    # plot_info,
    # plot_variance_comparison,
    process_images_tdann,
    report_mvpa_results,
    map_subcategory_to_categories,
    create_run_id,
    # apply_dimensionality_reduction_channels,
    # standardize_activations_channels,
    save_script_to_file as save_util_script_to_file,
    plot_lineplot_with_ci,
    visualize_projections
)

def save_script_to_file(output_directory):
    """
    Saves the script file that is calling this function to the specified output directory.

    This function automatically detects the script file that is executing this function
    and creates a copy of it in the output directory.
    It logs the process, indicating whether the saving was successful or if any error occurred.

    :param output_directory: The directory where the script file will be saved.
    :type output_directory: str
    """
    # Get the frame of the caller to this function
    caller_frame = inspect.stack()[1]
    # Get the file name of the script that called this function
    script_file = caller_frame.filename

    # Construct the output file path
    script_file_out = os.path.join(output_directory, os.path.basename(script_file))

    # Copy the script file to the output directory
    shutil.copy(script_file, script_file_out)


# Determine device: use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define configuration parameters
config = {
    "stimuli_path": "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2023/data/stimuli",
    "output_root": "/home/eik-tb/Desktop/New Folder/dnn_plots_TDANN/results",
    "noise_levels": np.arange(0,11, 1),  # Range of noise levels
    # "noise_levels": [0],  # Range of noise levels
    "dimensionality_reduction_methods": [None],  # Methods for dimensionality reduction
    "regularization_params": [1e-2],  # SVM regularization parameters
    "n_components": [None],  # Number of components for dimensionality reduction
    # "n_participants": 24,  # Number of participants
    "n_participants": 24,  # Number of participants
    "smooth_noise": True,  # Whether to smooth noise
    "generate_all_plots": False,  # Whether to generate all plots
    # "models_to_evaluate": [("Clip", "Semantic"), ("resnet50", "Perceptual")],  # List of models
    # "models_to_evaluate": [("TDANN", "V1")],  # List of models
    "models_to_evaluate": [("TDANN", "Perceptual"), ("Clip", "Semantic")],  # List of models
    "average_over_channels": True,
    "PCA_over_channels": False
}

# Iterate over combinations of parameters
for regularization_param in config["regularization_params"]:
    for reduction_method in config["dimensionality_reduction_methods"]:
        for n_components in config["n_components"]:
            for model_name, model_label in config["models_to_evaluate"]:
                proj_plotted_for_model=False

                all_noises_results_category = []
                all_noises_results_subcategory = []

                for noise_level in config["noise_levels"]:

                    # Create output directory for the current run
                    output_dir = (
                        f"{config['output_root']}/{create_run_id()}_{model_name}_"
                        f"noise-{noise_level}_components-{n_components}_"
                        f"method-{reduction_method}_C-{regularization_param}_averagechannel-{config['average_over_channels']}"
                    )
                    os.makedirs(output_dir, exist_ok=False)

                    # Save script to the output directory for reproducibility
                    save_util_script_to_file(output_dir)
                    save_script_to_file(output_dir)

                    # Initialize containers for true and predicted labels
                    true_labels = []
                    predicted_labels = []

                    # Set random seeds for reproducibility
                    np.random.seed(42)
                    random.seed(42)

                    # Process images to obtain activations
                    if model_name == "TDANN":
                        activations = process_images_tdann(config["stimuli_path"])
                    else:
                        model, preprocess_fn = load_model_and_preprocess(
                            model_name, device
                        )
                        activations = process_images(
                            model_name, model, preprocess_fn, config["stimuli_path"], device
                        )

                    # Sort and process activations
                    sorted_activations = sort_activations(activations)

                    # Initialize accuracy storage for participants
                    participant_category_accuracies = []
                    participant_subcategory_accuracies = []

                    # Iterate over participants
                    for participant_id in range(config["n_participants"]):
                        print(f"Processing PARTICIPANT: {participant_id}")

                        # Create a deep copy of activations for the current participant
                        participant_activations = deepcopy(activations)

                        class_labels = np.array(
                            [act["category"] for act in participant_activations.values()]
                        )

                        # Convert activations to a stacked NumPy array

                        if config["average_over_channels"]:
                            # Averaging activation values along the first axis
                            activation_arrays = [
                                np.expand_dims(np.mean(act["activation"], axis=0), axis=0)
                                for act in participant_activations.values()
                            ]
                            activations_np = np.array(activation_arrays)
                        else:

                            activation_arrays = [
                                v["activation"] for v in participant_activations.values()
                            ]
                            activations_np = np.array(activation_arrays)

                            # if config["PCA_over_channels"]:
                            #     # Standardize activations (zero mean, unit variance)
                            #     activations_np = standardize_activations_channels(activations_np)

                                # activations_np = apply_dimensionality_reduction_channels(
                                #     activations_np, n_components=1, method="PCA"
                                # )

                        if proj_plotted_for_model == False:

                            # # Plot MDS
                            # mds_proj = apply_dimensionality_reduction_np(
                            #     activations_np.reshape(activations_np.shape[0], -1),
                            #     n_components=2,
                            #     method="MDS"
                            # )
                            # mds_act = {idx:{
                            #             "category": class_labels[idx],
                            #             "activation": mds_proj[idx]}
                            #     for idx in range(len(class_labels))}

                            visualize_projections(activations_np, class_labels, alpha=.8, title=model_name)
                            proj_plotted_for_model=True


                        # # Plot variances before noise addition (optional)
                        # if config["generate_all_plots"]:
                        #     feature_variances_before, avg_sample_variances_before = plot_info(
                        #         activations_np, class_labels, title="Before Noise"
                        #     )

                        # Add Gaussian noise to activations
                        activations_np = add_gaussian_noise_np_smooth(
                            activations_np,
                            noise_level=noise_level,
                            smooth_noise=config["smooth_noise"],
                        )

                        # # Plot variances after noise addition (optional)
                        # if config["generate_all_plots"]:
                        #     feature_variances_after, avg_sample_variances_after = plot_info(
                        #         activations_np, class_labels, title="After Noise"
                        #     )
                        #     plot_variance_comparison(
                        #         feature_variances_before, feature_variances_after
                        #     )

                        # Flatten activations into 1D vectors
                        activations_np = np.array(
                            [act.flatten() for act in activations_np]
                        )

                        # Standardize activations (zero mean, unit variance)
                        activations_np = standardize_activations_np(activations_np)

                        # Apply dimensionality reduction if specified
                        if n_components is not None:
                            activations_np = apply_dimensionality_reduction_np(
                                activations_np, n_components=n_components, method=reduction_method
                            )

                            # if config["generate_all_plots"]:
                            #     plot_info(
                            #         activations_np,
                            #         class_labels,
                            #         title=f"After {reduction_method}",
                            #     )

                        # Update activations dictionary with processed data
                        for idx, stim_id in enumerate(participant_activations):
                            participant_activations[stim_id]["activation"] = (
                                activations_np[idx]
                            )

                        # Prepare dataset for classification
                        features, labels = prepare_dataset(participant_activations)

                        # Train and evaluate SVM classifier
                        (
                            test_accuracy,
                            y_true,
                            y_pred,
                            y_true_train,
                            y_pred_train,
                            fold_indices,
                            confusion_matrix,
                        ) = classify(features, labels, regularization_param)

                        # Collect predicted and true labels for overall analysis
                        predicted_labels.extend(y_pred)
                        true_labels.extend(y_true)

                        # Compute per-label accuracy
                        results_data = {
                            "Subject": participant_id,
                            "Fold": fold_indices,
                            "True Labels": y_true,
                            "Predicted Labels": y_pred,
                            "Correct": [true == pred for true, pred in zip(y_true, y_pred)],
                        }

                        # Create DataFrame for subcategory analysis
                        results_df_subcat = pd.DataFrame(results_data)

                        # Map subcategory labels to categories for broader analysis
                        results_data["True Labels"] = map_subcategory_to_categories(y_true)
                        results_data["Predicted Labels"] = map_subcategory_to_categories(y_pred)

                        # Create DataFrame for category analysis
                        results_df_cat = pd.DataFrame(results_data)

                        # Append accuracy data for the participant with metadata
                        participant_category_accuracies.append(
                            results_df_cat.assign(noise_level=noise_level, participant_id=participant_id)
                        )
                        participant_subcategory_accuracies.append(
                            results_df_subcat.assign(noise_level=noise_level, participant_id=participant_id)
                        )

                    # Plot average accuracy per category
                    report_mvpa_results(
                        participant_category_accuracies,
                        output_dir,
                        title=(
                            f"SVM accuracy category ({model_name.capitalize()}, "
                            f"{reduction_method}, noise: {noise_level}, "
                            f"dimensions: {n_components}, C: {regularization_param})"
                        ),
                    )

                    # Plot average accuracy per subcategory
                    report_mvpa_results(
                        participant_subcategory_accuracies,
                        output_dir,
                        title=(
                            f"SVM accuracy subcategory ({model_name.capitalize()}, "
                            f"{reduction_method}, noise: {noise_level}, "
                            f"dimensions: {n_components}, C: {regularization_param})"
                        ),
                    )

                    # Concatenate participant dataframes
                    all_participants_df = pd.concat(participant_subcategory_accuracies, axis=0, ignore_index=True)
                    true_labels_all_participants = all_participants_df["True Labels"]
                    predicted_labels_all_participants = all_participants_df["Predicted Labels"]

                    # Create confusion matrix display
                    disp = ConfusionMatrixDisplay.from_predictions(
                        true_labels_all_participants,
                        predicted_labels_all_participants,
                        cmap="Blues",
                        normalize="true"
                    )

                    computed_cm = disp.confusion_matrix
                    cm_labels = disp.display_labels

                    # Set vmin and vmax for the color map
                    disp.im_.set_clim(0, 1)  # Adjust color limits

                    # Add title and save the plot
                    plt.title(f"Normalized Confusion Matrix\n{model_name}, noise level: {noise_level}", pad=20)
                    plt.savefig(os.path.join(output_dir, f"{model_name.capitalize()}_noise-{noise_level}_confmat.png"))
                    plt.show()

                    # Create a DataFrame from the confusion matrix and labels
                    cm_df = pd.DataFrame(computed_cm, index=cm_labels, columns=cm_labels)

                    # Define the output file path
                    output_file_path = os.path.join(output_dir, f"{model_name.capitalize()}_noise-{noise_level}_confmat.csv")

                    # Save the DataFrame to a CSV file
                    cm_df.to_csv(output_file_path)

                    all_noises_results_subcategory.extend(participant_subcategory_accuracies)
                    all_noises_results_category.extend(participant_category_accuracies)

                all_category_data = pd.concat(all_noises_results_category).reset_index(drop=True)
                all_subcategory_data = pd.concat(all_noises_results_subcategory).reset_index(drop=True)

                # Save data for later use (optional)
                all_category_data.to_csv(os.path.join(config["output_root"], "f{model_name}_category_data.csv"), index=False)
                all_subcategory_data.to_csv(os.path.join(config["output_root"], "f{model_name}_subcategory_data.csv"), index=False)

                # Group by noise level and true labels, and calculate mean accuracy and confidence interval
                aggregated_data = all_category_data.groupby(["participant_id", "noise_level", "True Labels"]).agg(accuracy=("Correct", "mean")).reset_index()

                # Plot using the corrected data
                plot_lineplot_with_ci(
                    data=aggregated_data,
                    x_col="noise_level",
                    y_col="accuracy",
                    hue_col="True Labels",
                    title=f"Category-Level Accuracy Across Noise Levels - {model_name}",
                    output_dir=config["output_root"],
                )

                # Group by noise level and true labels, and calculate mean accuracy and confidence interval
                aggregated_data = all_subcategory_data.groupby(["participant_id", "noise_level", "True Labels"]).agg(accuracy=("Correct", "mean")).reset_index()

                # Plot using the corrected data
                plot_lineplot_with_ci(
                    data=aggregated_data,
                    x_col="noise_level",
                    y_col="accuracy",
                    hue_col="True Labels",
                    title=f"Sub-category-Level Accuracy Across Noise Levels - {model_name}",
                    output_dir=config["output_root"],
                )
