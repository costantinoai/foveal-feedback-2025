#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:16:05 2024

@author: costantino_ai
"""

import random
import torch
import os
from copy import deepcopy
import pandas as pd
import numpy as np
from utils import (
    add_gaussian_noise,
    load_model_and_preprocess,
    process_images,
    sort_activations,
    prepare_dataset,
    plot_label_accuracy_per_label,
    standardize_activations,
    classify,
    time_function
)
from cuml.decomposition import IncrementalPCA
from sklearnex import config_context
import cupy as cp

@time_function
def apply_dimensionality_reduction_rapids(activations, n_components, method="PCA", title=""):
    """
    Reduces the dimensionality of activations using MDS.

    Parameters:
        activations (dict): Dictionary with stim_id as keys and 'activation' data.
        n_components (int): Number of components to retain after reduction.

    Returns:
        dict: Updated activations with reduced dimensions.

    Raises:
        ValueError: For invalid inputs.
    """
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    # Flatten activations and stack into a numpy array
    flattened_activations = [v["activation"].reshape(-1) for v in activations.values()]
    activations_np = cp.vstack(flattened_activations)#.astype(np.float32)

    with config_context(target_offload="gpu:0"):
        # Apply MDS for dimensionality reduction
        reducer = IncrementalPCA(n_components=n_components)
        activations_reduced = reducer.fit_transform(activations_np)

    # Update the activations dictionary with reduced dimensions
    transformed_activations = deepcopy(activations)
    for i, stim_id in enumerate(activations):
        transformed_activations[stim_id]["activation"] = activations_reduced[i].get()

    return transformed_activations

# Determine device: use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define file paths for input data and output
stimuli_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2023/data/stimuli"
out_root = "/home/eik-tb/Desktop/New Folder/dnn_plots_stand-reduction-noise"
os.makedirs(out_root, exist_ok=True)

# Set parameters for analysis
# n_components = [5,10,50,100]
noise_levels = np.arange(1,10,2)
methods = ["IncrementalPCA"]
C = [0.1]

n_components = [10]
# noise_levels = [7]
# methods = ["MDS"]
# C = [0.001, 1, 10]

participants = 24

# Process models
model_tuples = [("clip", "Semantic"), ("alexnet", "Perceptual"), ("resnet50", "Perceptual")]
# model_tuples = [("alexnet", "Perceptual")]

for c in C:
    for method in methods:
        for n_component in n_components:
            for noise_level in noise_levels:
                for model_name, title_suffix in model_tuples:

                    np.random.seed(42)
                    random.seed(42)

                    # Load the model and preprocessing function
                    model, preprocess_fn = load_model_and_preprocess(model_name, device)

                    # Process images to get activations
                    activations = process_images(model_name, model, preprocess_fn, stimuli_path, device)

                    # Sort and process activations
                    sorted_activations = sort_activations(activations)

                    # Define number of participants
                    n_participants = 24
                    participant_label_accuracies = []

                    # Clean CUDA memory
                    del model

                    for participant in range(n_participants):
                        activations_sub = deepcopy(activations)

                        activations_sub = standardize_activations(activations_sub)

                        activations_sub = apply_dimensionality_reduction_rapids(
                            activations_sub, n_components=n_component, method=method
                        )

                        activations_sub = add_gaussian_noise(activations_sub, noise_level=noise_level)

                        # Prepare dataset with noise for each participant
                        features, labels = prepare_dataset(activations_sub)

                        # Train and evaluate SVM classifier
                        (
                            testing_accuracy,
                            y_true_aggregated,
                            y_pred_aggregated,
                            y_true_aggregated_train,
                            y_pred_aggregated_train,
                            fold_indices,
                        ) = classify(features, labels, c)

                        # Calculate accuracy per label for this participant
                        data = {
                            "Fold": fold_indices,
                            "True Labels": y_true_aggregated,
                            "Predicted Labels": y_pred_aggregated,
                        }
                        df = pd.DataFrame(data)
                        df["Correct"] = df["True Labels"] == df["Predicted Labels"]

                        # Aggregate accuracy per label for this participant
                        participant_accuracy_df = (
                            df.groupby("True Labels")
                            .agg(LabelAccuracy=("Correct", "mean"))
                            .reset_index()
                        )

                        # Append this participant's label accuracy DataFrame to the list
                        participant_label_accuracies.append(participant_accuracy_df)

                    # Plot the average accuracy per label across participants
                    plot_label_accuracy_per_label(
                        participant_label_accuracies,
                        out_root,
                        title=f"SVM accuracy ({model_name.capitalize()}, {method}, noise: {noise_level}, dimensions: {n_component}, C: {c})",
                    )
