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
    standardize_activations,
    add_gaussian_noise,
    apply_dimensionality_reduction,
    load_model_and_preprocess,
    process_images,
    sort_activations,
    classify,
    prepare_dataset,
    plot_label_accuracy_per_label
)
import cupy as cp

# Determine device: use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define file paths for input data and output
stimuli_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2023/data/stimuli"
out_root = "/home/eik-tb/Desktop/New Folder/dnn_plots_noise-stand-reduction"
os.makedirs(out_root, exist_ok=True)

# Set parameters for analysis
# n_components = [5,10,50,100]
noise_levels = np.arange(5,10,0.5)
# methods = ["IncrementalPCA", "MDS", "SVD"]
methods = ["IncrementalPCA"]
C = [1]

n_components = [10]
# noise_levels = [7]
# methods = ["MDS"]
# C = [0.001, 1, 10]

participants = 24


# Process models
model_tuples = [("clip", "Semantic"), ("alexnet", "Perceptual"), ("resnet50", "Perceptual")]

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

                    del model

                    for participant in range(n_participants):
                        activations_sub = deepcopy(activations)

                        # Flatten activations and stack into a numpy array
                        flattened_activations = [v["activation"].reshape(-1) for v in activations.values()]
                        # activations_sub = np.vstack(flattened_activations).astype(np.float32)
                        activations_np = cp.vstack(flattened_activations).astype(np.float32)

                        activations_np = add_gaussian_noise(activations_np, noise_level=noise_level)

                        activations_np = standardize_activations(activations_np)

                        activations_np = apply_dimensionality_reduction(
                            activations_np, n_components=n_component, method=method
                        )

                        # Update the activations dictionary
                        for i, stim_id in enumerate(activations):
                            activations_sub[stim_id]["activation"] = activations_np[i].get()

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
