#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:35:11 2024

@author: costantino_ai
"""

from sklearn.metrics import ConfusionMatrixDisplay
import random
import torch
import os
from copy import deepcopy
import pandas as pd
import numpy as np
from utils import (
    standardize_activations_np,
    add_gaussian_noise_np,
    apply_dimensionality_reduction_np,
    sort_activations,
    classify,
    prepare_dataset,
    plot_label_accuracy_per_label,
    load_model_and_preprocess,
    process_images,
    add_gaussian_noise_np_smooth,
    estimate_kernel_sigma,
    add_gaussian_noise_by_class,
    compute_class_variances
)
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE
code_path = (
    "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/TDANN-main/demo"
)
sys.path.append(code_path)

from src.positions import NetworkPositions
from src.model import load_model_from_checkpoint, LAYERS
from src.data import load_images_from_folder
from src.features import FeatureExtractor

# Mapping integers to categories
int_mapping = {"f": 0, "m": 1, "b": 2, "c": 3}

# Mapping categories to labels for plotting and interpreting results
label_mapping = {
    "f": "Female",
    "m": "Male",
    "b": "Bike",
    "c": "Car",
}

# Color mapping for different categories for visualization
color_mapping = {
    "Female": "#D34936",
    "Male": "#FCA95F",
    "Bike": "#3B8BCE",
    "Car": "#82CDA4",
}

def process_images_tdann(
    image_path,
    layers=LAYERS[0],
    n_batches=120,
    verbose=True,
):
    """
    Process images using a trained TDANN model, extracting features and activations.

    Args:
        positions_dir (str): Path to the directory containing network positions.
        weights_path (str): Path to the model checkpoint.
        image_path (str): Path to the folder containing images to process.
        layers (list): List of layers for which to extract features.
        label_mapping (dict): Mapping from labels to categories.
        n_batches (int): Number of batches to process in the dataloader.
        verbose (bool): If True, print progress information.

    Returns:
        dict: A dictionary containing activations for each label.
    """
    # Mapping categories to labels for plotting and interpreting results
    label_mapping = {
        "f": "Female",
        "m": "Male",
        "b": "Bike",
        "c": "Car",
    }

    positions_dir = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/TDANN-main/paths/positions"
    weights_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/TDANN-main/paths/weights/model_final_checkpoint_phase199.torch"
    image_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2023/data/stimuli"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load positions from each model layer
    _ = NetworkPositions.load_from_dir(positions_dir)

    # Load model from checkpoint and send to appropriate device
    model = load_model_from_checkpoint(weights_path)
    model = model.to(DEVICE)

    # Create a dataloader to serve the images
    dataloader, idx_to_label = load_images_from_folder(image_path)

    # Extract features for all layers, also storing the images and labels
    extractor = FeatureExtractor(dataloader, n_batches=n_batches, verbose=verbose)
    features, inputs, labels = extractor.extract_features(
        model, LAYERS, return_inputs_and_labels=True
    )

    activations = {}
    act_layer = "layer2.0"  # Specify the activation layer of interest

    for i, feature in enumerate(features[act_layer]):
        label_idx = labels[i]
        label = idx_to_label[label_idx]

        # Store the extracted features and label for the image
        activations[label] = {
            "activation": feature,  # Tensor as numpy for easier handling
            "category": label_mapping[label[0]],  # Map first character to category
        }

        # Debugging and visualization
        # View the input image
        # raw = inputs[i].squeeze().transpose(1, 2, 0)
        # normed = (raw - np.min(raw)) / np.ptp(raw)
        # fig, ax = plt.subplots(figsize=(1, 1))
        # ax.imshow(normed)
        # ax.axis("off")

        # Show response magnitude in each layer
        # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
        # for ax, layer in zip(axes.ravel(), layers):
        #     coordinates = network_positions.layer_positions[layer].coordinates
        #     responses = features[layer][i]
        #     extent = np.ptp(coordinates)
        #     ax.scatter(*coordinates.T, c=responses, cmap="magma", s=extent / 100)
        #     ax.set_title(layer)
        # plt.show()

    return activations

def compute_act_images(activations_np, classes):
    """Compute feature and spatial variances for each class."""
    # Feature variance
    class_act = {label: np.sum(np.sum(activations_np[classes == label], axis=1), axis=0) for label in np.unique(classes)}

    return class_act

def plot_feature_variances(class_variances, title):
    """Plot feature variances as line plots."""
    plt.figure(figsize=(10, 6))
    for label, variances in class_variances.items():
        plt.plot(variances.flatten(), label=label)
    plt.title(title)
    plt.ylabel("Variance")
    plt.xlabel("Feature Index")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, title="Classes")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spatial_heatmaps(spatial_variances, title):
    """Plot spatial heatmaps for each class with a consistent color scale."""
    # Compute global color scale
    all_variances = np.array(list(spatial_variances.values()))
    vmin = all_variances.min()
    vmax = all_variances.max()

    num_classes = len(spatial_variances)
    fig, axes = plt.subplots(1, num_classes, figsize=(20, 5), sharex=True, sharey=True)

    for ax, (label, variances) in zip(axes, spatial_variances.items()):
        ax.imshow(variances, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'{label} Variance')
        ax.set_xlabel('Width (x)')
        ax.set_ylabel('Height (y)')

    # Add colorbar outside the plot
    # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.1, pad=0.02)
    # cbar.set_label('Variance')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_tsne(X_2d, classes, title):
    """Plot t-SNE visualization with a legend."""
    c = [color_mapping[label] for label in classes]
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=c, s=100, alpha=0.8, edgecolor="k")
    legend_labels = list(label_mapping.values())
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[label], label=label)
                        for label in legend_labels],
               loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, title="Classes")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_centroids(centroids_before, centroids_after):
    """Plot centroids before and after noise."""
    plt.figure(figsize=(10, 8))
    for label, coord in centroids_before.items():
        plt.scatter(coord[0], coord[1], marker="o", label=f"{label} Before", color=color_mapping[label])
    for label, coord in centroids_after.items():
        plt.scatter(coord[0], coord[1], marker="x", label=f"{label} After", color=color_mapping[label])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, title="Classes")
    plt.title("Class Centroids Before and After Noise")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Variance analysis and bar plot for before-and-after comparisons
def plot_variance_comparison(mean_variances_before, mean_variances_after):
    labels = list(mean_variances_before.keys())
    before = [np.mean(mean_variances_before[label]) for label in labels]
    after = [np.mean(mean_variances_after[label]) for label in labels]
    change = [100 * (a - b) / b for b, a in zip(before, after)]  # % change

    # Bar plot comparison
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, before, width, label="Before Noise", color="#3B8BCE")
    plt.bar(x + width / 2, after, width, label="After Noise", color="#FCA95F")
    plt.xticks(x, labels)
    plt.ylabel("Mean Variance")
    plt.title("Comparison of Mean Feature Variance by Class")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Percentage change
    plt.bar(labels, change, color="#82CDA4")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Variance Change (%)")
    plt.title("Percentage Change in Variance by Class")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Determine device: use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define file paths for input data and output
stimuli_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2023/data/stimuli"
out_root = "/home/eik-tb/Desktop/New Folder/dnn_plots_TDANN"
os.makedirs(out_root, exist_ok=True)

# Set parameters for analysis
# n_components = [5,10,50,100]
# noise_levels = np.arange(5,10,0.5)
noise_levels = [3]
# methods = ["PCA", "MDS", "SVD"]
methods = ["MDS"]
C = [1e3]
# C = [1e2]

n_components = [10]
# noise_levels = [7]
# methods = ["MDS"]
# C = [0.001, 1, 10]

# n_participants = 24
n_participants = 5

# Process models
# model_tuples = [("TDANN", "V1"), ("alexnet", "Perceptual")]
# model_tuples = [("alexnet", "Perceptual")]
model_tuples = [("TDANN", "V1")]

# !!!: hint!! cars and bikes get way too much variance, that's why they drop. why and how to fix?? perthaps this is about the higher activations + smoothing?

# TODO: play with: smoothing kernel size, smoothing noise vs. noise+act, components, noise level

for c in C:
    for method in methods:
        for n_component in n_components:
            for noise_level in noise_levels:
                for model_name, title_suffix in model_tuples:

                    true_labels = []
                    predicted_labels = []

                    np.random.seed(42)
                    random.seed(42)

                    if model_name == "TDANN":

                        # Process images to get activations
                        activations = process_images_tdann(stimuli_path)

                    else:

                        # Load the model and preprocessing function
                        model, preprocess_fn = load_model_and_preprocess(
                            model_name, device
                        )

                        # Process images to get activations
                        activations = process_images(
                            model_name, model, preprocess_fn, stimuli_path, device
                        )

                    # Sort and process activations
                    sorted_activations = sort_activations(activations)

                    # Define number of participants
                    participant_label_accuracies = []

                    for participant in range(n_participants):

                        print(f"PARTICIPANT: {participant}")
                        activations_sub = deepcopy(activations)

                        # Step 1: Extract activations and stack them into a numpy array
                        # Stack into a numpy array
                        acts = [v["activation"] for v in activations_sub.values()]
                        classes = np.array([v["category"] for v in activations_sub.values()])
                        activations_np = np.array(acts)

                        # Flatten for next analysis
                        activations_np = np.array(
                            [act.flatten() for act in activations_np]
                        )

                        # # Step 3: Standardize the activations
                        # activations_np = standardize_activations_np(activations_np)

                        if n_component != None:
                            # Step 4: Apply dimensionality reduction
                            activations_np = apply_dimensionality_reduction_np(
                                activations_np, n_components=n_component, method=method
                            )
                        else:
                            method = "None"

                        # Step 3: Standardize the activations
                        activations_np = standardize_activations_np(activations_np)

                        # Step 4: Compute and plot variances after noise
                        class_variances_after, spatial_variances_after = compute_class_variances(activations_np, classes)
                        plot_feature_variances(class_variances_after, "Feature Variance by Class (Before noise PCA)")

                        # Step 5: t-SNE visualization after noise
                        flat_act_noisy = np.array([act.flatten() for act in activations_np])
                        tsne = TSNE(n_components=2, random_state=42)
                        X_2d_after = tsne.fit_transform(flat_act_noisy)
                        plot_tsne(X_2d_after, classes, "t-SNE Visualization of Classes (Before noise PCA)")

                        activations_np = add_gaussian_noise_np(activations_np, noise_level=noise_level)
                        # activations_np = add_gaussian_noise_by_class(activations_np, classes, noise_level=noise_level)

                        # Step 4: Compute and plot variances after noise
                        class_variances_after, spatial_variances_after = compute_class_variances(activations_np, classes)
                        plot_feature_variances(class_variances_after, "Feature Variance by Class (After noise PCA)")

                        # Step 5: t-SNE visualization after noise
                        flat_act_noisy = np.array([act.flatten() for act in activations_np])
                        tsne = TSNE(n_components=2, random_state=42)
                        X_2d_after = tsne.fit_transform(flat_act_noisy)
                        plot_tsne(X_2d_after, classes, "t-SNE Visualization of Classes (After noise PCA)")

                        # Update the activations dictionary
                        for i, stim_id in enumerate(activations):
                            activations_sub[stim_id]["activation"] = activations_np[i]

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
                            confmat
                        ) = classify(features, labels, c)

                        predicted_labels.extend(y_pred_aggregated)
                        true_labels.extend(y_true_aggregated)

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

                    ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels)
                    plt.show()
