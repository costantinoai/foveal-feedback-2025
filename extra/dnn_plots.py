#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:44:40 2024

@author: costantino_ai
"""

import os
from collections import OrderedDict
from copy import deepcopy
import pandas as pd
import numpy as np
from PIL import Image
import torch
import clip
from sklearn.svm import SVC
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import models, transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from natsort import natsorted
from sklearn.model_selection import StratifiedShuffleSplit

plt.rcParams['font.size'] = 20  # Set the font size
TITLE_FONT_SIZE = 26

def apply_dimensionality_reduction(activations, n_components, title=""):
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
    activations_np = np.vstack(flattened_activations)

    # Apply MDS for dimensionality reduction
    reducer = MDS(n_components=n_components, random_state=42)
    activations_reduced = reducer.fit_transform(activations_np)

    # Update the activations dictionary with reduced dimensions
    transformed_activations = deepcopy(activations)
    for i, stim_id in enumerate(activations):
        transformed_activations[stim_id]["activation"] = activations_reduced[i]

    return transformed_activations

def visualize_projections(activations, alpha=0.5, title=""):
    """
    Generates a 2D scatter plot of the reduced dimensionality data.

    Parameters:
        activations (dict): Dictionary containing reduced dimensionality data.
        alpha (float): Transparency level for the plot points.

    Outputs:
        Displays a 2D scatter plot.
    """
    plt.figure(figsize=(12,10))
    full_title=f"{title} MDS projection"
    plt.title(full_title, pad=20, fontsize=TITLE_FONT_SIZE)

    # Plot each category with the specified color and alpha transparency
    for category in color_mapping.keys():
        points = np.array(
            [v["activation"] for v in activations.values() if v["category"] == category]
        )
        plt.scatter(
            points[:, 0],
            points[:, 1],
            alpha=alpha,
            label=category,
            color=color_mapping[category],
            s=100  # Adjust this value to control the size of the points
        )

    # Hide x and y axis labels and ticks for a cleaner plot
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.xticks([])
    plt.yticks([])

    # Display the legend
    plt.legend(title="Sub-categories")
    plt.tight_layout()

    filename = full_title.replace(" ", "_").lower() + ".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def sort_activations_by_category(activations):
    """
    Sorts a dictionary of activations by custom category order: Female, Male, Bike, Car.

    Parameters:
        activations (dict): Dictionary to sort.

    Returns:
        OrderedDict: Sorted dictionary by category.
    """
    # Define the desired order
    category_order = ["Female", "Male", "Bike", "Car"]

    # Create a sorting key based on the custom order
    def get_sort_key(item):
        category_label = item[1]["category"]
        return category_order.index(category_label)

    # Sort the activations dictionary by the custom key
    sorted_activations = OrderedDict(natsorted(activations.items(), key=get_sort_key))

    return sorted_activations


def plot_rdm(activations, metric="cosine", title=""):
    """
    Plots the Representational Dissimilarity Matrix (RDM) using the specified metric.

    Parameters:
        activations (dict): Dictionary containing original or reduced dimensionality data.
        metric (str): Distance metric to use ('cosine' or 'correlation').
        colors (dict): Dictionary containing color codes for each activation.

    Outputs:
        Displays a heatmap of the RDM with color patches.
    """
    # Extract activations into a numpy array
    activation_matrix = np.vstack(
        [v["activation"].reshape(-1) for v in activations.values()]
    )
    rdm = squareform(pdist(activation_matrix, metric=metric))

    # Create a heatmap using seaborn
    fig, ax = plt.subplots(figsize=(12,10))  # Square figure
    sns.heatmap(
        rdm,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": f"{metric.capitalize()} Distance", "ticks": [0,1,2]},
        vmin=0,
        vmax=2
    )

    # Generate colored labels
    labels = list(activations.keys())

    # Define the width and offset of the color line
    line_width = 4
    offset = 0.5  # Distance to move lines away from the heatmap

    # Drawing color-coded lines next to each row and each column
    for i, label in enumerate(labels):
        color = color_mapping[label_mapping[label[0]]]
        # Add a rectangle for the row
        rect = Rectangle(
            (-line_width - offset, i), line_width, 1, color=color, clip_on=False
        )
        ax.add_patch(rect)
        if label[1:] == "15":
            ax.text(
                -line_width - 1.5 * offset,
                i + 0.5,
                label_mapping[label[0]],
                va="center",
                ha="right",
                color="black",
                rotation=90,
                fontsize=20,
            )

        # Add a rectangle for the column
        rect = Rectangle(
            (i, len(labels) + offset), 1, line_width, color=color, clip_on=False
        )
        ax.add_patch(rect)
        if label[1:] == "15":
            ax.text(
                i + 0.5,
                line_width + 4 + len(labels) + 1.5 * offset,
                label_mapping[label[0]],
                va="bottom",
                ha="center",
                color="black",
                fontsize=20,
            )

    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set title
    ax.set_title(f"{title} RDM", fontsize=TITLE_FONT_SIZE)

    # Adjust the axes to account for the added elements and moved lines
    ax.set_xlim(-line_width - 2 * offset, len(labels))
    ax.set_ylim(len(labels) + 2 * offset, -line_width - offset)

    plt.tight_layout()
    plt.show()


def sort_activations_by_key(activations):
    """
    Sorts a dictionary of activations by keys in ascending order.

    Parameters:
        activations (dict): Dictionary to sort.

    Returns:
        OrderedDict: Sorted dictionary by keys.
    """
    return OrderedDict(natsorted(activations.items(), key=lambda x: x[0]))


def load_model_and_preprocess(model_name, device):
    """
    Loads the specified model and its corresponding preprocessing function.

    Parameters:
    - model_name (str): The name of the model to load ("clip", "alexnet", or "resnet").
    - device (torch.device): The device to load the model onto (e.g., "cpu" or "cuda").

    Returns:
    - model (torch.nn.Module): The loaded model.
    - preprocess (callable): The preprocessing function for the model's input images.
    """
    if model_name.lower() == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device)
    elif model_name.lower() == "alexnet":
        model = models.alexnet(pretrained=True).to(device)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=True).to(device)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return model, preprocess

def process_images(model_name, model, preprocess, folder_path, device):
    """
    Processes images in a folder, extracting feature activations from the specified model.

    Parameters:
    - model_name (str): The name of the model ("clip", "alexnet", or "resnet50").
    - model (torch.nn.Module): The loaded model to use for feature extraction.
    - preprocess (callable): Preprocessing function for preparing images.
    - folder_path (str): Path to the folder containing images to process.
    - device (torch.device): The device (CPU or GPU) to use for processing.

    Returns:
    - image_activations (dict): A dictionary containing image activations and category labels for each image.
    """
    image_activations = {}  # Dictionary to store activations and categories for each image

    # Iterate over each file in the specified folder
    for filename in os.listdir(folder_path):
        # Check if file is an image (png, jpg, jpeg)
        if filename.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(folder_path, filename)  # Full path to the image

            # Open the image and ensure it is in RGB format
            image = Image.open(path).convert("RGB")

            # Apply preprocessing (resize, crop, normalize) and add batch dimension
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            # Disable gradient computation for faster inference
            with torch.no_grad():
                # Extract features based on the model type
                if model_name.lower() == "clip":
                    features = model.encode_image(image_tensor)  # CLIP's encoding method
                elif model_name.lower() == "alexnet":
                    features = model.features(image_tensor)  # Features extraction for AlexNet
                elif model_name.lower() == "resnet50":
                    # For ResNet, we can use the full model up to the penultimate layer
                    features = model(image_tensor)  # ResNet50 forward pass through the whole network

            # Use filename (without extension) as a unique key
            key = os.path.splitext(filename)[0]

            # Store the extracted features and label for the image
            image_activations[key] = {
                "activation": features.cpu().numpy(),  # Convert tensor to numpy for easier handling
                "category": label_mapping[key[0]],  # Map first character to category using label_mapping
            }

    return image_activations  # Return dictionary of activations and categories


def sort_and_process_activations(image_activations):
    image_activations = sort_activations_by_key(image_activations)
    image_activations = sort_activations_by_category(image_activations)
    return image_activations

def classify(X, y, C=1, pout=1):
    """
    Perform data classification using Support Vector Machines (SVM) and
    stratified leave-one-out cross-validation, printing and returning relevant classification metrics.

    Parameters:
    - X (array-like): Input data, where `n_samples` is the number of samples and
                      `n_features` is the number of features.
    - y (array-like): Target values.
    - pout : how many samples per class to be assigned in the test
    """

    # Define SVM kernel type
    kernel = "linear"

    # Define the stratified leave-one-out cross-validation
    n_splits = 1000
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=pout * len(np.unique(y)), random_state=42
    )

    # Initialize lists to store accuracy for each fold
    training_accuracy = []
    testing_accuracy = []

    # Prepare lists to aggregate true and predicted labels across all folds for confusion matrix and report
    y_true_aggregated = []
    y_pred_aggregated = []
    y_true_aggregated_train = []
    y_pred_aggregated_train = []
    # Lists to store results
    accuracies = []
    conf_matrices = []
    fold_indices = []  # To store fold index for each test instance

    fold_number = 0  # Initialize fold number

    # Begin cross-validation
    for train_idx, test_idx in sss.split(X, y):
        print("\n\n===================================================")
        print("Indices of train samples:", train_idx.tolist())
        print("Indices of test samples:", test_idx.tolist())

        # Define training and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Instantiate and fit the classifier on the training data
        clf = SVC(kernel=kernel, random_state=42, C=C).fit(X_train, y_train)

        # Predict labels for training and test sets
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        # Record training and testing accuracy
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        training_accuracy.append(train_acc)
        testing_accuracy.append(test_acc)

        # Evaluate the classifier
        acc = accuracy_score(y_test, y_pred_test)
        accuracies.append(acc)

        # Confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred_test)
        conf_matrices.append(conf_mat)

        # Aggregate true and predicted labels for test set
        y_true_aggregated.extend(y_test)
        y_pred_aggregated.extend(y_pred_test)
        y_true_aggregated_train.extend(y_train)
        y_pred_aggregated_train.extend(y_pred_train)
        fold_indices.extend(
            [fold_number] * len(test_idx)
        )  # Append fold number for each test instance

        fold_number += 1  # Increment fold number for each split

        # Output detailed predictions and performance for the current step
        print(f"\nTraining Accuracy: {train_acc:.2f}, Testing Accuracy: {test_acc:.2f}")
        print(f"\nTraining Predicted vs Actual: {list(zip(y_pred_train, y_train))}")
        print(f"\nTesting Predicted vs Actual: {list(zip(y_pred_test, y_test))}\n")

    # Display results
    print("Accuracy per fold:", accuracies)
    print("Mean accuracy:", np.mean(accuracies))

    # Assuming 'conf_matrices' is a list of confusion matrices from each fold
    sum_conf_matrix = np.sum(conf_matrices, axis=0)  # Sum across the matrices

    # Plotting the summed confusion matrix
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(
        sum_conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_title("Sum of Confusion Matrices Across All Folds", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")

    plt.show()

    # Calculate overall performance metrics
    avg_training_accuracy = np.mean(training_accuracy)
    avg_testing_accuracy = np.mean(testing_accuracy)

    # Print overall metrics
    print(f"Average Training Accuracy: {avg_training_accuracy:.2f}")
    print(f"Average Testing Accuracy: {avg_testing_accuracy:.2f}\n")

    # Return the performance metrics
    return (
        testing_accuracy,
        y_true_aggregated,
        y_pred_aggregated,
        y_true_aggregated_train,
        y_pred_aggregated_train,
        fold_indices,
    )


def prepare_dataset(activations, noise_level=0.0):
    labels = list(activations.keys())
    features = np.array(
        [activations[label]["activation"].flatten() for label in labels]
    ).squeeze()
    numeric_labels = [int_mapping[label[0]] for label in labels]

    # Standardize the noisy features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Add Gaussian noise to the features to degrade SVM performance
    noise = np.random.normal(0, noise_level, features.shape)
    features_noisy = features + noise

    y = np.array(numeric_labels)
    X = np.array(features_noisy)

    return X, y


def plot_label_accuracy_per_fold(
    y_true_aggregated,
    y_pred_aggregated,
    fold_indices,
    label_mapping,
    color_mapping,
    title="Accuracy per Label per Fold",
):
    """
    Plots a bar graph of accuracy per label per fold, with confidence intervals.

    Parameters:
        y_true_aggregated (list): List of true labels from the aggregated test sets.
        y_pred_aggregated (list): List of predicted labels from the aggregated test sets.
        fold_indices (list): List of fold indices corresponding to each element in y_true_aggregated and y_pred_aggregated.
        label_mapping (dict): Dictionary mapping numeric labels back to their string representations.
        color_mapping (dict): Dictionary mapping labels to colors.
        title (str): Title for the plot.
    """

    # Prepare the DataFrame
    data = {
        "Fold": fold_indices,
        "True Labels": y_true_aggregated,
        "Predicted Labels": y_pred_aggregated,
    }
    df = pd.DataFrame(data)
    df["Correct"] = df["True Labels"] == df["Predicted Labels"]

    # Map numeric labels to string labels
    reversed_dict = {value: key for key, value in int_mapping.items()}
    df["Label"] = df["True Labels"].map(reversed_dict).map(label_mapping)

    # Calculate accuracy per label per fold
    accuracy_df = (
        df.groupby(["Fold", "Label"])
        .agg(Accuracy=("Correct", "mean"), Count=("Correct", "size"))
        .reset_index()
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(8,10))
    sns.barplot(
        data=accuracy_df,
        x="Label",
        y="Accuracy",
        palette=color_mapping,
        ax=ax,
        capsize=0.3,
        errorbar=("ci", 95),
        err_kws={"linewidth": 1},
    )

    # Customize the plot aesthetics
    ax.set_ylabel("Accuracy")
    ax.set_title(title, pad=30, fontsize=TITLE_FONT_SIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.ylim([0., 1])

    # plt.tight_layout()
    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

def process_model(model_name, title_suffix):
    """
    Process images through the specified model, perform dimensionality reduction,
    visualization, and classification.

    Parameters:
    - model_name (str): Name of the model ('clip', 'alexnet', 'resnet50')
    - title_suffix (str): Suffix to add to plot titles
    """
    # Load the model and preprocessing function
    model, preprocess_fn = load_model_and_preprocess(model_name, device)

    # Process images to get activations
    activations = process_images(model_name, model, preprocess_fn, stimuli_path, device)

    # Sort and process activations
    sorted_activations = sort_and_process_activations(activations)

    # Apply 2D dimensionality reduction for visualization
    xy_activations = apply_dimensionality_reduction(sorted_activations, n_components=2)

    # Visualize the 2D projections
    visualize_projections(
        xy_activations,
        alpha=0.8,
        title=f"{title_suffix} ({model_name.capitalize()})"
    )

    # Apply dimensionality reduction to specified number of components
    reduced_activations = apply_dimensionality_reduction(
        sorted_activations, n_components=n_components
    )

    # Plot the Representational Dissimilarity Matrix (RDM)
    plot_rdm(
        reduced_activations,
        metric=distance_metric,
        title=f"{title_suffix} ({model_name.capitalize()})"
    )

    # Classification using SVM for each value of C
    for C in C_values:
        # Prepare dataset for classification
        features, labels = prepare_dataset(reduced_activations, noise_level=noise_level)

        # Train and evaluate SVM classifier
        (
            testing_accuracy,
            y_true_aggregated,
            y_pred_aggregated,
            y_true_aggregated_train,
            y_pred_aggregated_train,
            fold_indices,
        ) = classify(features, labels, C=C, pout=test_samples)

        # Plot label accuracy per fold
        plot_label_accuracy_per_fold(
            y_true_aggregated,
            y_pred_aggregated,
            fold_indices,
            label_mapping,
            color_mapping,
            title=f"SVM {title_suffix} {model_name.capitalize()}",
        )

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

# Determine device: use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define file paths for input data and output
stimuli_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2023/data/stimuli"
out_root = "/home/eik-tb/Desktop/New Folder"

# Set parameters for analysis
distance_metric = "correlation"  # Metric for computing RDM
n_components = 10                # Number of components for dimensionality reduction
C_values = [1]                   # List of C values for SVM parameter
test_samples = 10                # Number of samples in the test set per sub-cat (30 tot)
noise_level = 0                  # Level of noise to add to the dataset

# Process the CLIP model with semantic analysis
process_model("clip", "Semantic")

# Process the AlexNet model with perceptual analysis
process_model("alexnet", "Perceptual")

# Process the ResNet-50 model with perceptual analysis
process_model("resnet50", "Perceptual")
