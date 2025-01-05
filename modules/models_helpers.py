import os
import shutil
import inspect
from collections import defaultdict
from copy import deepcopy
import random

import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.svm import SVC
from sklearn.manifold import MDS
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter
import torch
import clip
from torchvision import models, transforms

from modules import (
    plt,
    TDANN_PATH,
    DEVICE,
    LETTER2INT_MAPPING,
    LABEL_MAPPING,
    COLOR_MAPPING,
)
from modules.utils import get_label_from_int, map_subcategory_to_categories

# TDANN-specific imports
from src.positions import NetworkPositions
from src.model import load_model_from_checkpoint, LAYERS
from src.data import load_images_from_folder
from src.features import FeatureExtractor


def load_clean_dnn_activations(model_name: str, stimuli_path: str):
    """
    Load or compute clean (noise-free) activations for the specified model.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "TDANN", "Clip", "resnet50").
    stimuli_path : str
        Path to the stimuli images.

    Returns
    -------
    dict
        Dictionary of activations keyed by stimulus ID, each containing:
        {
            "activation": np.ndarray,
            "category": str,
            "subcategory": str,
            ...
        }
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    if model_name == "TDANN":
        return process_images_tdann(stimuli_path)
    else:
        model, preprocess_fn = load_model_and_preprocess(model_name, DEVICE)
        return process_images(model_name, model, preprocess_fn, stimuli_path, DEVICE)


def plot_activations_MDS(clean_activations, model_output_dir, title):
    # Convert the clean activations into a NumPy array for MDS
    # We'll average along the first dimension if each entry is 2D or 3D
    activation_arrays_clean = [
        np.expand_dims(np.mean(act["activation"], axis=0), axis=0)
        for act in clean_activations.values()
    ]
    activations_clean_np = np.array(activation_arrays_clean)

    # Extract labels to color the MDS plot
    class_labels_clean = np.array([act["category"] for act in clean_activations.values()])

    # Visualize MDS projection of the noise-free activations
    # This MDS function internally reduces dimensions and plots the 2D embedding
    visualize_projections(
        activations_clean_np,
        class_labels_clean,
        alpha=0.8,
        title=title,
        out_dir=model_output_dir,
    )
    return


def prepare_dnn_participant_data(
    participant_id: int,
    clean_activations: dict,
    noise_level: float,
    config: dict,
):
    """
    Prepares data for a single participant:
      1) Adds noise to the activations,
      2) Prepares features and labels.

    Parameters
    ----------
    participant_id : int
        Numeric ID/index of the participant.
    clean_activations : dict
        A dictionary of original (noise-free) activations, keyed by stimulus ID.
    noise_level : float
        Magnitude of the Gaussian noise to inject.
    config : dict
        Experiment configuration dictionary, including "smooth_noise".
    add_gaussian_noise_np_smooth : function
        Function to add Gaussian (optionally smoothed) noise to a NumPy array.
    standardize_activations_np : function
        Function to standardize activations.

    Returns
    -------
    features : np.ndarray
        Feature matrix (processed activations) for the participant.
    labels : np.ndarray
        Ground-truth labels corresponding to the features.
    """

    def _prepare_dataset(activations):
        labels = list(activations.keys())
        features = np.array(
            [activations[label]["activation"].flatten() for label in labels]
        ).squeeze()
        numeric_labels = [LETTER2INT_MAPPING[label[0]] for label in labels]

        y = np.array(numeric_labels)
        X = np.array(features)
        return X, y

    print(
        f"Preparing data for PARTICIPANT: {participant_id} | Noise level: {noise_level}"
    )

    # Deep copy the original activations to avoid overwriting them
    participant_activations = deepcopy(clean_activations)

    # Convert activation arrays to NumPy for noise injection
    activation_arrays = [
        np.expand_dims(np.mean(act["activation"], axis=0), axis=0)
        for act in participant_activations.values()
    ]
    activations_np = np.array(activation_arrays)

    # Add Gaussian noise (smoothed if specified in config)
    activations_np = add_gaussian_noise_np_smooth(
        activations_np,
        noise_level=noise_level,
        smooth_noise=True,
    )

    # Flatten and standardize the activations
    activations_np = np.array([act.flatten() for act in activations_np])
    activations_np = standardize_activations_np(activations_np)

    # Place processed activations back into the participant_activations dictionary
    for idx, stim_id in enumerate(participant_activations):
        participant_activations[stim_id]["activation"] = activations_np[idx]

    # Prepare dataset (features and labels)
    features, labels = _prepare_dataset(participant_activations)

    return features, labels


def get_dnn_results_df(
    mvpa_results,
    participant_id: int,
    noise_level: float,
):
    """
    Classifies data and records results:
      1) Classifies the data at the category and sub-category levels,
      2) Creates results DataFrames.

    Parameters
    ----------
    participant_id : int
        Numeric ID/index of the participant.
    features : np.ndarray
        Feature matrix for the participant.
    labels : np.ndarray
        Ground-truth labels for the participant.
    noise_level : float
        Noise level used for this participant's data.
    config : dict
        Experiment configuration dictionary, including "regularization".
    classify : function
        Classification function that returns predictions and metrics.
    map_subcategory_to_categories : function
        Function to map sub-category labels to higher-level category labels.

    Returns
    -------
    results_df_subcat : pd.DataFrame
        DataFrame containing sub-category-level classification results.
    results_df_cat : pd.DataFrame
        DataFrame containing category-level classification results.
    """
    print(
        f"Classifying data for PARTICIPANT: {participant_id} | Noise level: {noise_level}"
    )

    (
        _,
        y_true,
        y_pred,
        _,
        _,
        fold_indices,
        _,
    ) = mvpa_results

    # Create DataFrame for sub-category results
    results_data = {
        "Subject": participant_id,
        "Fold": fold_indices,
        "True Labels": y_true,
        "Predicted Labels": y_pred,
        "Correct": [t == p for t, p in zip(y_true, y_pred)],
    }
    results_df_subcat = pd.DataFrame(results_data)

    # Create DataFrame for category-level results by mapping sub-categories to categories
    results_data_cat = {
        **results_data,
        "True Labels": map_subcategory_to_categories(y_true),
        "Predicted Labels": map_subcategory_to_categories(y_pred),
    }
    results_df_cat = pd.DataFrame(results_data_cat)

    # Annotate DataFrames with noise level and participant ID
    results_df_subcat["noise_level"] = noise_level
    results_df_subcat["participant_id"] = participant_id
    results_df_cat["noise_level"] = noise_level
    results_df_cat["participant_id"] = participant_id

    return results_df_subcat, results_df_cat


def process_images_tdann(
    image_path,
    layer="layer2.0",
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
        LABEL_MAPPING (dict): Mapping from labels to categories.
        n_batches (int): Number of batches to process in the dataloader.
        verbose (bool): If True, print progress information.

    Returns:
        dict: A dictionary containing activations for each label.
    """

    positions_dir = os.path.join(TDANN_PATH, "paths/positions")
    weights_path = os.path.join(
        TDANN_PATH, "paths/weights/model_final_checkpoint_phase199.torch"
    )
    image_path = "./data/stimuli"

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

    for i, feature in enumerate(features[layer]):
        label_idx = labels[i]
        label = idx_to_label[label_idx]

        # Here we trim some pixels because, when we plotted the activations, we
        # notices some artifacts at the edges, likely due to kernel padding. We
        # decided to trim these pixels since they showed to be particularly
        # indicative of the class category, biasing the SVM performance.
        trimmed_feature = feature[:, 3:-3, 3:-3]

        # Store the extracted features and label for the image
        activations[label] = {
            "activation": trimmed_feature,  # Tensor as numpy for easier handling
            "category": LABEL_MAPPING[label[0]],  # Map first character to category
        }

    return activations


def visualize_projections(
    activations_np, class_labels, alpha=0.5, method="MDS", title="", out_dir=None
):
    """
    Generates a 2D scatter plot of the reduced dimensionality data.

    Parameters:
        activations (dict): Dictionary containing reduced dimensionality data.
        alpha (float): Transparency level for the plot points.

    Outputs:
        Displays a 2D scatter plot.
    """

    # Perform MDS on the flattened activations
    mds_proj = apply_dimensionality_reduction_np(
        activations_np.reshape(activations_np.shape[0], -1), n_components=2, method=method
    )
    # Create a new dictionary structured like participant_activations, but storing 2D MDS coordinates
    activations = {
        idx: {"category": class_labels[idx], "activation": mds_proj[idx]}
        for idx in range(len(class_labels))
    }

    labels = sorted(set([str(item["category"]) for key, item in activations.items()]))

    plt.figure()
    full_title = f"{title} {method} projection"
    plt.title(full_title, pad=20)

    # Plot each category with the specified color and alpha transparency
    for category in labels:
        points = np.array(
            [v["activation"] for v in activations.values() if v["category"] == category]
        )
        plt.scatter(
            points[:, 0],
            points[:, 1],
            alpha=alpha,
            label=category,
            color=COLOR_MAPPING[category],
            s=100,  # Adjust this value to control the size of the points
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
    filename = os.path.join(out_dir, filename) if out_dir is not None else filename
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def standardize_activations_np(activations):
    """
    Standardizes the activations to have mean 0 and variance 1.

    Parameters:
        activations (dict): Dictionary with stim_id as keys and 'activation' data.

    Returns:
        dict: Updated activations with standardized values.
    """
    # Apply standard scaling
    scaler = StandardScaler()
    activations_scaled = scaler.fit_transform(activations)

    return activations_scaled


def apply_smoothing_kernel(array, kernel_size=4, stride=1):
    if stride != 1:
        raise ValueError("Stride must be 1 to preserve input dimensions.")

    if len(array.shape) == 3:
        size = (1, kernel_size, kernel_size)
    else:
        size = (1, kernel_size)

    # Apply the uniform filter across height and width dimensions
    smoothed_array = uniform_filter(
        array,
        size=size,
        mode="constant",
        cval=0.0,
    )

    return smoothed_array


def add_gaussian_noise_np_smooth(activations, noise_level=0.1, smooth_noise=False):
    """
    Adds Gaussian noise to the activations with a standard deviation
    proportional to the original standard deviation of each activation.

    Parameters:
        activations (np.ndarray): 2D array where each row represents an activation sample.
        noise_level (float): Multiplier for the standard deviation of the noise to be added.

    Returns:
        np.ndarray: Updated activations with added Gaussian noise.
    """

    original_std = np.std(activations)
    original_avg = 0

    for i, act in enumerate(activations):

        if noise_level != 0:
            # Scale noise level by the original standard deviation
            sigma = noise_level * original_std
            noise = np.random.normal(original_avg, sigma, act.shape)

            if smooth_noise:
                Ks = estimate_kernel_sigma(act)
                noise = apply_smoothing_kernel(noise, kernel_size=Ks, stride=1)
        else:
            noise = np.zeros_like(act)

        activations[i] = act + noise

    return activations


def estimate_kernel_sigma(data):
    """
    Estimate the kernel sigma used in Gaussian smoothing of a matrix or vector.

    Parameters:
        data (1D, 2D, or 3D array): Input smoothed data.

    Returns:
        float: Estimated kernel sigma.
        int: Estimated kernel size (assumes 3σ cutoff).
    """
    # Handle 3D matrices by averaging across the first dimension
    data = np.squeeze(data)
    if len(data.shape) == 3:
        data = np.mean(data, axis=0)

    # If data is 1D, process directly
    if len(data.shape) == 1:
        # Center the vector by removing its mean
        data -= np.mean(data)

        # Compute 1D autocorrelation using FFT
        f = np.fft.fft(data)
        acf = np.fft.ifft(f * np.conj(f)).real
        acf = np.fft.fftshift(acf)

        # Normalize the autocorrelation
        acf /= np.max(acf)

        # Define the x-axis for fitting
        center = len(acf) // 2
        x = np.arange(len(acf)) - center

    # If data is 2D, process as before
    elif len(data.shape) == 2:
        # Center the data by removing its mean
        data -= np.mean(data)

        # Apply a window to reduce edge artifacts (optional)
        window = np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1]))
        data *= window

        # Compute the 2D autocorrelation using FFT
        f = np.fft.fft2(data)
        acf = np.fft.ifft2(f * np.conj(f)).real
        acf = np.fft.fftshift(acf)

        # Normalize the autocorrelation
        acf /= np.max(acf)

        # Extract central slices along both axes
        center = acf.shape[0] // 2
        acf_x = acf[center, :]
        acf_y = acf[:, center]

        # Average the slices to get a 1D profile
        acf = (acf_x + acf_y) / 2

        # Define the x-axis for fitting
        x = np.arange(len(acf)) - center
    else:
        raise ValueError("Input data must be 1D, 2D, or 3D.")

    # Define the Gaussian function for fitting
    def gaussian(x, a, sigma):
        return a * np.exp(-(x**2) / (2 * sigma**2))

    # Initial guess for fitting parameters
    p0 = [1, 1]

    # Perform the curve fitting
    try:
        popt, _ = curve_fit(gaussian, x, acf, p0=p0)
    except RuntimeError:
        raise ValueError(
            "Curve fitting failed. Check the input data for noise or irregularities."
        )

    # Extract the fitted sigma
    sigma_fitted = popt[1]

    # Convert fitted sigma to kernel sigma
    estimated_sigma = sigma_fitted / np.sqrt(2)

    # Estimate the kernel size (3σ cutoff)
    estimated_kernel_size = int(2 * np.ceil(3 * estimated_sigma) + 1)

    return estimated_kernel_size


def apply_dimensionality_reduction_np(activations, n_components, method="PCA", title=""):
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
    else:
        if method == "PCA":
            reducer = PCA(n_components=n_components, random_state=42)
        if method == "SVD":
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        elif method == "MDS":
            reducer = MDS(n_components=n_components, random_state=42)

        activations_reduced = reducer.fit_transform(activations)
        activations_reduced = np.array(activations_reduced)

    return activations_reduced


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
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    elif model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=True).to(device)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
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
    image_activations = (
        {}
    )  # Dictionary to store activations and categories for each image

    # Define a hook function to capture activations
    activations = defaultdict(list)

    def hook(module, input, output):
        activations["activation"] = output.cpu().detach().numpy()

    # Register hook for the ReLU layer after the first convolutional layer
    if model_name.lower() == "alexnet":
        layer = model.features[1]  # ReLU after the first conv layer in AlexNet
        hook_handle = layer.register_forward_hook(hook)

    elif model_name.lower() == "resnet50":
        layer = model.relu  # ReLU after the first conv layer in ResNet
        hook_handle = layer.register_forward_hook(hook)

    # Process each image
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(folder_path, filename)

            # Open the image and preprocess
            image = Image.open(path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            # Run the model to get activations
            with torch.no_grad():
                if model_name.lower() == "clip":
                    features = model.encode_image(
                        image_tensor
                    )  # Direct encoding for CLIP
                    activation = features.cpu().numpy()
                else:
                    # Forward pass through model to trigger hook
                    model(image_tensor)
                    activation = activations["activation"][0]

            # Use filename (without extension) as a unique key
            key = os.path.splitext(filename)[0]
            image_activations[key] = {
                "activation": activation,
                "category": LABEL_MAPPING[key[0]],  # Map first character to category
            }

            # Clear the activations for the next image
            activations.clear()

    if (model_name.lower() == "resnet50") or (model_name.lower() == "alexnet"):
        # Remove the hook after processing all images
        hook_handle.remove()

    return image_activations


def classify(X, y, folds_idxs=None, C=1):
    """
    Perform data classification using Support Vector Machines (SVM) and
    stratified 5-fold cross-validation, printing and returning relevant classification metrics.

    Parameters:
    - X (array-like): Input data, where `n_samples` is the number of samples and
                      `n_features` is the number of features.
    - y (array-like): Target values.

    Returns:
    - testing_accuracy (list): List of testing accuracy across all folds.
    - y_true_aggregated, y_pred_aggregated (list): Aggregated true and predicted labels for test sets.
    - y_true_aggregated_train, y_pred_aggregated_train (list): Aggregated true and predicted labels for training sets.
    - fold_indices (list): Fold index for each test instance.
    """

    # Shuffle the data beforehand to prevent any ordered class bias
    indices = np.random.permutation(len(y))
    X, y, folds_idxs = (
        X[indices].astype(np.float64),
        y[indices].astype(np.float64),
        folds_idxs[indices].astype(np.float64) if folds_idxs is not None else None,
    )

    # Define SVM kernel type
    kernel = "linear"
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store results
    training_accuracy = []
    testing_accuracy = []
    y_true_aggregated = []
    y_pred_aggregated = []
    y_true_aggregated_train = []
    y_pred_aggregated_train = []
    fold_indices = []

    # Begin cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=folds_idxs)):

        # Output the current step and related indices information
        print("\n\n===================================================")
        print(f"## STEP: {fold_idx+1} of {n_splits} ##")
        print("Indices of train samples:", train_idx.tolist())
        print("Indices of test samples:", test_idx.tolist())
        if folds_idxs is not None:
            print(
                "... corresponding to the following runs:", folds_idxs[test_idx].tolist()
            )

        # Get train/test data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Instantiate and fit the classifier on the training data
        clf = SVC(kernel=kernel, random_state=42, C=C, cache_size=1024 * 4).fit(
            X_train, y_train
        )

        # Predict labels for training and test sets
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        # Record training and testing accuracy
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        training_accuracy.append(train_acc)
        testing_accuracy.append(test_acc)

        # Aggregate true and predicted labels for test set
        y_true_aggregated.extend([get_label_from_int(y) for y in y_test])
        y_pred_aggregated.extend([get_label_from_int(y) for y in y_pred_test])
        y_true_aggregated_train.extend([get_label_from_int(y) for y in y_train])
        y_pred_aggregated_train.extend([get_label_from_int(y) for y in y_pred_train])
        fold_indices.extend([fold_idx] * len(test_idx))

        # Output detailed predictions and performance for the current step
        print(
            f"Fold {fold_idx} - Training Accuracy: {train_acc:.2f}, Testing Accuracy: {test_acc:.2f}"
        )
        # print(f"Training Predicted vs Actual: {list(zip(y_pred_train, y_train))}")
        # print(f"Testing Predicted vs Actual: {list(zip(y_pred_test, y_test))}\n")

    # Generate and print a classification report for the test set
    print("\nClassification Report (Test Set):")
    print(classification_report(y_true_aggregated, y_pred_aggregated))

    # Print a confusion matrix for the test set
    confmat = confusion_matrix(y_true_aggregated, y_pred_aggregated)
    # print("\nConfusion Matrix (Test Set):")
    # print(confmat)

    return (
        testing_accuracy,
        y_true_aggregated,
        y_pred_aggregated,
        y_true_aggregated_train,
        y_pred_aggregated_train,
        fold_indices,
        confmat,
    )


def plot_lineplot_with_ci(
    data, x_col, y_col, hue_col, title="", output_dir=None, palette=COLOR_MAPPING
):
    """
    Generate a polished line plot with shaded 95% confidence intervals for each line.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data to plot.
        x_col (str): Column name for the x-axis (e.g., "noise_level").
        y_col (str): Column name for the y-axis (e.g., "accuracy").
        hue_col (str): Column name for the line grouping (e.g., "category").
        title (str): Plot title.
        output_dir (str): Directory to save the plot. If None, the plot will not be saved.
    """
    plt.figure()

    # Create the lineplot with enhanced aesthetics
    sns.lineplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=hue_col,
        markers=True,
        dashes=False,
        linewidth=2.5,
        err_style="band",
        errorbar=("ci", 95),
        palette=COLOR_MAPPING,
    )

    # Enhance plot aesthetics
    plt.title(title, pad=20)
    plt.xlabel(x_col.replace("_", " ").capitalize(), labelpad=10)
    plt.ylabel(y_col.replace("_", " ").capitalize(), labelpad=10)
    plt.xticks()
    plt.yticks()
    plt.legend(
        title=hue_col.replace("_", " ").capitalize(),
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
        frameon=False,
    )
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    plt.tight_layout()

    # Add a horizontal line at chance level (0.25) and set y-axis limits
    plt.axhline(0.25, color="black", linestyle="--", linewidth=1, label="Chance level")
    plt.ylim(0, 1)

    # Save the plot if output directory is provided
    if output_dir:
        filename = f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.show()


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
