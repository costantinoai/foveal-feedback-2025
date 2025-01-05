import inspect
import shutil
import random
import os
import pandas as pd
import numpy as np
from modules import CAT_ORDER, SUBCAT_ORDER
from modules.utils import (
    save_and_plot_confusion_matrix,
    create_run_id,
    save_script_to_file as save_util_script_to_file,
    report_mvpa_results,
)
from modules.models_helpers import (
    classify,
    save_script_to_file as save_models_script_to_file,
    plot_lineplot_with_ci,
    load_clean_dnn_activations,
    plot_activations_MDS,
    prepare_dnn_participant_data,
    get_dnn_results_df,
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


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
config = {
    "stimuli_path": "./data/stimuli",
    "output_root": "./results",
    "noise_levels": np.arange(0, 11, 1),  # Range of noise levels from 0 to 10
    # "noise_levels": [10],  # Range of noise levels from 0 to 10
    "n_participants": 24,  # Number of participants to simulate (same as fMRI)
    "models_to_evaluate": [
        ("TDANN", "Perceptual"),
        ("CLIP", "Semantic"),
        # ("resnet50", "Perceptual")
    ],
    "regularization": 1e-2,  # Regularization parameter for SVM
}

# ----------------------------------------------------------------------------
# Main Analysis
# ----------------------------------------------------------------------------
for model_name, model_label in config["models_to_evaluate"]:

    model_output_dir = f"{config['output_root']}/{create_run_id()}_{model_name}_svm"
    os.makedirs(model_output_dir, exist_ok=False)

    # Save the script here for reproducibility
    save_util_script_to_file(model_output_dir)
    save_models_script_to_file(model_output_dir)
    save_script_to_file(model_output_dir)

    # ------------------------------------------------------------------------
    # 1) Load Clean (Noise-Free) Activations Once
    # ------------------------------------------------------------------------
    clean_activations = load_clean_dnn_activations(model_name, config["stimuli_path"])

    # ------------------------------------------------------------------------
    # 2) Plot MDS for Clean Activations (Only Once) --> Fig. 5A, 5B
    # ------------------------------------------------------------------------
    plot_activations_MDS(
        clean_activations, model_output_dir, f"{model_name}"
    )

    # Lists to store aggregated results (across noise levels)
    all_noises_results_category = []
    all_noises_results_subcategory = []

    # ------------------------------------------------------------------------
    # 3) For Each Noise Level
    # ------------------------------------------------------------------------
    for noise_level in config["noise_levels"]:
        # Create a unique directory for results at this noise level
        output_dir = (
            f"{model_output_dir}/{create_run_id()}_{model_name}_" f"noise-{noise_level}"
        )
        os.makedirs(output_dir, exist_ok=False)

        # Prepare containers for overall predictions (across participants)
        true_labels_all = []
        predicted_labels_all = []

        # Set random seeds for reproducibility
        np.random.seed(42)
        random.seed(42)

        # --------------------------------------------------------------------
        # 3a) For Each Participant, Add smoothed noise and classify
        # --------------------------------------------------------------------
        participant_category_accuracies = []
        participant_subcategory_accuracies = []

        for participant_id in range(config["n_participants"]):
            print(
                f"Processing PARTICIPANT: {participant_id} | Model: {model_name} | Noise level: {noise_level}"
            )

            # Prepare data (add noise and generate features/labels)
            features, labels = prepare_dnn_participant_data(
                participant_id, clean_activations, noise_level, config
            )

            # Perform classification
            mvpa_results = classify(features, labels, C=config["regularization"])

            # Get results dataframes
            results_df_subcat, results_df_cat = get_dnn_results_df(
                mvpa_results,
                participant_id,
                noise_level,
            )

            # Collect results
            participant_subcategory_accuracies.append(results_df_subcat)
            participant_category_accuracies.append(results_df_cat)

        # --------------------------------------------------------------------
        # 3b) Plot Accuracies & Confusion Matrices Across all participants  --> Fig. 5C, 5D, 5E, 5F
        # --------------------------------------------------------------------
        # Plot average accuracy at the category level + Latex tables
        report_mvpa_results(
            data=pd.concat(participant_category_accuracies).reset_index(drop=True),
            title=f"Category decoding accuracy - {model_name}",
            chance_level=0.25,
            accuracy_col="accuracy",
            true_col="True Labels",
            pred_col="Predicted Labels",
            subject_col="Subject",
            hue_order=CAT_ORDER,
            x_groups=None,
            x_groups_order=None,
            outroot=output_dir,
        )

        # Plot average accuracy at the sub-category level
        report_mvpa_results(
            data=pd.concat(participant_subcategory_accuracies).reset_index(drop=True),
            title=f"Sub-category decoding accuracy - {model_name}",
            chance_level=0.25,
            accuracy_col="accuracy",
            true_col="True Labels",
            pred_col="Predicted Labels",
            subject_col="Subject",
            hue_order=SUBCAT_ORDER,
            x_groups=None,
            x_groups_order=None,
            outroot=output_dir,
        )

        # Construct confusion matrix from all participants at sub-cat level
        all_subcat_df = pd.concat(participant_subcategory_accuracies, ignore_index=True)
        y_true_all = all_subcat_df["True Labels"]
        y_pred_all = all_subcat_df["Predicted Labels"]

        # Just call the function
        save_and_plot_confusion_matrix(
            y_true_all,
            y_pred_all,
            output_dir=output_dir,
            title=f"Confusion Matrix - {model_name} | Noise: {noise_level}",
            fmt=".2f",
            colorbar=True,
        )

        # Accumulate all participant results across noise levels
        all_noises_results_category.extend(participant_category_accuracies)
        all_noises_results_subcategory.extend(participant_subcategory_accuracies)

    # ------------------------------------------------------------------------
    # 4) Aggregate & Plot Across All Noise Levels --> Suppl Fig. 2A, 2B, 4
    # ------------------------------------------------------------------------
    # Category-level data (across subjects for all noise levels)
    all_category_data = pd.concat(all_noises_results_category).reset_index(drop=True)
    all_subcategory_data = pd.concat(all_noises_results_subcategory).reset_index(
        drop=True
    )

    # Save aggregated data
    all_category_data.to_csv(
        os.path.join(output_dir, f"{model_name}_category_data.csv"), index=False
    )
    all_subcategory_data.to_csv(
        os.path.join(output_dir, f"{model_name}_subcategory_data.csv"), index=False
    )

    # Group by noise level and true labels, compute mean accuracy (for categories)
    aggregated_data_cat = (
        all_category_data.groupby(["participant_id", "noise_level", "True Labels"])
        .agg(accuracy=("Correct", "mean"))
        .reset_index()
    )
    # Plot line plot for category-level
    plot_lineplot_with_ci(
        data=aggregated_data_cat,
        x_col="noise_level",
        y_col="accuracy",
        hue_col="True Labels",
        title=f"Category-Level Accuracy Across Noise Levels - {model_name}",
        output_dir=output_dir,
    )

    # Group by noise level and true labels, compute mean accuracy (for sub-categories)
    aggregated_data_subcat = (
        all_subcategory_data.groupby(["participant_id", "noise_level", "True Labels"])
        .agg(accuracy=("Correct", "mean"))
        .reset_index()
    )

    # Plot line plot for sub-category-level
    plot_lineplot_with_ci(
        data=aggregated_data_subcat,
        x_col="noise_level",
        y_col="accuracy",
        hue_col="True Labels",
        title=f"Sub-category-Level Accuracy Across Noise Levels - {model_name}",
        output_dir=output_dir,
    )
