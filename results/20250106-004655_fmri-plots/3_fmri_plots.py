import os, inspect, shutil
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from modules import plt, CAT_ORDER, SUBCAT_ORDER, ROIS_ORDER, FOV_ROIS_ORDER
from modules.utils import (
    save_and_plot_confusion_matrix,
    create_run_id,
    save_script_to_file as save_util_script_to_file,
    report_mvpa_results,
)
from modules.fmri_helpers import (
    load_fmri_classification_data,
    save_script_to_file as save_fmri_utils_to_file,
    plot_significant_correlation_barplot,
    generate_correlation_table_latex,
    plot_foveal_roi_accuracy,
    load_and_prepare_bh_data,
    perform_two_way_anova,
    plot_behavioral_accuracy_boxplot,
    compute_confusion_matrices,
    compute_correlations_and_significance,
    load_and_add_dnn_confmats,
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
params = {
    "data_path": "results/20250106-004549_fmri_mvpa/20250106-004606_Bike-Car-Female-Male.pkl",
    "clip_csv_path": "results/20250106-004452_CLIP_svm/20250106-004531_CLIP_noise-10/confusion_matrix_Confusion Matrix - CLIP | Noise: 10.csv",
    "tdann_csv_path": "results/20250106-004403_TDANN_svm/20250106-004445_TDANN_noise-10/confusion_matrix_Confusion Matrix - TDANN | Noise: 10.csv",
    "bh_data_path": "data/behavioural_acc.csv",
    "output_root": "results",
}

output_dir = os.path.join(params["output_root"], f"{create_run_id()}_fmri-plots")
os.makedirs(output_dir, exist_ok=False)

# Save the script here for reproducibility
save_util_script_to_file(output_dir)
save_fmri_utils_to_file(output_dir)
save_script_to_file(output_dir)

# Import the classification data
filtered_data = load_fmri_classification_data(os.path.join(params["data_path"]))

##### Plot 1: Category-Level MVPA Accuracy ---> Fig. 6B
report_mvpa_results(
    data=filtered_data,
    title="MVPA Accuracy by ROI and Category",
    chance_level=0.25,
    accuracy_col="Correct",
    true_col="y_true_cat",
    pred_col="y_pred_cat",
    subject_col="Subject",
    x_groups="ROI",
    x_groups_order=ROIS_ORDER,
    hue_order=CAT_ORDER,
    outroot=output_dir,
)

##### Plot 2: Sub-Category-Level MVPA Accuracy ---> Fig. 6A
report_mvpa_results(
    data=filtered_data,
    title="MVPA Accuracy by ROI and Sub-category",
    chance_level=0.25,
    accuracy_col="Correct",
    true_col="y_true",
    pred_col="y_pred",
    subject_col="Subject",
    x_groups="ROI",
    x_groups_order=ROIS_ORDER,
    hue_order=SUBCAT_ORDER,
    outroot=output_dir,
)

##### Plot 3: Confusion Matrix for each ROI ---> Fig. 6C
# Create a row of 5 subplots for confusion matrices
fig3, axes = plt.subplots(1, 5, figsize=(50, 13))
axes = axes.flatten()

# Define a shared normalization and colormap
cmap = cm.Blues
norm = mcolors.Normalize(vmin=0, vmax=1)

# Plot each ROI's confusion matrix
for i, ROI in enumerate(ROIS_ORDER):
    # Filter data for the current ROI
    cm_data = filtered_data[filtered_data["ROI"] == ROI]

    # Extract true and predicted labels
    y_true = cm_data["y_true"]
    y_pred = cm_data["y_pred"]

    # Plot directly on the subplot axis
    save_and_plot_confusion_matrix(
        y_true,
        y_pred,
        output_dir=None,
        title=f"{ROI}",
        fmt=".2f",
        colorbar=False,  # Suppress individual colorbars
        ax=axes[i],
    )

# Adjust layout to add more space between subplots
plt.subplots_adjust(wspace=0.5)  # Increase space between subplots

# Add a single colorbar
cbar = fig3.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, location="right", shrink=0.6
)
cbar.set_label("Normalized Count")

# Add a global title and adjust layout
plt.suptitle("Sub-category Confusion Matrices per ROI", y=0.8)
plt.tight_layout(rect=[0, 0, 0.01, 1])  # Adjust layout to make room for the colorbar
plt.savefig(
    os.path.join(output_dir, "ROI_confusion_matrices.png"), dpi=300, bbox_inches="tight"
)
plt.show()

##### Plot 4: Correlation between ROIs and models ---> Fig. 6D, 6E
# Compute confusion matrices for ROIs
all_cms = compute_confusion_matrices(filtered_data, ROIS_ORDER, SUBCAT_ORDER)

# Add DNN confusion matrices
all_cms = load_and_add_dnn_confmats(all_cms, params["tdann_csv_path"], params["clip_csv_path"])

# Compute correlations and significance
correlation_matrix, p_values, significance_matrix = compute_correlations_and_significance(
    all_cms
)

# Plot significant correlations
plot_significant_correlation_barplot(
    correlation_matrix, list(all_cms.keys()), significance_matrix, output_path=output_dir
)

# Generate and print LaTeX table
latex_table = generate_correlation_table_latex(correlation_matrix, p_values, all_cms)
print(latex_table)

##### Plot 5: Line Plot of Accuracy over Foveal ROIs --> Fig. 3A
fov_data = load_fmri_classification_data(
    os.path.join(params["data_path"]),
    roi_order=FOV_ROIS_ORDER,
)

# Plot the average decoding accuracy per Foveal ROI
plot_foveal_roi_accuracy(
    fov_data,
    x_hue="ROI",
    y_col="Correct",
    chance_level=0.25,
    title="Decoding Accuracy over Foveal ROIs",
    hue_order=FOV_ROIS_ORDER,
    out_dir=output_dir,
)

##### Plot 6: Behavioral Accuracy Across Categories ---> Fig. 4

runs = ["Run1", "Run2", "Run3", "Run4", "Run5"]

data_path = os.path.join(params["data_path"], "behavioural_acc.csv")

# Load and preprocess data
behavioral_data, category_means, overall_mean = load_and_prepare_bh_data(
    params["bh_data_path"], list(SUBCAT_ORDER), runs
)

#  Print summary statistics
print(f"Overall mean accuracy: {overall_mean:.2f}")
print("Mean accuracy for each category:")
for category, mean in category_means.items():
    print(f"{category}: {mean:.2f}")

# Perform two-way ANOVA
anova_table = perform_two_way_anova(behavioral_data, list(SUBCAT_ORDER), runs)

# Create and save boxplot
plot_behavioral_accuracy_boxplot(
    behavioral_data, list(SUBCAT_ORDER), overall_mean, output_path=output_dir
)
