import random
import os, shutil, inspect, glob
from scipy.io import loadmat
import nibabel as nb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold

from modules import (
    SUBCAT_ORDER,
    ROIS_ORDER,
    COLOR_MAPPING,
    FOV_ROIS_COLORMAP,
)
from modules.utils import (
    map_subcategory_to_categories,
    perform_ttest,
    annotate_bar,
    create_run_id,
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


def load_fmri_classification_data(
    file_path, roi_order=ROIS_ORDER, category_order=SUBCAT_ORDER
):
    """
    Load and process classification data from a pickle file into a structured DataFrame.

    This function extracts region-of-interest (ROI) classification results from
    a given pickle file, constructs a unified DataFrame, and processes the data
    for further analysis, including accuracy computation and category mapping.
    It validates the provided ROI and category orders to ensure they match
    the data, processing all strings in lowercase for validation and
    returning a DataFrame with capitalized strings.

    Parameters
    ----------
    file_path : str
        Path to the pickle file containing classification data.

    roi_order : list
        List of relevant ROIs to filter and analyze (e.g., ["Foveal", "Peripheral"]).

    category_order : list
        Ordered list of categories to ensure consistency (e.g., ["female", "male", "bike", "car", "face", "vehicle"]).

    Returns
    -------
    pd.DataFrame
        A processed and filtered DataFrame with classification results.
        Columns include:
        - "roi": Region of Interest
        - "subject": Subject ID
        - "y_true": True labels
        - "y_pred": Predicted labels
        - "accuracy": Classification accuracy
        - "y_true_cat": True category ("Face" or "Vehicle")
        - "y_pred_cat": Predicted category ("Face" or "Vehicle")
    """
    # Load dataset from pickle file
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # Extract ROI keys and ensure all ROIs in roi_order are present
    roi_keys = [
        roi
        for roi in data["results"]["classification"][
            "bike", "car", "female", "male"
        ].keys()
    ]

    missing_rois = set(roi_order) - set(roi_keys)
    if missing_rois:
        raise ValueError(f"ROIs in roi_order are missing from the data: {missing_rois}")

    # Determine the number of subjects
    n_subjects = len(
        data["results"]["classification"]["bike", "car", "female", "male"][
            list(roi_keys)[0]
        ]["y_true_aggregated"]
    )

    # Initialize an empty list to store individual DataFrames
    data_frames = []

    # Iterate over ROIs and subjects to construct the DataFrame
    for roi in roi_keys:
        for subject in range(n_subjects):
            y_true = data["results"]["classification"]["bike", "car", "female", "male"][
                roi
            ]["y_true_aggregated"][subject]
            y_pred = data["results"]["classification"]["bike", "car", "female", "male"][
                roi
            ]["y_pred_aggregated"][subject]

            data_frames.append(
                pd.DataFrame(
                    {
                        "ROI": roi,
                        "Subject": subject,
                        "y_true": [label.capitalize() for label in y_true],
                        "y_pred": [label.capitalize() for label in y_pred],
                    }
                )
            )

    # Combine all individual DataFrames into one unified DataFrame
    full_data = pd.concat(data_frames, ignore_index=True)

    # Validate that all unique labels are included in category_order
    unique_labels = set(full_data["y_true"].unique()).union(full_data["y_pred"].unique())

    missing_labels = unique_labels - set(category_order)
    if missing_labels:
        raise ValueError(
            f"Labels in the data are missing from category_order: {missing_labels}"
        )

    # Calculate accuracy for each subject in each ROI
    full_data["Correct"] = (full_data["y_true"] == full_data["y_pred"]).astype(float)

    # Categorize true and predicted values into "Face" or "Vehicle"
    full_data["y_true_cat"] = map_subcategory_to_categories(full_data["y_true"])
    full_data["y_pred_cat"] = map_subcategory_to_categories(full_data["y_pred"])

    # Filter data for specified ROIs
    filtered_data = full_data[full_data["ROI"].isin([roi for roi in roi_order])].copy()

    return filtered_data


def load_new_fmri_classification_data(
    file_path, roi_order=ROIS_ORDER, category_order=SUBCAT_ORDER
):
    """
    Load and process classification data from a pickle file into a structured DataFrame.

    This function extracts region-of-interest (ROI) classification results from
    a given pickle file, constructs a unified DataFrame, and processes the data
    for further analysis, including accuracy computation and category mapping.
    It validates the provided ROI and category orders to ensure they match
    the data, processing all strings in lowercase for validation and
    returning a DataFrame with capitalized strings.

    Parameters
    ----------
    file_path : str
        Path to the pickle file containing classification data.

    roi_order : list
        List of relevant ROIs to filter and analyze (e.g., ["Foveal", "Peripheral"]).

    category_order : list
        Ordered list of categories to ensure consistency (e.g., ["female", "male", "bike", "car", "face", "vehicle"]).

    Returns
    -------
    pd.DataFrame
        A processed and filtered DataFrame with classification results.
        Columns include:
        - "roi": Region of Interest
        - "subject": Subject ID
        - "y_true": True labels
        - "y_pred": Predicted labels
        - "accuracy": Classification accuracy
        - "y_true_cat": True category ("Face" or "Vehicle")
        - "y_pred_cat": Predicted category ("Face" or "Vehicle")
    """
    # Load dataset from pickle file
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # Extract ROI keys and ensure all ROIs in roi_order are present
    roi_keys = data["ROI"].unique()

    missing_rois = set(roi_order) - set(roi_keys)

    if missing_rois:
        raise ValueError(f"ROIs in roi_order are missing from the data: {missing_rois}")

    # Validate that all unique labels are included in category_order
    unique_labels = set(data["True Labels"].unique()).union(
        data["Predicted Labels"].unique()
    )

    missing_labels = unique_labels - set(category_order)
    if missing_labels:
        raise ValueError(
            f"Labels in the data are missing from category_order: {missing_labels}"
        )

    # Categorize true and predicted values into "Face" or "Vehicle"
    data["y_true_cat"] = map_subcategory_to_categories(data["True Labels"])
    data["y_pred_cat"] = map_subcategory_to_categories(data["Predicted Labels"])
    data["y_true"] = data["True Labels"]
    data["y_pred"] = data["Predicted Labels"]

    data = data.drop(["Predicted Labels", "True Labels"], axis=1)

    # Filter data for specified ROIs
    filtered_data = data[data["ROI"].isin([roi for roi in roi_order])].copy()

    return filtered_data


def calculate_cv_upper_bound(n):
    """
    Calculate the theoretical upper bound of the Coefficient of Variation (CV)
    for a row in a confusion matrix with n categories. This is the CV when
    all observations are concentrated in one cell.
    """
    return np.sqrt(n - 1)


def permutation_test_correlation(x, y, n_permutations=1000, random_state=None):
    """
    Perform a permutation test to determine the significance of the correlation
    between two arrays x and y.

    Parameters
    ----------
    x, y : array-like
        Input arrays for correlation calculation.
    n_permutations : int
        Number of permutations to perform.
    random_state : int or None
        If int, sets the random seed for reproducibility.

    Returns
    -------
    corr : float
        The observed correlation.
    p_value : float
        The p-value from the permutation test.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Compute the observed correlation
    corr = np.corrcoef(x, y)[0, 1]

    # Null distribution: shuffle y and recalculate correlation
    null_distributions = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        null_corr = np.corrcoef(x, y_perm)[0, 1]
        null_distributions.append(null_corr)

    null_distributions = np.array(null_distributions)
    # P-value: proportion of null correlations that exceed or equal the observed correlation in absolute value
    p_value = np.mean(np.abs(null_distributions) >= np.abs(corr))

    return corr, p_value


def load_dnn_confmats(tdann, clip):
    tdann_cm = pd.read_csv(tdann,index_col=0).values
    clip_cp = pd.read_csv(clip,index_col=0).values
    return tdann_cm, clip_cp


def plot_significant_correlation_barplot(
    correlation_matrix, roi_names, significance_matrix, output_path=None
):
    """
    Plot barplots for the correlations of each ROI with "TDANN" and "CLIP".

    Parameters
    ----------
    correlation_matrix : ndarray
        NxN correlation matrix.
    roi_names : list of str
        Names corresponding to rows/columns of the matrix.
    significance_matrix : ndarray (object)
        NxN matrix with significance levels: NaN for not significant, "*", "**", "***" for significance levels.
    output_path : str or None
        Path to save the plots if provided.
    """
    # Identify the indices for "TDANN" and "CLIP"
    try:
        tdann_index = roi_names.index("TDANN")
        clip_index = roi_names.index("CLIP")
    except ValueError as e:
        raise ValueError("Both 'TDANN' and 'CLIP' must be present in roi_names.") from e

    # Remove "TDANN" and "CLIP" from ROI names for x-axis
    filtered_roi_names = [roi for roi in roi_names if roi not in ["TDANN", "CLIP"]]

    # Extract correlations for TDANN and CLIP, excluding their own entries
    tdann_correlations = np.delete(
        correlation_matrix[:, tdann_index], [tdann_index, clip_index]
    )
    clip_correlations = np.delete(
        correlation_matrix[:, clip_index], [tdann_index, clip_index]
    )

    # Extract significance for TDANN and CLIP, excluding their own entries
    tdann_significance = np.delete(
        significance_matrix[:, tdann_index], [tdann_index, clip_index]
    )
    clip_significance = np.delete(
        significance_matrix[:, clip_index], [tdann_index, clip_index]
    )

    def annotate_bars(ax, bars, significance):
        """
        Annotate bars with significance asterisks if significant.

        Parameters
        ----------
        ax : matplotlib Axes
            The axis to annotate.
        bars : list of BarContainer
            The bars to annotate.
        significance : list of str
            Significance levels corresponding to the bars ("*", "**", "***", or NaN).
        """
        for bar, sig in zip(bars, significance):
            if sig and type(sig) == str:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # Center of the bar
                    height + 0.02,  # Slightly above the bar
                    sig,
                    ha="center",
                    va="bottom",
                    color="black",
                )

    # Plot barplot for TDANN
    plt.figure()
    ax = plt.gca()
    bars = ax.bar(filtered_roi_names, tdann_correlations, color="#FF5733")
    annotate_bars(ax, bars, tdann_significance)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Correlations with TDANN Confusion Matrix")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.1)  # Set y-axis limits
    plt.tight_layout()
    if output_path:
        plt.savefig(
            os.path.join(output_path, "Correlations_with_TDANN.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()

    # Plot barplot for CLIP
    plt.figure()
    ax = plt.gca()
    bars = ax.bar(filtered_roi_names, clip_correlations, color="#3498DB")
    annotate_bars(ax, bars, clip_significance)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Correlations with CLIP Confusion Matrix")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.1)  # Set y-axis limits
    plt.tight_layout()
    if output_path:
        plt.savefig(
            os.path.join(output_path, "Correlations_with_CLIP.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def generate_correlation_table_latex(
    correlation_matrix, p_values, all_cms, alpha=0.05, decimals_r=3, decimals_p=4
):
    """
    Generate a LaTeX table for pairwise correlations with significance testing.

    Parameters:
    - correlation_matrix: 2D array-like, correlation coefficients matrix.
    - p_values: 2D array-like, p-values matrix.
    - all_cms: dict, keys are ROI names.
    - alpha: float, significance threshold for p-values.
    - decimals_r: int, number of decimal places for correlation coefficients.
    - decimals_p: int, number of decimal places for p-values.

    Returns:
    - str: LaTeX formatted string for the correlation table.
    """
    roi_names = list(all_cms.keys())
    table = []

    # Table start
    table.append(r"\begin{table}[htbp]")
    table.append(r"    \centering")
    table.append(
        r"    \caption{Pairwise correlations between ROI models with permutation-based significance testing.}"
    )
    table.append(r"    \label{tab:correlation_matrix}")
    table.append(r"    \begin{tabular}{" + ("l" + "c" * len(roi_names)) + "}")
    table.append(r"    \toprule")

    # Header row
    header = " & " + " & ".join([r"\textbf{" + roi + "}" for roi in roi_names]) + r" \\"
    table.append("    " + header)
    table.append(r"    \midrule")

    # Rows
    for i, roi_i in enumerate(roi_names):
        row_entries = [r"\textbf{" + roi_i + "}"]
        for j in range(len(roi_names)):
            if i == j:
                # Diagonal: self-comparison
                entry = r"$r=1.000$ \\—"
            else:
                r_val = correlation_matrix[i, j]
                p_val = p_values[i, j]
                r_str = f"{r_val:.{decimals_r}f}"
                p_str = f"{p_val:.{decimals_p}g}"
                if p_val < alpha:
                    entry = rf"$r={r_str}$ \\ $p={p_str}^*$"
                else:
                    entry = rf"$r={r_str}$ \\ $p={p_str}$"
            row_entries.append(entry)
        table.append("    " + " & ".join(row_entries) + r" \\")

    # Footer and note
    table.append(r"    \bottomrule")
    table.append(r"    \end{tabular}")
    table.append(r"    \vspace{1em}")
    table.append(r"    \small")
    table.append(
        r"    \textit{Note}: The table presents Pearson correlation coefficients ($r$) and their associated"
    )
    table.append(
        r"    p-values ($p$) obtained via a permutation test (10,000 permutations). Entries on the diagonal"
    )
    table.append(
        r"    represent self-comparisons. An asterisk ($^*$) indicates $p < 0.05$."
    )
    table.append(r"\end{table}")

    return "\n".join(table)


def load_and_prepare_bh_data(bh_data_path, subcat_order, runs):
    """Load and preprocess behavioral data."""
    # Load data
    behavioral_data = pd.read_csv(bh_data_path, sep="\t")

    # Capitalize column names
    behavioral_data.columns = [col.capitalize() for col in behavioral_data.columns]

    # Calculate mean accuracy for categories and overall
    category_means = behavioral_data[subcat_order].mean()
    overall_mean = category_means.mean()

    return behavioral_data, category_means, overall_mean


def perform_two_way_anova(behavioral_data, subcat_order, runs):
    """Perform two-way ANOVA and calculate effect sizes."""

    def _print_anova_results(anova_table):
        """Print detailed ANOVA results."""
        print("\nTwo-way ANOVA results:")
        print(anova_table)

        print("\nDetailed results:")
        for effect in ["C(Condition)", "C(Run)"]:
            df_effect = anova_table.loc[effect, "df"]
            df_residual = anova_table.loc["Residual", "df"]
            f_value = anova_table.loc[effect, "F"]
            p_value = anova_table.loc[effect, "PR(>F)"]
            eta_squared = anova_table.loc[effect, "eta2_p"]

            print(f"\n{effect}:")
            print(f"F({df_effect}, {df_residual}) = {f_value:.2f}")
            print(f"p = {p_value:.3f}")
            print(f"η²p = {eta_squared:.2f}")
        return

    # Reshape data for two-way ANOVA
    data_long = pd.melt(
        behavioral_data.reset_index(),
        id_vars=["index"],
        value_vars=subcat_order + runs,
        var_name="Variable",
        value_name="Accuracy",
    )

    # Create 'Condition' and 'Run' columns
    data_long["Condition"] = data_long["Variable"].apply(
        lambda x: x if x in subcat_order else "Other"
    )
    data_long["Run"] = data_long["Variable"].apply(lambda x: x if x in runs else "Other")

    # Perform two-way ANOVA
    model = ols("Accuracy ~ C(Condition) + C(Run)", data=data_long).fit()
    anova_table = anova_lm(model, typ=2)

    # Calculate partial eta-squared
    anova_table["eta2_p"] = anova_table["sum_sq"] / (
        anova_table["sum_sq"] + anova_table.loc["Residual", "sum_sq"]
    )

    _print_anova_results(anova_table)

    return anova_table


def plot_behavioral_accuracy_boxplot(
    behavioral_data,
    subcat_order,
    overall_mean,
    title="Behavioral Accuracy Across Categories",
    color_mapping=COLOR_MAPPING,
    output_path=None,
):
    """Plot a boxplot of behavioral accuracy across sub-categories."""
    fig, ax = plt.subplots(figsize=(8, 12))
    flierprops = dict(marker="o", markersize=10, linestyle="none", markerfacecolor="w")
    sns.boxplot(
        data=behavioral_data[subcat_order],
        palette=color_mapping,
        ax=ax,
        flierprops=flierprops,
    )

    # Customize plot aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Sub-category")
    ax.set_title("Behavioral Accuracy Across Categories", pad=20)
    ax.axhline(overall_mean, color="black", linestyle="--", label="Overall Mean")
    ax.legend()

    # Set vmin and vmax for the y-axis
    ax.set_ylim(0, 1.1)  # Replace 0 and 1.2 with your desired minimum and maximum values

    # Save and show plot
    plt.tight_layout()
    if output_path:
        plt.savefig(
            os.path.join(output_path, title + ".png"), dpi=300, bbox_inches="tight"
        )
    plt.show()


def compute_confusion_matrices(filtered_data, rois, subcat_order):
    """
    Compute confusion matrices for each ROI.

    Parameters:
    - filtered_data: DataFrame, the filtered dataset containing 'roi', 'y_true', and 'y_pred'.
    - rois: list, list of ROI names.
    - subcat_order: list, order of subcategories for labels.

    Returns:
    - dict: Confusion matrices for each ROI.
    """
    return {
        roi: confusion_matrix(
            filtered_data[filtered_data["ROI"] == roi]["y_true"],
            filtered_data[filtered_data["ROI"] == roi]["y_pred"],
            labels=subcat_order,
        )
        for roi in rois
    }


def load_and_add_dnn_confmats(all_cms, tdann, clip):
    """
    Load DNN confusion matrices and append them to the ROI confusion matrices.

    Parameters:
    - all_cms: dict, existing confusion matrices for ROIs.
    - load_dnn_confmats_func: function, callable to load DNN confusion matrices.

    Returns:
    - dict: Updated confusion matrices with DNN entries.
    """
    tdann_cm, clip_cm = load_dnn_confmats(tdann, clip)
    all_cms["TDANN"] = tdann_cm
    all_cms["CLIP"] = clip_cm
    return all_cms


def compute_correlations_and_significance(all_cms, n_permutations=10000, alpha=0.05):
    """
    Compute correlation matrix and significance testing using permutation tests.

    Parameters:
    - all_cms (dict): A dictionary where keys are ROI/model names and values are confusion matrices.
    - n_permutations (int): Number of permutations for the correlation significance test.
    - alpha (float): Significance level threshold for p-values.

    Returns:
    - correlation_matrix (np.array): Matrix of Pearson correlation coefficients.
    - p_values (np.array): Matrix of p-values corresponding to correlations.
    - significance_matrix (np.array): Matrix indicating significance levels:
        - NaN if not significant.
        - "*" for p < 0.05.
        - "**" for p < 0.01.
        - "***" for p < 0.001.
    """
    import numpy as np

    # Number of ROIs/models
    n = len(all_cms.keys())

    # Initialize matrices for correlation coefficients and p-values
    correlation_matrix = np.zeros((n, n))  # To store Pearson correlation coefficients
    p_values = np.ones((n, n))  # To store p-values

    # Extract keys (names of ROIs/models) from the dictionary
    keys = list(all_cms.keys())

    # Iterate over all pairs of ROIs/models
    for i in range(n):
        for j in range(i, n):
            # Flatten the confusion matrices for correlation calculation
            vec_i = all_cms[keys[i]].flatten()
            vec_j = all_cms[keys[j]].flatten()

            # Compute correlation and p-value using permutation testing
            corr, p_val = permutation_test_correlation(
                vec_i, vec_j, n_permutations=n_permutations, random_state=42
            )

            # Assign the correlation and p-value to the symmetric matrix positions
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
            p_values[i, j] = p_val
            p_values[j, i] = p_val

    # Initialize the significance matrix with NaN
    significance_matrix = np.full((n, n), np.nan, dtype=object)

    # Populate the significance matrix based on p-value thresholds
    for i in range(n):
        for j in range(n):
            if p_values[i, j] < 0.001:
                significance_matrix[i, j] = "***"
            elif p_values[i, j] < 0.01:
                significance_matrix[i, j] = "**"
            elif p_values[i, j] < 0.05:
                significance_matrix[i, j] = "*"

    return correlation_matrix, p_values, significance_matrix


def plot_foveal_roi_accuracy(
    fov_data, x_hue, y_col, chance_level=0.25, title="", hue_order=None, out_dir=None
):
    """
    Plot MVPA bar plot with statistical annotations.

    Parameters:
    - data (pd.DataFrame): Input data for plotting.
    - x_hue (str): Column for hue categories (e.g., true labels), shown in the legend.
    - y_col (str): Column for y-axis values (e.g., accuracy).
    - chance_level (float): Reference chance level for significance testing.
    - title (str): Title of the plot.
    - hue_order (list or None): Order of hue categories for plotting and operations.

    Returns:
    - stats_results (list): A list of dictionaries with t-test results for each category or group.
    """

    def get_lines(_ax):
        # Get all lines in the plot
        lines = [line for line in _ax.lines]
        return lines

    def get_lines_coordinates(_ax):
        # Get all lines in the plot
        lines = get_lines(_ax)

        # Get hues (categories) from the legend
        hues = tuple([label.get_text() for label in _ax.get_xmajorticklabels()])

        # Create a list of tuples: (x, y, y_upper, y_lower, index)
        line_tuples = [
            (
                lines[i].get_xdata()[0],  # x-coordinate
                lines[i + 1].get_ydata()[0],  # y-coordinate
                lines[i + 2].get_ydata()[0],  # y upper bound
                lines[i + 3].get_ydata()[0],  # y lower bound
                (
                    None,  # x_major tick (ROI)
                    hues[i // 4],  # Hue assignment based on index
                ),
            )
            for i in range(0, len(lines), 4)  # Process groups of 4 lines
        ]
        return line_tuples

    # Calculate average accuracy per participant per ROI
    fov_data_accuracy = (
        fov_data.groupby([x_hue, "Subject"])["Correct"].mean().reset_index()
    )

    # Ensure accuracy is a float for proper plotting
    fov_data_accuracy["Correct"] = fov_data_accuracy["Correct"].astype(float)

    # Determine figure size dynamically based on the number of x_groups
    fig, ax = plt.subplots()

    # Plot the bars with seaborn
    sns.lineplot(
        data=fov_data_accuracy,
        x=x_hue,
        y="Correct",
        marker="o",
        palette=FOV_ROIS_COLORMAP,
        hue=x_hue,
        err_style="bars",
        err_kws={"elinewidth": 2, "capsize": 5},  # Error bars appearance
        markers=True,
        legend=False,
        ax=ax,
        errorbar=("ci", 95),
        n_boot=10000,
    )

    # Get x and y coordinates of each line in the plot
    line_tuples = get_lines_coordinates(ax)

    # Perform t-tests
    stats_results = []

    # For each ROI, let's perform a ttest and annotate the respective bar if needed
    for roi in hue_order:

        # Slice the data belonging to this ROI
        roi_data = fov_data_accuracy[fov_data_accuracy[x_hue] == roi]

        # Make sure we have a unidimensional vector --> 1 group of observations
        group_data_vector = np.squeeze(roi_data["Correct"].values)

        # Perform ttest against chance level --> is the average acc > than chance?
        stats_results_roi = perform_ttest(
            group_data_vector, chance_level=0.25, group_label=(None, roi)
        )

        # Annotate this bar in the plot
        line_tuple = next(
            (line_tuple for line_tuple in line_tuples if line_tuple[-1] == (None, roi)),
            None,
        )

        # Annotate this ROI bar in the plot
        annotate_bar(stats_results_roi, line_tuple, ax)

        # Store results
        stats_results.append(stats_results_roi)

    # Adjust labels, ticks, and titles
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Region of Interest")
    ax.set_xticks(range(len(hue_order)))
    ax.set_xticklabels(hue_order, rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_title("Decoding Accuracy over Foveal ROIs", pad=30)
    ax.axhline(0.25, color="black", linestyle="--", label="Overall Mean")

    # Show the plot
    plt.tight_layout()

    if out_dir is not None:
        # Save the plot
        filename = title.replace(" ", "_").lower() + ".png"
        plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")

    plt.show()

    return stats_results


def initialize_graph_dataframe(params):
    """
    Initialize a DataFrame for storing graph-related data.

    Args:
        params (dict): Dictionary containing parameters including 'rois' (list of ROIs)
                       and 'conditions_list' (list of condition comparisons).

    Returns:
        pd.DataFrame: A pandas DataFrame with initialized columns for graph data.
    """
    columns = ("avg", "std", "comparison", "roi")  # Columns needed in the graph dataframe
    graph_df = pd.DataFrame(columns=columns)  # Initialize the DataFrame

    # Repeat ROIs for each condition
    graph_df["roi"] = params["rois"] * len(params["conditions_list"])

    # Assign condition comparisons, repeated for each ROI
    graph_df["comparison"] = [
        comparison
        for comparison in params["conditions_list"]
        for _ in range(len(params["rois"]))
    ]

    return graph_df


def initialize_reports_and_results(params):
    """
    Initialize nested dictionaries for storing reports and results.

    Args:
        params (dict): Dictionary containing 'rois' (list of ROIs) and
                       'conditions_list' (list of condition comparisons).

    Returns:
        dict: A dictionary to hold results for all ROIs and conditions.
    """
    conditions = tuple([condition.lower() for condition in SUBCAT_ORDER])
    # Create a nested dictionary to hold results for each condition and ROI
    results = {
        conditions: {
            roi: {
                "acc": [],  # List to store accuracy values
                "y_true_aggregated": [],  # Aggregated true labels
                "y_pred_aggregated": [],  # Aggregated predicted labels
                "y_true_aggregated_train": [],  # Aggregated true train labels
                "y_pred_aggregated_train": [],  # Aggregated predicted train labels
            }
            for roi in params["rois"]
        }
    }

    return results


def get_mask_files(params, sub):
    """
    Retrieve paths to mask files for a specific subject.

    Args:
        params (dict): Dictionary containing the root directory for ROIs.
        sub (str): Subject identifier (e.g., 'sub-02').

    Returns:
        dict: Mapping of ROI names to mask file paths.
    """

    def parse_roi_label(roi_label):
        """
        Convert ROI label to a human-readable name.

        Args:
            roi_label (str): Label string extracted from the file name.

        Returns:
            str: Human-readable name for the ROI.
        """
        if "FOV+" in roi_label:
            size = roi_label.split("+")[-1]
            degrees = f"{size[0]}.{size[1]}" if size[1] != "0" else size[0]
            return f"Foveal {degrees}°"
        mapping = {
            "FFA": "FFA",
            "LOC": "LOC",
            "OPP": "Opposite",
            "PER": "Peripheral",
        }
        return mapping.get(roi_label, "Unknown")

    if "rois_root" not in params:
        raise ValueError("The 'rois_root' key is missing in parameters.")

    sub_path = os.path.join(params["rois_root"], sub)  # Subject directory path

    # Find all ROI mask files for the subject
    roi_files = glob.glob(os.path.join(sub_path, f"{sub}*_label-*_roi.*nii"))

    # Map ROI names to file paths
    masks = {
        parse_roi_label(
            os.path.basename(path).split("_label-")[-1].split("_roi")[0]
        ): path
        for path in roi_files
    }

    return masks


def get_beta_dataframes(beta_loc):
    """
    Generate a DataFrame containing information about beta values.

    Args:
        beta_loc (str): Directory path containing beta files and SPM.mat file.

    Returns:
        pd.DataFrame: DataFrame containing beta file paths and associated metadata.
    """
    # Initialize the DataFrame
    betas_df = pd.DataFrame(
        None,
        columns=("beta_path", "spm_filename", "condition", "run", "bin", "array"),
    )

    # Populate the beta file paths
    betas_df["beta_path"] = sorted(glob.glob(os.path.join(beta_loc, "*beta*.?ii")))

    # Load SPM.mat data
    mat = loadmat(os.path.join(beta_loc, "SPM.mat"))

    # Match beta file names with SPM.mat information
    matching_indices = [
        idx
        for idx, entry in enumerate(mat["SPM"]["Vbeta"][0][0][0])
        if any(
            str(entry[0][0]) in os.path.basename(path) for path in betas_df["beta_path"]
        )
    ]

    # Extract and populate metadata
    betas_df["spm_filename"] = [
        str(mat["SPM"]["Vbeta"][0][0][0][idx][0][0]) for idx in matching_indices
    ]
    betas_df["condition"] = [
        str(mat["SPM"]["Vbeta"][0][0][0][idx][5][0]).split(" ")[-1].split("*")[0]
        for idx in matching_indices
    ]
    betas_df["run"] = [
        str(mat["SPM"]["Vbeta"][0][0][0][idx][5][0]).split(" ")[-2].split("(")[-1][0]
        for idx in matching_indices
    ]
    betas_df["bin"] = [
        str(mat["SPM"]["Vbeta"][0][0][0][idx][5][0]).split(" ")[-1].split("(")[-1][0]
        for idx in matching_indices
    ]
    betas_df["array"] = [
        np.array(nb.load(beta).get_fdata()) for beta in betas_df["beta_path"]
    ]

    return betas_df


def filter_and_sort_betas(betas_df, conditions):
    """
    Filter and sort beta values DataFrame based on specified conditions.

    Args:
        betas_df (pd.DataFrame): Original DataFrame containing beta data.
        conditions (tuple): Tuple of conditions to filter by (e.g., ('face', 'vehicle')).

    Returns:
        pd.DataFrame: Filtered and sorted DataFrame by 'condition' and 'run'.
    """
    # Filter rows where 'condition' matches any condition in the provided tuple
    filtered_betas_df = betas_df[betas_df["condition"].str.match("|".join(conditions))]

    # Sort the filtered DataFrame by 'condition' and 'run', reset the index for clean output
    sorted_betas_df = filtered_betas_df.sort_values(["condition", "run"]).reset_index(
        drop=True
    )

    return sorted_betas_df


def demean_data_per_run(betas_df):
    """
    Demean beta data for each run in the DataFrame.

    This function calculates the mean beta value for each run, subtracts it from
    the beta values of that run, and updates the DataFrame with demeaned values.

    Args:
        betas_df (pd.DataFrame): DataFrame containing beta data arrays and run indices.

    Returns:
        pd.DataFrame: Updated DataFrame with demeaned beta data.
    """
    # Extract all beta arrays into a single numpy array
    all_betas_data = np.array([array for array in betas_df["array"]])

    # Get the unique run identifiers
    unique_runs = np.unique(betas_df["run"].values)

    # Iterate over each run to calculate and apply the demeaning operation
    for run in unique_runs:
        # Boolean mask for selecting rows corresponding to the current run
        run_mask = betas_df["run"] == run

        # Extract beta data for the current run
        run_data = all_betas_data[run_mask.values]

        # Compute the mean beta array for the current run
        run_mean = np.mean(run_data, axis=0)

        # Subtract the mean from all beta arrays in the run
        demeaned_data = run_data - run_mean

        # Update the DataFrame with the demeaned beta arrays
        betas_df.loc[run_mask, "array"] = list(demeaned_data)

    return betas_df


def prepare_dataset_for_classification(betas_df, mask_files, roi_name):
    """
    Prepare and preprocess dataset for classification by applying an ROI mask and shuffling data.

    Args:
        betas_df (pd.DataFrame): DataFrame containing beta data arrays.
        mask_files (dict): Dictionary mapping ROI names to corresponding mask file paths.
        roi_name (str): Name of the ROI to use for masking.

    Returns:
        tuple: Three numpy arrays - X (features), y (labels), runs_idx (run indices).
    """
    # Load the ROI mask for the specified ROI
    mask = nb.load(mask_files[roi_name]).get_fdata() > 0

    # Apply the mask to each beta data array, handling NaN values
    masked_betas = [np.nan_to_num(beta[mask], nan=0.0) for beta in betas_df["array"]]

    # Extract conditions (labels) and run indices
    labels = list(betas_df["condition"])
    runs = list(betas_df["run"])

    # Combine features, labels, and run indices for shuffling
    dataset = list(zip(masked_betas, labels, runs))

    # Shuffle the dataset to randomize order
    random.seed(42)  # Ensure reproducibility with a fixed seed
    random.shuffle(dataset)

    # Unzip the shuffled dataset into separate arrays
    features, labels, run_indices = zip(*dataset)

    # Convert the lists to numpy arrays for compatibility with downstream tasks
    return np.array(features), np.array(labels), np.array(run_indices)


def save_results_to_file(params):
    """
    Save results to a pickle file if logging is enabled.

    This function generates a unique filename based on the current conditions and other parameters,
    serializes the `params` dictionary into a pickle file, and saves it to the specified output directory.

    Args:
        params (dict): Dictionary containing experiment parameters and results.
            Expected keys:
                - 'out_dir': Directory to save the pickle file.
                - 'results': Dictionary of results to be saved.
                - 'conditions_list': List of condition comparisons.

    Returns:
        None. The results are saved to a file if logging is enabled.
    """
    # Ensure required parameters are present
    if "out_dir" not in params:
        raise ValueError("'out_dir' is missing from parameters.")

    # Create a unique filename for the results file
    run_id = create_run_id()  # Generate a unique identifier for the run
    conditions_str = "-".join(SUBCAT_ORDER)  # Combine conditions into a string
    filename = f"{run_id}_{conditions_str}.pkl"  # Construct the full filename

    # Determine the full path for the pickle file
    file_path = os.path.join(params["out_dir"], filename)

    # Save the results dictionary to the pickle file
    with open(file_path, "wb") as fp:
        pickle.dump(params, fp)

    # Notify the user of the saved file location
    print(f"Results dictionary saved as: {file_path}")


def perform_classification(X, y, runs_idx, C=1):
    """
    Perform data classification using Support Vector Machines (SVM) and
    cross-validation, printing and returning relevant classification metrics.

    Parameters:
    - X (array-like): Input data, where `n_samples` is the number of samples and
                      `n_features` is the number of features.
    - y (array-like): Target values.
    - runs_idx (array-like): Group labels for the samples.
    - roi_name (str): Name of the Region of Interest.
    - conditions (list): Conditions under which classification is performed.
    - subject_id (str): Identifier of the subject being analyzed.

    """

    # Define SVM kernel type
    kernel = "linear"

    # Initialize GroupKFold with the number of unique groups
    gkf = GroupKFold(n_splits=len(set(runs_idx)))

    # Initialize lists to store accuracy for each fold
    training_accuracy = []
    testing_accuracy = []

    # Prepare lists to aggregate true and predicted labels across all folds for confusion matrix and report
    y_true_aggregated = []
    y_pred_aggregated = []
    y_true_aggregated_train = []
    y_pred_aggregated_train = []

    # Loop through each split of the data into training and test sets
    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=runs_idx)):
        # Output the current step and related indices information
        print("\n\n===================================================")
        print(f"## FOLD: {i+1} ##")
        print("Indices of train samples:", train_idx.tolist())
        print("Indices of test samples:", test_idx.tolist())
        print("... corresponding to the following runs:", runs_idx[test_idx].tolist())

        # Define training and test sets
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
        y_true_aggregated.extend(y_test)
        y_pred_aggregated.extend(y_pred_test)
        y_true_aggregated_train.extend(y_train)
        y_pred_aggregated_train.extend(y_pred_train)

        # Output detailed predictions and performance for the current step
        print(f"\nTraining Accuracy: {train_acc:.2f}, Testing Accuracy: {test_acc:.2f}")
        print(f"\nTraining Predicted vs Actual: {list(zip(y_pred_train, y_train))}")
        print(f"\nTesting Predicted vs Actual: {list(zip(y_pred_test, y_test))}\n")

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
    )
