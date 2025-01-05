import os, inspect, shutil
import random
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from modules import (
    plt,
    SUBCAT_ORDER,
    CAT_ORDER,
    COLOR_MAPPING,
    LETTER2INT_MAPPING,
    LABEL_MAPPING,
)

def set_rnd_seed(seed=42):
    # Set random seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

def perform_ttest(vector, chance_level=0.25, group_label=None):
    """
    Perform a one-sample t-test on a single vector and calculate the 95% confidence interval.

    Parameters:
    - vector (array-like): Input data (e.g., list, numpy array, or pandas series).
    - chance_level (float): Value to compare the sample mean against.

    Returns:
    - result (dict): Dictionary containing t-test results and confidence interval.
    """

    # Validate input is a unidimensional vector
    if not isinstance(vector, (list, np.ndarray, pd.Series)) or np.ndim(vector) != 1:
        raise ValueError("Input 'vector' must be a unidimensional array-like object.")

    # Convert input to numpy array
    vector = np.array(vector)

    # Perform t-test
    ttest_result = stats.ttest_1samp(vector, chance_level, alternative="greater")

    # Calculate statistics
    mean = np.mean(vector)
    sem = stats.sem(vector)
    n = len(vector)

    # Calculate confidence interval using the t-distribution

    # This is different from what seaborn does, so the ci_up and ci_down may differ
    # between this and the  plots. To match seaborn, we would need to do a bootstrap
    # (non-parametric) and get the  low and high perceptiles, such as:
    # # Calculate bootstrap confidence intervals
    # res = bootstrap((data,), statistic, confidence_level=0.95, n_resamples=1000, method='percentile')

    # # Extract the lower and upper bounds
    # ci_low, ci_high = res.confidence_interval

    # Here we do a simpler ci over the mean, without bootstrapping, which assumes normal data
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)

    # Return results as a dictionary
    result = {
        "mean": mean,
        "sem": sem,
        "t_stat": ttest_result.statistic,
        "p_value": ttest_result.pvalue,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "group": group_label,
    }

    return result


def annotate_bar(stats_result, line_center, _ax):
    """
    Annotate bars with significance markers based on t-test results.

    Parameters:
    - stats_results (list): List of t-test results.
    - line_center (list): List of tuples (line_x, line_y, line_lower, line_upper, group).
    - _ax (matplotlib.axes.Axes): Axis object for the plot.
    """
    line_x, line_y, line_lower, line_upper, group = line_center
    p_value = stats_result["p_value"]
    ci_up = stats_result["ci_high"]

    significance = (
        "***"
        if p_value < 0.001
        else "**" if p_value < 0.01 else "*" if p_value < 0.05 else None
    )
    if significance:
        _ax.text(line_x, ci_up + 0.01, significance, ha="center")


def plot_mvpa_barplot(
    data,
    x_hue,
    y_col,
    x_groups=None,
    chance_level=0.25,
    title="",
    hue_order=None,
    x_groups_order=None,
    out_dir=None,
):
    """
    Plot MVPA bar plot with statistical annotations.

    Parameters:
    - data (pd.DataFrame): Input data for plotting.
    - x_hue (str): Column for hue categories (e.g., true labels), shown in the legend.
    - y_col (str): Column for y-axis values (e.g., accuracy).
    - x_groups (str or None): Column for x-axis groups (stacked bars), or None if ungrouped.
    - chance_level (float): Reference chance level for significance testing.
    - title (str): Title of the plot.
    - hue_order (list or None): Order of hue categories for plotting and operations.
    - x_groups_order (list or None): Order of x-axis groups for plotting.

    Returns:
    - stats_results (list): A list of dictionaries with t-test results for each category or group.
    """

    def get_bars(_ax):
        # Filter out bars with height or width equal to 0 (e.g., legend placeholders)
        bars = [
            bar for bar in _ax.patches if bar.get_height() > 0 and bar.get_width() > 0
        ]
        return bars

    def get_lines(_ax):
        # Get all lines in the plot
        lines = [line for line in _ax.lines]
        return lines

    def find_xmajortick_for_bar(_ax, bar):
        """
        Find the x-major tick label corresponding to a given bar element.

        Parameters:
        - ax: matplotlib Axes object containing the bar chart.
        - bar: The bar element (matplotlib Rectangle object).

        Returns:
        - The x-major tick label (string) if found, otherwise None.
        """
        # Calculate the bar's x-center
        bar_center = bar.get_center()[0]

        # Get x-tick positions and labels
        x_ticks = _ax.get_xticks()
        x_tick_labels = [tick.get_text() for tick in _ax.get_xticklabels()]

        # Find the index of the closest x-tick
        closest_index = np.argmin([abs(bar_center - tick) for tick in x_ticks])

        return x_tick_labels[closest_index]

    def get_bars_coordinates(_ax):
        """
        Calculate bar details for annotation, ensuring alignment with x_groups_order and hue_order.

        Parameters:
        - ax (matplotlib.axes.Axes): Axis object containing the bars.

        Returns:
        - List of tuples (bar_x, bar_y, ci_upper, ci_lower, group_tuple) for each bar:
          bar_x: x-coordinate of the bar center
          bar_y: y-coordinate (height) of the bar
          ci_upper: upper limit of the confidence interval
          ci_lower: lower limit of the confidence interval
          group_tuple: tuple of (hue, group) or (hue, None) if ungrouped
        """

        # Filter out bars with height or width equal to 0 (e.g., legend placeholders)
        bars = get_bars(_ax)

        # Get all lines in the plot
        lines = get_lines(_ax)

        # Make sure we have one error line per bar
        assert len(bars) == len(lines)

        # Get hues (categories) from the legend
        hues = tuple(
            [label.get_text() for label in _ax.get_legend().get_texts()]
            if _ax.get_legend()
            else [label.get_text() for label in _ax.get_xmajorticklabels()]
        )

        # Calculate the number of bars per hue
        bars_per_hue = len(bars) // len(hues)

        # Create a list of tuples: (x, y, y_upper, y_lower, index)
        bar_tuples = [
            (
                bars[i].get_x()
                + bars[i].get_width() / 2,  # x-coordinate (center of the bar)
                bars[i].get_height(),  # y-coordinate (height of the bar)
                np.nanmax(lines[i].get_ydata()),  # y upper bound (error bar upper)
                np.nanmin(lines[i].get_ydata()),  # y lower bound (error bar lower)
                (
                    find_xmajortick_for_bar(_ax, bars[i]),  # x_major tick (ROI)
                    hues[i // bars_per_hue],  # Hue assignment based on index
                ),
            )
            for i in range(len(bars))
        ]

        return bar_tuples

    # Determine figure size dynamically based on the number of x_groups
    figsizex = 3 * (len(x_groups_order) if x_groups else len(hue_order)) + 1
    fig, ax = plt.subplots(figsize=(figsizex, 12))

    # Plot the bars with seaborn
    x_group_col = x_groups if x_groups else x_hue
    x_groups_order = x_groups_order if x_groups else hue_order
    sns.barplot(
        data=data,
        x=x_group_col,
        y=y_col,
        hue=x_hue,
        palette=COLOR_MAPPING,
        errorbar=("ci", 95),
        err_kws={"linewidth": 1},
        capsize=0.1,
        ax=ax,
        order=x_groups_order,
        hue_order=hue_order,
    )

    # Get coordinates for bars and error bars
    bar_tuples = get_bars_coordinates(ax)

    # Perform t-tests and annotate bars
    stats_results = []

    # Define the order of bars based on provided arguments
    bars_order = (
        [(x, h) for h in hue_order for x in x_groups_order]
        if x_groups_order != hue_order
        else [(h, h) for h in hue_order]
    )

    if x_group_col is None:
        x_group_col = x_hue

    # Decide the iteration structure: single loop if orders are the same, else nested loops
    for idx, (x_group, hue_group) in enumerate(bars_order):

        group_data_vector = np.squeeze(
            data[(data[x_hue] == hue_group) & (data[x_group_col] == x_group)][
                "Accuracy"
            ].values
        )

        # Perform ttest against chance level --> is the average acc > than chance?
        stats_result_group = perform_ttest(
            group_data_vector, chance_level=0.25, group_label=(x_group, hue_group)
        )

        # Annotate this bar in the plot
        bar_tuple = next(
            (
                bar_tuple
                for bar_tuple in bar_tuples
                if bar_tuple[-1] == (x_group, hue_group)
            ),
            None,
        )

        annotate_bar(stats_result_group, bar_tuple, ax)

        # Store results
        stats_results.append(stats_result_group)

    # Add chance level reference line
    ax.axhline(chance_level, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(x_groups if x_groups else x_hue)
    ax.set_title(title, pad=30)
    ax.set_ylim(0.0, 1.1)

    # Show the plot
    plt.tight_layout()

    if out_dir is not None:
        # Save the plot
        filename = title.replace(" ", "_").lower() + ".png"
        plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")

    plt.show()

    return stats_results


def generate_latex_table(
    stats_results, caption="Statistical Results", label="tab:stats_results"
):
    """
    Generate a LaTeX table summarizing the statistical results.

    Parameters:
    - stats_results (list): List of dictionaries containing:
        - 'group': The group/category (e.g., ROI).
        - 'hue': The hue/label (e.g., Face, Vehicle).
        - 'mean': Mean accuracy.
        - 'sem': Standard error of the mean.
        - 't_stat': t-statistic.
        - 'p_value': p-value.
        - 'error_95': 95% confidence interval error.
    - caption (str): Caption for the LaTeX table.
    - label (str): Label for referencing the table in LaTeX.

    Returns:
    - str: LaTeX table string.
    """
    # Start building the LaTeX table
    table_header = r"""
    \begin{table}[htbp]
    \centering
    \caption{%s}
    \label{%s}
    \begin{tabular}{lccccc}
    \hline
    Group & Hue & Mean Accuracy & SEM & t-statistic & p-value & 95\%% CI \\
    \hline
    """ % (
        caption,
        label,
    )

    # Generate rows for each result
    table_rows = []
    for res in stats_results:
        significance = ""
        if res["p_value"] < 0.001:
            significance = r"$^{***}$"
        elif res["p_value"] < 0.01:
            significance = r"$^{**}$"
        elif res["p_value"] < 0.05:
            significance = r"$^{*}$"

        row = (
            f"{res['group']} & {res['mean']:.2f}{significance} & "
            f"{res['sem']:.2f} & {res['t_stat']:.2f} & {res['p_value']:.3g} & ±{res['sem']*1.96:.2f} \\"
        )
        table_rows.append(row)

    # Footer for the LaTeX table
    table_footer = r"""
    \hline
    \multicolumn{7}{l}{\textsuperscript{*}$p<0.05$, \textsuperscript{**}$p<0.01$, \textsuperscript{***}$p<0.001$} \\
    \end{tabular}
    \end{table}
    """

    # Combine header, rows, and footer
    full_table = table_header + "\n".join(table_rows) + table_footer

    return print(full_table)


def generate_results_paragraph(stats_results):
    """
    Generate a textual summary of the statistical results.

    Parameters:
    - stats_results (list): List of dictionaries containing:
        - 'group': The group/category (e.g., ROI).
        - 'hue': The hue/label (e.g., Face, Vehicle).
        - 'mean': Mean accuracy.
        - 'sem': Standard error of the mean.
        - 't_stat': t-statistic.
        - 'p_value': p-value.

    # Example Input:
    stats_results = [
        {'group': 'FFA', 'hue': 'Face', 'mean': 0.4958, 'sem': 0.0406, 't_stat': 6.056, 'p_value': 1.774e-6, 'error_95': 0.0796},
        {'group': 'FFA', 'hue': 'Vehicle', 'mean': 0.4958, 'sem': 0.0419, 't_stat': 5.866, 'p_value': 2.801e-6, 'error_95': 0.0821},
        {'group': 'Foveal 0.5\degree', 'hue': 'Face', 'mean': 0.2708, 'sem': 0.0415, 't_stat': 0.502, 'p_value': 0.31, 'error_95': 0.0813}
    ]

    Returns:
    - str: A formatted paragraph summarizing the results.
    """
    paragraph = "Statistical analysis revealed the following results:\n\n"
    for res in stats_results:
        significance = "NOT statistically significant"
        if res["p_value"] < 0.05:
            significance = "statistically SIGNIFICANT"

        paragraph += (
            f"For group '{res['group']}', the mean accuracy was "
            f"{res['mean']:.2f} (SEM = {res['sem']:.2f}). A one-sample t-test against chance level "
            f"yielded t({len(stats_results) - 1}) = {res['t_stat']:.2f}, p = {res['p_value']:.3g}, "
            f"indicating that the result was {significance}.\n\n"
        )
    return print(paragraph)


def save_and_plot_confusion_matrix(
    y_true,
    y_pred,
    output_dir=None,
    title="confusion_matrix",
    fmt=".2f",
    colorbar=True,
    ax=None,
):
    """
    Creates and saves a styled confusion matrix plot.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - output_dir (str, optional): Directory to save the plot and CSV. If None, files won't be saved.
    - title (str, optional): Title for the confusion matrix plot.
    - fmt (str, optional): Format for annotations, default is ".2f".
    - colorbar (bool, optional): Whether to include a colorbar in the heatmap.
    - ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis for plotting. If None, a new figure is created.

    Returns:
    - cm_normalized (numpy.ndarray): The normalized confusion matrix data.
    """
    # Determine labels based on the unique predictions
    group_labels = SUBCAT_ORDER if len(set(y_pred)) == 4 else CAT_ORDER

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=group_labels)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)  # Normalize

    # Use the provided axis or create a new figure
    if ax is None:
        fig, ax = plt.subplots()

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=colorbar,
        vmin=0,
        vmax=1,
        xticklabels=group_labels,
        yticklabels=group_labels,
        square=True,
        ax=ax,
    )

    # Remove leading zeros in annotations
    for text in ax.texts:
        text.set_text(
            text.get_text().lstrip("0") if "." in text.get_text() else text.get_text()
        )

    # Set titles and labels
    ax.set_title(title, pad=20)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(group_labels, rotation=30, ha="right")
    ax.set_yticklabels(group_labels, rotation=30)

    # Save the confusion matrix data to a CSV file
    if output_dir:
        csv_filename = os.path.join(output_dir, f"confusion_matrix_{title}.csv")
        cm_df = pd.DataFrame(cm_normalized, index=group_labels, columns=group_labels)
        cm_df.to_csv(csv_filename)
        print(f"Confusion matrix data saved to: {csv_filename}")

    return cm_normalized


def create_run_id():
    """
    Generate a unique run identifier based on the current date and time.

    This function creates a string representing the current date and time in the format 'YYYYMMDD-HHMMSS'.
    It can be used to create unique identifiers for different runs or experiments.

    :returns: A string representing the current date and time.
    :rtype: str
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def map_subcategory_to_categories(labels_list):
    return [
        (
            "Vehicle"
            if label.lower() in ["bike", "car"]
            else "Face" if label.lower() in ["male", "female"] else None
        )
        for label in labels_list
    ]


def get_label_from_int(i, INT_MAPPING=LETTER2INT_MAPPING, LABEL_MAPPING=LABEL_MAPPING):
    """
    Given an integer, returns the respective label.

    Parameters:
        i (int): The integer to lookup in INT_MAPPING.
        INT_MAPPING (dict): A dictionary mapping keys to integers.
        LABEL_MAPPING (dict): A dictionary mapping keys to labels.

    Returns:
        str: The corresponding label, or None if the integer is not found.
    """
    # Get the key corresponding to the integer
    key = next((key for key, value in INT_MAPPING.items() if value == i), None)

    if key is None:
        return None  # Handle the case where the integer is not in INT_MAPPING

    # Get the label corresponding to the key
    return LABEL_MAPPING.get(key, None)


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


def report_mvpa_results(
    data,
    title="",
    chance_level=0.25,
    accuracy_col="accuracy",
    true_col="True Labels",
    pred_col="Predicted Labels",
    subject_col="Subject",
    x_groups=None,
    x_groups_order=None,
    hue_order=None,
    outroot=None,
):
    """
    Generates plots and statistical results for MVPA analysis.

    Parameters:
    - data (pd.DataFrame): Input data.
    - title (str): Plot title.
    - chance_level (float): Reference chance level for significance testing.
    - group_by (str or None): Grouping variable for analysis (e.g., ROI).
    - accuracy_col (str): Column name for accuracy values. If not provided, it will be calculated.
    - true_col (str): Column name for true labels.
    - pred_col (str): Column name for predicted labels.
    - subject_col (str): Column name for subject identifiers.
    - x_groups (str or None): Column to group x-axis (e.g., ROI), optional.
    - x_groups_order (list or None): Custom order for x-axis groups.
    - hue_order (list or None): Custom order for hues (labels).
    - outroot (str or None): Directory to save plots.

    Returns:
    - stats_results (list): Statistical results.
    """
    # Rename columns for consistency
    data = data.rename(
        columns={
            true_col: "True Labels",
            pred_col: "Predicted Labels",
            subject_col: "Subject",
            x_groups: "Group" if x_groups else None,
            accuracy_col: "Accuracy" if accuracy_col in data.columns else None,
        }
    )

    # Calculate the accuracy column only if it is not explicitly provided
    if "Accuracy" not in data.columns:
        data["Accuracy"] = (data["True Labels"] == data["Predicted Labels"]).astype(float)

    # Group the data appropriately based on input
    group_cols = ["Subject", "True Labels"]
    if x_groups:
        group_cols.append("Group")

    grouped_data = data.groupby(group_cols).mean(numeric_only=True).reset_index()

    # Determine hue order dynamically if not provided
    if hue_order is None:
        hue_order = (
            CAT_ORDER if len(grouped_data["True Labels"].unique()) == 2 else SUBCAT_ORDER
        )

    # Call the plot function
    stats_results = plot_mvpa_barplot(
        data=grouped_data,
        x_hue="True Labels",
        y_col="Accuracy",
        x_groups="Group" if x_groups else None,
        chance_level=chance_level,
        title=title,
        hue_order=hue_order,
        x_groups_order=x_groups_order,
    )

    # Generate reports
    generate_results_paragraph(stats_results)
    generate_latex_table(stats_results)
