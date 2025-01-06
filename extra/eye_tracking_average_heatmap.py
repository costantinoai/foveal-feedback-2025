#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:12:48 2024

@author: costantino_ai
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import numpy as np
import math
import pandas as pd
import glob
from scipy.signal import convolve2d
from PIL import Image

# Increase font size to 15 for all text in plots
plt.rcParams.update({"font.size": 15})

# Load and preprocess the image
def preprocess_image(image_path, target_size_px):
    """
    Load and resize an image to the target size while preserving aspect ratio.

    Parameters:
        image_path (str): Path to the image file.
        target_size_px (int): Desired size in pixels.

    Returns:
        ndarray: Resized image.
    """
    with Image.open(image_path) as img:
        # Resize image while maintaining aspect ratio
        img = img.resize((target_size_px, target_size_px), Image.Resampling.LANCZOS)
        return np.array(img)


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def visual_deg_to_px(
    degree, diagonal_size_inch=31.55, screen_distance_mm=1120, screen_resolution_px=(1920, 1080)
):
    """
    Convert visual degrees to pixels.

    Parameters:
        degree (float): The size in visual degrees.
        diagonal_size_inch (float): The diagonal size of the screen in inches.
        screen_distance_mm (int): The viewing distance from the viewer's eye to the screen in millimeters via a mirror.
        screen_resolution_px (int): The horizontal resolution of the screen.

    Returns:
        float: The corresponding size in pixels.
    """
    diagonal_size_mm = diagonal_size_inch * 25.4  # Convert inches to millimeters
    screen_width_mm = (
        diagonal_size_mm
        / (1 + (screen_resolution_px[0] / screen_resolution_px[1]) ** 2) ** 0.5
        * (screen_resolution_px[0] / screen_resolution_px[1])
    )  # Calculate width from diagonal and aspect ratio
    rad = np.deg2rad(degree)  # Convert degree to radians
    px_per_degree_w = (
        np.tan(rad) * screen_distance_mm * screen_resolution_px[0] / screen_width_mm
    )
    return px_per_degree_w


def load_and_preprocess_data(file_pattern):
    """Load data from CSV files and preprocess."""
    # Load data
    data_files = glob.glob(file_pattern)
    data = pd.concat([pd.read_csv(f) for f in data_files])

    # Filter data
    data = data[data["screen"] == "STIM"]
    data = data.sort_values(["sub", "run", "trial_n", "time"])

    # Calculate sample time from the minimum time in each trial
    data["samp"] = data.groupby(["sub", "run", "trial_n"])["time"].transform(
        lambda x: (x - x.min()).to_numpy()
    )

    # Replace missing gaze values with NaN
    data.loc[data["gaze_x"] == 1e8, "gaze_x"] = np.nan
    data.loc[data["gaze_y"] == 1e8, "gaze_y"] = np.nan

    # Normalize gaze data by subtracting the mean
    data["gaze_x"] = data.groupby(["sub", "run"])["gaze_x"].transform(
        lambda x: x - x.mean()
    )
    data["gaze_y"] = data.groupby(["sub", "run"])["gaze_y"].transform(
        lambda x: x - x.mean()
    )

    return data


def remove_outliers(data):
    """Remove outliers based on gaze coordinates."""

    data["gaze_outlier_x"] = data.groupby(["sub", "run", "trial_n"])["gaze_x"].transform(
        lambda x: any([np.isnan(x).sum() > 0.5 * len(x)])
    )

    data["gaze_outlier_y"] = data.groupby(["sub", "run", "trial_n"])["gaze_y"].transform(
            lambda y: any([np.isnan(y).sum() > 0.5 * len(y)])
        )

    data = data[(data["gaze_outlier_x"] == False) & (data["gaze_outlier_y"] == False)]
    return data


def interpolate_missing_values(data):
    """Interpolate missing gaze values."""

    data["gaze_x"] = (
        data.groupby(["sub", "run", "trial_n"])["gaze_x"]
        .apply(lambda x: x.interpolate("cubic").ffill().bfill())
        .reset_index(level=["sub", "run", "trial_n"], drop=True)
    )

    data["gaze_y"] = (
        data.groupby(["sub", "run", "trial_n"])["gaze_y"]
        .apply(lambda x: x.interpolate("cubic").ffill().bfill())
        .reset_index(level=["sub", "run", "trial_n"], drop=True)
    )

    return data


def calculate_average_heatmap(data, screen_width, screen_height):
    """
    Calculate the average heatmap of gaze data across subjects and runs.

    This function creates a heatmap for each combination of subject and run,
    accumulates all heatmaps, and computes their average. The resulting heatmap
    matches the screen resolution and represents the average gaze density across
    all subjects and runs.

    Parameters:
        data (pd.DataFrame): A DataFrame containing gaze data with columns
            ['gaze_x', 'gaze_y', 'sub', 'run'], where:
            - 'gaze_x' and 'gaze_y' represent gaze coordinates in pixels.
            - 'sub' indicates the subject identifier.
            - 'run' indicates the run identifier.
        screen_width (int): The width of the screen in pixels.
        screen_height (int): The height of the screen in pixels.

    Returns:
        np.ndarray: A 2D array of shape (screen_height, screen_width) representing
        the average heatmap of gaze data.
    """
    # Initialize an accumulator canvas with the same resolution as the screen
    total_canvas = np.zeros((screen_height, screen_width))
    count = 0  # Track the number of heatmaps created

    # Define screen limits for validation
    xlim = (-screen_width // 2, screen_width // 2)
    ylim = (-screen_height // 2, screen_height // 2)

    # Group data by subject and run
    grouped = data.groupby(["sub", "run"])

    for (sub, run), group in grouped:
        # Extract gaze data for the current subject and run
        xdata = group["gaze_x"].values
        ydata = group["gaze_y"].values

        # Map gaze data to screen pixel indices
        x_indices = (xdata - xlim[0]).astype(int)
        y_indices = (ydata - ylim[0]).astype(int)

        # Initialize a canvas for the current subject and run
        canvas = np.zeros((screen_height, screen_width))

        # Populate the canvas with gaze data
        for x, y in zip(x_indices, y_indices):
            if 0 <= x < screen_width and 0 <= y < screen_height:  # Ensure valid indices
                canvas[y, x] += 1

        # Accumulate the canvas into the total canvas
        total_canvas += canvas
        count += 1

    # Compute the average heatmap
    average_canvas = total_canvas / count
    return average_canvas


def apply_smoothing(canvas, kernel_size, kernel_half_width):
    """Smooth the heatmap using a Gaussian kernel."""

    def fwhm_kernel_2d(size, half_width):
        x, y = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1))
        sigma = half_width / (2 * np.sqrt(2 * np.log(2)))
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    kernel = fwhm_kernel_2d(kernel_size, kernel_half_width)
    smoothed_canvas = convolve2d(canvas, kernel, mode="same")
    return smoothed_canvas


def draw_stimuli(
    diagonal_size_inch,
    screen_distance_mm,
    screen_resolution_px,
    image_path,
    heatmap_canvas,
):
    """
    Draws visual stimuli based on a specific screen setup for an fMRI study, including an image at specified locations and overlaying a heatmap.

    Parameters:
        diagonal_size_inch (float): The diagonal size of the screen in inches.
        screen_distance_mm (int): The viewing distance from the viewer's eye to the screen in millimeters via a mirror.
        screen_resolution_px (int): The screen resolution in pixels (width, height).
        image_path (str): Path to the image file to be used in the stimulus.
        heatmap_canvas (ndarray): 2D heatmap data to overlay on the background.
    """
    # Visual degrees specifications
    plus_size_deg = 0.5
    image_size_deg = 0.9  # Image size in visual degrees
    circle_start_deg = 2
    circle_increment_deg = 5
    radius_deg = 7.0
    circles_degs = list(
        np.arange(
            circle_start_deg, radius_deg + circle_increment_deg, circle_increment_deg
        )
    )

    # Convert visual degrees to pixel dimensions
    plus_size_px = visual_deg_to_px(
        plus_size_deg, diagonal_size_inch, screen_distance_mm, screen_resolution_px
    )
    image_size_px = visual_deg_to_px(
        image_size_deg, diagonal_size_inch, screen_distance_mm, screen_resolution_px
    )
    radius_px = visual_deg_to_px(
        radius_deg, diagonal_size_inch, screen_distance_mm, screen_resolution_px
    )

    # Create figure and gridspec layout
    fig = plt.figure(figsize=(10, 8))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.4)  # Main plot and colorbar space

    ax = fig.add_subplot(gs[0])  # Main plot
    ax.set_aspect("equal")

    # Limit visible range for X and Y axes
    ax.set_xlim([screen_resolution_px[0] // 2 - 500, screen_resolution_px[0] // 2 + 500])
    ax.set_ylim([screen_resolution_px[1] // 2 - 500, screen_resolution_px[1] // 2 + 500])

    # Draw "+" at the center
    center = (screen_resolution_px[0] / 2, screen_resolution_px[1] / 2)
    ax.plot(
        [center[0] - plus_size_px / 2, center[0] + plus_size_px / 2],
        [center[1], center[1]],
        "k-",
    )
    ax.plot(
        [center[0], center[0]],
        [center[1] - plus_size_px / 2, center[1] + plus_size_px / 2],
        "k-",
    )

    # Load and place the image at specific positions
    image = preprocess_image(image_path, int(image_size_px))
    for offset in [(-1, -1), (1, 1)]:
        imagebox = OffsetImage(image, zoom=1)  # Pre-scaled image
        new_x = screen_resolution_px[0] / 2 + offset[0] * radius_px * (2**0.5) / 2
        new_y = screen_resolution_px[1] / 2 + offset[1] * radius_px * (2**0.5) / 2
        ab = AnnotationBbox(imagebox, (new_x, new_y), frameon=False)
        ax.add_artist(ab)

    # Draw circles with divergent color scale
    cmap = plt.get_cmap("GnBu")
    for i, circle_deg in enumerate(circles_degs):
        radius = visual_deg_to_px(
            circle_deg, diagonal_size_inch, screen_distance_mm, screen_resolution_px
        )
        if circle_deg > 4.0 and math.modf(circle_deg)[0] != 0:
            continue
        color_index = i / (len(circles_degs) - 1)
        edgecolor = (
            adjust_lightness(cmap(color_index), amount=0.5)
            if circle_deg <= 4.0
            else (0, 0, 0, 0.2)
        )
        circle = Circle(
            (screen_resolution_px[0] / 2, screen_resolution_px[1] / 2),
            radius,
            edgecolor=edgecolor,
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(circle).set_zorder(10)

    # Overlay the heatmap with visual degree labels
    sns.heatmap(
        heatmap_canvas,
        cmap=plt.get_cmap("Spectral").reversed(),
        alpha=0.6,
        cbar=True,
        cbar_ax=fig.add_subplot(gs[1]),  # Add colorbar to second grid
        cbar_kws={
            "label": "Average Fixation Density (samples per pixel)"
        },  # Label for colorbar
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        mask=np.isnan(heatmap_canvas),  # Mask NaN values
    ).set_facecolor(
        (0.49, 0.49, 0.49)
    )  # Set NaN values to gray

    # Simplify colorbar ticks
    cbar = ax.collections[0].colorbar  # Access the colorbar
    cbar.set_ticks(
        [
            np.nanmin(heatmap_canvas),
            (np.nanmax(heatmap_canvas) - np.nanmin(heatmap_canvas)) // 2,
            np.nanmax(heatmap_canvas),
        ]
    )  # Lowest, midpoint, highest
    cbar.set_ticklabels(
        [
            f"{np.nanmin(heatmap_canvas):.2f}",
            f"{np.nanmean(heatmap_canvas):.2f}",
            f"{np.nanmax(heatmap_canvas):.2f}",
        ]
    )  # Custom labels

    # Generate tick positions and labels in degrees
    degree_ticks = np.arange(-10, 11, 2)  # Degrees from -10 to 10, stepping by 2
    tick_positions_x = [screen_resolution_px[0] // 2 + visual_deg_to_px(deg) for deg in degree_ticks]
    tick_positions_y = [screen_resolution_px[1] // 2 - visual_deg_to_px(deg) for deg in degree_ticks]

    # Set the ticks and labels
    ax.set_xticks(tick_positions_x)
    ax.set_xticklabels(degree_ticks)
    ax.set_yticks(tick_positions_y)
    ax.set_yticklabels(degree_ticks)

    # Adjust axis limits based on visual degrees
    max_deg = 10  # Maximum visual angle to display
    ax.set_xlim([
        screen_resolution_px[0] // 2 - visual_deg_to_px(max_deg),
        screen_resolution_px[0] // 2 + visual_deg_to_px(max_deg)
    ])
    ax.set_ylim([
        screen_resolution_px[1] // 2 - visual_deg_to_px(max_deg),
        screen_resolution_px[1] // 2 + visual_deg_to_px(max_deg)
    ])

    # Add labels for the axes in degrees
    ax.set_xlabel("Horizontal Position (degrees of visual angle)")
    ax.set_ylabel("Vertical Position (degrees of visual angle)")

    # Finalize and save the figure
    plt.tight_layout()
    output_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2023/fMRI_stimuli_ET_heatmap.png"
    plt.savefig(output_path, dpi=300)
    plt.show()


# Parameters
screen_width = 1920
screen_height = 1080

# Main execution
file_pattern = "/home/eik-tb/Desktop/misc/new_cosmomvpa_phd/eye_tracking/data/*.csv"
data = load_and_preprocess_data(file_pattern)
data = remove_outliers(data)
data = data.reset_index()

data = interpolate_missing_values(data)

# Calculate the average heatmap
average_canvas = calculate_average_heatmap(data, screen_width, screen_height)

# Replace NaN with 0
thresholded_canvas = np.where(average_canvas < 1e-2, np.nan, average_canvas)

# Visualize the stimuli with the heatmap
draw_stimuli(
    31.55,
    1120,
    (1920, 1080),
    "/home/eik-tb/Desktop/misc/new_cosmomvpa_phd/stim/raw/c20.png",
    thresholded_canvas,
)
