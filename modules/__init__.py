import matplotlib.pyplot as plt
import torch
import sys
import os

# ========================================================================
# CONFIGURATION & GLOBAL VARIABLES
# ========================================================================

TDANN_PATH = "./external/TDANN-main"
sys.path.append(os.path.join(TDANN_PATH, "demo"))

# Device configuration: Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global settings for publication-quality plots
BASE_FONT_SIZE = 30  # Base font size for scaling (change this to scale all fonts)

plt.rcParams["figure.figsize"] = (14, 10)  # Slightly larger figure size
plt.rcParams["figure.dpi"] = 300  # High DPI for publication-ready images
plt.rcParams["font.size"] = BASE_FONT_SIZE  # Base font size for text
plt.rcParams["axes.titlesize"] = BASE_FONT_SIZE * 1.2  # Title font size (larger for emphasis)
plt.rcParams["axes.labelsize"] = BASE_FONT_SIZE  # Axis label font size
plt.rcParams["xtick.labelsize"] = BASE_FONT_SIZE * 0.9  # X-axis tick font size
plt.rcParams["ytick.labelsize"] = BASE_FONT_SIZE * 0.9  # Y-axis tick font size
plt.rcParams["legend.fontsize"] = BASE_FONT_SIZE * 0.7  # Legend font size (slightly smaller)
plt.rcParams["legend.title_fontsize"] = BASE_FONT_SIZE * 0.7  # Legend title font size
plt.rcParams["legend.frameon"] = False  # Make legend borderless
plt.rcParams["legend.loc"] = "upper right"  # Legend in the top-right corner
plt.rcParams["savefig.bbox"] = "tight"  # Ensure plots save tightly cropped
plt.rcParams["savefig.pad_inches"] = 0.1  # Add small padding around saved plots
plt.rcParams["savefig.format"] = "png"  # Default save format

# Mapping integers to categories
LETTER2INT_MAPPING = {
    "f": 0,  # Female
    "m": 1,  # Male
    "b": 2,  # Bike
    "c": 3,  # Car
}

INT2LETTER_MAPPING = {
    0: "f",  # Female
    1: "m",  # Male
    2: "b",  # Bike
    3: "c",  # Car
}

# Mapping categories to readable labels for interpretation and plotting
LABEL_MAPPING = {
    "f": "Female",
    "m": "Male",
    "b": "Bike",
    "c": "Car",
}

# Color mapping for visualization purposes
COLOR_MAPPING = {
    "Female": "#D34936",
    "Male": "#FCA95F",
    "Bike": "#3B8BCE",
    "Car": "#82CDA4",
    "Face": "#F76C5E",
    "Vehicle": "#76CDD8",
}

# Define specific foveal ROIs
FOV_ROIS_ORDER = (
    "Foveal 0.5°",
    "Foveal 1°",
    "Foveal 1.5°",
    "Foveal 2°",
    "Foveal 2.5°",
    "Foveal 3°",
    "Foveal 3.5°",
    "Foveal 4°",
)

FOV_ROIS_COLORMAP = [
(0.5359477124183007, 0.8039215686274516, 0.16078431372548962),
(0.3167368908347655, 0.6725821308915723, 0.22568776530219936),
(0.24178904158039635, 0.5996488356164026, 0.23033963035822258),
(0.19913514325858017, 0.5125988052227732, 0.38395549025279124),
(0.1427049975242376, 0.4054713535893345, 0.4486444834446205),
(0.10422914263744715, 0.29670126874279124, 0.3844982698961938),
(0.024567474048442894, 0.21305651672433681, 0.34182237600922727),
(0.01568627450980392, 0.12549019607843137, 0.2529411764705882),
]

# Define category order for sub-categories and broader categories
SUBCAT_ORDER = ("Bike", "Car", "Female", "Male")
CAT_ORDER = ("Vehicle", "Face")
ROIS_ORDER = ("Foveal 0.5°", "Peripheral", "Opposite", "FFA", "LOC")
