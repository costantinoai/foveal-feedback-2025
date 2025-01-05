# Foveal Feedback 2025

This repository contains the code and data for reproducing the results and figures presented in the manuscript **"Foveal Feedback 2025"**. It includes scripts for MVPA classification, fMRI analysis, PPI analysis, and their associated plots.

---

## Project Overview

### Directory Structure

'''
.
├── 1_models_mvpa.py          # MVPA classification
├── 2_fmri_mvpa.py            # Processes and saves MVPA analysis results on fMRI data
├── 3_fmri_plots.py           # Generates plots from fMRI MVPA results
├── 4_perform_PPI.py          # Performs PPI analysis
├── 5_PPI_plot.R              # R script for PPI plots
├── data/                     # Data directory
│   ├── stimuli/              # Experimental stimuli images
│   ├── ppi_results.csv       # PPI analysis results
│   ├── behavioural_acc.csv   # Behavioral accuracy data
│   ├── 2024-04-16_122451_bike-car-female-male.pkl # Human MVPA results
├── external/                 # External dependencies (e.g., TDANN)
├── modules/                  # Helper modules
├── results/                  # Generated figures and results
├── extra/                    # Additional scripts and utilities
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── setup.sh                  # Automated setup script
'''

---

## Automated Setup

The `setup.sh` script automates the installation process (tested on Linux), including:
- Cloning the repository.
- Setting up a Python environment.
- Installing dependencies (including TDANN).
- Setting the required environment variables.

### Run the Setup Script

To use the setup script, execute the following commands in a terminal:

'''
chmod +x setup.sh
./setup.sh
'''

This script handles the entire installation process. After running the script, you only need to download and set up the dataset (instructions below) before running the analyses.

If you prefer to perform the steps manually or are using a different operating system, follow the instructions below.

---

## Manual Installation and Setup

### Step 1: Clone the Repository

'''
git clone https://github.com/costantinoai/foveal-feedback-2025.git
cd foveal-feedback-2025
'''

### Step 2: Set Up the Environment

1. **Create and activate a Python environment**:
   '''
   conda create -n foveal-feedback python=3.8 -y
   conda activate foveal-feedback
   '''

2. **Install Python dependencies**:
   '''
   pip install -r requirements.txt
   '''

### Step 3: Install External Dependencies (TDANN)

1. **Clone and install TDANN**:
   '''
   mkdir -p external && cd external
   git clone https://github.com/neuroailab/TDANN.git TDANN-main
   cd TDANN-main

   pip install -r requirements.txt
   pip install -e .
   cd ../../
   '''

2. Ensure the TDANN framework is installed in the `external/TDANN-main/` directory, as required by the scripts.

IMPORTANT: TDANN has its own required dependenciec. Make sure you follow the correct installation process (see the [neuroailab/TDANN](https://github.com/neuroailab/TDANN) repository)

### Step 4: Download and Extract the Dataset

1. **Download the dataset** from [OSF](https://osf.io/h95a2/).
2. **Extract the dataset**:
   - For Linux/macOS:
     '''
     cat BIDS.zip.* > BIDS_combined.zip
     unzip BIDS_combined.zip -d ./data
     '''
   - For Windows:
     - Select all dataset parts (e.g., `BIDS.zip.001`, `BIDS.zip.002`, etc.).
     - Right-click and choose "Extract All".
     - Extract the contents into the `data/` directory.

3. Verify that the `BIDS/` folder is in the `data/` directory.

---

## Verification

After installation, verify the setup:

'''
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import spacetorch; print('Spacetorch installation successful')"
'''

Activate your environment with:

'''
conda activate foveal-feedback
'''

---

## Running the Scripts

### General Workflow

1. Run the scripts in order from **1** to **5** for a complete analysis.
2. Outputs are saved in dedicated subfolders within the `results/` directory, which include:
   - Generated figures and data files.
   - A copy of the executed script for reproducibility.
3. Update directory paths in the scripts if performing new analyses or using custom datasets.
4. If you only want to plot existing data, use as input the files in `results/` or the backups in `data/`.

---

### Script Overview

#### **Script 1: MVPA Models**

Performs MVPA classification on DNN data, and generates:
- MDS plots for clean activations (Figures 5A, 5B).
- Accuracy and confusion matrices across participants (Figures 5C-F).
- Aggregated results across noise levels (Supplementary Figures 2A, 2B, 4).

#### **Script 2: fMRI MVPA Analysis**

Processes and saves MVPA analysis results on fMRI data.

#### **Script 3: fMRI Plots**

Generates plots from the results saved by `2_fmri_mvpa.py` producing:
- Sub-category and category-level accuracies (Figures 6A, 6B).
- ROI-specific confusion matrices and correlations (Figures 6C-E).
- Behavioral accuracy and foveal ROI analysis (Figures 3A, 4).

#### **Script 4: PPI Analysis**

Executes PPI analysis on the dataset.

#### **Script 5: PPI Plots**

Generates PPI plot using R/RStudio (Figures 7).

---

## Citation

If you use this repository in your work, please cite:

> [Full citation details here]
