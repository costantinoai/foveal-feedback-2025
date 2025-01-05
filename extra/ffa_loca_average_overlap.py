import os
import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass
import nilearn.image as image
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from matplotlib.colors import ListedColormap
import pandas as pd


# Helper function to create binary masks from left and right hemispheres
def create_combined_mask(path_left, path_right, target_affine, target_shape):
    """
    Create a combined binary mask from left and right hemisphere images.

    Parameters:
    - path_left: Path to the left hemisphere image.
    - path_right: Path to the right hemisphere image.
    - target_affine: Affine matrix for resampling.
    - target_shape: Target shape for resampling.

    Returns:
    - A binary mask combining left and right hemispheres.
    """
    img_left = nib.load(path_left)
    img_right = nib.load(path_right)

    img_left_resampled = image.resample_img(
        img_left,
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation="nearest",
    )
    img_right_resampled = image.resample_img(
        img_right,
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation="nearest",
    )

    mask_left = img_left_resampled.get_fdata() > 0
    mask_right = img_right_resampled.get_fdata() > 0

    combined_mask = mask_left | mask_right
    return combined_mask.astype(np.uint8)


# Function to analyze ROIs for all subjects
def analyze_subject_rois(subject_dir, y_image_paths):
    """
    Analyze subject-specific ROIs against target ROIs and generate overlap statistics.

    Parameters:
    - subject_dir: Directory containing subject-specific ROI images.
    - y_image_paths: Dictionary mapping target ROI names to their left and right hemisphere paths.

    Returns:
    - DataFrame containing subject-level and group-level results.
    """

    results = []  # Initialize a list to store results for all subjects and ROIs

    # Iterate over all subjects in the directory
    for subject in sorted(os.listdir(subject_dir)):
        subject_path = os.path.join(subject_dir, subject)

        # Ensure the current entry is a directory
        if os.path.isdir(subject_path):
            # Iterate over all ROI files within the subject's directory
            for roi_file in sorted(os.listdir(subject_path)):
                # Process only files related to "FFA" or "LOC" ROIs
                if "FFA" in roi_file or "LOC" in roi_file:
                    roi_path = os.path.join(
                        subject_path, roi_file
                    )  # Full path to the ROI file

                    # Load the subject's ROI image (X) and create a binary mask
                    img_x = nib.load(roi_path)
                    data_x = (
                        img_x.get_fdata() > 0
                    )  # Binary mask: 1 for ROI, 0 for outside

                    # Determine whether the current ROI is FFA or LOC and get corresponding target masks
                    if "FFA" in roi_file:
                        y_paths = y_image_paths["FFA"]
                    elif "LOC" in roi_file:
                        y_paths = y_image_paths["LOC"]

                    # Create a combined binary mask for the target ROI (Y)
                    combined_mask = create_combined_mask(
                        y_paths["left"], y_paths["right"], img_x.affine, img_x.shape
                    )

                    # Calculate overlap and non-overlap regions
                    overlap = data_x & combined_mask  # Voxels in both X and Y
                    x_not_y = data_x & ~combined_mask  # Voxels in X but not in Y

                    # Compute statistics
                    overlap_voxels = np.sum(overlap)  # Number of overlapping voxels
                    non_overlap_voxels = np.sum(
                        x_not_y
                    )  # Number of non-overlapping voxels

                    # Calculate center of mass for X and Y in voxel space
                    com_voxel_x = center_of_mass(data_x)
                    com_voxel_y = center_of_mass(combined_mask)

                    # Convert voxel coordinates to MNI space
                    com_mni_x = nib.affines.apply_affine(img_x.affine, com_voxel_x)
                    com_mni_y = nib.affines.apply_affine(img_x.affine, com_voxel_y)

                    # Calculate the size (number of voxels) for X and Y
                    size_x = np.sum(data_x)
                    size_y = np.sum(combined_mask)

                    # Append the results for this comparison
                    results.append(
                        {
                            "Subject": subject,  # Subject identifier
                            "ROI": roi_file.split("_")[
                                2
                            ],  # Extract ROI type from the filename
                            "Target ROI": (
                                "FFA" if "FFA" in roi_file else "LOC"
                            ),  # Target ROI type
                            "Center of Mass X (MNI)": com_mni_x,  # Center of mass for X in MNI space
                            "Center of Mass Y (MNI)": com_mni_y,  # Center of mass for Y in MNI space
                            "Size X (voxels)": size_x,  # Size of X in voxels
                            "Size Y (voxels)": size_y,  # Size of Y in voxels
                            "Overlap (voxels)": overlap_voxels,  # Number of overlapping voxels
                            "X not Y (voxels)": non_overlap_voxels,  # Number of non-overlapping voxels
                        }
                    )

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Group-level reports
    group_report = df.groupby("Target ROI")[
        ["Overlap (voxels)", "X not Y (voxels)", "Size X (voxels)", "Size Y (voxels)"]
    ].mean()
    print("\nGroup-level Report:")
    print(group_report)

    print("\nDetailed Per-Subject Results:")
    print(df.head())

    return df


# Plot function to visualize overlaps
def plot_overlap(img_x_path, img_y_combined_mask, subject, roi_name):
    """
    Generate a plot showing the overlap and non-overlap regions.

    Parameters:
    - img_x_path: Path to the subject-specific ROI image.
    - img_y_combined_mask: Combined binary mask for the target ROI.
    - subject: Subject identifier.
    - roi_name: Name of the subject-specific ROI.
    """
    img_x = nib.load(img_x_path)
    data_x = img_x.get_fdata() > 0

    combined_map = np.where(
        data_x & img_y_combined_mask, 1, np.where(data_x & ~img_y_combined_mask, 2, 0)
    )

    combined_img = nib.Nifti1Image(combined_map.astype(np.float32), img_x.affine)

    plot_stat_map(
        combined_img,
        title=f"{subject} {roi_name} Overlap",
        cmap=ListedColormap(["blue", "red"]),
        colorbar=True,
        vmin=1,
        draw_cross=False,
        alpha=0.5,
    )
    plt.show()


# Main script
subject_dir = "/data/projects/fov/data/BIDS/derivatives/rois"
y_image_paths = {
    "FFA": {
        "left": "/home/eik-tb/Downloads/face_parcels (1)/face_parcels/face_parcels/lFFA.hdr",
        "right": "/home/eik-tb/Downloads/face_parcels (1)/face_parcels/face_parcels/rFFA.hdr",
    },
    "LOC": {
        "left": "/home/eik-tb/Downloads/face_parcels (1)/object_parcels/lLOC.hdr",
        "right": "/home/eik-tb/Downloads/face_parcels (1)/object_parcels/rLOC.hdr",
    },
}

# Analyze ROIs
df = analyze_subject_rois(subject_dir, y_image_paths)

# # Individual subject visualizations
# for _, row in df.iterrows():
#     subject = row["Subject"]
#     roi_name = row["ROI"]
#     roi_path = os.path.join(subject_dir, subject, f"{subject}_space-MNI152NLin2009cAsym_{roi_name}_roi.nii")
#     combined_mask = create_combined_mask(
#         y_image_paths[row["Target ROI"]]["left"],
#         y_image_paths[row["Target ROI"]]["right"],
#         nib.load(roi_path).affine,
#         nib.load(roi_path).shape
#     )
#     plot_overlap(roi_path, combined_mask, subject, roi_name)


def compute_average_overlap(subject_dir, y_image_paths, target_roi):
    """
    Compute the average ROI overlap across subjects for a given target ROI.

    Parameters:
    - subject_dir: Directory containing subject-specific ROI images.
    - y_image_paths: Dictionary of left and right hemisphere paths for the target ROIs.
    - target_roi: The target ROI to analyze (e.g., "FFA" or "LOC").

    Returns:
    - Tuple containing:
      - average_overlap_map: Averaged overlap map for all subjects.
      - combined_mask: Combined binary mask for the target ROI.
    """
    all_subject_rois = []
    affine = None  # Will store the affine of the first subject's ROI
    shape = None  # Will store the shape of the first subject's ROI

    # Iterate through all subjects
    for subject in sorted(os.listdir(subject_dir)):
        subject_path = os.path.join(subject_dir, subject)

        if os.path.isdir(subject_path):
            for roi_file in sorted(os.listdir(subject_path)):
                # Process only files related to the target ROI
                if target_roi in roi_file:
                    roi_path = os.path.join(subject_path, roi_file)

                    # Load the subject ROI and create a binary mask
                    img_x = nib.load(roi_path)
                    data_x = img_x.get_fdata() > 0  # Binary mask
                    all_subject_rois.append(data_x)

                    # Save affine and shape for reference
                    if affine is None:
                        affine = img_x.affine
                        shape = img_x.shape

    # Average the ROI maps across subjects
    average_overlap_map = np.mean(
        np.array([roi for i, roi in enumerate(all_subject_rois) if i not in {14, 15}]),
        axis=0,
    )

    # Create the combined mask for the target ROI
    combined_mask = create_combined_mask(
        y_image_paths[target_roi]["left"],
        y_image_paths[target_roi]["right"],
        affine,
        shape,
    )

    return average_overlap_map, combined_mask, affine

from scipy.ndimage import label, center_of_mass
def compute_dice_coefficient(mask_x, mask_y):
    """
    Compute the Dice Coefficient between two binary masks.

    Parameters:
    - mask_x: Binary mask of X.
    - mask_y: Binary mask of Y.

    Returns:
    - Dice coefficient.
    """
    intersection = np.sum(mask_x & mask_y)
    total_voxels = np.sum(mask_x) + np.sum(mask_y)
    return 2 * intersection / total_voxels if total_voxels > 0 else 0
def compute_containment(mask_x, mask_y):
    """
    Compute the percentage of X voxels that are within Y.

    Parameters:
    - mask_x: Binary mask of X.
    - mask_y: Binary mask of Y.

    Returns:
    - Containment percentage.
    """
    inside_y = np.sum(mask_x & mask_y)
    total_x = np.sum(mask_x)
    return inside_y / total_x * 100 if total_x > 0 else 0
def compute_jaccard_index(mask_x, mask_y):
    """
    Compute the Jaccard Index between two binary masks.

    Parameters:
    - mask_x: Binary mask of X.
    - mask_y: Binary mask of Y.

    Returns:
    - Jaccard index.
    """
    intersection = np.sum(mask_x & mask_y)
    union = np.sum(mask_x | mask_y)
    return intersection / union if union > 0 else 0

def compute_average_map_stats(average_overlap_map, combined_mask, affine, roi_name):
    """
    Compute statistics comparing the average overlap map to the combined ROI mask.

    Parameters:
    - average_overlap_map: Averaged ROI map across all subjects.
    - combined_mask: Combined binary mask for the target ROI.
    - affine: Affine transformation for the ROI maps.
    - roi_name: The name of the ROI (e.g., "FFA" or "LOC").

    Returns:
    - A list of dictionaries, one per cluster, containing cluster-specific statistics.
    """
    # Threshold the average map to create a binary representation
    thresholded_map = average_overlap_map > 0

    # Label connected clusters in the thresholded map (X clusters)
    labeled_map_x, n_clusters_x = label(thresholded_map)

    # Label connected clusters in the combined mask (Y clusters)
    labeled_map_y, n_clusters_y = label(combined_mask)

    # Get MNI-space centers of mass for Y clusters
    y_clusters = []
    for y_cluster_id in range(1, n_clusters_y + 1):
        cluster_mask_y = labeled_map_y == y_cluster_id
        com_voxel_y = center_of_mass(cluster_mask_y)
        com_mni_y = nib.affines.apply_affine(affine, com_voxel_y)
        hemisphere_y = "LEFT" if com_mni_y[0] < 0 else "RIGHT"
        y_clusters.append({
            "Cluster ID": y_cluster_id,
            "Hemisphere": hemisphere_y,
            "Center of Mass (MNI)": com_mni_y,
            "Mask": cluster_mask_y,
        })

    # Process X clusters
    cluster_stats = []
    for cluster_id_x in range(1, n_clusters_x + 1):
        # Extract the current X cluster
        cluster_mask_x = labeled_map_x == cluster_id_x
        cluster_size_x = np.sum(cluster_mask_x)

        # Compute the center of mass for the X cluster
        com_voxel_x = center_of_mass(cluster_mask_x)
        com_mni_x = nib.affines.apply_affine(affine, com_voxel_x)
        hemisphere_x = "LEFT" if com_mni_x[0] < 0 else "RIGHT"

        # Match with the corresponding Y cluster in the same hemisphere
        matching_y_cluster = next(
            (y for y in y_clusters if y["Hemisphere"] == hemisphere_x), None
        )

        if matching_y_cluster is not None:
            overlap = cluster_mask_x & matching_y_cluster["Mask"]
            x_not_y = cluster_mask_x & ~matching_y_cluster["Mask"]

            overlap_voxels = np.sum(overlap)
            non_overlap_voxels = np.sum(x_not_y)
            cluster_size_y = np.sum(matching_y_cluster["Mask"])
            com_mni_y = matching_y_cluster["Center of Mass (MNI)"]

            overlap = cluster_mask_x & matching_y_cluster["Mask"]
            containment = compute_containment(cluster_mask_x, matching_y_cluster["Mask"])
            dice = compute_dice_coefficient(cluster_mask_x, matching_y_cluster["Mask"])
            jaccard = compute_jaccard_index(cluster_mask_x, matching_y_cluster["Mask"])
        else:
            containment = 0
            dice = 0
            jaccard = 0

            overlap_voxels = 0
            non_overlap_voxels = cluster_size_x
            cluster_size_y = 0
            com_mni_y = None

        # Assign cluster name based on ROI and hemisphere
        cluster_name = f"{roi_name} {hemisphere_x}"

        # Collect cluster statistics
        cluster_stats.append({
            "Cluster ID": cluster_id_x,
            "Cluster Name": cluster_name,
            "Size X (voxels)": cluster_size_x,
            "Size Y (voxels)": cluster_size_y,
            "Overlap (voxels)": overlap_voxels,
            "X not Y (voxels)": non_overlap_voxels,
            "Center of Mass Average ROI (MNI)": com_mni_x,
            "Center of Mass Julian et al. (2012) resampled (MNI)": com_mni_y,
            "Containment (%)": containment,
            "Dice Coefficient": dice,
            "Jaccard Index": jaccard,
        })

    return cluster_stats
def plot_average_overlap_map(average_overlap_map, combined_mask, affine, target_roi):
    """
    Plot the average ROI overlap across subjects and the combined target ROI mask.

    Parameters:
    - average_overlap_map: Averaged ROI map across all subjects.
    - combined_mask: Combined binary mask for the target ROI.
    - affine: Affine transformation for the ROI maps.
    - target_roi: The target ROI (e.g., "FFA" or "LOC").
    """
    # Combine the average overlap map and the target ROI mask for visualization
    overlap = np.where(average_overlap_map > 0.5, 1, 0)
    combined_map = combined_mask * 2 + overlap
    combined_img = nib.Nifti1Image(combined_map.astype(np.float32), affine)

    # Plot the combined map
    plot_stat_map(
        combined_img,
        title=f"Average Overlap for {target_roi} (Subjects vs Target ROI)",
        cmap=ListedColormap(["blue", "red", "purple"]),  # Custom colors
        colorbar=True,
        draw_cross=False,
        vmin=1,
        alpha=0.5,
        vmax=3
    )
    plt.show()

from scipy.stats import ttest_rel
# Compute stats and clusters for FFA
average_overlap_map, combined_mask, affine = compute_average_overlap(
    subject_dir, y_image_paths, "FFA"
)
plot_average_overlap_map(average_overlap_map, combined_mask, affine, "FFA")

stats_ffa = compute_average_map_stats(average_overlap_map, combined_mask, affine, "FFA")
print("\nCluster Statistics for FFA:")
for cluster_stat in stats_ffa:
    for key, value in cluster_stat.items():
        print(f"{key}: {value}")
    print("-" * 40)

# Compute stats and clusters for LOC
average_overlap_map, combined_mask, affine = compute_average_overlap(
    subject_dir, y_image_paths, "LOC"
)
plot_average_overlap_map(average_overlap_map, combined_mask, affine, "LOC")

stats_loc = compute_average_map_stats(average_overlap_map, combined_mask, affine, "LOC")
print("\nCluster Statistics for LOC:")
for cluster_stat in stats_loc:
    for key, value in cluster_stat.items():
        print(f"{key}: {value}")
    print("-" * 40)
