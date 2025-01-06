import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass
from scipy import ndimage

# Load the NIfTI file and calculate the center of mass for each ROI cluster
def calculate_com_mni(nifti_path):
    """
    Calculate the center of mass (CoM) and size for each cluster in an ROI NIfTI mask and
    convert the CoM coordinates to MNI space.

    Parameters:
    - nifti_path (str): Path to the NIfTI ROI mask file.

    Returns:
    - List of dictionaries, each containing the CoM coordinates in MNI space and size for a cluster.
    """
    # Load the NIfTI image
    img = nib.load(nifti_path)

    # Get the affine matrix and image data
    affine = img.affine
    data = img.get_fdata()

    # Ensure binary data (handle masks with values slightly deviating from 1 or 0)
    binary_data = (data > 0.5).astype(np.uint8)

    # Find unique clusters in the binary mask (labeling each cluster if needed)
    labeled_data, num_features = ndimage.label(binary_data)

    # List to store center of mass coordinates and sizes
    cluster_info = []

    for cluster_index in range(1, num_features + 1):
        # Isolate the current cluster
        cluster_mask = (labeled_data == cluster_index)

        # Compute the center of mass in voxel coordinates
        com_voxel = center_of_mass(cluster_mask)

        # Convert voxel coordinates to MNI coordinates using the affine matrix
        com_mni = nib.affines.apply_affine(affine, com_voxel)

        # Calculate the size of the cluster (number of voxels)
        cluster_size = np.sum(cluster_mask)

        # Append the result
        cluster_info.append({
            "center_of_mass_mni": tuple(com_mni),
            "size": cluster_size
        })

    return cluster_info

if __name__ == "__main__":
    # Example usage
    nifti_path = "/media/costantino_ai/Samsung_T5/2exp_fMRI/Exp/Data/Data/fmri/BIDS/derivatives/parcels/nii/Vehicle_probability_map_thresh2subjs_smoothed_parcels_sig.nii"  # Replace with your NIfTI file path

    try:
        cluster_info = calculate_com_mni(nifti_path)
        print("Cluster Information (MNI Coordinates and Sizes):")
        for i, cluster in enumerate(cluster_info, start=1):
            print(f"Cluster {i}: Center of Mass: {cluster['center_of_mass_mni']}, Size: {cluster['size']} voxels")
    except Exception as e:
        print(f"An error occurred: {e}")
