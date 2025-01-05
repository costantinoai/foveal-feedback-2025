from typing import Optional
import os
from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from natsort import natsorted

def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    xforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # add empty batch dim before returning
    return torch.unsqueeze(xforms(img), dim=0)


def create_dataloader(
    input_tensor: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    num_workers: int = 1,
) -> DataLoader:
    # create a dataset with fake labels
    labels = labels or torch.Tensor([-1] * len(input_tensor))
    dataset = TensorDataset(input_tensor, labels)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

def load_images_from_folder(
    folder_path: str,
    batch_size: int = 1,
    num_workers: int = 1,
) -> DataLoader:
    xforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    image_tensors = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):  # Only process .png files
            img_path = os.path.join(folder_path, file_name)
            label = file_name.split(".")[0]
            labels.append(label)
            img = Image.open(img_path).convert("RGB")
            image_tensor = xforms(img)
            image_tensors.append(image_tensor)

    if not image_tensors:
        raise ValueError(f"No .png files found in the folder: {folder_path}")

    # Stack all tensors into a single batch and add the batch dimension
    input_tensor = torch.stack(image_tensors)

    # Create and return the DataLoader
    label_to_idx = {label: idx for idx, label in enumerate(natsorted(set(labels)))}
    idx_to_label = {idx: label for idx, label in enumerate(natsorted(set(labels)))}
    labels_torch = torch.tensor([label_to_idx[label] for label in labels])
    dataset = TensorDataset(input_tensor, labels_torch)

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers), idx_to_label
