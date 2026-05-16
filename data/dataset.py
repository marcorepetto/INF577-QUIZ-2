import numpy as np
import torch
from torch.utils.data import Dataset

class FacialAttributesDataset(Dataset):
    def __init__(self, npz_path, transform=None, normalization=None, normalization_fun=None):
        """
        Args:
            npz_path (str): Path to the .npz file.
            transform (callable, optional): Optional transform to be applied on a sample.
            normalization (str): Type of normalization to apply.
            normalization_fun (callable, optional): Custom normalization function.
        """
        data = np.load(npz_path)
        self.images = data['images']
        
        if 'labels' in data:
            self.labels = data['labels']
            assert len(self.images) == len(self.labels), "Images and labels must have the same length"
        else:
            self.labels = None
            
        self.transform = transform
        self.normalization = normalization

        if self.normalization == "mean":
            # Pre-calculate global channel-wise mean and std (N, H, W, C) -> (C,)
            self.channel_means = np.mean(self.images, axis=(0, 1, 2)).astype(np.float32)
            self.channel_stds = np.std(self.images, axis=(0, 1, 2)).astype(np.float32) + 1e-7

            self.normalization_function = lambda image: (image - self.channel_means) / self.channel_stds
        elif self.normalization == "intrgb":
            self.normalization_function = lambda image: image / 255.0
        elif self.normalization is None:
            if normalization_fun is not None:
                self.normalization_function = normalization_fun
            else:
                self.normalization_function = lambda x: x  # No normalization
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Convert image to float32
        image = image.astype(np.float32)
        
        image = self.normalization_function(image)

        # PyTorch expects (C, H, W). Assuming input is (H, W, C)
        if image.ndim == 3 and image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))
        
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
            return image, label
        else:
            return image

    def get_normalization_function(self):
        return self.normalization_function
