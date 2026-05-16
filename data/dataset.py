import numpy as np
import torch
from torch.utils.data import Dataset

class FacialAttributesDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        """
        Args:
            npz_path (str): Path to the .npz file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        data = np.load(npz_path)
        self.images = data['images']
        
        if 'labels' in data:
            self.labels = data['labels']
            assert len(self.images) == len(self.labels), "Images and labels must have the same length"
        else:
            self.labels = None
            
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Convert image to float32 and normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

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

