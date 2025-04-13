import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Resize

class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.resize = Resize((256, 256))  # Resize both image and mask to 256x256

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize both image and mask
        image = self.resize(image)
        mask = self.resize(mask)

        # Convert mask to binary format
        mask = np.array(mask, dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=np.array(image), mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask = mask.unsqueeze(0)

        return image, mask