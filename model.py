"""
model.py
--------
Defines:
1. PlantDataset  -> reads data/train or data/test folders with structure:
                    data/<split>/<plant_name>/<stage>/*.jpg
2. PlantNet      -> CNN with a shared backbone (MobileNetV2, pretrained)
                    and TWO output heads:
                        - species head  (predicts plant name)
                        - stage head    (predicts baby / mature)
"""

import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, models


# ----------------------------------------------------------------------
# 1. DATASET
# ----------------------------------------------------------------------
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class PlantDataset(Dataset):
    """
    Expects folder structure:
        root/
            plant_name_1/
                baby/    *.jpg
                mature/  *.jpg
            plant_name_2/
                baby/
                mature/
            ...
    Automatically discovers plant classes and stage classes from folder names.
    """

    def __init__(self, root_dir, transform=None,
                 species_classes=None, stage_classes=None):
        self.root_dir = root_dir
        self.transform = transform

        # Discover species (plant) folders
        self.species_classes = species_classes or sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        # Discover stage folders (usually just baby / mature, but flexible)
        self.stage_classes = stage_classes or ["baby", "mature"]

        self.species_to_idx = {c: i for i, c in enumerate(self.species_classes)}
        self.stage_to_idx = {c: i for i, c in enumerate(self.stage_classes)}

        self.samples = []  # list of (filepath, species_idx, stage_idx)
        for species in self.species_classes:
            species_dir = os.path.join(root_dir, species)
            for stage in self.stage_classes:
                stage_dir = os.path.join(species_dir, stage)
                if not os.path.isdir(stage_dir):
                    continue
                for fname in os.listdir(stage_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((
                            os.path.join(stage_dir, fname),
                            self.species_to_idx[species],
                            self.stage_to_idx[stage],
                        ))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {root_dir}. "
                f"Check your folder structure matches: "
                f"root/<plant_name>/<baby|mature>/*.jpg"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, species_idx, stage_idx = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, species_idx, stage_idx


# ----------------------------------------------------------------------
# 2. MODEL
# ----------------------------------------------------------------------
class PlantNet(nn.Module):
    """
    Shared CNN backbone (MobileNetV2, ImageNet-pretrained) with two heads:
        - species_head : predicts which of the 6 plants it is
        - stage_head   : predicts growth stage (baby / mature)
    """

    def __init__(self, num_species, num_stages=2, pretrained=True):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Keep only the feature extractor (drop the original classifier)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = backbone.last_channel  # 1280 for MobileNetV2

        self.species_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_species),
        )

        self.stage_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_stages),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        species_logits = self.species_head(x)
        stage_logits = self.stage_head(x)
        return species_logits, stage_logits
