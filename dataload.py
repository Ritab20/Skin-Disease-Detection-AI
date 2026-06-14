# STEP 4: Dataset + DataLoader (UPGRADED VERSION)

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("balanced_data.csv")

IMAGE_DIR = "images"


# -----------------------------
# CUSTOM DATASET CLASS
# -----------------------------
class SkinDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        img_name = self.df.loc[idx, "image_id"] + ".jpg"

        img_path = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(img_path):

            raise FileNotFoundError(
                f"{img_name} not found in {IMAGE_DIR}"
            )

        image = Image.open(img_path).convert("RGB")

        label = self.df.loc[idx, "label"]

        if self.transform:

            image = self.transform(image)

        return image, label


# -----------------------------
# IMAGE TRANSFORMATIONS
# -----------------------------
train_transform = transforms.Compose([

    transforms.Resize((300, 300)),  # EfficientNet optimal size

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(30),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------
# STRATIFIED TRAIN/VAL SPLIT
# -----------------------------
train_df, val_df = train_test_split(

    df,

    test_size=0.2,

    stratify=df["label"],

    random_state=42

)


# -----------------------------
# CREATE DATASETS
# -----------------------------
train_dataset = SkinDataset(

    train_df,

    transform=train_transform

)


val_dataset = SkinDataset(

    val_df,

    transform=val_transform

)


# -----------------------------
# DATALOADERS (GPU-OPTIMIZED)
# -----------------------------
train_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=True,

    num_workers=0,

    pin_memory=True

)


val_loader = DataLoader(

    val_dataset,

    batch_size=16,

    shuffle=False,

    num_workers=0,

    pin_memory=True

)


# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == "__main__":

    print("Train samples:", len(train_dataset))

    print("Validation samples:", len(val_dataset))


    for images, labels in train_loader:

        print("Image batch shape:", images.shape)

        print("Label batch shape:", labels.shape)

        break


    print("\nSTEP 4 COMPLETED SUCCESSFULLY!")