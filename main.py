import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# Hyperparameters and paths

train_data_path = "D:/Engineering/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/train"
test_data_path = "D:/Engineering/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/val"

batch_size = 64
lr = 0.001
epochs = 2


# GPU availability

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Transforms for the Train and Test datasets

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Train and Test datasets. Generalizing transforms applied to train set

raw_dataset = datasets.ImageFolder(
    root= train_data_path,
    transform= transform
)

train_size = int(0.85 * len(raw_dataset))
val_size = len(raw_dataset) - train_size

train_dataset, val_dataset = random_split(raw_dataset, [train_size, val_size])


test_dataset = datasets.ImageFolder(
    root= test_data_path,
    transform=transform_test
)

# Dataloading

train_dataloader = DataLoader(
    dataset= train_dataset,
    batch_size= batch_size,
    shuffle= True,
    num_workers= 3,
    pin_memory= True
)

val_dataloader = DataLoader(
    dataset= val_dataset,
    batch_size= batch_size,
    shuffle= False,                 
    num_workers= 3,
    pin_memory= True
)


test_dataloader = DataLoader(
    dataset= test_dataset,
    batch_size= batch_size,
    shuffle= False,                 # Not shuffling the Test set
    num_workers= 3,
    pin_memory= True
)