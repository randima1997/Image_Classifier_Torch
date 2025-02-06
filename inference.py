import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from main import christmasClassifier, test


test_data_path = "D:/Engineering/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/val"
weights_path = "weights/Resnet34_weights_unfrozen.pth"

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

print(f"Using {device} device")


transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

test_dataset = datasets.ImageFolder(
        root= test_data_path,
        transform=transform_test
    )

test_dataloader = DataLoader(
        dataset= test_dataset,
        batch_size= 64,
        shuffle= False,                 # Not shuffling the Test set
        num_workers= 0,
        pin_memory= True
    )

model = christmasClassifier()
model.load_state_dict(torch.load(weights_path, weights_only=True))
model = model.to(device)

loss_func = nn.CrossEntropyLoss()

test(test_dataloader, model, device, loss_func)