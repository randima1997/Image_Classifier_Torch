import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from main import christmasClassifier, train, test
from multiprocessing import freeze_support
import time





if __name__ == '__main__':
    freeze_support()

    weights_path = "weights/Resnet34_weights_rtr.pth"

# Hyperparameters and paths

    train_data_path = "D:/Engineering/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/train"
    test_data_path = "D:/Engineering/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/val"

    batch_size = 64
    lr = 0.0001
    epochs = 4
    num_workers = 5


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
        num_workers= num_workers,
        pin_memory= True
    )

    val_dataloader = DataLoader(
        dataset= val_dataset,
        batch_size= batch_size,
        shuffle= False,                 
        num_workers= num_workers,
        pin_memory= True
    )


    test_dataloader = DataLoader(
        dataset= test_dataset,
        batch_size= batch_size,
        shuffle= False,                 # Not shuffling the Test set
        num_workers= num_workers,
        pin_memory= True
    )


# Initialize model and load the weights

    model = christmasClassifier()
    model.load_state_dict(torch.load(weights_path, weights_only=True))

    for params in model.parameters():
            params.requires_grad = True

    model = model.to(device)


# Loss function

    loss_func = nn.CrossEntropyLoss()

# Defining optimizers

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma= 0.1)


    try:
        for t in range(epochs):


            print(f"Running Epoch {t+1}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            start_time = time.time()
            train(train_dataloader, model, device, loss_func, optimizer)
            test(val_dataloader, model, device, loss_func)
            scheduler.step()
            print("Elapsed time: ", (time.time() - start_time))




    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving the model...")
        torch.save(model.state_dict(), "weights/Resnet34_weights_unfrozen.pth")
        print("Model saved!")