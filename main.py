import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from multiprocessing import freeze_support
import time


# Model definition

class christmasClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet34(weights= "IMAGENET1K_V1")

        for params in self.base_model.parameters():
            params.requires_grad = False

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 8)

    def forward(self, x):
        x = self.base_model(x)

        return x


# Training function

def train(train_dataloader, model, device, loss_fn, optim):
    size = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size

    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device) , y.to(device)
        prediction = model(X)

        loss = loss_fn(prediction, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_loss, current = loss.item(), batch * batch_size + len(X)
        print(f"loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")


# Test function

def test(val_dataloader, model, device, loss_fn):
        
    model.eval()
    size = len(val_dataloader.dataset)
    num_batches = len(val_dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in val_dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")



if __name__ == '__main__':
    freeze_support()

# GPU availability

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")


# Hyperparameters and paths

    train_data_path = "/media/randima/Data/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/train"
    test_data_path = "/media/randima/Data/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/val"

    batch_size = 64
    lr = 0.001
    epochs = 2
    num_workers = 5


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

    

    model = christmasClassifier().to(device)

# Loss function

    loss_func = nn.CrossEntropyLoss()

# Defining optimizers

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))



# Main execution block


    try:
        for t in range(epochs):
            
            start_time = time.time()
            print(f"Running Epoch {t+1}")
            train(train_dataloader, model, device, loss_func, optimizer)
            test(val_dataloader, model, device, loss_func)
            print("Elapsed time: ", (time.time() - start_time))


    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving the model...")
        torch.save(model.state_dict(), 'weights/Resnet34_weights.pth')
        print("Model saved!")

        