import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torchvision.io import decode_image, read_image, write_png





path_train =  "D:/Engineering/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/train"
path_test = "D:/Engineering/Uni Siegen/Semester 3/Deep Learning/Project 1/Image_Classifier_TF/data/val"

transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness= (0.4,1.5), contrast=0.2, saturation=0.2, hue=0.1),
    #transforms.GaussianBlur(kernel_size=3, sigma=((0.1, 4.0))),
    #transforms.RandomRotation(30),
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    
])

train_raw = datasets.ImageFolder(
    root = path_train,
    transform= transform,
)

test_raw = datasets.ImageFolder(
    root = path_test,
    transform= transform,
)

train_dataloader = DataLoader(dataset= train_raw, batch_size= 4, shuffle= True)
test_dataloader = DataLoader(dataset= test_raw, batch_size= 4, shuffle= False)


data_iter = iter(test_dataloader)

for i in range(10):
    images, labels = next(data_iter)

images, labels = next(data_iter)

def show_images(imgs, labels, class_names):
    # Undo normalisation if applicable (optional)
    imgs = imgs.numpy().transpose(0, 2, 3, 1)  # Convert to (batch_size, height, width, channels)
    
    # Plot the images
    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    for i, (img, label) in enumerate(zip(imgs, labels)):
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(class_names[label])
    plt.tight_layout()
    plt.show()

class_names = ('christmas_cookies', 'christmas_presents', 'christmas_tree', 'fireworks', 'penguin', 'reindeer', 'santa', 'snowman')

show_images(images, labels, class_names)



