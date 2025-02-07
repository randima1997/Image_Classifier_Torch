# Image Classifier using PyTorch

This project was mainly undertaken as part of the University of Siegen Christmas Image Classification Challenge. 
The challenge involved classifying a test dataset of 160 Christmas-themed images to into 8 different categories. 
A training dataset comprising 3,726 images were provided. 

# Classification categories:
  0. Christmas Cookies
  1. Christmas Presents
  2. Christmas Tree
  3. Fireworks
  4. Penguins
  5. Reindeer
  6. Santa
  7. Snowman

# Training Implementation
Due to the relatively smaller dataset, a transfer learning approach was taken. 
The main optimizer: Adam
Loss function: CrossEntropyLoss

# Resnet34
* Training was initially done treating the pre-trained model as a feature extractor.
* Achieved an accuracy of 84%
* Unfroze and finetuned the remainder of the layers to achieve an accuracy of 90.5%

# Resnet50
* Training was initially done treating the pre-trained model as a feature extractor.
* Achieved an accuracy of 90.8%
* Unfroze and finetuned the remainder of the layers with lr scheduling with a step size of 1 epoch
* Reached 92%
* Tested with batch_size 32 and step lr scheduling
* Reached 93.8%
* Tested batch_size 16 and 8



