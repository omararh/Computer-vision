# Deep Image Classification Project

## Project Overview

This project applies deep learning techniques for computer vision tasks to a new dataset. The goal is to test the performance of different classification models on the Caltech101 dataset.

## 1. Data Loading

We use the [Caltech101 dataset](https://data.caltech.edu/records/mzrjq-6wc02), which contains approximately 10,000 images (300 x 200 pixels) of objects across 101 categories.

### Loading Procedure

The dataset is loaded using PyTorch's `ImageFolder` ([documentation](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)). We'll use 90% of the data for training and 10% for testing.

Steps to create the dataloaders:

1. Load the entire dataset with `ImageFolder` using transformations for training (potentially including random modifications for data augmentation).
2. Randomly select 90% of the image indices for training, reserving the remaining indices for testing.
3. Use `torch.utils.data.Subset` ([documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset)) to create the training set with the selected indices.
4. Reload the entire dataset with test transformations (using only Resize, tensor transformation, and normalization as done in the practical work for pre-trained models on ImageNet), then reuse `torch.utils.data.Subset` to create the test set with the reserved indices.

### Working with Google Colab

When working in Google Colab, ensure you specify the GPU runtime. Also, run these commands before loading the data:

```python
!pip install Pillow==4.0.0
!pip install PIL
!pip install image
from PIL import Image
```

## 2. Comparing Different Models

We'll compare various pre-trained PyTorch models:

* resnet18
* alexnet
* squeezenet1_0
* vgg16
* densenet161
* inception_v3

For each model, we need to:

1. Resize images to match the format used during pre-training (224×224 for all models except inception, which requires 299×299).
2. Normalize images as they were during training (use `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`).
3. Adapt the model output so the number of outputs corresponds to the expected number of classes (101).
4. Define an optimizer that only modifies non-pre-trained parameters.

## 3. Model Evaluation

Models will be evaluated based on their classification accuracy on test data. To obtain an average score independent of the train/test split, we'll use Cross-Validation:

1. Repeat for k iterations (e.g., k=10):
   a. Build train and test datasets using random splitting (as defined in the data loading section).
   b. Train models on the training set.
   c. Record the classification accuracy on the test set for each model.
2. Return the average classification accuracy on test data for each model.

Optionally, we can also record the average loss curves for training and testing for each model across all folds.
