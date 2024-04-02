from torch.utils.data import Dataset
import os
import torch as t
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter("ignore")

# mean and standard deviation of the solar cell RGB images in the training set across the three channels.
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    """
    Custom dataset class for the solar cell challenge dataset.
    
    Attributes:
        to_tensor (torchvision.transforms.Compose): Transformation to convert images to PyTorch tensors.
        val_transform (torchvision.transforms.Compose): Transformation for validation dataset.
        train_transform (torchvision.transforms.Compose): Transformation for training dataset.
        images (pandas.DataFrame): Pandas DataFrame containing image paths.
        labels (pandas.DataFrame): Pandas DataFrame containing labels where each row is a tuple.
        mode (str): Dataset mode, either 'train' or 'val'.
    """
    
    def __init__(self, data, mode):
        """
        Initializes the ChallengeDataset.

        Parameters:
            data (pandas.DataFrame): Pandas DataFrame containing image paths and labels where each row contains the image path and the tuple of labels.
            mode (str): Dataset mode, either 'train' or 'val'.
        """
        self.mode = mode 
        # Compose is the callable class which does chain of transformations on the data.
        self.to_tensor = tv.transforms.Compose([tv.transforms.ToTensor()])

        # Consider creating two different transforms based on whether you are in the training or validation dataset.
        self.val_transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(mean=train_mean, std=train_std)])
        self.train_transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.RandomHorizontalFlip(),
                                                      tv.transforms.ToTensor(), tv.transforms.Normalize(mean=train_mean, std=train_std)])

        self.images= data.iloc[:, 0]
        self.labels = data.iloc[:, 1:]
        
    def __len__(self):
        """
        Gets the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.images)
        
    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Parameters:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the image and its corresponding labels.
        """
        if self.mode == 'val':
            temp_img = imread(self.images.iloc[index])
            img = gray2rgb(temp_img)
            img = self.val_transform(img)
            # the labels have a shape of (1, 2) because it is a multi-class problem (whether solar cell shows a crack and if it can be considered inactive).
            labels = self.to_tensor(np.asarray(self.labels.iloc[index]).reshape(1, 2))
            return (img, labels)
        if self.mode == 'train':
            temp_img = imread(self.images.iloc[index])
            img = gray2rgb(temp_img)
            img = self.train_transform(img)
            # the labels have a shape of (1, 2) because it is a multi-class problem (whether solar cell shows a crack and if it can be considered inactive).
            labels = self.to_tensor(np.asarray(self.labels.iloc[index]).reshape(1, 2))
            return (img, labels)

