import numpy as np
import glob2 as glob
import PIL.Image as Image
import random
import h5py
import matplotlib.pyplot as plt
import os

import torch
import torchvision.transforms.functional as TF
import cv2
import pickle


"""
ISIC Dataset
Class only contains the dataset consisting of those ISIC images with exactly 3 annoations.
Note that there are images with less and more annotations.
"""
# Folder Structure (Files are >50GB and are therefore stored in scratch directory on theia!)
# datapath = /scratch/kmze/isic
# --- Images
# --- Segmentations
class LIDC(torch.utils.data.Dataset):
    images = []
    labels = []
    series_uid = []
    #def __init__(self, transform, apply_symmetric_transforms, data_path):
    def __init__(self, transform, apply_symmetric_transforms, data_path):
        self.images=[]
        self.labels=[]
        self.series_uid=[]
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(data_path):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = data_path + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def symmetric_augmentation(self, images_and_masks=[]):
        # Random Horizontal Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [TF.affine(x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0)
                            for x in images_and_masks]

        return images_and_masks

    def __getitem__(self, index):
        #image = np.expand_dims(self.images[index], axis=0)
        image = self.images[index]
        #Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0,3)].astype(float)
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        #image = torch.from_numpy(image)
        #label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, [label], torch.as_tensor([0])

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)


class LIDC2(torch.utils.data.Dataset):
    images = []
    labels = []
    series_uid = []
    #def __init__(self, transform, apply_symmetric_transforms, data_path):
    def __init__(self, transform, apply_symmetric_transforms, data_path):
        self.images=[]
        self.labels=[]
        self.series_uid=[]
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(data_path):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = data_path + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def symmetric_augmentation(self, images_and_masks=[]):
        # Random Horizontal Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if (np.random.random() > 0.5):
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [TF.affine(x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0)
                            for x in images_and_masks]

        return images_and_masks

    def __getitem__(self, index):
        #image = np.expand_dims(self.images[index], axis=0)
        image = self.images[index]
        #Randomly select one of the four labels for this image
        #label = self.labels[index][random.randint(0,3)].astype(float)
        label1 = self.labels[index][0].astype(float)
        label2 = self.labels[index][1].astype(float)
        label3 = self.labels[index][2].astype(float)
        label4 = self.labels[index][3].astype(float)


        if self.transform is not None:
            image = self.transform(image)
            label1 = self.transform(label1)
            label2 = self.transform(label2)
            label3 = self.transform(label3)
            label4 = self.transform(label4)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        #image = torch.from_numpy(image)
        #label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label1 = label1.type(torch.FloatTensor)
        label2 = label2.type(torch.FloatTensor)
        label3 = label3.type(torch.FloatTensor)
        label4 = label4.type(torch.FloatTensor)

        label = []
        label.append(label1)
        label.append(label2)
        label.append(label3)
        label.append(label4)

        return image, label, [label], torch.as_tensor([0])

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
