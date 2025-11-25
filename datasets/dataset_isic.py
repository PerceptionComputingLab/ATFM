import numpy as np
import glob2 as glob
import PIL.Image as Image
import random


import torch
import torchvision.transforms.functional as TF


"""
ISIC Dataset
Class only contains the dataset consisting of those ISIC images with exactly 3 annoations.
Note that there are images with less and more annotations.
"""
# Folder Structure (Files are >50GB and are therefore stored in scratch directory on theia!)
# datapath = /scratch/kmze/isic
# --- Images
# --- Segmentations


class ISIC(torch.utils.data.Dataset):
    def __init__(self, transform, apply_symmetric_transforms, data_path):

        # Initialization
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms

        # Image List
        # all image paths stored in a list
        all_image_paths = sorted(glob.glob(data_path + '/Images/*'))

        image_paths = []
        annotator1_paths = []
        annotator2_paths = []
        annotator3_paths = []

        for entry in all_image_paths:
            # gives substring with unique identifier ISIC_0000023
            ident = entry[-16:-4]
            annotations_per_ident = sorted(
                glob.glob(data_path + f'/Segmentations/{ident}*'))

            if len(annotations_per_ident) == 3:
                image_paths.append(entry)
                annotator1_paths.append(annotations_per_ident[0])
                annotator2_paths.append(annotations_per_ident[1])
                annotator3_paths.append(annotations_per_ident[2])

        self.image_paths = 3*image_paths
        self.all_annotator_paths = annotator1_paths+annotator2_paths+annotator3_paths
        self.annotator1_paths = 3*annotator1_paths
        self.annotator2_paths = 3*annotator2_paths
        self.annotator3_paths = 3*annotator3_paths

    def symmetric_augmentation(self, images_and_masks=[]):
        """
        Applies affine transformations for image and masks.

        Args:
            images_and_masks: (list of torch.tensors) images and masks to transform
        Returns:
            List of torch.tensors that are transformed in the same way (symmetric) and therefore suitable for semantic segmentation

        """

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

    def get_pathnahme(self, idx):
        image_path = self.image_paths[idx]
        target1_path = self.annotator1_paths[idx]
        target2_path = self.annotator2_paths[idx]
        target3_path = self.annotator3_paths[idx]
        return image_path, target1_path, target2_path, target3_path

    def __len__(self):
        # Returns the total number of samples in the DataSet
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Generates one sample of data

        image_path = self.image_paths[idx]
        target_path = self.all_annotator_paths[idx]
        target1_path = self.annotator1_paths[idx]
        target2_path = self.annotator2_paths[idx]
        target3_path = self.annotator3_paths[idx]

        path_list = [image_path, target_path, target1_path, target2_path,
                     target3_path]

        # helper function reads a path into a img with nibabel package

        def path_to_image(path):
            img = Image.open(path)  # Read in with Pillow
            return img

        # Prepares images
        def pipe(x):
            x = path_to_image(x)
            #x = x.astype('int16')
            x = self.transform(x)
            x = x.float()
            return x

        transformed_list = [pipe(x) for x in path_list]
        image, target, target1, target2, target3 = transformed_list
        # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
        if self.symmetric_transforms:
            image, target, target1, target2, target3 = self.symmetric_augmentation(
                transformed_list)

        # Normalize each patch by its mean and variance and set intensities to 0 that are more then 3 stds away
        #mean = torch.mean(image)
        #std = torch.std(image)
        #image = (image-mean)/(3*std)
       # image = TF.normalize(image, torch.mean(image), torch.std(image))
        return image, target, [target1, target2, target3], None
