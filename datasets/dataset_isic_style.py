from matplotlib.colors import ListedColormap
import numpy as np
import glob2 as glob
import PIL.Image as Image
import random

# import cv2


import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

"""
ISIC Dataset
Class only contains the dataset consisting of those ISIC images with exactly 3 annoations.
Note that there are images with less and more annotations.
"""
# Folder Structure
# datapath = /scratch/kmze/isic
# --- Images
# --- Segmentations


class ISIC_style(torch.utils.data.Dataset):
    def __init__(self, transform, apply_symmetric_transforms, data_path):
        # Initialization
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms

        # Image List
        # all image paths stored in a list
        all_image_paths = sorted(glob.glob(data_path + "/Images/*"))

        image_paths = []
        annotator1_paths = []
        annotator2_paths = []
        annotator3_paths = []
        annotator1_style = []
        annotator2_style = []
        annotator3_style = []

        for entry in all_image_paths:
            # gives substring with unique identifier ISIC_0000023
            ident = entry[-16:-4]
            annotations_per_ident = sorted(
                glob.glob(data_path + f"/Segmentations/{ident}*")
            )

            if len(annotations_per_ident) == 3:
                image_paths.append(entry)
                annotator1_paths.append(annotations_per_ident[0])
                annotator1_style.append(annotations_per_ident[0][-5:-4])
                annotator2_paths.append(annotations_per_ident[1])
                annotator2_style.append(annotations_per_ident[1][-5:-4])
                annotator3_paths.append(annotations_per_ident[2])
                annotator3_style.append(annotations_per_ident[2][-5:-4])

        self.image_paths = 3 * image_paths
        self.all_annotator_paths = (
            annotator1_paths + annotator2_paths + annotator3_paths
        )
        all_annotator_styles = annotator1_style + annotator2_style + annotator3_style
        all_annotator_styles = [
            0 if x == "f" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        all_annotator_styles = [
            1 if x == "m" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        all_annotator_styles = [
            2 if x == "p" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        self.all_annotator_styles = all_annotator_styles
        self.annotator1_paths = 3 * annotator1_paths
        self.annotator2_paths = 3 * annotator2_paths
        self.annotator3_paths = 3 * annotator3_paths

    def symmetric_augmentation(self, images_and_masks=[]):
        """
        Applies affine transformations for image and masks.

        Args:
            images_and_masks: (list of torch.tensors) images and masks to transform
        Returns:
            List of torch.tensors that are transformed in the same way (symmetric) and therefore suitable for semantic segmentation

        """

        # Random Horizontal Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [
            TF.affine(
                x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0
            )
            for x in images_and_masks
        ]

        return images_and_masks

    def get_pathnahme(self, idx):
        image_path = self.image_paths[idx]
        all_path = self.all_annotator_paths[idx]
        target1_path = self.annotator1_paths[idx]
        target2_path = self.annotator2_paths[idx]
        target3_path = self.annotator3_paths[idx]
        return image_path, all_path, target1_path, target2_path, target3_path

    def __len__(self):
        # Returns the total number of samples in the DataSet
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Generates one sample of data

        image_path = self.image_paths[idx]
        target_path = self.all_annotator_paths[idx]
        style = self.all_annotator_styles[idx]
        target1_path = self.annotator1_paths[idx]
        target2_path = self.annotator2_paths[idx]
        target3_path = self.annotator3_paths[idx]

        path_list = [image_path, target_path, target1_path, target2_path, target3_path]

        # helper function reads a path into a img with nibabel package

        def path_to_image(path):
            img = Image.open(path)  # Read in with Pillow
            return img

        # Prepares images
        def pil_image(x):
            x = path_to_image(x)
            # x = x.astype('int16')
            return x

        def to_tensor_trafo(x):
            x = self.transform(x)
            x = x.float()
            return x

        pil_image_list = [pil_image(x) for x in path_list]
        image, target, target1, target2, target3 = pil_image_list
        # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
        if self.symmetric_transforms:
            pil_image_list = self.symmetric_augmentation(pil_image_list)

        # Transform style variable to numerical values and Torch Tensor (f =0, m=1, p=2)
        style = torch.as_tensor(style)
        final_list = [to_tensor_trafo(x) for x in pil_image_list]
        image, target, target1, target2, target3 = final_list
        # Normalize each patch by its mean and variance and set intensities to 0 that are more then 3 stds away
        # mean = torch.mean(image)
        # std = torch.std(image)
        # image = (image-mean)/(3*std)
        # image = TF.normalize(image, torch.mean(image), torch.std(image))
        return image, target, [target1, target2, target3], style


class ISIC_style_subset(torch.utils.data.Dataset):
    def __init__(self, transform, apply_symmetric_transforms, data_path, style):
        # Initialization
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms

        # Image List
        # all image paths stored in a list
        all_image_paths = sorted(glob.glob(data_path + "/Images/*"))

        image_paths = []
        annotator1_paths = []
        annotator2_paths = []
        annotator3_paths = []
        annotator1_style = []
        annotator2_style = []
        annotator3_style = []

        for entry in all_image_paths:
            # gives substring with unique identifier ISIC_0000023
            ident = entry[-16:-4]
            annotations_per_ident = sorted(
                glob.glob(data_path + f"/Segmentations/{ident}*")
            )

            if len(annotations_per_ident) == 3:
                image_paths.append(entry)
                annotator1_paths.append(annotations_per_ident[0])
                annotator1_style.append(annotations_per_ident[0][-5:-4])
                annotator2_paths.append(annotations_per_ident[1])
                annotator2_style.append(annotations_per_ident[1][-5:-4])
                annotator3_paths.append(annotations_per_ident[2])
                annotator3_style.append(annotations_per_ident[2][-5:-4])

        self.image_paths = 3 * image_paths
        self.all_annotator_paths = (
            annotator1_paths + annotator2_paths + annotator3_paths
        )
        self.annotator1_paths = 3 * annotator1_paths
        self.annotator2_paths = 3 * annotator2_paths
        self.annotator3_paths = 3 * annotator3_paths
        all_annotator_styles = annotator1_style + annotator2_style + annotator3_style
        all_annotator_styles = [
            0 if x == "f" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        all_annotator_styles = [
            1 if x == "m" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        all_annotator_styles = [
            2 if x == "p" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        self.all_annotator_styles = all_annotator_styles

        # List of positions of the chosen style in the all_annotator_styles list
        indices = [i for i, x in enumerate(self.all_annotator_styles) if x == style]
        # Only keep image paths, annotator paths and styles on those positions in the lists
        self.image_paths = [self.image_paths[index] for index in indices]
        self.all_annotator_paths = [
            self.all_annotator_paths[index] for index in indices
        ]
        self.all_annotator_styles = [
            self.all_annotator_styles[index] for index in indices
        ]
        self.annotator1_paths = [self.annotator1_paths[index] for index in indices]
        self.annotator2_paths = [self.annotator2_paths[index] for index in indices]
        self.annotator3_paths = [self.annotator3_paths[index] for index in indices]

    def symmetric_augmentation(self, images_and_masks=[]):
        """
        Applies affine transformations for image and masks.

        Args:
            images_and_masks: (list of torch.tensors) images and masks to transform
        Returns:
            List of torch.tensors that are transformed in the same way (symmetric) and therefore suitable for semantic segmentation

        """

        # Random Horizontal Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [
            TF.affine(
                x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0
            )
            for x in images_and_masks
        ]

        return images_and_masks

    def get_pathnahme(self, idx):
        image_path = self.image_paths[idx]
        all_path = self.all_annotator_paths[idx]

        return image_path, all_path

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
        style = self.all_annotator_styles[idx]

        path_list = [image_path, target_path, target1_path, target2_path, target3_path]

        # helper function reads a path into a img with nibabel package

        def path_to_image(path):
            img = Image.open(path)  # Read in with Pillow
            return img

        # Prepares images
        def pil_image(x):
            x = path_to_image(x)
            # x = x.astype('int16')
            return x

        def to_tensor_trafo(x):
            x = self.transform(x)
            x = x.float()
            return x

        pil_image_list = [pil_image(x) for x in path_list]
        image, target, target1, target2, target3 = pil_image_list
        # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
        if self.symmetric_transforms:
            pil_image_list = self.symmetric_augmentation(pil_image_list)

        # Transform style variable to numerical values and Torch Tensor (f =0, m=1, p=2)
        style = torch.as_tensor(style)
        final_list = [to_tensor_trafo(x) for x in pil_image_list]
        image, target, target1, target2, target3 = final_list
        # Normalize each patch by its mean and variance and set intensities to 0 that are more then 3 stds away
        # mean = torch.mean(image)
        # std = torch.std(image)
        # image = (image-mean)/(3*std)
        # image = TF.normalize(image, torch.mean(image), torch.std(image))
        return image, target, [target1, target2, target3], style

class ISIC_style_subset2(torch.utils.data.Dataset):
    def __init__(self, transform, apply_symmetric_transforms, data_path, style):
        # Initialization
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms

        # Image List
        # all image paths stored in a list
        all_image_paths = sorted(glob.glob(data_path + "/Images/*"))

        image_paths = []
        annotator1_paths = []
        annotator2_paths = []
        annotator3_paths = []
        annotator1_style = []
        annotator2_style = []
        annotator3_style = []

        for entry in all_image_paths:
            # gives substring with unique identifier ISIC_0000023
            ident = entry[-16:-4]
            annotations_per_ident = sorted(
                glob.glob(data_path + f"/Segmentations/{ident}*")
            )

            if len(annotations_per_ident) == 3:
                image_paths.append(entry)
                annotator1_paths.append(annotations_per_ident[0])
                annotator1_style.append(annotations_per_ident[0][-5:-4])
                annotator2_paths.append(annotations_per_ident[1])
                annotator2_style.append(annotations_per_ident[1][-5:-4])
                annotator3_paths.append(annotations_per_ident[2])
                annotator3_style.append(annotations_per_ident[2][-5:-4])

        self.image_paths = 3 * image_paths
        self.all_annotator_paths = (
            annotator1_paths + annotator2_paths + annotator3_paths
        )
        self.annotator1_paths = 3 * annotator1_paths
        self.annotator2_paths = 3 * annotator2_paths
        self.annotator3_paths = 3 * annotator3_paths
        all_annotator_styles = annotator1_style + annotator2_style + annotator3_style
        all_annotator_styles = [
            0 if x == "f" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        all_annotator_styles = [
            1 if x == "m" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        all_annotator_styles = [
            2 if x == "p" else x for x in all_annotator_styles
        ]  # (f =0, m=1, p=2)
        self.all_annotator_styles = all_annotator_styles

        # List of positions of the chosen style in the all_annotator_styles list
        indices = [i for i, x in enumerate(self.all_annotator_styles) if x == style]
        # Only keep image paths, annotator paths and styles on those positions in the lists
        self.image_paths = [self.image_paths[index] for index in indices]
        self.all_annotator_paths = [
            self.all_annotator_paths[index] for index in indices
        ]
        self.all_annotator_styles = [
            self.all_annotator_styles[index] for index in indices
        ]
        self.annotator1_paths = [self.annotator1_paths[index] for index in indices]
        self.annotator2_paths = [self.annotator2_paths[index] for index in indices]
        self.annotator3_paths = [self.annotator3_paths[index] for index in indices]

    def symmetric_augmentation(self, images_and_masks=[]):
        """
        Applies affine transformations for image and masks.

        Args:
            images_and_masks: (list of torch.tensors) images and masks to transform
        Returns:
            List of torch.tensors that are transformed in the same way (symmetric) and therefore suitable for semantic segmentation

        """

        # Random Horizontal Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [
            TF.affine(
                x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0
            )
            for x in images_and_masks
        ]

        return images_and_masks

    def get_pathnahme(self, idx):
        image_path = self.image_paths[idx]
        all_path = self.all_annotator_paths[idx]

        return image_path, all_path

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
        style = self.all_annotator_styles[idx]

        path_list = [image_path, target_path, target1_path, target2_path, target3_path]

        # helper function reads a path into a img with nibabel package

        def path_to_image(path):
            img = Image.open(path)  # Read in with Pillow
            return img

        # Prepares images
        def pil_image(x):
            x = path_to_image(x)
            # x = x.astype('int16')
            return x

        def to_tensor_trafo(x):
            x = self.transform(x)
            x = x.float()
            return x

        pil_image_list = [pil_image(x) for x in path_list]
        image, target, target1, target2, target3 = pil_image_list
        # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
        if self.symmetric_transforms:
            pil_image_list = self.symmetric_augmentation(pil_image_list)

        # Transform style variable to numerical values and Torch Tensor (f =0, m=1, p=2)
        style = torch.as_tensor(style)
        final_list = [to_tensor_trafo(x) for x in pil_image_list]
        image, target, target1, target2, target3 = final_list
        # Normalize each patch by its mean and variance and set intensities to 0 that are more then 3 stds away
        # mean = torch.mean(image)
        # std = torch.std(image)
        # image = (image-mean)/(3*std)
        # image = TF.normalize(image, torch.mean(image), torch.std(image))
        return image, [target1, target2, target3], target, style

class ISIC_dynamic_augmentation(torch.utils.data.Dataset):
    def __init__(self, transform, apply_symmetric_transforms, data_path, style=0):
        # Initialization
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms

        # Image List
        # all image paths stored in a list
        all_image_paths = sorted(glob.glob(data_path + "/Images/*"))

        image_paths = []
        annotator1_paths = []
        annotator2_paths = []
        annotator3_paths = []
        annotator1_style = []
        annotator2_style = []
        annotator3_style = []

        for entry in all_image_paths:
            # gives substring with unique identifier ISIC_0000023
            ident = entry[-16:-4]
            annotations_per_ident = sorted(
                glob.glob(data_path + f"/Segmentations/{ident}*")
            )

            if len(annotations_per_ident) == 3:
                image_paths.append(entry)
                annotator1_paths.append(annotations_per_ident[0])
                annotator1_style.append(annotations_per_ident[0][-5:-4])
                annotator2_paths.append(annotations_per_ident[1])
                annotator2_style.append(annotations_per_ident[1][-5:-4])
                annotator3_paths.append(annotations_per_ident[2])
                annotator3_style.append(annotations_per_ident[2][-5:-4])

        # substitute letter by integers

        annotator1_style = [
            0 if x == "f" else x for x in annotator1_style
        ]  # (f =0, m=1, p=2)
        annotator1_style = [
            1 if x == "m" else x for x in annotator1_style
        ]  # (f =0, m=1, p=2)
        annotator1_style = [
            2 if x == "p" else x for x in annotator1_style
        ]  # (f =0, m=1, p=2)

        annotator2_style = [
            0 if x == "f" else x for x in annotator2_style
        ]  # (f =0, m=1, p=2)
        annotator2_style = [
            1 if x == "m" else x for x in annotator2_style
        ]  # (f =0, m=1, p=2)
        annotator2_style = [
            2 if x == "p" else x for x in annotator2_style
        ]  # (f =0, m=1, p=2)

        annotator3_style = [
            0 if x == "f" else x for x in annotator3_style
        ]  # (f =0, m=1, p=2)
        annotator3_style = [
            1 if x == "m" else x for x in annotator3_style
        ]  # (f =0, m=1, p=2)
        annotator3_style = [
            2 if x == "p" else x for x in annotator3_style
        ]  # (f =0, m=1, p=2)

        # Filter out cases with no style 0 annotations because we cannot do dynamic augmentation then
        indices2 = [
            i
            for i, x in enumerate(image_paths)
            if (annotator1_style[i] == 0)
            or (annotator2_style[i] == 0)
            or (annotator3_style[i] == 0)
        ]
        image_paths = [image_paths[index] for index in indices2]
        annotator1_style = [annotator1_style[index] for index in indices2]
        annotator2_style = [annotator2_style[index] for index in indices2]
        annotator3_style = [annotator3_style[index] for index in indices2]
        annotator1_paths = [annotator1_paths[index] for index in indices2]
        annotator2_paths = [annotator2_paths[index] for index in indices2]
        annotator3_paths = [annotator3_paths[index] for index in indices2]

        self.image_paths = 3 * image_paths
        self.all_annotator_paths = (
            annotator1_paths + annotator2_paths + annotator3_paths
        )
        self.annotator1_paths = 3 * annotator1_paths
        self.annotator2_paths = 3 * annotator2_paths
        self.annotator3_paths = 3 * annotator3_paths
        self.annotator1_style = 3 * annotator1_style
        self.annotator2_style = 3 * annotator2_style
        self.annotator3_style = 3 * annotator3_style
        self.all_annotator_styles = (
            annotator1_style + annotator2_style + annotator3_style
        )

    def dynamic_augmentation(self, mask, style):
        # mask = mask.rotate(90, PIL.Image.NEAREST, expand=1)
        if style == 1:
            mask = np.array(mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(
                mask, (0, 0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT
            )
            mask = mask > 0.5
            mask = mask * 255
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)

        if style == 2:
            mask = np.array(mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=3)
            mask = cv2.GaussianBlur(
                mask, (0, 0), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT
            )
            mask = mask > 0.5
            mask = mask * 255
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)

        return mask

    def symmetric_augmentation(self, images_and_masks=[]):
        """
        Applies affine transformations for image and masks.

        Args:
            images_and_masks: (list of torch.tensors) images and masks to transform
        Returns:
            List of torch.tensors that are transformed in the same way (symmetric) and therefore suitable for semantic segmentation

        """

        # Random Horizontal Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # Random Vertical Flip
        if np.random.random() > 0.5:
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # Shift/Scale/Rotate Randomly
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        scale = random.uniform(0.9, 1.1)  # prev 0.9 1.1
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [
            TF.affine(
                x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0
            )
            for x in images_and_masks
        ]

        return images_and_masks

    def get_pathnahme(self, idx):
        image_path = self.image_paths[idx]
        all_path = self.all_annotator_paths[idx]

        return image_path, all_path

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
        target_style = self.all_annotator_styles[idx]
        target1_style = self.annotator1_style[idx]
        target2_style = self.annotator2_style[idx]
        target3_style = self.annotator3_style[idx]

        path_list = [image_path, target_path, target1_path, target2_path, target3_path]
        style_list = [target_style, target1_style, target2_style, target3_style]

        # helper function reads a path into a img with nibabel package

        def path_to_image(path):
            img = Image.open(path)  # Read in with Pillow
            return img

        # Prepares images
        def pil_image(x):
            x = path_to_image(x)
            # x = x.astype('int16')
            return x

        def to_tensor_trafo(x):
            x = self.transform(x)
            x = x.float()
            return x

        # case 1: all annotators have the same style 0
        if target_style == 0:
            # no dynamic augmentation

            pil_image_list = [pil_image(x) for x in path_list]
            image, target, target1, target2, target3 = pil_image_list
            # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
            if self.symmetric_transforms:
                pil_image_list = self.symmetric_augmentation(pil_image_list)

            # Transform style variable to numerical values and Torch Tensor (f =0, m=1, p=2)
            style = torch.as_tensor(target_style)
            final_list = [to_tensor_trafo(x) for x in pil_image_list]
            image, target, target1, target2, target3 = final_list
            return image, target, [target1, target2, target3], style

        if target_style != 0:
            # dynamic augmentation
            if style_list.count(0) == 2:  # case 2: 2 annotators have style 0
                # apply dynamic augmentation

                style0_index = [i for i, e in enumerate(style_list) if e == 0]

                # index of chosen style 0 mask in path list
                choice = np.random.choice(style0_index)

                target_path = path_list[choice + 1]  # chosen style 0 mask path
                path_list = [
                    image_path,
                    target_path,
                    target1_path,
                    target2_path,
                    target3_path,
                ]
                style_list = [target_style, target1_style, target2_style, target3_style]

                pil_image_list = [pil_image(x) for x in path_list]
                image, target, target1, target2, target3 = pil_image_list
                # Apply dynamic augmentation to target mask

                target = self.dynamic_augmentation(target, target_style)
                pil_image_list = [image, target, target1, target2, target3]
                # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
                if self.symmetric_transforms:
                    pil_image_list = self.symmetric_augmentation(pil_image_list)

                # Transform style variable to numerical values and Torch Tensor (f =0, m=1, p=2)
                style = torch.as_tensor(target_style)
                final_list = [to_tensor_trafo(x) for x in pil_image_list]
                image, target, target1, target2, target3 = final_list
                return image, target, [target1, target2, target3], style

            if style_list.count(0) == 1:  # case 3: 1 annotator has style 0
                # apply dynamic augmentation

                style0_index = [i for i, e in enumerate(style_list) if e == 0]

                # index of chosen style 0 mask in path list
                choice = np.random.choice(style0_index)
                target_path = path_list[choice + 1]  # chosen style 0 mask path
                path_list = [
                    image_path,
                    target_path,
                    target1_path,
                    target2_path,
                    target3_path,
                ]
                style_list = [target_style, target1_style, target2_style, target3_style]

                pil_image_list = [pil_image(x) for x in path_list]
                image, target, target1, target2, target3 = pil_image_list
                # Apply dynamic augmentation to target mask
                target = self.dynamic_augmentation(target, target_style)
                pil_image_list = [image, target, target1, target2, target3]
                # Activate symmetric transformations for Data Augmentation, see method description of symmetric_augmentation
                if self.symmetric_transforms:
                    pil_image_list = self.symmetric_augmentation(pil_image_list)

                # Transform style variable to numerical values and Torch Tensor (f =0, m=1, p=2)
                style = torch.as_tensor(target_style)
                final_list = [to_tensor_trafo(x) for x in pil_image_list]
                image, target, target1, target2, target3 = final_list
                return image, target, [target1, target2, target3], style


if __name__ == "__main__":
    trafo = transforms.Compose([transforms.ToTensor()])
    data_path = "/home/kmze/style_probunet/data/isic3/isic256_3_style"
    dataset = ISIC_dynamic_augmentation(
        trafo, apply_symmetric_transforms=False, data_path=data_path, style=0
    )
    print(len(dataset))
