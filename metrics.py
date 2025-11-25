from itertools import product

import torch
import torch.nn as nn
import numpy as np


def IoU(target, predicted_mask):
    """
    Args:
        target: (torch.tensor (batchxCxHxW)) Binary Target Segmentation from training set
        predicted_mask: (torch.tensor (batchxCxHxW)) Predicted Segmentation Mask

    Returns:
        IoU: (Float) Average IoUs over Batch
    """

    target = target.detach()
    predicted_mask = predicted_mask.detach()
    smooth = 1e-8
    true_p = (torch.logical_and(target == 1, predicted_mask == 1)).sum()
    # true_n = (torch.logical_and(target == 0, predicted_mask == 0)).sum().item() #Currently not needed for IoU
    false_p = (torch.logical_and(target == 0, predicted_mask == 1)).sum()
    false_n = (torch.logical_and(target == 1, predicted_mask == 0)).sum()
    sample_IoU = (smooth+float(true_p))/(float(true_p) +
                                         float(false_p)+float(false_n)+smooth)

    return sample_IoU


"""
    Metrics for the similarity of Distributions over Segmentations
"""


# Aasas GED functions
def aasa_IoU(mask0, mask1):
    if (np.sum(mask0.flatten()) == 0 & np.sum(mask1.flatten() == 0)):
        IoU = 0
    else:
        I = np.sum(np.multiply(mask0, mask1).flatten())
        U = np.sum((mask0+mask1 > 0).flatten())
        IoU = I/U
    return IoU


def aasa_IoU_dist(mask0, mask1):
    return 1 - aasa_IoU(mask0, mask1)


def aasa_GED(annotations, predictions):
    idx_predictions = range(predictions.shape[0])  # a
    idx_annotations = range(annotations.shape[0])  # b

    e_ab = np.stack([1 - aasa_IoU(predictions[i, :, :], annotations[j, :, :])
                     for i, j in product(idx_predictions, idx_annotations)]).mean(axis=0)

    e_aa = np.stack([1. - aasa_IoU(predictions[i, :, :], predictions[j, :, :])
                     for i, j in product(idx_predictions, idx_predictions)]).mean(axis=0)

    e_bb = np.stack([1 - aasa_IoU(annotations[i, :, :], annotations[j, :, :])
                     for i, j in product(idx_annotations, idx_annotations)]).mean(axis=0)
    ged = 2 * e_ab - e_aa - e_bb
    return ged


def ged_list(annotations, predictions, threshold=0.5):
    ged_list = []
    # Iterate over each image in the testset
    for n in range(annotations.shape[0]):
        ged_list.append(
            aasa_GED(annotations[n, :, :, :], predictions[n, :, :, :]))  # Calculate GED between available annotations and predictions for each image
    return ged_list


# 3 functions taken from Steffens code, citation!!

# Steffen

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def iou(a, b, eps=1e-6):
    """Intersection over Union operation
    Args:
        a: Long-Tensors of shape Bx1xHxW of segmentation maps
        b: Long-Tensors of shape Bx1xHxW of segmentation maps
    Returns:
        List of IoUs per sample in batch, length B
    """
    if a.max() > 1:
        raise Exception('IoU is only implemented for 2 classes')

    #intersection = (a & b).float().sum(dim=(1, 2, 3))
    intersection = (torch.logical_and(a, b)).float().sum(dim=(1, 2, 3))

    #union = (a | b).float().sum(dim=(1, 2, 3))
    union = (torch.logical_or(a, b)).float().sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou

# Steffen


def ged(samples_a, samples_b):
    """Calculates the Generalized Energy Distance.
    Args:
        samples_a: List of Long-Tensors of shape Bx1xHxW of segmentation maps
        samples_b: List of Long-Tensors of shape Bx1xHxW of segmentation maps
    Returns:
        ged, sample_diversity: Lists of distances, length B
    """
    idx_a = range(len(samples_a))
    idx_b = range(len(samples_b))

    e_ab = torch.stack([1 - iou(samples_a[i], samples_b[j])
                        for i, j in product(idx_a, idx_b)]).mean(dim=0)

    e_aa = torch.stack([1. - iou(samples_a[i], samples_a[j])
                        for i, j in product(idx_a, idx_a)]).mean(dim=0)

    e_bb = torch.stack([1 - iou(samples_b[i], samples_b[j])
                        for i, j in product(idx_b, idx_b)]).mean(dim=0)

    ged = 2 * e_ab - e_aa - e_bb
    sample_diversity = e_aa

    return ged, sample_diversity

# Steffen


def generalized_energy_distance(model, x, ys, which_model, sample_count=16, treshold=0.5):
    """Calculates the Generalized Energy Distance.
    Args:
        model: the model, implementing a sample_prediction method
        x: The images, Float-Tensor of shape BxCxHxW
        ys: The annotations, List of Long-Tensors of shape Bx1xHxW of segmentation maps
        sample_count: the amount of samples to draw from the model
    Returns:
        ged, sample_diversity: Lists of distances, length B
    """

    if which_model == "mcdropout":
        y_hats = [(torch.sigmoid(model.forward(x))).ge(treshold)
                  for _ in range(sample_count)]
        ys = [ys[torch.randint(len(ys), ())] for _ in range(sample_count)]
    if which_model == "probunet":
        # No need to hand over the images, since the probabilistic Unet class has already build up the prior and the UNet prediction when being called by model.forward(images,masks, training=False)
        y_hats = [(torch.sigmoid(model.sample(testing=True))).ge(treshold)
                  for _ in range(sample_count)]
        ys = [ys[torch.randint(len(ys), ())] for _ in range(sample_count)]
    if which_model == "gtr":
        # in this case we do not hand over the model but the output_dict! So model = output_dict here. That is becaues we already have generated the logit distribution for that batch. Otherwise we would calculate it again.
        # this is actually output_dict["distribution"] for the given batch of images
        logit_distribution = model["distribution"]
        y_hats = [(torch.sigmoid(logit_distribution.sample())).ge(treshold)
                  for _ in range(sample_count)]
        ys = [ys[torch.randint(len(ys), ())] for _ in range(sample_count)]
    return ged(y_hats, ys)


# Steffen
def entropy(p):
    """
    Calculates the entropy (uncertainty) of p
    Args:
        p (Tensor BxCxHxW): probability per class
    Returns:
        Tensor Bx1xHxW
    """
    mask = p > 0.00001
    h = torch.zeros_like(p)
    h[mask] = torch.log2(1 / p[mask])
    H = torch.sum(p * h, dim=1, keepdim=True)
    return H


def iou_from_mean_spu(model, test_loader, threshold=0.5):
    sum_IoU = 0
    sum_loss = 0
    counter = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, masks, seg_dist, style_label in test_loader:
            counter += 1
            # Send tensors to Cuda
            images = images.to(device)
            masks = masks.to(device)
            style_label = style_label.to(device)

            # IoU/Loss on Image Level
            model.forward(images, masks, style=style_label,
                          training=False)  # outputs logits
            logits = model.sample_the_mean()
            pred_mask = (torch.sigmoid(logits)).ge(threshold)
            # Calculate Loss
            loss_function = nn.BCEWithLogitsLoss()
            sum_IoU += IoU(masks, pred_mask)
            sum_loss += loss_function(logits, masks)
    return sum_IoU/len(test_loader)


def variance_of_iou_spu(model, test_loader):
    list_IoU = []
    sum_loss = 0
    counter = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, masks, seg_dist, style_label in test_loader:
            counter += 1
            # Send tensors to Cuda
            images = images.to(device)
            masks = masks.to(device)
            style_label = style_label.to(device)

            # IoU/Loss on Image Level
            model.forward(images, masks, style=style_label,
                          training=False)  # outputs logits

            logit_list = []
            for i in range(100):
                logits = model.sample(testing=True)
                logit_list.append(logits)
            logits = torch.stack(logit_list)
            mean_logits = torch.mean(logits, dim=0)
            pred_mask = (torch.sigmoid(mean_logits)).ge(0.5)
            # Calculate Loss
            loss_function = nn.BCEWithLogitsLoss()
            list_IoU.append(IoU(masks, pred_mask))
            sum_loss += loss_function(mean_logits, masks)
    return np.std(list_IoU)


def iou_from_mean_pu(model, test_loader, threshold=0.5):
    sum_IoU = 0
    sum_loss = 0
    counter = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, masks, seg_dist, style_label in test_loader:
            counter += 1
            # Send tensors to Cuda
            images = images.to(device)
            masks = masks.to(device)
            style_label = style_label.to(device)

            # IoU/Loss on Image Level
            model.forward(images, masks,
                          training=False)  # outputs logits
            logits = model.sample_the_mean()
            pred_mask = (torch.sigmoid(logits)).ge(threshold)
            # Calculate Loss
            loss_function = nn.BCEWithLogitsLoss()
            sum_IoU += IoU(masks, pred_mask)
            sum_loss += loss_function(logits, masks)
    return sum_IoU/len(test_loader)


def variance_of_iou_pu(model, test_loader):
    list_IoU = []
    sum_loss = 0
    counter = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, masks, seg_dist, style_label in test_loader:
            counter += 1
            # Send tensors to Cuda
            images = images.to(device)
            masks = masks.to(device)
            style_label = style_label.to(device)

            # IoU/Loss on Image Level
            model.forward(images, masks,
                          training=False)  # outputs logits
            logit_list = []
            for i in range(100):
                logits = model.sample(testing=True)
                logit_list.append(logits)
            logits = torch.stack(logit_list)
            mean_logits = torch.mean(logits, dim=0)
            pred_mask = (torch.sigmoid(mean_logits)).ge(0.5)
            # Calculate Loss
            loss_function = nn.BCEWithLogitsLoss()
            list_IoU.append(IoU(masks, pred_mask))
            sum_loss += loss_function(mean_logits, masks)
    return np.std(list_IoU)


# Volume Calculations
def volume_difference_mean_pred(annotations, predictions, threshold=0.5):
    assert annotations.shape[0] == predictions.shape[0]
    differences = []
    for i in range(annotations.shape[0]):  # for each image

        vol_annotation = vol(annotations[i])
        vol_mean_prediction = vol(np.mean(predictions[i], axis=1) > threshold)
        diff = vol_mean_prediction - vol_annotation
        differences.append(diff)
    return np.array(differences)


def volume_difference(annotations, predictions, threshold=0.5):
    assert annotations.shape[0] == predictions.shape[0]
    differences = []
    for i in range(annotations.shape[0]):  # for each image
        differences_per_image = []
        vol_annotation = vol(annotations[i])
        for j in range(predictions.shape[1]):  # for each prediction
            # volume of prediction j of image i
            vol_prediction = vol(predictions[i, j] > threshold)
            diff = vol_prediction - vol_annotation
            differences_per_image.append(diff)
        differences.append(differences_per_image)
    differences = np.array(differences)
    return differences


def vol(segmentation):
    vol = np.sum(segmentation.flatten())
    return vol


def vol_set(segmentation_set):
    # Input N_ann x dim1 x dim2
    N_ann = segmentation_set.shape[0]
    vols = []
    for n in range(N_ann):
        vols.append(vol(segmentation_set[n, :, :]))
    return np.array(vols)


if __name__ == '__main__':
    predictions = np.random.rand(22, 100, 256, 256)
    annotations = np.random.rand(22, 1, 1, 256, 256)
    vol_diff = volume_difference(annotations, predictions)
    print(vol_diff.shape)
