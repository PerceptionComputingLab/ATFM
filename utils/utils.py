import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
from typing import Tuple

import torch
from torch._C import StringType, device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
import torch.distributions as td
import torch


"""
    Functions for padding the images to admissible input shapes of the U-Net
"""

def scale_array(array, min_value, max_value):     # 找到数组中的最小值和最大值
    min_val = np.min(array)
    max_val = np.max(array)          # 将数组的元素缩放到目标范围
    scaled_array = min_value + (max_value - min_value) * (array - min_val) / (max_val - min_val)
    return scaled_array

def data_normal(origin_data):
    d_min = origin_data.min()
    if d_min < 0:
        origin_data += torch.abs(d_min)
        d_min = origin_data.min()
    d_max = origin_data.max()
    dst = d_max - d_min
    norm_data = (origin_data - d_min).true_divide(dst)
    return norm_data

def pad_to_admissible_size(x, image_size, admissible_size):
    if isinstance(image_size, tuple):
        v_margin = admissible_size[0] - image_size[0]
        h_margin = admissible_size[1] - image_size[1]
        # if margin is not even, pad one more pixel on the right/up than on the left/down
        pads = (
            int(math.floor(h_margin / 2)),
            int(math.ceil(h_margin / 2)),
            int(math.floor(v_margin / 2)),
            int(math.ceil(v_margin / 2)),
        )
        out = F.pad(x, pads, "constant", 0)
    else:
        margin = admissible_size - image_size
        # if margin is not even, pad one more pixel on the right/up than on the left/down
        pads = (
            int(math.floor(margin / 2)),
            int(math.ceil(margin / 2)),
            int(math.floor(margin / 2)),
            int(math.ceil(margin / 2)),
        )
        out = F.pad(x, pads, "constant", 0)
    return out, pads


def pad_to_image_size(output, image_size, output_size):
    if isinstance(image_size, tuple):
        v_margin = image_size[0] - output_size[0]
        h_margin = image_size[1] - output_size[1]
        # if margin is not even, pad one more pixel on the right/up than on the left/down
        pads = (
            int(math.floor(h_margin / 2)),
            int(math.ceil(h_margin / 2)),
            int(math.floor(v_margin / 2)),
            int(math.ceil(v_margin / 2)),
        )
        out = F.pad(output, pads, "constant", 0)
    else:
        margin = image_size - output_size
        # if margin is not even, pad one more pixel on the right/up than on the left/down
        pads = (
            int(math.floor(margin / 2)),
            int(math.ceil(margin / 2)),
            int(math.floor(margin / 2)),
            int(math.ceil(margin / 2)),
        )
        out = F.pad(output, pads, "constant", 0)
    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2] : -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0] : -pad[1]]
    return x


def get_pads_to_original_size(image_size, output_size):
    """
    Args
        image_size: (int) size of input image
        output_size: (int) size of the image that comes out of the U-Net
    Returns:
        pads: (tuple) padding that is neccesary to get to the input size from the output_size
    """
    if isinstance(image_size, tuple):
        v_margin = image_size[0] - output_size[0]
        h_margin = image_size[1] - output_size[1]
        # if margin is not even, pad one more pixel on the right/up than on the left/down
        pads = (
            int(math.floor(h_margin / 2)),
            int(math.ceil(h_margin / 2)),
            int(math.floor(v_margin / 2)),
            int(math.ceil(v_margin / 2)),
        )
        return pads
    else:
        margin = image_size - output_size
        # if margin is not even, pad one more pixel on the right/up than on the left/down
        pads = (
            int(math.floor(margin / 2)),
            int(math.ceil(margin / 2)),
            int(math.floor(margin / 2)),
            int(math.ceil(margin / 2)),
        )

        return pads


"""
    Functions for plotting
"""


def make_image_grid(images, masks, predictions, required_padding):
    """
    Args
        X_batch: (torch.tensor BxCxHxW) Tensor contains the input images
        target_batch: (torch.tensor BxCxHxW) Tensor contains the target segmentations
        pred_batch: (torch.tensor BxCxHxW) Tensor contains the predictions

    Returns:
        grid: grid object to be plotted in wandb
    """
    grid_img = torchvision.utils.make_grid(images, len(images))
    grid_target = torchvision.utils.make_grid(
        F.pad(masks, required_padding, "constant", 0), len(masks)
    )
    grid_pred = torchvision.utils.make_grid(
        F.pad(predictions, required_padding, "constant", 0), len(predictions)
    )
    grid = torch.stack([grid_img, grid_target, grid_pred])
    grid = torchvision.utils.make_grid(grid, 1)

    return grid


def make_image_grid_with_heatmaps(images, masks, predictions, required_padding):
    """
    Args
        X_batch: (torch.tensor BxCxHxW) Tensor contains the input images
        target_batch: (torch.tensor BxCxHxW) Tensor contains the target segmentations
        pred_batch: (torch.tensor BxCxHxW) Tensor contains the predictions

    Returns:
        grid: grid object to be plotted in wandb
    """
    grid_img = torchvision.utils.make_grid(images, len(images))
    grid_target = torchvision.utils.make_grid(masks, len(masks))
    grid_pred = torchvision.utils.make_grid(
        F.pad(predictions, required_padding, "constant", 0), len(predictions)
    )
    grid = torch.stack([grid_img, grid_target, grid_pred])
    grid = torchvision.utils.make_grid(grid, 1)

    return grid


"""
Functions for Probabilistic U-Net based on implementation of https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/blob/master/utils.py
"""


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def save_mask_prediction_example(mask, pred, iter):
    plt.imshow(pred[0, :, :], cmap="Greys")
    plt.savefig("images/" + str(iter) + "_prediction.png")
    plt.imshow(mask[0, :, :], cmap="Greys")
    plt.savefig("images/" + str(iter) + "_mask.png")




class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
    ):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, logits: torch.tensor, target: torch.tensor, **kwargs):
        return super().forward(logits, target)


class GTRLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 1):
        super().__init__()
        self.num_mc_samples = num_mc_samples

    @staticmethod
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

    def forward(self, logits, target, distribution, **kwargs):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        logit_sample = self.fixed_re_parametrization_trick(
            distribution, self.num_mc_samples
        )
        target = target.unsqueeze(1)
        target = target.expand((self.num_mc_samples,) + target.shape)
        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        target = target.reshape((flat_size, -1))
        target = target.unsqueeze(1)
        log_prob = -F.binary_cross_entropy_with_logits(
            logit_sample, target, reduction="none"
        ).view((self.num_mc_samples, batch_size, -1))

        loglikelihood = torch.mean(
            torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0)
            - math.log(self.num_mc_samples)
        )
        loss = -loglikelihood

        return loss


class ReshapedDistribution(td.Distribution):
    def __init__(
        self,
        base_distribution: td.Distribution,
        new_event_shape: Tuple[int, ...],
        validate_args=None,
    ):
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=new_event_shape,
            validate_args=validate_args,
        )
        self.base_distribution = base_distribution
        self.new_shape = base_distribution.batch_shape + new_event_shape

    @property
    def support(self):
        return self.base_distribution.support

    @property
    def arg_constraints(self):
        return self.base_distribution.arg_constraints()

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape).view(
            sample_shape + self.new_shape
        )

    def log_prob(self, value):
        return self.base_distribution.log_prob(value.view(self.batch_shape + (-1,)))

    def entropy(self):
        return self.base_distribution.entropy()
