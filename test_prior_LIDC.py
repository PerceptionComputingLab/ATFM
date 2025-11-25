import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from flows import ConditionalFlow, OTConditionalFlow
from inference_LIDC import infer
from models.Condition_Unet import Unet
from dataloaders import *
from metrics import *
from models.GTR import GaussianTruncationRepresentation
from scipy.optimize import linear_sum_assignment

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_hm_iou(Pred, Masks):
    lcm = np.lcm(len(Pred), len(Masks))
    len1 = len(Pred)
    len2 = len(Masks)
    for i in range((lcm // len1) - 1):
        for j in range(len1):
            Pred.append(Pred[j])
    for i in range((lcm // len2) - 1):
        for j in range(len2):
            Masks.append(Masks[j])
    #print(len(Pred))
    #print(len(Masks))
    cost_matrix = np.zeros((lcm, lcm))
    for i in range(lcm):
        for j in range(lcm):
            cost_matrix[i][j] = 1 - IoUIoU(Pred[i], Masks[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    HM_IoU = np.mean([(1 - cost_matrix[i][j]) for i, j in zip(row_ind, col_ind)])
    return HM_IoU

def compute_max_Dice(Pred, Masks):
    len1 = len(Pred)
    len2 = len(Masks)
    mx_D = 0
    for j in range(len2):
        mx = 0
        for i in range(len1):
            Diceij = Dice(Pred[i], Masks[j])
            if Diceij > mx:
                mx = Diceij
        mx_D = mx_D + mx
    return mx_D / len2

def Dice(target, predicted_mask):
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
    sample_IoU = (smooth + float(true_p) + float(true_p)) / (float(true_p) + float(true_p) +
                                         float(false_p) + float(false_n) + smooth)

    return sample_IoU

def IoUIoU(target, predicted_mask):
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

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def loss_fn(
    model,
    target_flow,
    x_0,
    x_1,
    t,
    cond
):
    """Counts MSE loss between predicted and target conditional vector fields.

    Check eq. (9) in paper: https://arxiv.org/abs/2210.02747

    Args:
        model: Model that predicts conditional vector field.
        target_flow: Object that models target conditional vector field.
        x_0: Samples from base distribution, [batch_size, 1, h, w].
        x_1: Samples from target distribution, [batch_size, 1, h, w].
        t: Time samples, [batch_size].

    Returns:
        MSE loss between predicted and target conditional vector fields.
    """
    x_t = target_flow.sample_p_t(x_0=x_0, x_1=x_1, t=t).to(device)
    predicted_cond_vector_field = model(x_t, t, cond)

    target_cond_vector_field = target_flow.get_conditional_vector_field(x_0=x_0, x_1=x_1, t=t)

    return F.mse_loss(predicted_cond_vector_field, target_cond_vector_field)


def euler_sampler(model, sampling_times, cond, x, BS):
    """The probability flow ODE sampler with simple Euler discretization.

    Args:
      model: A velocity model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
        dt = 1./sampling_times
        eps = 1e-3
        for i in range(sampling_times):
            num_t = i / sampling_times
            t = torch.ones(BS, device=device) * num_t
            pred = model(x, t, cond)
            #print(f'pmin:{torch.min(pred)}, pmax:{torch.max(pred)}, xmin:{torch.min(x)}, xmax:{torch.max(x)}')
            sigma_t = 0
            pred_sigma = pred
            x = x.detach().clone() + pred_sigma * dt
    return x


def test(
    prior_model,
    model,
    target_flow,
    testloader,
    optimizer,
    num_epochs,
    save_path
):
    """Infers conditional vector field model.

    Args:
        model: Model that predicts conditional vector field.
        target_flow: Object that models target conditional vector field.
        dataloader: Dataloader.
        optimizer: Optimizer.
        device: Target device.
        num_epochs: Num epochs to train.
        save_path: Where to save checkpoints and intermediate results.
    """
    time_distribution = Uniform(0, 1)
    base_distribution = Normal(0, 1)

    GED = 0.0
    IoU = 0.0
    mxD = 0.0
    print("\nTesting...")
    prior_model.eval()
    model.eval()
    tcnt = 0
    for images, masks, _, _ in tqdm(testloader):
        tcnt += 1
        BS = images.shape[0]
        Masks = []
        Pred = []

        images = images.to(device1)
        logits, output_dict, _ = prior_model(images)
        logit_distribution = output_dict["distribution"]

        images = images.to(device)
        for i in range(4):
            Masks.append(masks[i].to(device).int())

        num_samples = 1
        for i in range(num_samples):
            x0 = logit_distribution.sample()
            x0 = torch.sigmoid(x0).to(device)
            result = euler_sampler(model=model, sampling_times=25, cond=images, x=x0, BS=BS)
            To = result.ge(0.5).int()
            Pred.append(To.int())


        #GED1, _ = ged(Masks, Pred)
        #GED += GED1
        #print(f'Current Mean GED = {GED/tcnt}')

        #now = compute_hm_iou(Pred, Masks)
        #IoU += now
        #print(f'This HM-IoU = {now}, Current Mean HM-IoU = {IoU/tcnt}')

        now2 = compute_max_Dice(Pred, Masks)
        mxD += now2
        print(f'This maxDice = {now2}, Current Mean maxDice = {mxD/tcnt}')



parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--output_path", type=str, default="results_LIDC")
parser.add_argument("--resume_training", type=bool, default=False)
parser.add_argument("--resume_filepath", type=str, default=None)
if __name__ == "__main__":
    args = parser.parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader, _ = get_dataloader_2(
        task="LIDC", split="test", batch_size=1, shuffle=False, splitratio=[0.8, 0.0, 0.2], randomsplit=False
    )

    GTR = GaussianTruncationRepresentation(
        name='GTR',
        num_channels=1,
        rank=10,
        num_filters=[32, 64, 128, 192],
        diagonal=False,
    ).to(device1)
    checkpoint1 = torch.load("/root/ATFM/saved_models/LIDC/GTR_LIDC.pt")
    GTR.load_state_dict(checkpoint1["model_state_dict"])

    # Setup model and optimizer
    model = Unet(
        channels=1,
        dim_mults=(1, 2, 4),
        dim=args.image_size,
        resnet_block_groups=1,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    checkpoint = torch.load("/root/ATFM/saved_models/LIDC/SFM-for-ATFM-LIDC.pth")
    print(f'Testing: 1e-3, sampling_time=25, model=199')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    target_flow = OTConditionalFlow(sigma_min=0)

    os.makedirs(args.output_path, exist_ok=True)

    test(
        prior_model=GTR,
        model=model,
        target_flow=target_flow,
        testloader=test_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_path=Path(args.output_path),
    )
