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

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_hm_iou(preds_0, preds_1):
    # Preds are assumed to be binary (foreground/background), reshaping them for comparison
    preds_0 = preds_0.view(preds_0.shape[0], -1)
    preds_1 = preds_1.view(preds_1.shape[0], -1)

    # Calculate IoU matrix
    intersection = (preds_0 & preds_1).float().sum(dim=1)
    union = (preds_0 | preds_1).float().sum(dim=1)
    iou = intersection / union

    # Apply Hungarian Matching
    cost_matrix = 1 - iou
    _, col_indices = torch.linear_sum_assignment(cost_matrix)

    # Calculate HM-IoU
    hm_iou = (1 - iou[col_indices]).mean()

    return hm_iou

def Dice(target, predicted_mask):
    target = target.detach()
    predicted_mask = predicted_mask.detach()
    smooth = 1e-8
    true_p = (torch.logical_and(target == 1, predicted_mask == 1)).sum()
    # true_n = (torch.logical_and(target == 0, predicted_mask == 0)).sum().item() #Currently not needed for Dice
    Dice = (2. * true_p + smooth) / (torch.sum(target) + torch.sum(predicted_mask) + smooth)

    return Dice

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

    X_1 = x_t + predicted_cond_vector_field * ((1. - t).view(t.shape[0], 1, 1, 1))
    #print(X_1)
    X_1 = X_1.ge(0.5)

    target_cond_vector_field = target_flow.get_conditional_vector_field(x_0=x_0, x_1=x_1, t=t)

    Loss1 = F.mse_loss(predicted_cond_vector_field, target_cond_vector_field)
    Loss2 = Dice(X_1, x_1).mean()
    Loss3 = IoU(X_1, x_1)
    #Final Loss = Loss1 - alpha * Loss2
    #print(f"Loss1:{Loss1}, Loss2:{Loss2}")

    return Loss1, Loss2, Loss3


def train(
    prior_model,
    model,
    target_flow,
    dataloader,
    optimizer,
    device,
    num_epochs,
    resume_epoch,
    save_path
):
    """Trains conditional vector field model.

    Args:
        model: Model that predicts conditional vector field.
        target_flow: Object that models target conditional vector field.
        dataloader: Dataloader.
        optimizer: Optimizer.
        device: Target device.
        num_epochs: Num epochs to train.
        save_path: Where to save checkpoints and intermediate results.
    """
    #base_distribution = Normal(0, 1)

    for epoch in range(num_epochs - resume_epoch):
        Loss = 0
        Dice = 0
        IoU = 0
        prior_model.eval()
        model.train()
        print(f"Epochs:{epoch+1+resume_epoch}/{num_epochs} ... ")
        print("Training")

        for images, masks, _, _ in tqdm(dataloader):
            back_loss = 0
            optimizer.zero_grad()

            images = images.to(device1)
            logits, output_dict, _ = prior_model(images)
            logit_distribution = output_dict["distribution"]

            images = images.to(device)
            #print(images.shape)
            x_1 = masks[0]
            batch_size = x_1.shape[0]
            x_1 = x_1.to(device)
            x_2 = masks[1].to(device)
            x_3 = masks[2].to(device)
            x_4 = masks[3].to(device)

            x_0_1 = logit_distribution.sample()
            x_0_1 = torch.sigmoid(x_0_1).to(device)

            t = np.random.randint(low=0, high=1000, size=(batch_size,))
            t = 1.0 * t / 1000
            t = torch.tensor(t).float().to(device)

            loss1, dice1, iou1 = loss_fn(model=model, target_flow=target_flow, x_0=x_0_1, x_1=x_1, t=t, cond=images)
            loss2, dice2, iou2 = loss_fn(model=model, target_flow=target_flow, x_0=x_0_1, x_1=x_2, t=t, cond=images)
            loss3, dice3, iou3 = loss_fn(model=model, target_flow=target_flow, x_0=x_0_1, x_1=x_4, t=t, cond=images)
            loss4, dice4, iou4 = loss_fn(model=model, target_flow=target_flow, x_0=x_0_1, x_1=x_3, t=t, cond=images)
            loss = 0.25 * (loss1 + loss2 + loss3 + loss4)
            dice = 0.25 * (dice1 + dice2 + dice3 + dice4)
            iou = 0.25 * (iou1 + iou2 + iou3 +iou4)
            Loss += loss
            Dice += dice
            IoU += iou
            back_loss += loss + 0.001 * (1 - dice)

            #pbar.update(1)
            #pbar.set_postfix({"loss": f"{loss.item():.3f}"})

            back_loss.backward()
            optimizer.step()

        Loss = Loss / (len(dataloader))
        Dice = Dice / (len(dataloader))
        IoU = IoU / (len(dataloader))
        print(f'epoch:{epoch+1+resume_epoch}, Loss={Loss}, Dice={Dice}, IoU={IoU}')

        checkpoint = {
            "epoch": epoch+resume_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": Loss,
        }
        torch.save(checkpoint, save_path / f"{epoch+resume_epoch}-checkpoint.pth")

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--output_path", type=str, default="/root/FlowMatchingLIDC/DiceLossLogs/")
parser.add_argument("--resume_training", type=bool, default=False)
parser.add_argument("--resume_filepath", type=str, default=None)
if __name__ == "__main__":
    args = parser.parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = get_dataloader_2(
        task="LIDC", split="train", batch_size=args.batch_size, shuffle=True, randomsplit=True
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

    if args.resume_training == True:
        checkpoint = torch.load(args.resume_filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_epoch = checkpoint["epoch"]
        print(f"Resume Training on Epoch {resume_epoch}:")
    else:
        resume_epoch = 0
        print("Training from Scratch")

    # Setup conditional flow
    target_flow = OTConditionalFlow(sigma_min=0)

    os.makedirs(args.output_path, exist_ok=True)

    train(
        prior_model=GTR,
        model=model,
        target_flow=target_flow,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        resume_epoch=resume_epoch,
        save_path=Path(args.output_path),
    )
