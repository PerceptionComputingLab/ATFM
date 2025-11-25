import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from tqdm import tqdm
import argparse
from types import SimpleNamespace
from datetime import datetime

import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim

from metadata_manager import *
from utils.utils import *
from utils.metrics import *
from models.GTR import GaussianTruncationRepresentation
from dataloaders import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--what",
    default="isic3_style_concat",
    help="Dataset to train on.",
)
parser.add_argument(
    "--lr",
    default=0.0001,
    type=float,
    help="Learning Rate for Training. Default is 0.0001",
)
parser.add_argument(
    "--rank",
    default=10,
    type=int,
    help="Rank for Covoriance decomposition. Default is 10",
)
parser.add_argument(
    "--epochs", default=200, type=int, help="Number of Epochs to train. Default is 200"
)
parser.add_argument(
    "--batchsize", default=6, type=int, help="Number of Samples per Batch. Default is 6"
)
parser.add_argument(
    "--weightdecay",
    default=1e-4,
    type=float,
    help="Parameter for Weight Decay. Default is 1e-4",
)
parser.add_argument(
    "--resume_epoch",
    default=0,
    type=int,
    help="Resume training at the specified epoch. Default is 0",
)
parser.add_argument(
    "--save_model",
    default=False,
    type=bool,
    help="Set True if checkpoints should be saved. Default is False",
)
parser.add_argument(
    "--testit",
    default=False,
    type=bool,
    help="Set True testing the trained model on the testset. Default is False",
)
parser.add_argument(
    "--test_treshold",
    default=0.5,
    type=float,
    help="Treshold for masking the logid/sigmoid predictions. Only use with --testit. Default is 0.5",
)
parser.add_argument(
    "--N", default=16, type=int, help="Number of Samples for GED Metric. Default is 16"
)
parser.add_argument(
    "--W",
    default=1,
    type=int,
    help="Set 0 to turn off Weights and Biases. Default is 1 (tracking)",
)
parser.add_argument(
    "--transfer",
    default="None",
    help="Activates transfer learning when given a model's name. Default is None (no transfer learning)",
)
parser.add_argument(
    '--log_dir',
    default='loggers',
    help='Store logs in this directory during training.',
    type=str
)
parser.add_argument(
    '--save_model_step',
    type=int,
    default=50
)
parser.add_argument(
    '--write',
    help='Saves the training logs',
    dest='write',
    action='store_true'
)
parser.set_defaults(
    write=True
)
parser.add_argument(
    "--num_filters",
    default=[32, 64, 128, 192],
    nargs="+",
    help="Number of filters per layer. Default is [32,64,128,192]",
    type=int,
)

def train(
    model,
    resume_epoch,
    epochs,
    opt,
    train_loader,
    val_loader,
    save_checkpoints,
    transfer_model,
    metadata,
    forward_passes,
    log_dir,
    save_model_step,
    write,
    W=True
):
    # Set device to Cuda if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Check if want to resume prior checkpoints
    if resume_epoch > 0:
        print(f"Resuming training on epoch {resume_epoch} ... \n")
        # Load Checkpoint
        checkpoint = torch.load(
            f"checkpoints/{meta.directory_name}/{model.name}/{resume_epoch}_checkpoint.pt"
        )
        # Inject checkpoint to model and optimizer
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    if transfer_model != "None":
        print(f"Continue Training on the model {transfer_model}...\n")
        transfer_dict = torch.load(
            f"saved_models/{meta.directory_name}/{transfer_model}.pt"
        )
        model = transfer_dict["model"]
        opt = transfer_dict["optimizer"]
        loss = transfer_dict["loss"]
    else:
        print(f"Training from scratch...\n")

    iterations = 0
    max_iou=-1
    for epoch in range(resume_epoch, epochs):  # may be error in range
        print(f"Epochs:{epoch+1}/{epochs} ... ")
        print("Training")
        sum_batch_loss = 0
        sum_batch_IoU = 0
        counter = 0
        model.train()
        for images, masks, _, _ in tqdm(train_loader):
        #for batch in tqdm(train_loader):
            #print(batch)
            counter += 1
            iterations += 1
            # Send tensors to Cuda
            images = images.to(device)
            masks = masks.to(device)

            # Set parameter gradients to None
            opt.zero_grad()
            # Forward pass
            logits, output_dict, logging_infos_of_that_step = model(
                images
            )  # outputs logits

            logit_distribution = output_dict["distribution"]
            # Treshold (default 0.5)
            pred_mask = torch.sigmoid(logits).ge(meta.masking_threshold)

            # Calculate Loss
            loss_function = GTRLossMCIntegral(
                num_mc_samples=20
            )
            loss = loss_function(logits, masks, logit_distribution)
            sum_batch_loss += float(loss)

            # Calculate IoU for this prediction
            batch_IoU = IoU(masks, pred_mask)
            sum_batch_IoU += float(batch_IoU)
            # Backward pass & weight update
            loss.backward()
            opt.step()

        if save_checkpoints == True and (epoch % save_model_step == 0):
            os.makedirs(
                f"saved_models/{meta.directory_name}/{model.name}", exist_ok=True
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": loss,
                },
                f"saved_models/{meta.directory_name}/{model.name}/{epoch+1}_checkpoint.pt",
            )

        """
        Evaluate on the validation set and track to see if overfitting happens
        """
        print("\nValidating")
        sum_IoU = 0
        sum_loss = 0
        counter = 0
        model.eval()
        with torch.no_grad():
            for images, masks, seg_dist, _ in tqdm(val_loader):
                counter += 1
                # Send tensors to cuda
                images = images.to(device)
                masks = masks.to(device)
                seg_dist = [x.to(device) for x in seg_dist]

                # IoU/Loss on Image Level
                # outputs logits (the mean of the distribution)
                logits, output_dict, _ = model(images)
                logit_distribution = output_dict["distribution"]
                pred_mask = (torch.sigmoid(logits)).ge(meta.masking_threshold)

                loss_function = GTRLossMCIntegral(
                    num_mc_samples=20
                )
                loss = loss_function(logits, masks, logit_distribution)
                sum_IoU += IoU(masks, pred_mask)
                #print(sum_IoU)
                sum_loss += loss

        if max_iou <= sum_IoU:
            max_iou=sum_IoU
            os.makedirs(f"saved_models/{meta.directory_name}/{model.name}", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": loss,
                },
                f"saved_models/{meta.directory_name}/{model.name}/best_model_IoU={max_iou/len(val_loader)}.pt",
            )

        print(f"Epoch {epoch+1} Finished")
        print(f"Loss Function:{sum_loss / len(val_loader)}")
        print(f"IoU:{sum_IoU / len(val_loader)}\n")
    print(f"Train finished! max_Iou={max_iou/len(val_loader)}")

if __name__ == "__main__":
    # Load parsed arguments from command lind
    args = parser.parse_args()

    what_task = args.what
    resume_epoch = args.resume_epoch
    epochs = args.epochs
    batch_size = args.batchsize
    learning_rate = args.lr
    weight_decay = args.weightdecay
    save_checkpoints = args.save_model
    forward_passes = args.N
    log_dir = args.log_dir
    rank = args.rank
    W = bool(args.W)  # Bool for turning off wandb tracking
    transfer_model = args.transfer
    num_filters = args.num_filters
    save_model_step = args.save_model_step
    write = args.write

    # Read in Metadata for the task chosen in command line
    meta_dict = get_meta(what_task)
    meta = SimpleNamespace(**meta_dict)

    # Hand some information about the current run to Wandb Panel
    config = dict(
        epochs=epochs,
        resumed_at=resume_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss="See Paper",
        architecture="GTR",
        dataset=meta.description,
        N_for_metrics=forward_passes,
        rank=rank,
        filter=num_filters,
    )

    training_run_name = (
        str(datetime.now())[:16]
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
    )

    print(f"Modelname: {training_run_name}")
    # Check for GPU

    if torch.cuda.is_available():
        print("\nThe model will be run on GPU.")
    else:
        print("\nNo GPU available!")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\nUsing the {meta.description} dataset.\n")

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda":
        torch.cuda.manual_seed(230)

    # Init a model
    GTR = GaussianTruncationRepresentation(
        name=training_run_name,
        num_channels=meta.channels,
        rank=rank,
        num_filters=num_filters,
        diagonal=False,
    ).to(device)
    # Count number of total parameters in the model and log
    pytorch_total_params = sum(p.numel() for p in GTR.parameters())

    # Note that Weight Decay and L2 Regularization are not the same (except for SGD) see paper: Hutter 2019 'Decoupled Weight Decay Regularization'
    # AdamW implements the correct weight decay as shown in their paper
    opt = optim.AdamW(GTR.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Fetch Dataloaders
    train_loader, _ = get_dataloader(
        task=what_task, split="train", batch_size=batch_size, shuffle=True, randomsplit=True
    )
    val_loader, _ = get_dataloader(
        task=what_task, split="val", batch_size=4, shuffle=False, randomsplit=False
    )
    # Empty GPU Cache
    torch.cuda.empty_cache()
    # Start Training
    train(
        GTR,
        resume_epoch,
        epochs,
        opt,
        train_loader,
        val_loader,
        save_checkpoints,
        transfer_model,
        meta,
        forward_passes,
        log_dir,
        save_model_step,
        write,
        W=W
    )

    print(f"Saved: {training_run_name} Data: {what_task} Model: GTR")
    # End Training Run
