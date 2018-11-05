import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import datetime
import torch
import time
import PIL
import os

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from architectures import Model
from utils import CellDataset
from torch import optim

###

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

### Load balanced data set comprosed of 74 635 images per batch, i.e. BalancedData.npy is of shape (746350, 128, 128, 3).

dataset = CellDataset("./data/BalancedData.npy", "./data/Balanced.npy",
					  transformations =  transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomAffine(degrees = 0,shear = 0.05),
                                          transforms.ColorJitter(brightness = 0.03, contrast = 0.05, saturation = 0.03, hue = 0.03),
                                          transforms.ToTensor()]))

### Split data between training, validation, and test sets.

n_images = dataset.data.shape[0]

indices = list(range(n_images))

non_train_ratio = 0.1

non_train_size = int(np.floor(non_train_ratio * n_images))

###

np.random.seed(42)

np.random.shuffle(indices)

train_indices, non_train_indices = indices[non_train_size :], indices[: non_train_size]

###

train_sampler = SubsetRandomSampler(train_indices)

###

valid_test_ratio = int(np.floor(0.05 * n_images))

test_indices, valid_indices = non_train_indices[valid_test_ratio :], non_train_indices[: valid_test_ratio]

test_sampler, valid_sampler = SubsetRandomSampler(test_indices), SubsetRandomSampler(valid_indices)


###

b_size = 1024

train_loader = DataLoader(dataset, batch_size = b_size, sampler = train_sampler, num_workers = 40)

valid_loader = DataLoader(dataset, batch_size = b_size, sampler = valid_sampler, num_workers = 40)

test_loader = DataLoader(dataset, batch_size = b_size, sampler = test_sampler, num_workers = 40)

###

print("\n")
print("Data Statistics")
print("=" * 50)
print("Number of Training Examples: %d/%d" %(len(train_loader.sampler), len(indices)))
print("Number of Validation Examples: %d/%d" %(len(valid_loader.sampler), len(indices)))
print("Number of Test Examples: %d/%d" %(len(test_loader.sampler), len(indices)))
print("=" * 50, "\n")

data_loaders = {"train": train_loader, "valid": valid_loader}

###

latent_dim = 2048

CAE = Model(z_dim = latent_dim)

CAE = nn.DataParallel(CAE).cuda()

### Training

n_epochs = 200

optimiser = optim.Adam(CAE.parameters(), lr = 0.0001, weight_decay = 1e-6, amsgrad = True)

###

best_val_loss = np.infty

### Data frames for logging traininng stats.

train_names = ["Epoch", "Train Loss (Ave)"]

train_stats = pd.DataFrame(columns = train_names)

valid_names = ["Epoch", "Valid Loss (Ave)"]

valid_stats = pd.DataFrame(columns = valid_names)

###

criterion = nn.MSELoss()

for epoch in range(n_epochs):
    print("\n")
    print("Epoch: [%d/%d]" %(epoch + 1, n_epochs))
    print("=" * 50, "\n")
    for phase in ["train", "valid"]:
        if phase == "train":
            CAE.train()
        else:
            CAE.eval()
        ###
        running_loss = 0.0
        ###
        start = time.time()
        for batch_index, batch in enumerate(data_loaders[phase]):
            x = batch["image"].cuda()
            ###
            optimiser.zero_grad()
            ###
            with torch.set_grad_enabled(phase == "train"):
                z, x_recon = CAE(x)
                loss = criterion(x_recon,x)
                ###
                if phase == "train":
                    loss.backward()
                    optimiser.step()
            ### Running stats
            running_loss = running_loss + loss.item()
        ### End of epoch stats
        delta = (time.time() - start)/60
        n_steps_per_epoch = batch_index + 1
        ave_epoch_loss = running_loss/n_steps_per_epoch
        ###
        if phase == "train":
            print("Time: %.3f Minutes | Train Loss (Ave): %.4f" %(delta, ave_epoch_loss))
            row = pd.Series([epoch + 1, ave_epoch_loss], index = train_names)
            train_stats = train_stats.append(row, ignore_index = True)
        else:
            print("Time: %.3f Minutes | Valid Loss (Ave): %.4f" %(delta, ave_epoch_loss))
            row = pd.Series([epoch + 1, ave_epoch_loss], index = valid_names)
            valid_stats = valid_stats.append(row, ignore_index = True)
            ###
            if ave_epoch_loss < best_val_loss:
                print("\nValid Loss Decreased, Saving New Best Model...")
                #print("=" * 50)
                best_val_loss = ave_epoch_loss
                path = "./models/BroadCAE.pth"
                torch.save(CAE.state_dict(), path)

print("\n")
print("Saving Logs")
print("=" * 50)
print("\n")

train_stats.to_csv("./logs/TrainStats.csv", index = False)
valid_stats.to_csv("./logs/ValidStats.csv", index = False)

