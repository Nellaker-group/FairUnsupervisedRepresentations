import torch.nn as nn
import pandas as pd
import numpy as np
import datetime
import warnings
import random
import GPUtil
import torch
import time
import PIL
import os

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from architectures import Model, NeuralNet
from torchvision import transforms
from utils import CellDataset
from torch import optim

###

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

###

dataset = CellDataset("./data/BalancedData.npy", "./data/BalancedLabels.npy", transformations = transforms.ToTensor())

###

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

CAE = Model(z_dim = 2048);

CAE = nn.DataParallel(CAE).cuda();

CAE.load_state_dict(torch.load("./models/BroadCAE.pth"));

###

NN = NeuralNet();

NN = nn.DataParallel(NN).cuda();

NN.load_state_dict(torch.load("./models/PretrainedNeuralNetwork.pth"));

###

def get_random_batch(loader):
    for batch in loader:
        pass
    b = batch["image"].cuda()
    t = batch["class"].cuda()
    return b, t

###

adv_optimiser = optim.Adam(NN.parameters(), lr = 1e-3, amsgrad = True)

CAE_optimiser = optim.Adam(CAE.parameters(), lr = 1e-3, amsgrad = True)

###

adv_criterion = nn.CrossEntropyLoss()

###

Lambda = 50

###

train_names = ["Epoch", "Neural Network Loss (Ave)", "CAE Loss (1 Random Batch)", "Total Loss", "Lambda"]

train_logs = pd.DataFrame(columns = train_names)

###

n_epochs = 200

for epoch in range(n_epochs):
    print("\n")
    print("Epoch [%d/%d]" %(epoch + 1, n_epochs))
    print("=" * 50, "\n")
    for phase in ["train"]:
        ###################################################################
        # Step 1: Train adversarial for a single epoch and keep CAE fixed
        ###################################################################
        if phase == "train":
            CAE.eval()
            NN.train()
        else:
            CAE.eval()
            NN.eval()
        start = time.time()
        adv_running_loss = 0.0
        for batch_index, batch in enumerate(data_loaders[phase]):
            x = batch["image"].cuda()
            target = batch["class"].cuda()
            latent_z, _ = CAE(x)
            adv_optimiser.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                output = NN(latent_z)
                adv_loss = adv_criterion(output, target)
                if phase == "train":
                    adv_loss.backward()
                    adv_optimiser.step()
            adv_running_loss = adv_running_loss + adv_loss.item()
        ### End of epoch for adversarial classifier
        delta = (time.time() - start)/60
        n_steps_per_epoch = batch_index + 1
        adv_ave_epoch_loss = adv_running_loss/n_steps_per_epoch
        if phase == "train":
            print("1. Completed Training for Neural Network (Adversary)\n")
            print("Train Loss (Ave): %.3f | Time (Min): %.3f" %(adv_ave_epoch_loss, delta))
            row = [epoch + 1, adv_ave_epoch_loss]
        else:
            print("Valid Loss for Adversary (Ave): %.3f | Time (Min): %.3f" %(adv_ave_epoch_loss, delta))
            row = [epoch + 1, adv_ave_epoch_loss]
        ####################################################################
        # Step 2: Train CAE on signle random batch whilst keeping adversary fixed
        ####################################################################
        if phase == "train":
            NN.eval()
            CAE.train()
            random_batch, random_labels = get_random_batch(train_loader)
        else:
            NN.eval()
            CAE.eval()
            random_batch, random_labels = get_random_batch(valid_loader)
        start = time.time()
        x = random_batch
        target = random_labels
        CAE_optimiser.zero_grad()
        with torch.set_grad_enabled(phase == "train"):
            latent_z, x_recon = CAE(x)
            CAE_loss = (x_recon - x).pow(2).mean()
            ###
            output = NN(latent_z)
            NN_loss = adv_criterion(output, target)
            ###
            CAE_loss = CAE_loss.cuda()
            combined_loss = CAE_loss - Lambda * NN_loss
            if phase == "train":
                combined_loss.backward()
                CAE_optimiser.step()
        delta = (time.time() - start)/60
        if phase == "train":
            print("\n2. Completed Training of CAE (1 Random Batch)\n")
            print("CAE Loss: %.3f\n" %(CAE_loss))
            print("*" * 50)
            print("\nCombined Train Loss: %.3f | Time (Min): %.3f" %(combined_loss, delta))
            row = row + [CAE_loss.item(), combined_loss.item(), Lambda]
            row = pd.Series(row, index = train_names)
            train_logs = train_logs.append(row, ignore_index = True)
        else:
            print("Combined Valid Loss (One Random Batch): %.3f | Time (Min): %.3f" %(combined_loss, delta))

###

train_logs.to_pickle("./logs/AdversarialTrainingLogs.pkl")

###

path = "./models/FairBroadCAE.pth"

torch.save(CAE.state_dict(), path)
