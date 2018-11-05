import torch.nn as nn
import pandas as pd
import numpy as np
import datetime
import GPUtil
import torch
import time
import PIL
import os

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from architectures import Model, NeuralNet
from torchvision import transforms
from torch import optim

###

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

###

dataset = CellDataset("./data/BalancedData.npy", "./data/BalancedLabels.npy", transformations = transforms.ToTensor())

###

NN = NeuralNet();

NN = nn.DataParallel(NN).cuda()

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

b_size = 1000

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

CAE = Model(z_dim = latent_dim);

CAE = nn.DataParallel(CAE).cuda()

CAE.load_state_dict(torch.load("./models/BroadCAE.pth"));

CAE.eval()

###

n_epochs = 200

optimiser = optim.SGD(NN.parameters(), lr = 1e-1, weight_decay = 1e-3)

best_val_loss = np.infty

###

train_names = ["Epoch", "Train Loss (Ave)"]

train_stats = pd.DataFrame(columns = train_names)

###

valid_names = ["Epoch", "Valid Loss (Ave)"]

valid_stats = pd.DataFrame(columns = valid_names)

###

criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    print("\n")
    print("Epoch [%d/%d]" %(epoch + 1, n_epochs))
    print("=" * 50, "\n")
    for phase in ["train", "valid"]:
        if phase == "train":
            NN.train()
        else:
            NN.eval()
        ###
        start = time.time()
        ###
        running_loss = 0.0
        ###
        for batch_index, batch in enumerate(data_loaders[phase]):
            x = batch["image"].cuda()
            target = batch["class"].cuda()
            ###
            latent_z, _ = CAE(x)
            ###
            optimiser.zero_grad()
            ###
            with torch.set_grad_enabled(phase == "train"):
                latent_z = latent_z.cuda()
                output = NN(latent_z)
                loss = criterion(output, target)
                if phase == "train":
                    loss.backward()
                    optimiser.step()
            running_loss = running_loss + loss.item()
        ### End of epoch
        delta = (time.time() - start)/60
        n_steps_per_epoch = batch_index + 1
        ave_epoch_loss = running_loss/n_steps_per_epoch
        if phase == "train":
            print("Time: %.3f Minutes | Train Loss (Ave): %.3f" %(delta, ave_epoch_loss))
            row = pd.Series([epoch + 1, ave_epoch_loss], index = train_names)
            train_stats = train_stats.append(row, ignore_index = True)
        else:
            print("Time: %.3f Minutes | Valid Loss (Ave): %.3f" %(delta, ave_epoch_loss))
            row = pd.Series([epoch + 1, ave_epoch_loss], index = valid_names)
            valid_stats = valid_stats.append(row, ignore_index = True)
            if ave_epoch_loss < best_val_loss:
                print("\nValid Loss Decreased, Saving New Best Model...")
                #print("=" * 50)
                best_val_loss = ave_epoch_loss
                path = "./models/PretrainedNeuralNework.pth"
                torch.save(NN.state_dict(), path)

###

print("Saving Logs...")

train_stats.to_csv("./logs/TrainStatsNN.csv", index = False)
valid_stats.to_csv("./logs/ValidStatsNN.csv", index = False)
