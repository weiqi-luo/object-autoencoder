import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
import signal
import click
# torch
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# my own functions
from utils import *
from preprocess import ObjectPoseDataset


@click.command()
@click.argument('filename', type=click.Path())
@click.argument('suffix', required=False)
@click.option('--augment', is_flag=True)


def main(filename,suffix,augment):
    """
    filename: path to the input checkpoints file in directory ../data/models  
    [suffix]: test (only keyword test is supported)  
              if it is set then use test set for testing, otherwise use training set)  
    --augment: augment the test set
    """
    # Hyper Parameters
    BATCH_SIZE = 64
    N_TEST_IMG = 10


    #! load model and dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder, dataloader, info = load_model_dataset(filename,suffix,BATCH_SIZE,augment)


    #! Visualize the testing process
    # initialize figure
    f, a = plt.subplots(3, N_TEST_IMG, figsize=(5, 2))
    # original data (first row) for viewing
    for i,data in enumerate(dataloader):
        sample_plot = data['image'][:N_TEST_IMG]
        sample_aug = data['image_aug'][:N_TEST_IMG]
    for i in range(N_TEST_IMG):
        imshow(sample_plot[i],a[1][i])
        imshow(sample_aug[i],a[0][i])
    # plotting decoded image (second row)
    _, decoded_data = autoencoder(sample_aug.to(device))
    for i in range(N_TEST_IMG):
        imshow(decoded_data[i],a[2][i]) 

    #! Compute the testing MSE loss
    sum_top = 0; sum_bottom = 0
    for step, data in enumerate(dataloader):
        inputImg = data['image_aug'].to(device)
        encoded, decoded = autoencoder(inputImg)               
        sum_top += (inputImg-decoded).pow(2).sum().detach().to('cpu').numpy()
        sum_bottom += inputImg.numel()
    loss = sum_top/sum_bottom

    print('Finished Testing. Training loss is {}, testing loss is {}'.format(info['loss'],loss))
    plt.show()

if __name__ == '__main__':
    main()
