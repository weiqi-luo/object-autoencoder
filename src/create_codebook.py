import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
import signal
import click
from tqdm import tqdm
import pickle
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
def create_codebook(filename,suffix):
    """
    filename: path to the input checkpoints file in directory ../data/models  
    [suffix]: test (only keyword test is supported, if set then create codebook for test set)
    """
    # Hyper Parameters
    BATCH_SIZE = 64
    N_TEST_IMG = 10

    #! load model and dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder, dataloader, info = load_model_dataset(filename,suffix,BATCH_SIZE,False)
    

    #! Create codebook
    for step, data in enumerate(tqdm(dataloader)):
        inputImg = data['image'].to(device)
        encoded, _ = autoencoder(inputImg)
        codeword = encoded.detach().to('cpu').numpy()
        cpose = data['cpose'].reshape((-1,16)).numpy()
        try:
            codebook = np.vstack((codebook,codeword))
            cposes = np.vstack((cposes,cpose))
        except NameError:
            print("init codebook")
            codebook = codeword
            cposes = cpose
    codedict = {'codebook':codebook, 'cposes':cposes}

    if suffix:
        direct = "test_codebook"
    else:
        direct = "train_codebook"
    save_dir = path.join( "/",*filename.split("/")[:-1], direct)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = path.join(save_dir,"{}.txt".format(filename.split("/")[-1].split(".")[0]))
    print("saving codebook to ", save_path)
    with open(save_path,"wb") as f:
        pickle.dump(codedict,f)

    print('Finished creating codebook')
    plt.show()


if __name__ == '__main__':
    create_codebook()
