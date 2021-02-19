import sys
from os import path
import os
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
from tqdm import tqdm,trange,tqdm_notebook
from datetime import datetime
now = datetime.now()


@click.command()
@click.argument('model')
@click.argument('dataset')
@click.argument('length', default=28)
@click.option('--augment', is_flag=True)
def main(model,dataset,length,augment):
    """
    model: linear, conv, convrgb0, convrgb1, convrgb2(final version)  
    dataset: mnist(deperaceted), 019_pitcher_base, 024_bowl, 037_scissors  
    [length]: resize the images before forwarding them to neural network  
    --augment: augment the training dataset    
    """

    # hyper parameters
    EPOCH = 100
    BATCH_SIZE = 64
    LR = 0.0001         # learning rate
    N_TEST_IMG = 10
    if N_TEST_IMG > BATCH_SIZE:
        N_TEST_IMG = BATCH_SIZE
        
    root_dir =  path.join("../data/", dataset)
    npy_file =  path.join(root_dir, "camera_poses.npy")
    pth_dir = "../checkpoints/{}_{}".format(dataset,now.strftime("%Y-%m-%d-%H-%M-%S"))
    img_dir = path.join(pth_dir,"img") 
    os.mkdir(pth_dir)
    os.mkdir(img_dir)
    print("Loading data from ", root_dir)
    print("Saving checkpoint to ", pth_dir)
    print("Saving images to ", img_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #! TODO Load and normalize the training dataset
    transform, transform_aug = get_transform(model=model, dataset=dataset, length=length, augment=augment)
    traindataset = ObjectPoseDataset(npy_file=npy_file, root_dir=root_dir, transform=transform, transform_aug=transform_aug)
    trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=4)


    #! TODO Define a Convolutional Neural Network
    autoencoder = get_model(model,length)
    # autoencoder = autoencoder.double()
    print(autoencoder)
    autoencoder.to(device)


    #! TODO Define a loss function
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)


    #! TODO Visualize the training process
    f, a = plt.subplots(3, N_TEST_IMG, figsize=(N_TEST_IMG*length, 3*length))
    plt.ion()   # continuously plot
    for i,data in enumerate(trainloader):
        sample_plot = data['image'][:N_TEST_IMG]
        sample_aug = data['image_aug'][:N_TEST_IMG]
    for i in range(N_TEST_IMG):
        imshow(sample_plot[i],a[1][i])
        imshow(sample_aug[i],a[0][i])
    plt.draw(); plt.pause(0.05)


    #! TODO Save checkpoints
    def save_checkponts():
        checkpoint = {'info':{'dataset':dataset, 'epoch':epoch, 'step':step, 'loss':loss, 'length':length, 'augment':augment},
            'autoencoder': autoencoder, 
            'transform': transform,
            'transform_aug': transform_aug,
            'state_dict': autoencoder.state_dict(),
            'optimizer' : optimizer.state_dict()}
        pth_file =  path.join(pth_dir,'checkpoint_{}_{}_aug{}_len{}_epo{}_step{}.pth').format(model,dataset,augment,length,epoch,step)
        torch.save(checkpoint,pth_file)


    #! TODO Train the network on the training data
    loss_list = []
    try:
        for epoch in range(EPOCH):
            pbar = tqdm(trainloader)
            for step, data in enumerate(pbar):
                inputImg = data['image_aug'].to(device)
                refImg = data['image'].to(device)
                encoded, decoded = autoencoder(inputImg)
                loss = loss_func(decoded, refImg)      # mean square error
                optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                optimizer.step()                    # apply gradients
                loss_list.append(loss.data.to('cpu').numpy())


                #! Visualize the result and save checkpoints
                if step % 100 == 0:
                    print('Epoch: {}/{} '.format(epoch,EPOCH), '| train loss: %.4f' % loss.data.to('cpu').numpy())                    
                    # pbar.set_description('Epoch: {}/{} '.format(epoch,EPOCH)+'| train loss: %.4f' % loss.data.to('cpu').numpy())
                    # plotting decoded image (second row)
                    _, decoded_data = autoencoder(sample_plot.to(device))
                    for i in range(N_TEST_IMG):
                        imshow(decoded_data[i],a[2][i])                
                    plt.draw(); plt.pause(0.05)

                    if step % 500 == 0:
                        def save_img():
                            img_file = path.join(img_dir,"epo{}_step{}_loss{}.png".format(epoch,step,loss.data.to('cpu').numpy()))
                            plt.savefig(img_file,format="png")
                        save_img()
            
            save_checkponts()
            

    except KeyboardInterrupt:
        save_checkponts()
        print("KeyboardInterrupt has been caught.")


    #! Visualize the loss
    save_img()
    plt.ioff()
    fig_loss = plt.figure();ax_loss = plt.axes()
    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("MSE loss")
    x_loss = np.linspace(0,len(loss_list),len(loss_list))
    loss_file = path.join(img_dir,"loss.png")
    ax_loss.plot(x_loss,loss_list)
    plt.savefig(loss_file,format="png")
    plt.show()

    print('Training finished!')


if __name__ == '__main__':
    main()