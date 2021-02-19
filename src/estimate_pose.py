import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
import signal
import click
import pickle
import cv2
# torch
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# my own functions
from utils import *
from offRenderer import render_images
from preprocess import ObjectPoseDataset
from other.tfaxis import *
from scipy.spatial.transform import Rotation as R
from create_codebook import create_codebook
from tqdm import tqdm


@click.command()
@click.argument('filename', type=click.Path())
@click.argument('suffix', required=False)
@click.option('--augment', is_flag=True)

def main(filename, suffix, augment):
    """
    filename: path to the input checkpoints file in directory ../data/models  
    [suffix]: test (only keyword test is supported)  
    --augment: augment the estimation set
    """
    # Hyper Parameters
    BATCH_SIZE = 64
    N_TEST_IMG = 10
    fig = plt.figure()
    plt.ion()
    ax = Axes3D(fig)
    ax.set_xlim3d([-2.0, 2.0])
    ax.set_ylim3d([-2.0, 2.0])
    ax.set_zlim3d([-2.0, 2.0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('TfAxis Slerp Interpoation')


    #! load model and dataset
    autoencoder, dataloader, info = load_model_dataset(filename,suffix,BATCH_SIZE,augment)
    autoencoder.to('cpu')
    
    codebook_path = path.join("/",*filename.split("/")[:-1],"train_codebook",filename.split("/")[-1].split(".")[0]+".txt")
    print("Loading codebook from ", codebook_path)

    if not path.exists(codebook_path):
        print("No codebook exists, please create codebook first.")
        sys.exit()
        
    with open(codebook_path,"rb") as f: 
        codedict = pickle.load(f)
    codebook = codedict['codebook']; cposes = codedict['cposes']
    path_mesh = '../data/raw/models/{}/textured_simple.obj'.format(info['dataset']) # path of mesh
    print("Loading 3d mesh file from ", path_mesh)

    #! find the best match in codebook according to cosine similarity
    vis = visualize_poseEstimation(path_mesh)
    error1_list = []
    error2_list = []
    iteration = 0
    try:
        for step, data in enumerate(dataloader): 
            # Forward the augmented dataset to the loaded network to generate codeword and decoded image 
            cwBatch, dcBatch = autoencoder(data['image_aug']) # get codeword and decoded images
            cwBatch = cwBatch.detach().numpy()
            matBatch = data['cpose'].numpy() # get ground truth of object pose

            # loop through (codeword, decoded image, ground truth pose, input augmented image) in one batch
            for cw, dc, matTrue,ip in zip(cwBatch,dcBatch,matBatch,data['image_aug']):
                iteration+=1
                # compute the best match of codeword in the codebook
                ind_cwMatch,cwMatch = max(enumerate(codebook), key=lambda x: cos_sim(x[1],cw))
                matMatch = cposes[ind_cwMatch].reshape(4,4)
                error = rotation_error(matTrue, matMatch)
                error1_list.append(error)
                print(iteration, "error between best match with codeword", error)

                # # compute the best match according to the difference of rotation
                # ind_rMatch,rMatch = min(enumerate(cposes), key=lambda x: rotation_error(x[1],matTrue))
                # rMatch = rMatch.reshape((4,4))
                # error2 = rotation_error(rMatch, matMatch)
                # error3 = rotation_error(rMatch, matTrue)
                # error2_list.append(error2)
                # print("error between best match with rotation matrix", error2)
                # print("error between best match with rotation matrix", error3)

                # # break
                # # visualize the images
                # ip = np.transpose(ip.numpy(), (1, 2, 0))
                # dc = dc.detach().to('cpu').numpy()
                # dc = np.transpose(dc, (1, 2, 0))
                # pic = vis.call(matTrue)
                # pic_match = vis.call(matMatch)
                # dc = cv2.cvtColor(dc, cv2.COLOR_RGB2BGR)
                # pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
                # pic_match = cv2.cvtColor(pic_match, cv2.COLOR_RGB2BGR)
                # ip = cv2.cvtColor(ip, cv2.COLOR_RGB2BGR)
                
                # cv2.imshow("rendered truth from the rotation matrix",pic)
                # cv2.imshow("best match according to codeword",pic_match)
                # cv2.imshow("decoded picture from codeword", dc)
                # cv2.imshow("ground truth of original picture",ip)
                # key = cv2.waitKey(10)

                # #visualize the ground truth pose and best match pose in 3d plotting
                # quatMatch = R.from_matrix(matMatch[:3,:3]).as_quat()
                # quatTrue = R.from_matrix(matTrue[:3,:3]).as_quat()
                # quatrMatch = R.from_matrix(rMatch[:3,:3]).as_quat()
                # tf1 = TfAxis(origin=matMatch[:3,3], quat=quatMatch, scale=0.4)
                # # tf2 = TfAxis(origin=matTrue[:3,3], quat=quatTrue, scale=0.4)
                # tf3 = TfAxis(origin=rMatch[:3,3], quat=quatrMatch, scale=0.4)
                # tf1.plot(ax)
                # # tf2.plot(ax)
                # tf3.plot(ax)
                # plt.draw()
                # plt.pause(0.5)
                # print("============================================================")
                # if key==27:    # Esc key to stop
                #     return
    except KeyboardInterrupt:
        print("keyboard interrupt")            

    print("average error between best match pose and ground truth pose is: ", np.mean(error1_list) )
    # print("average error between best match pose and pose with smallest rotation error is: ", np.mean(error2_list) )
    # print(error2_list.count(0),len(error2_list))
    # all_error = [[error1_list],[error2_list]]
    # np.save("../pitcherbase_aug.npy",all_error)

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()



# augment autoencoder on augment dataset : 0.097961389794742
# ... augment dataset :  0.05772636899230092
# nonaugment autoencoder on augment dataset : 2.471527949051328
# ... non-augmented 0.055561889671674206 


# aug ... aug ... :  0.502774630666909
# aug ... nonaug ... : 0.1276884973564403
# nonaug ... aug ... : 2.1878652177062166
# nonaug ... nonaug ... : 0.221245394454483


